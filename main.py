import sys
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import ta
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QCalendarWidget, QCheckBox, QGroupBox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

class DatePickerView(QWidget):
    def __init__(self, is_start_date_picker, stock, selected_indicators):
        super().__init__()

        self.is_start_date_picker = is_start_date_picker
        self.stock = stock
        self.selected_indicators = selected_indicators

        self.indicator_checkboxes = {}  # Initialize indicator_checkboxes dictionary

        layout = QVBoxLayout()

        # Add labels for displaying selected dates
        self.start_date_label = QLabel("Start Date: ", self)
        layout.addWidget(self.start_date_label)

        self.end_date_label = QLabel("End Date: ", self)
        layout.addWidget(self.end_date_label)

        # Add calendar widget for selecting dates
        self.calendar = QCalendarWidget(self)
        self.calendar.selectionChanged.connect(self.update_selected_dates)
        layout.addWidget(self.calendar)

        # Group boxes to organize indicator checkboxes
        indicator_groupboxes = {}

        # Add indicator selection checkboxes
        indicator_types = {
            'Trend': ['Simple Moving Average (SMA)', 'Exponential Moving Average (EMA)', 'Moving Average Convergence Divergence (MACD)', 'Average Directional Index (ADX)'],
            'Momentum': ['Relative Strength Index (RSI)', 'Stochastic Oscillator', 'Williams %R'],
            'Volume': ['On Balance Volume (OBV)', 'Accumulation/Distribution Line (A/D Line)'],
            'Volatility': ['Average True Range (ATR)', 'Bollinger Bands Middle (BBM)', 'Bollinger Bands High (BBH)', 'Bollinger Bands Low (BBL)', 'Keltner’s Channel Center (KCC)', 'Keltner’s Channel High (KCH)', 'Keltner’s Channel Low (KCL)']
        }

        for category, indicators in indicator_types.items():
            indicator_groupboxes[category] = QGroupBox(category)
            indicator_layout = QVBoxLayout()
            for indicator in indicators:
                checkbox = QCheckBox(indicator, self)
                checkbox.setChecked(indicator in self.selected_indicators)
                indicator_layout.addWidget(checkbox)
                self.indicator_checkboxes[indicator] = checkbox  # Add checkbox to indicator_checkboxes dictionary
            indicator_groupboxes[category].setLayout(indicator_layout)
            layout.addWidget(indicator_groupboxes[category])

        # Add predict button (initially disabled)
        # Enable if both dates != null
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button, alignment=Qt.AlignmentFlag.AlignRight)

        # Add select all button below the predict button
        self.select_all_button = QPushButton("Select All", self)
        self.select_all_button.clicked.connect(self.select_all)
        layout.addWidget(self.select_all_button, alignment=Qt.AlignmentFlag.AlignRight)

        # Add reset button
        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.reset_dates)
        layout.addWidget(self.reset_button, alignment=Qt.AlignmentFlag.AlignRight)

        self.setLayout(layout)

        # Member variables to track selected dates
        self.start_date_selected = False
        self.end_date_selected = False
        
        # Connect slot to restrict date selection
        self.calendar.selectionChanged.connect(self.restrict_date_selection)

    def select_all(self):
        for checkbox in self.indicator_checkboxes.values():
            checkbox.setChecked(True)

    def restrict_date_selection(self):
        selected_date = self.calendar.selectedDate()
        current_date = QDate.currentDate()
        if selected_date > current_date:
            QMessageBox.warning(self, "Warning", "Please select a date not greater than the current date.")
            self.calendar.setSelectedDate(current_date)
        

    def update_selected_dates(self):
        selected_date = self.calendar.selectedDate()
        
        # Check if it's start date picker mode
        if self.is_start_date_picker:
            self.start_date_label.setText("Start Date: " + selected_date.toString(Qt.DateFormat.ISODate))
            self.start_date_selected = True
            self.start_date = selected_date  # Set the start date
            self.close()  # Close the calendar after selecting the start date
            self.is_start_date_picker = False  # Change to end date picker mode
            end_date_picker = DatePickerView(False, self.stock, self.selected_indicators)
            end_date_picker.setWindowTitle("End Date Picker")
            end_date_picker.show()
        else:
            self.end_date_label.setText("End Date: " + selected_date.toString(Qt.DateFormat.ISODate))
            self.end_date_selected = True
            self.end_date = selected_date  # Set the end date
            self.close()  # Close the calendar after selecting the end date
            self.is_start_date_picker = True  # Reset to start date picker mode

        # Enable predict button if both start and end dates are selected
        if self.start_date_selected and self.end_date_selected:
            self.predict_button.setEnabled(True)


    def reset_dates(self):
        self.calendar.setSelectedDate(QDate.currentDate())
        self.start_date_label.setText("Start Date: ")
        self.end_date_label.setText("End Date: ")
        self.start_date_selected = False
        self.end_date_selected = False
        self.predict_button.setEnabled(False)
        self.is_start_date_picker = True  # Reset to start date picker mode

        # Reset all checkbox states to unchecked
        for checkbox in self.indicator_checkboxes.values():
            checkbox.setChecked(False)




    def predict(self):
        if not (self.start_date_selected and self.end_date_selected):
            QMessageBox.warning(self, "Warning", "Please select both start and end dates.")
            return

        # Download historical data from Yahoo Finance
        stock_data = yf.download(self.stock, start=self.start_date_label.text()[len("Start Date: "):], end=self.end_date_label.text()[len("End Date: "):])

        # Calculate technical indicators
        stock_data = ta.add_all_ta_features(stock_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

        # Selecting relevant features including technical indicators
        features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Drop selected indicators not included
        selected_indicators = [indicator for indicator, checkbox in self.indicator_checkboxes.items() if checkbox.isChecked()]
        for indicator in selected_indicators:
            stock_data.drop(columns=[f"{indicator.lower()}_{param}" for param in ta.utils.dropna(stock_data) if f"{indicator.lower()}_{param}" in stock_data.columns], inplace=True)

        # Print column names to verify correct column names
        print(stock_data.columns)

        # Scaling features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        # Define input and target data
        X = scaled_features[:-1]  # Input features
        y = scaled_features[1:, 0]  # Target variable (open price shifted by 1 day)

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Reshape data for LSTM input
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # Building the LSTM model
        model = Sequential()
        model.add(LSTM(units=500, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.50))
        model.add(LSTM(units=500, return_sequences=True))
        model.add(Dropout(0.25))
        model.add(LSTM(units=500))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compiling the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Define early stopping
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

        # Training the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # Predicting the open price
        predicted_open_price_scaled = model.predict(X_test[-1:])
        predicted_open_price = predicted_open_price_scaled * (features['Open'].max() - features['Open'].min()) + features['Open'].min()

        # Extracting the open prices for the last week
        recent_open_prices = stock_data['Open']

        # Extracting the last observed open price
        last_observed_open_price = recent_open_prices.iloc[-1]

        # Determine if the predicted trend is upward or downward
        trend = "Upward" if predicted_open_price > last_observed_open_price else "Downward"

        print("Predicted Open Price:", predicted_open_price)
        print("Trend:", trend)

        # Plotting the recent open prices
        plt.figure(figsize=(10, 6))
        plt.plot(recent_open_prices.index, recent_open_prices.values, label='Recent Open Prices', marker='o')

        # Plotting the predicted open price
        plt.axhline(y=predicted_open_price, color='r', linestyle='--', label=f'Predicted Open Price ({trend} trend)')

        # Adding labels and title
        plt.xlabel('Date')
        plt.ylabel('Open Price')
        plt.title('Recent Open Prices and Predicted Open Price')
        plt.legend()

        # Rotating x-axis labels for better readability
        plt.xticks(rotation=45)

        # Displaying the plot
        plt.grid(True)
        plt.tight_layout()
        plt.show()




class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stock Prediction Model")
        self.resize(800, 600)
        self.setStyleSheet("""
            background-color: #aa557f;
        """)

        layout = QVBoxLayout()

        stckprdmodel = QLabel("Stock Prediction Model", self)
        stckprdmodel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stckprdmodel.setStyleSheet("""
            color: #ffffff;
            font-size: 70px;
            font-family: Verdana;
        """)
        layout.addWidget(stckprdmodel)

        select_stock_label = QLabel("Select a stock", self)
        select_stock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        select_stock_label.setStyleSheet("""
            color: #ffffff;
            font-size: 45px;
            font-weight: bold;
        """)
        layout.addWidget(select_stock_label)

        # Create stock buttons
        self.rblay_button = QPushButton("RBLAY", self)
        self.set_button_style(self.rblay_button)
        self.rblay_button.clicked.connect(lambda: self.open_date_picker_for_stock('RBLAY'))  # Pass 'RBLAY' as argument
        layout.addWidget(self.rblay_button)

        self.alaay_button = QPushButton("AYAAF", self)
        self.set_button_style(self.alaay_button)
        self.alaay_button.clicked.connect(lambda: self.open_date_picker_for_stock('AYAAF'))  # Pass 'AYAAF' as argument
        layout.addWidget(self.alaay_button)

        footer_label = QLabel("The Survivors (?)", self)
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_label.setStyleSheet("""
            color: #ffffff;
            font-size: 22px;
            font-weight: bold;
        """)
        layout.addWidget(footer_label)

        self.setLayout(layout)

        # Member variable to hold DatePickerView instance
        self.date_picker_view = None

    def open_date_picker_for_stock(self, stock):
        selected_indicators = []
        if self.date_picker_view:
            selected_indicators = self.date_picker_view.selected_indicators
            self.date_picker_view.close()

        # Disable the other button
        if stock == 'RBLAY':
            self.alaay_button.setEnabled(False)
            self.set_button_style(self.alaay_button, enabled=False)
            other_stock = 'AYAAF'
        else:
            self.rblay_button.setEnabled(False)
            self.set_button_style(self.rblay_button, enabled=False)
            other_stock = 'RBLAY'

        # Create an instance of DatePickerView if not already created
        if not self.date_picker_view:
            self.date_picker_view = DatePickerView(True, stock, selected_indicators)

        # Show DatePickerView
        self.date_picker_view.setWindowTitle("Start Date Picker")
        self.date_picker_view.show()

    def reset_buttons(self):
        self.rblay_button.setEnabled(True)
        self.set_button_style(self.rblay_button, enabled=True)
        self.alaay_button.setEnabled(True)
        self.set_button_style(self.alaay_button, enabled=True)

    def set_button_style(self, button, enabled=True):
        if enabled:
            button.setStyleSheet("""
                QPushButton {
                    color: #ffffff;
                    background-color: #342d4f;
                    border-radius: 20px;
                    font-size: 36px;
                    font-weight: bold;
                    padding: 10px 20px;
                }
                QPushButton:hover {
                    background-color: #483f62;
                }
                QPushButton:pressed {
                    background-color: #241e32;
                }
            """)
        else:
            button.setStyleSheet("""
                QPushButton {
                    color: #ffffff;
                    background-color: #888888;
                    border-radius: 20px;
                    font-size: 36px;
                    font-weight: bold;
                    padding: 10px 20px;
                }
            """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())