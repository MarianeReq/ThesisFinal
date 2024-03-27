import sys
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QCalendarWidget, QCheckBox
from PyQt6.QtGui import QFont
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import ta

class TechnicalIndicatorSelection(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Technical Indicator Selection")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        # Main heading
        main_heading = QLabel("Technical Indicator Selection")
        main_heading.setFont(QFont("Arial", 20))  # Adjust "Arial" to your preferred font family
        layout.addWidget(main_heading)

        # Trend Indicators container
        self.trend_container = self.create_indicator_container("Trend Indicators", ["SMA", "EMA", "MACD", "ADX", "CCI"])
        layout.addWidget(self.trend_container)

        # Momentum Indicators container
        self.momentum_container = self.create_indicator_container("Momentum Indicators", ["RSI", "Stochastic Oscillator", "Williams %R"])
        layout.addWidget(self.momentum_container)

        # Volume Indicators container
        self.volume_container = self.create_indicator_container("Volume Indicators", ["OBV", "A/D Line"])
        layout.addWidget(self.volume_container)

        # Volatility Indicators container
        self.volatility_container = self.create_indicator_container("Volatility Indicators", ["ATR", "Bollinger Bands", "Keltner's Channel"])
        layout.addWidget(self.volatility_container)

        # Submit button
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.on_submit) 
        layout.addWidget(self.on_submit)

        self.setLayout(layout)

    def on_submit(self):
          # Clear the list before gathering new checked indicators
        self.checked_indicators.clear()

        # Gather the checked technical indicators from each container
        for container in [self.trend_container, self.momentum_container, self.volume_container, self.volatility_container]:
            category_name = container.title()
            for checkbox in container.findChildren(QCheckBox):
                if checkbox.isChecked():
                    indicator_name = checkbox.text()
                    self.checked_indicators.append((category_name, indicator_name))

        # Print the checked indicators
        print("Checked Indicators for Prediction:", self.checked_indicators)


   
    def create_indicator_container(self, heading, indicators):
        container = QWidget()
        container.setObjectName("indicator-container")

        heading_label = QLabel(heading)
        indicators_checkboxes = [QCheckBox(indicator) for indicator in indicators]

        container_layout = QVBoxLayout()
        container_layout.addWidget(heading_label)

        for checkbox in indicators_checkboxes:
            container_layout.addWidget(checkbox)

        container.setLayout(container_layout)
        return container

class DatePickerView(QWidget):
    def __init__(self, is_start_date_picker, stock):
        super().__init__()

        self.is_start_date_picker = is_start_date_picker
        self.stock = stock

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

       #Select TI button
        self.ti_selection_button = QPushButton("Select TIs", self)
        self.ti_selection_button.clicked.connect(self.go_to_ti_selection)
        layout.addWidget(self.ti_selection_button)
       
        # Add predict button (initially disabled)
        # Enable if both dates != null
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button, alignment=Qt.AlignmentFlag.AlignRight)

        # Add reset button
        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.reset_dates)
        layout.addWidget(self.reset_button)

        self.setLayout(layout)

        # Member variables to track selected dates
        self.start_date_selected = False
        self.end_date_selected = False
        
        # Connect slot to restrict date selection
        self.calendar.selectionChanged.connect(self.restrict_date_selection)
        

    def go_to_ti_selection(self):
        
        # Create an instance of the TechnicalIndicatorSelection widget
        self.ti_selection_widget = TechnicalIndicatorSelection()

        # Show the TechnicalIndicatorSelection widget
        self.ti_selection_widget.show()
    
    


    def restrict_date_selection(self):
        selected_date = self.calendar.selectedDate()
        current_date = QDate.currentDate()
        if selected_date > current_date:
            QMessageBox.warning(self, "Warning", "Please select a date not greater than the current date.")
            self.calendar.setSelectedDate(current_date)

    def update_selected_dates(self):
        selected_date = self.calendar.selectedDate()
        if self.is_start_date_picker:
            self.start_date_label.setText("Start Date: " + selected_date.toString(Qt.DateFormat.ISODate))
            self.start_date_selected = True
            self.close()  # Close the calendar after selecting the start date
            self.is_start_date_picker = False  # Change to end date picker mode
            end_date_picker = DatePickerView(False, self.stock)
            end_date_picker.setWindowTitle("End Date Picker")
            end_date_picker.show()
        else:
            self.end_date_label.setText("End Date: " + selected_date.toString(Qt.DateFormat.ISODate))
            self.end_date_selected = True
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


    def predict(self):
        if not (self.start_date_selected and self.end_date_selected):
            QMessageBox.warning(self, "Warning", "Please select both start and end dates.")
            return

        # Download historical data from Yahoo Finance
        stock_data = yf.download(self.stock, start=self.start_date_label.text()[len("Start Date: "):], end=self.end_date_label.text()[len("End Date: "):])

        # Selecting relevant features
        features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

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

        if ("Trend Indicators", "SMA") in self.checked_indicators:
            # Calculate SMA
            sma_window = 20 
            stock_data["SMA"] = ta.trend.sma_indicator(close=stock_data["Close"], window=sma_window)

            last_sma = stock_data["SMA"].iloc[-1]

            if last_sma > stock_data["Close"].iloc[-1]:
                print("SMA predicts upward movement")
            else:
                print("SMA predicts downward movement")


        if ("Trend Indicators", "EMA") in self.checked_indicators:
            # Calculate EMA
            ema_window = 20  
            stock_data["EMA"] = ta.trend.ema_indicator(close=stock_data["Close"], window=ema_window)

            # Extracting the last observed EMA value
            last_ema = stock_data["EMA"].iloc[-1]

            # Determine if the predicted trend is upward or downward for EMA
            if last_ema > stock_data["Close"].iloc[-1]:
                print("EMA predicts upward movement")
            else:
                print("EMA predicts downward movement")
        
        if ("Trend Indicators", "MACD") in self.checked_indicators:
            # Calculate MACD
            macd = ta.trend.macd(close=stock_data["Close"], window_slow=26, window_fast=12)
            stock_data["MACD"] = macd.macd()
            stock_data["MACD_signal"] = macd.macd_signal()

            last_macd = stock_data["MACD"].iloc[-1]
            last_macd_signal = stock_data["MACD_signal"].iloc[-1]

            # Determine if the predicted trend is upward or downward for MACD
            if last_macd > last_macd_signal:
                print("MACD predicts upward movement")
            else:
                print("MACD predicts downward movement")
        
        if("Trend Indicators", "ADX") in self.checked_indicators:
            # Calculate Average Directional Index (ADX)
            adx_period = 14  # Example period for ADX calculation
            stock_data["ADX"] = ta.trend.adx(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], window=adx_period)

            # Extracting the last observed ADX value
            last_adx = stock_data["ADX"].iloc[-1]

            # Interpret ADX for upward or downward movement
            if last_adx > 25:
                print("Strong trend (ADX > 25)")
            else:
                print("Weak trend or no clear trend (ADX <= 25)") 

        if("Trend Indicators", "CCI") in self.checked_indicators:
            # Calculate Commodity Channel Index (CCI)
            cci_period = 20  # Example period for CCI calculation
            stock_data["CCI"] = ta.trend.cci(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], window=cci_period)

            # Extracting the last observed CCI value
            last_cci = stock_data["CCI"].iloc[-1]

            # Interpret CCI for upward or downward movement
            if last_cci > 0:
                print("Upward (CCI > 0)") 
            elif last_cci < 0:
                print("Downward (CCI < 0)")
            else:
                print("No clear directional signal (CCI = 0)")

        if ("Momentum Indicators", "Williams %R") in self.checked_indicators:
            williams_r_period = 14  # Example period for Williams %R
            stock_data["Williams %R"] = ta.momentum.WilliamsRIndicator(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], lbp=williams_r_period)

            # Extracting the last observed Williams %R value
            last_williams_r = stock_data["Williams %R"].iloc[-1]

            if last_williams_r < -50:
                print("Williams %R indicates upward movement")
            elif last_williams_r > -50:
                print("Williams %R indicates downward movement")
            else:
                print("Williams %R is neutral")

        if ("Momentum Indicators", "RSI") in self.checked_indicators:
            # Calculate RSI
            rsi_period = 14  # Example period for RSI calculation
            stock_data["RSI"] = ta.momentum.RSIIndicator(close=stock_data["Close"], window=rsi_period)

            # Extracting the last observed RSI value
            last_rsi = stock_data["RSI"].iloc[-1]

            # Interpret the RSI value for upward or downward movement
            if last_rsi > 70:
                print("RSI indicates overbought and predicts downward movement")
            elif last_rsi < 30:
                print("RSI indicates oversold and predicts upward movement")
            else:
                print("RSI is within normal range and does not provide a clear prediction")
        
        if ("Momentum Indicators", "Stochastic Oscillator") in self.checked_indicators:
            # Calculate Stochastic Oscillator
            stoch_period = 14  # Example period for Stochastic Oscillator calculation
            stock_data["%K"], stock_data["%D"] = ta.momentum.stoch(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], window=stoch_period)

            # Extracting the last observed %K and %D values
            last_percent_k = stock_data["%K"].iloc[-1]
            last_percent_d = stock_data["%D"].iloc[-1]

            # Interpret the Stochastic Oscillator for upward or downward movement
            if last_percent_k > last_percent_d and last_percent_k > 80:
                print("Stochastic Oscillator indicates overbought and potential downward movement")
            elif last_percent_k < last_percent_d and last_percent_k < 20:
                print("Stochastic Oscillator indicates oversold and potential upward movement")
            else:
                print("Stochastic Oscillator is within normal range and does not provide a clear directional signal")
        
        if ("Volume Indicators", "OBV") in self.checked_indicators:
            stock_data["OBV"] = ta.volume.on_balance_volume(close=stock_data["Close"], volume=stock_data["Volume"])

            # Extracting the last observed OBV value
            last_obv = stock_data["OBV"].iloc[-1]

            # Extracting the previous OBV value for comparison
            previous_obv = stock_data["OBV"].iloc[-2]

            # Interpret the OBV for upward or downward movement
            if last_obv > previous_obv:
                print("OBV indicates upward movement")
            elif last_obv < previous_obv:
                print("OBV indicates downward movement")
            else:
                print("OBV is neutral")

        if ("Volume Indicators", "AD Line") in self.checked_indicators:
            # Calculate Accumulation/Distribution Line (A/D Line)
            stock_data["A/D Line"] = ta.volume.acc_dist_index(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], volume=stock_data["Volume"])

            # Extracting the last observed A/D Line value
            last_ad_line = stock_data["A/D Line"].iloc[-1]

            # Extracting the previous A/D Line value for comparison
            previous_ad_line = stock_data["A/D Line"].iloc[-2]

            # Interpret the A/D Line for upward or downward movement
            if last_ad_line > previous_ad_line:
                print("A/D Line indicates upward movement")
            elif last_ad_line < previous_ad_line:
                print("A/D Line indicates downward movement")
            else:
                print("A/D Line is neutral")
        
        if ("Volatility Indicators", "ATR") in self.checked_indicators:
            # Calculate Average True Range (ATR)
            atr_period = 14  # Example period for ATR calculation
            stock_data["ATR"] = ta.volatility.average_true_range(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], window=atr_period)

            # Extracting the last observed ATR value
            last_atr = stock_data["ATR"].iloc[-1]

            # Extracting the previous ATR value for comparison
            previous_atr = stock_data["ATR"].iloc[-2]  # Assuming you have at least two ATR values in your dataset

            # Interpret the ATR for upward or downward movement
            if last_atr > previous_atr:
                print("ATR indicates increased volatility, which may lead to a larger price movement (upward or downward)")
            else:
                print("ATR indicates normal or decreased volatility, which may result in smaller price movements or consolidation")

        
        if ("Volatility Indicators", "Bollinger Bands") in self.checked_indicators:
            # Calculate Bollinger Bands
            bollinger_period = 20  # Example period for Bollinger Bands calculation
            stock_data["BB_upper"], stock_data["BB_middle"], stock_data["BB_lower"] = ta.volatility.bollinger_hband(close=stock_data["Close"], window=bollinger_period, std=2)
            stock_data["BB_width"] = stock_data["BB_upper"] - stock_data["BB_lower"]

            # Extracting the last observed Bollinger Bands values
            last_bb_upper = stock_data["BB_upper"].iloc[-1]
            last_bb_lower = stock_data["BB_lower"].iloc[-1]
            last_bb_width = stock_data["BB_width"].iloc[-1]

            # Interpret the Bollinger Bands for upward or downward movement
            if last_bb_upper > last_bb_upper.shift(1) and last_bb_lower < last_bb_lower.shift(1):
                print("Bollinger Bands indicate a potential upward breakout")
            elif last_bb_upper < last_bb_upper.shift(1) and last_bb_lower > last_bb_lower.shift(1):
                print("Bollinger Bands indicate a potential downward breakout")
            else:
                print("Bollinger Bands are within normal range and do not provide a clear directional signal")
        
        if("Volatility Indicators", "Keltner's Channel") in self.checked_indicators:
            # Calculate Keltner Channel
            keltner_period = 20  # Example period for Keltner Channel calculation
            stock_data["upper_band"], stock_data["middle_band"], stock_data["lower_band"] = ta.volatility.keltner_channel(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], window=keltner_period)

            # Extracting the last observed upper band, middle band, and lower band values
            last_upper_band = stock_data["upper_band"].iloc[-1]
            last_middle_band = stock_data["middle_band"].iloc[-1]
            last_lower_band = stock_data["lower_band"].iloc[-1]

            # Extracting the last observed close price
            last_close_price = stock_data["Close"].iloc[-1]

            # Interpret Keltner Channel for upward or downward movement
            if last_close_price > last_upper_band:
                print("Price is above the upper band of Keltner Channel, indicating potential downward movement")
            elif last_close_price < last_lower_band:
                print("Price is below the lower band of Keltner Channel, indicating potential upward movement")
            else:
                print("Price is within the Keltner Channel bands, suggesting a neutral or ranging market")



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

        self.alaay_button = QPushButton("ALAAY", self)
        self.set_button_style(self.alaay_button)
        self.alaay_button.clicked.connect(lambda: self.open_date_picker_for_stock('AYAAY'))  # Pass 'AYAAY' as argument
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
        # Disable the other button
        if stock == 'RBLAY':
            self.alaay_button.setEnabled(False)
            self.set_button_style(self.alaay_button, enabled=False)
            other_stock = 'AYAAY'
        else:
            self.rblay_button.setEnabled(False)
            self.set_button_style(self.rblay_button, enabled=False)
            other_stock = 'RBLAY'

        # Create an instance of DatePickerView if not already created
        if not self.date_picker_view:
            self.date_picker_view = DatePickerView(True, stock)

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
