import sys
import yfinance as yf
import numpy as np
import ta
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

class DatePickerView(QWidget):
    def __init__(self, stock):
        super().__init__()

        self.stock = stock

        layout = QVBoxLayout()

        # Add labels for displaying selected dates
        self.start_date_label = QLabel("Start Date: ", self)
        layout.addWidget(self.start_date_label)

        self.end_date_label = QLabel("End Date: ", self)
        layout.addWidget(self.end_date_label)

        self.setLayout(layout)

        # Predict for each date
        self.predict_for_dates()

    def predict_for_dates(self):
        start_dates = [
            "01/01/2014", "02/01/2014", "03/01/2014", "04/01/2014", "05/01/2014", "06/01/2014", "07/01/2014", "08/01/2014",
            "09/01/2014", "10/01/2014", "11/01/2014", "12/01/2014", "13/01/2014", "14/01/2014", "15/01/2014", "16/01/2014",
            "17/01/2014", "18/01/2014", "19/01/2014", "20/01/2014", "21/01/2014", "22/01/2014", "23/01/2014", "24/01/2014",
            "25/01/2014", "26/01/2014", "27/01/2014", "28/01/2014", "29/01/2014", "30/01/2014", "31/01/2014", "01/02/2014",
            "02/02/2014", "03/02/2014", "04/02/2014", "05/02/2014", "06/02/2014", "07/02/2014", "08/02/2014", "09/02/2014",
            "10/02/2014", "11/02/2014", "12/02/2014", "13/02/2014", "14/02/2014", "15/02/2014", "16/02/2014", "17/02/2014",
            "18/02/2014", "19/02/2014", "20/02/2014", "21/02/2014", "22/02/2014", "23/02/2014", "24/02/2014", "25/02/2014",
            "26/02/2014", "27/02/2014", "28/02/2014", "01/03/2014", "02/03/2014", "03/03/2014", "04/03/2014", "05/03/2014",
            "06/03/2014", "07/03/2014", "08/03/2014", "09/03/2014", "10/03/2014", "11/03/2014", "12/03/2014", "13/03/2014",
            "14/03/2014", "15/03/2014", "16/03/2014", "17/03/2014", "18/03/2014", "19/03/2014", "20/03/2014", "21/03/2014",
            "22/03/2014", "23/03/2014", "24/03/2014", "25/03/2014", "26/03/2014"
        ]

        end_dates = [
            "01/01/2024", "02/01/2024", "03/01/2024", "04/01/2024", "05/01/2024", "06/01/2024", "07/01/2024", "08/01/2024",
            "09/01/2024", "10/01/2024", "11/01/2024", "12/01/2024", "13/01/2024", "14/01/2024", "15/01/2024", "16/01/2024",
            "17/01/2024", "18/01/2024", "19/01/2024", "20/01/2024", "21/01/2024", "22/01/2024", "23/01/2024", "24/01/2024",
            "25/01/2024", "26/01/2024", "27/01/2024", "28/01/2024", "29/01/2024", "30/01/2024", "31/01/2024", "01/02/2024",
            "02/02/2024", "03/02/2024", "04/02/2024", "05/02/2024", "06/02/2024", "07/02/2024", "08/02/2024", "09/02/2024",
            "10/02/2024", "11/02/2024", "12/02/2024", "13/02/2024", "14/02/2024", "15/02/2024", "16/02/2024", "17/02/2024",
            "18/02/2024", "19/02/2024", "20/02/2024", "21/02/2024", "22/02/2024", "23/02/2024", "24/02/2024", "25/02/2024",
            "26/02/2024", "27/02/2024", "28/02/2024", "29/02/2024", "01/03/2024", "02/03/2024", "03/03/2024", "04/03/2024",
            "05/03/2024", "06/03/2024", "07/03/2024", "08/03/2024", "09/03/2024", "10/03/2024", "11/03/2024", "12/03/2024",
            "13/03/2024", "14/03/2024", "15/03/2024", "16/03/2024", "17/03/2024", "18/03/2024", "19/03/2024", "20/03/2024",
            "21/03/2024", "22/03/2024", "23/03/2024", "24/03/2024", "25/03/2024", "26/03/2024"
        ]

        # Initialize lists to hold predictions and dates
        predictions = []

        # Loop through each start and end date, predict and append results
        for start_date, end_date in zip(start_dates, end_dates):
            stock_data = yf.download(self.stock, start=start_date, end=end_date)
            stock_data = ta.add_all_ta_features(stock_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
            features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)
            X = scaled_features[:-1]
            y = scaled_features[1:, 0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            model = Sequential()
            model.add(LSTM(units=500, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.50))
            model.add(LSTM(units=500, return_sequences=True))
            model.add(Dropout(0.25))
            model.add(LSTM(units=500))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
            predicted_open_price_scaled = model.predict(X_test[-1:])
            predicted_open_price = predicted_open_price_scaled * (features['Open'].max() - features['Open'].min()) + features['Open'].min()
            predictions.append(predicted_open_price.item())

        # Display results in a columnar format
        print("Start Date\tEnd Date\tPredicted Value")
        for start_date, end_date, prediction in zip(start_dates, end_dates, predictions):
            print(f"{start_date}\t{end_date}\t{prediction}")

        # Close the application after displaying predictions
        QApplication.instance().quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    date_picker_view = DatePickerView("AYAAF")
    sys.exit(app.exec())

