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
            "2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08",
            "2021-01-09", "2021-01-10", "2021-01-11", "2021-01-12", "2021-01-13", "2021-01-14", "2021-01-15", "2021-01-16",
            "2021-01-17", "2021-01-18", "2021-01-19", "2021-01-20", "2021-01-21", "2021-01-22", "2021-01-23", "2021-01-24",
            "2021-01-25", "2021-01-26", "2021-01-27", "2021-01-28", "2021-01-29", "2021-01-30", "2021-01-31", "2021-02-01",
            "2021-02-02", "2021-02-03", "2021-02-04", "2021-02-05", "2021-02-06", "2021-02-07", "2021-02-08", "2021-02-09",
            "2021-02-10", "2021-02-11", "2021-02-12", "2021-02-13", "2021-02-14", "2021-02-15", "2021-02-16", "2021-02-17",
            "2021-02-18", "2021-02-19", "2021-02-20", "2021-02-21", "2021-02-22", "2021-02-23", "2021-02-24", "2021-02-25",
            "2021-02-26", "2021-02-27", "2021-02-28", "2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04", "2021-03-05",
            "2021-03-06", "2021-03-07", "2021-03-08", "2021-03-09", "2021-03-10", "2021-03-11", "2021-03-12", "2021-03-13",
            "2021-03-14", "2021-03-15", "2021-03-16", "2021-03-17", "2021-03-18", "2021-03-19", "2021-03-20", "2021-03-21",
            "2021-03-22", "2021-03-23", "2021-03-24", "2021-03-25", "2021-03-26"
        ]
        end_dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08",
            "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-13", "2024-01-14", "2024-01-15", "2024-01-16",
            "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-20", "2024-01-21", "2024-01-22", "2024-01-23", "2024-01-24",
            "2024-01-25", "2024-01-26", "2024-01-27", "2024-01-28", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01",
            "2024-02-02", "2024-02-03", "2024-02-04", "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-09",
            "2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
            "2024-02-18", "2024-02-19", "2024-02-20", "2024-02-21", "2024-02-22", "2024-02-23", "2024-02-24", "2024-02-25",
            "2024-02-26", "2024-02-27", "2024-02-28", "2024-02-29", "2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04",
            "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08", "2024-03-09", "2024-03-10", "2024-03-11", "2024-03-12",
            "2024-03-13", "2024-03-14", "2024-03-15", "2024-03-16", "2024-03-17", "2024-03-18", "2024-03-19", "2024-03-20",
            "2024-03-21", "2024-03-22", "2024-03-23", "2024-03-24", "2024-03-25", "2024-03-26"
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
    date_picker_view = DatePickerView("RBLAY")
    sys.exit(app.exec())

