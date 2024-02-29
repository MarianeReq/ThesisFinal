import sys
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QCalendarWidget

class DatePickerView(QWidget):
    def __init__(self, is_start_date_picker):
        super().__init__()

        self.is_start_date_picker = is_start_date_picker

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

        # Add reset button
        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.reset_dates)
        layout.addWidget(self.reset_button)

        self.setLayout(layout)

    def update_selected_dates(self):
        selected_date = self.calendar.selectedDate()
        if self.is_start_date_picker:
            self.start_date_label.setText("Start Date: " + selected_date.toString(Qt.DateFormat.ISODate))
            self.close()  # Close the calendar after selecting the start date
            self.is_start_date_picker = False  # Change to end date picker mode
            end_date_picker = DatePickerView(False)
            end_date_picker.setWindowTitle("End Date Picker")
            end_date_picker.show()
        else:
            self.end_date_label.setText("End Date: " + selected_date.toString(Qt.DateFormat.ISODate))
            self.close()  # Close the calendar after selecting the end date

    def reset_dates(self):
        self.calendar.setSelectedDate(QDate.currentDate())
        self.start_date_label.setText("Start Date: ")
        self.end_date_label.setText("End Date: ")


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

        rlc_button = QPushButton("RLC", self)
        rlc_button.setStyleSheet("""
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
        layout.addWidget(rlc_button)

        alaay_button = QPushButton("ALAAY", self)
        alaay_button.setStyleSheet("""
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
        alaay_button.clicked.connect(self.open_date_picker)
        layout.addWidget(alaay_button)

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

    def open_date_picker(self):
        # Create an instance of DatePickerView if not already created
        if not self.date_picker_view:
            self.date_picker_view = DatePickerView(True)

        # Show DatePickerView
        self.date_picker_view.setWindowTitle("Start Date Picker")
        self.date_picker_view.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
