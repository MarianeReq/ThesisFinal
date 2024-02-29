import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtQuick import QQuickView

if __name__ == "__main__":
    app = QApplication(sys.argv)
    view = QQuickView()
    view.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
    view.setSource("your_qml_file.qml")
    view.show()
    sys.exit(app.exec())
