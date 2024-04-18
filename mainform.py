from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QMainWindow, QProgressBar, QAction, \
    QVBoxLayout, QMessageBox, QDialog, QFileDialog, QCheckBox, QTextEdit, QPushButton, QToolTip
from PyQt5.QtWidgets import QLineEdit, QTreeWidgetItem, QTreeWidget, QInputDialog
from PyQt5.QtCore import QCoreApplication, Qt, QAbstractTableModel, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap

from forms.main_form import Ui_MainWindow
from forms.input import Ui_Dialog


class MainForm(QMainWindow):
    """Главное окно программы"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('Получение субтитров по картинке')

        self.ui.action.triggered.connect(self.open_input)

    def open_input(self):
        self.dialog_input = InputDialog()  # Создаем экземпляр дочернего диалогового окна
        self.dialog_input.show()
        self.dialog_input.ok_button_clicked_input.connect(self.import_files)

    def import_files(self):
        self.dialog_input.close()
        # Загружаем изображение из файла
        pixmap = QPixmap(self.dialog_input.ui_input.coeff_file_input_line.text())
        pixmap = pixmap.scaledToWidth(self.ui.Photo.width())

        # Устанавливаем изображение в QLabel
        self.ui.Photo.setPixmap(pixmap)

        # Устанавливаем размеры QLabel в соответствии с изображением
        self.ui.Photo.resize(pixmap.width(), pixmap.height())


class InputDialog(QDialog):
    """Класс для первого диалогового окна (импорт данных)"""
    ok_button_clicked_input = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui_input = Ui_Dialog()
        self.ui_input.setupUi(self)

        # обозреватель проводника для выбора файлов
        self.ui_input.toolButton_coeff.clicked.connect(lambda: self.on_tool_button_clicked(self.ui_input.coeff_file_input_line))

        self.ui_input.buttonBox.accepted.connect(self.ok_button_clicked)

    def on_tool_button_clicked(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбор файла", '')
        line_edit.setText(file_path)

    def ok_button_clicked(self):
        # Обработка события нажатия кнопки "Ok"
        self.ok_button_clicked_input.emit()  # Отправляем сигнал при нажатии кнопки "Ok"
