import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QMainWindow, QProgressBar, QAction, \
    QVBoxLayout, QMessageBox, QDialog, QFileDialog, QCheckBox, QTextEdit, QPushButton, QToolTip
from PyQt5.QtWidgets import QLineEdit, QTreeWidgetItem, QTreeWidget, QInputDialog
from PyQt5.QtCore import QCoreApplication, Qt, QAbstractTableModel, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap
from keras import Model
from keras.src.applications.vgg16 import preprocess_input, VGG16
from keras.src.utils import load_img, img_to_array

from forms.main_form import Ui_MainWindow
from forms.input import Ui_Dialog
from main import predict_caption, model, tokenizer, max_length, loaded_model


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

        image_path = self.dialog_input.ui_input.coeff_file_input_line.text()
        pixmap = QPixmap(image_path)

        self.ui.Photo.setPixmap(pixmap)
        self.ui.Photo.resize(pixmap.width(), pixmap.height())

        vgg_model = VGG16()
        # restructure the model
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

        # load image
        image = load_img(image_path, target_size=(224, 224))
        # convert image pixels to numpy array
        image = img_to_array(image)
        # reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # preprocess image for vgg
        image = preprocess_input(image)
        # extract features
        feature = vgg_model.predict(image, verbose=0)
        # predict from the trained model





        # Генерируем подпись с помощью обученной модели
        predicted_caption = predict_caption(loaded_model, feature, tokenizer, max_length)

        # Показываем изображение

        # Отображаем сгенерированную подпись в GUI
        self.ui.description.setText(predicted_caption)


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