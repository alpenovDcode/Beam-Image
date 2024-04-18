# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

import os
import sys
import datetime
import time

from mainform import MainForm

def main():
    def replaceSlash(filePath):
        platform = sys.platform
        slashMap = {'win32': '\\',
                    'cygwin': '\\',
                    'darwin': '/',
                    'linux2': '/',
                    'linux': '/'}
        if platform not in slashMap.keys(): platform = 'linux2'
        return filePath.replace('/', slashMap[platform])

    def my_excepthook(type, value, tback):
        window.currThread = None
        print(str(value))
        QMessageBox.critical(
            window, "Ошибка", "Возникла непредвиденная ошибка",
            QMessageBox.Cancel
        )
        window.progress_bar.setVisible(False)
        window.statusBar().showMessage('')
        sys.__excepthook__(type, value, tback)

    # Создание приложение
    app = QApplication(sys.argv)
    window = MainForm()
    window.app = app
    window.showMaximized()

    translator = QTranslator(app)
    translator.load("qtbase_ru", QLibraryInfo.location(
        QLibraryInfo.TranslationsPath))
    app.installTranslator(translator)
    sys.excepthook = my_excepthook

    app.exec_()

if __name__ == '__main__':
    main()