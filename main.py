import sys
from datetime import datetime

from PyQt5.QtCore import QTranslator, QLibraryInfo
from PyQt5.QtWidgets import QApplication, QMessageBox

from mainform import MainForm


def main():
    """Главная функция"""

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
        file = replaceSlash("Logs\\log_for_develop.txt")
        with open(file, "a", encoding="utf-8") as f:
            f.write(
                "[" + str(datetime.now().replace(
                    microsecond=0)) + "]" + " " + "CRITICAL" + " " + "Main" + ": " + str(
                    value) + "\n")
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
