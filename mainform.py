import os
import sys
import datetime
import pandas as pd
from app.utils.logger import Logger as logger
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QMainWindow, QProgressBar, QAction, \
    QVBoxLayout, QMessageBox, QDialog, QFileDialog, QCheckBox, QTextEdit, QPushButton
from PyQt5.QtWidgets import QLineEdit, QTreeWidgetItem, QTreeWidget, QInputDialog
from PyQt5.QtCore import QCoreApplication, Qt, QAbstractTableModel, pyqtSignal, QThread
from forms.mainform_ui import Ui_MainWindow
from forms.input import Ui_Dialog as Ui_Input
from forms.save import Ui_Dialog as Ui_Save
from forms.config_input import Ui_Dialog as Ui_Config
from PyQt5.QtGui import QColor, QFont
from app.reading.read_with_threads import IMPORT, CALCULATION
from app.fill_json import create_json
import json
import traceback

titleProgram = "Алгоритм целевой закачки"

from enum import Enum, auto
class NodeType(Enum):
    """Типы узлов в дереве скважин"""
    PROJECT = auto()
    DATA = auto()
    TSK = auto()
    NAG_DOB = auto()
    NGT = auto()
    KU = auto()
    PLASTS = auto()
    SPECTR = auto()
    OPZ = auto()
    VPP = auto()
    PVT = auto()
    NAG_NAG = auto()
    MODES = auto()
    TAB_RES = auto()
    RESULT = auto()


class ProjectView(QWidget):
    def __init__(self, parent:'MainForm'=None):
        super().__init__(parent)
        self.main_form = parent
        layout = QVBoxLayout()
        self.tree = QTreeWidget(self)
        self.tree.header().setVisible(False)
        self.fill_tree()
        self.tree.itemClicked.connect(self.showInputDialog)
        layout.addWidget(self.tree)
        self.setLayout(layout)
        self.setMaximumWidth(400)

    def showInputDialog(self, item=None):
        """Сохранение результатов"""
        if int(item.text(5)) == NodeType.RESULT.value:
            self.dialog_save = QtWidgets.QDialog(self)
            self.ui_save = Ui_Save()
            self.ui_save.setupUi(self.dialog_save)
            self.dialog_save.show()
            self.ui_save.toolButton.clicked.connect(lambda: self.on_tool_button_clicked(self.ui_save.lineEdit))
            self.ui_save.buttonBox.accepted.connect(self.ok_button_clicked_save)
        else:
            pass

    def on_tool_button_clicked(self, line_edit):
        folder_path = QFileDialog.getExistingDirectory(self.dialog_save, "Выберите папку", options=QFileDialog.ShowDirsOnly)
        if folder_path:
            line_edit.setText(folder_path)

    def ok_button_clicked_save(self):
        self.main_form.output_path = self.ui_save.lineEdit.text()
        if not os.path.exists(self.main_form.output_path):
            os.makedirs(self.main_form.output_path)

        # Получение текущей даты и времени
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")
        # Имя файла с текущей датой и временем
        file_name = f'Итоги_расчета_{formatted_datetime}.xlsx'

        output_path_results = os.path.join(self.main_form.output_path, file_name)

        with pd.ExcelWriter(output_path_results, engine='xlsxwriter') as writer:
            self.main_form.output_result_df.to_excel(writer, sheet_name='Результаты расчета', index=False)

            # Сохраните второй датафрейм в другом листе
            self.main_form.block1_result_df.to_excel(writer, sheet_name='Блок 1', index=False)

            # датафрейм с донорами и реципиентами
            self.main_form.donors_recipients_result_df.to_excel(writer, sheet_name='Поиск доноров для реципиентов', index=False)

            self.main_form.df_all_don_and_rec.to_excel(writer, sheet_name='Доноры и реципиенты', index=False)

            # датафрейм с настройками
            self.main_form.settings_df.to_excel(writer, sheet_name='Настройки', index=False)

        # Проверка успешного сохранения файлов
        if os.path.exists(output_path_results):
            self.messageShowAfterSave(1)
        else:
            self.messageShowAfterSave(0)

        app_name = f'app_{formatted_datetime}.log'
        output_path_logs = os.path.join(self.main_form.output_path, app_name)
        logger.save(output_path_logs)

    def messageShowAfterSave(self, code:int):
        """Сообщение после импорта"""
        if code == 1:
            QMessageBox.information(
                self,
                "Информация",
                "Сохранение данных завершено")
            return
        QMessageBox.information(
            self,
            "Информация",
            "Ошибка при сохранении данных")

    def fill_tree(self):
        """Заполнение дерева"""
        # Создаем объект QColor для красного цвета (255, 0, 0)
        red_color = QColor(255, 0, 0)
        green_color = QColor(0, 190, 0)

        # Создаем объект QFont с нужным размером шрифта
        font = QFont('Bahnschrift SemiBold', 10)

        self.tree.clear()
        
        project_item = QTreeWidgetItem()
        self.tree.addTopLevelItem(project_item)
        project_item.setText(0, 'Проект')
        project_item.setText(5, str(NodeType.PROJECT.value))
        project_item.setFont(0, font)
        # project_item.setText(5, str(NodeType.PROJECT.value))

        # Данные проекта
        pr_data_item = QTreeWidgetItem(project_item)
        pr_data_item.setText(0, f'Исходные данные')
        pr_data_item.setText(5, str(NodeType.DATA.value))
        pr_data_item.setFont(0, font)
        # pr_data_item.setText(5, str(NodeType.PROJECT.value))

        pr_data_level = QTreeWidgetItem(pr_data_item)
        pr_data_level.setText(0, 'ЦК')
        pr_data_level.setFont(0, font)

        # Перекраска в красный
        if not self.main_form.df_tsk.empty:
            pr_data_level.setForeground(0, green_color)
        else:
            pr_data_level.setForeground(0, red_color)
        pr_data_level.setFont(0, font)

        pr_data_level.setText(5, str(NodeType.TSK.value))
        pr_data_level.setText(6, str('ЦК'))

        pr_data_level = QTreeWidgetItem(pr_data_item)
        pr_data_level.setText(0, 'Окружение НАГ-ДОБ')

        # Перекраска в красный
        if not self.main_form.df_nag_dob.empty:
            pr_data_level.setForeground(0, green_color)
        else:
            pr_data_level.setForeground(0, red_color)
        pr_data_level.setFont(0, font)

        pr_data_level.setText(5, str(NodeType.NAG_DOB.value))
        pr_data_level.setText(6, str('Окружение НАГ-ДОБ'))

        pr_data_level = QTreeWidgetItem(pr_data_item)
        pr_data_level.setText(0, 'Выгрузка из NGT')

        # Перекраска в красный
        if not self.main_form.df_ngt.empty:
            pr_data_level.setForeground(0, green_color)
        else:
            pr_data_level.setForeground(0, red_color)
        pr_data_level.setFont(0, font)

        pr_data_level.setText(5, str(NodeType.NGT.value))
        pr_data_level.setText(6, str('Выгрузка из NGT'))

        pr_data_level = QTreeWidgetItem(pr_data_item)
        pr_data_level.setText(0, 'КУ')

        # Перекраска в красный
        if not self.main_form.df_ku.empty:
            pr_data_level.setForeground(0, green_color)

            for plast in self.main_form.plast_list:
                plast_data_level = QTreeWidgetItem(pr_data_level)
                plast_data_level.setText(0, plast)
                plast_data_level.setFont(0, font)
                plast_data_level.setText(5, str(NodeType.PLASTS.value))

            self.tree.expandItem(pr_data_level)

        else:
            pr_data_level.setForeground(0, red_color)
        pr_data_level.setFont(0, font)

        pr_data_level.setText(5, str(NodeType.KU.value))
        pr_data_level.setText(6, str('КУ'))

        pr_data_level = QTreeWidgetItem(pr_data_item)
        pr_data_level.setText(0, 'СПЕКТР')

        # Перекраска в красный
        if not self.main_form.df_spectr.empty:
            pr_data_level.setForeground(0, green_color)
        else:
            pr_data_level.setForeground(0, red_color)
        pr_data_level.setFont(0, font)

        pr_data_level.setText(5, str(NodeType.SPECTR.value))
        pr_data_level.setText(6, str('СПЕКТР'))

        pr_data_level = QTreeWidgetItem(pr_data_item)
        pr_data_level.setText(0, 'ОПЗ')

        # Перекраска в красный
        if not self.main_form.df_opz.empty:
            pr_data_level.setForeground(0, green_color)
        else:
            pr_data_level.setForeground(0, red_color)
        pr_data_level.setFont(0, font)

        pr_data_level.setText(5, str(NodeType.OPZ.value))
        pr_data_level.setText(6, str('ОПЗ'))

        pr_data_level = QTreeWidgetItem(pr_data_item)
        pr_data_level.setText(0, 'ВПП')

        # Перекраска в красный
        if not self.main_form.df_vpp.empty:
            pr_data_level.setForeground(0, green_color)
        else:
            pr_data_level.setForeground(0, red_color)
        pr_data_level.setFont(0, font)

        pr_data_level.setText(5, str(NodeType.VPP.value))
        pr_data_level.setText(6, str('ВПП'))

        pr_data_level = QTreeWidgetItem(pr_data_item)
        pr_data_level.setText(0, 'PVT')

        # Перекраска в красный
        if not self.main_form.df_pvt.empty:
            pr_data_level.setForeground(0, green_color)
        else:
            pr_data_level.setForeground(0, red_color)
        pr_data_level.setFont(0, font)

        pr_data_level.setText(5, str(NodeType.PVT.value))
        pr_data_level.setText(6, str('PVT'))

        pr_data_level = QTreeWidgetItem(pr_data_item)
        pr_data_level.setText(0, 'Окружение НАГ-НАГ')

        # Перекраска в красный
        if not self.main_form.df_nag_nag.empty:
            pr_data_level.setForeground(0, green_color)
        else:
            pr_data_level.setForeground(0, red_color)
        pr_data_level.setFont(0, font)

        pr_data_level.setText(5, str(NodeType.NAG_NAG.value))
        pr_data_level.setText(6, str('Окружение НАГ-НАГ'))

        pr_data_level = QTreeWidgetItem(pr_data_item)
        pr_data_level.setText(0, 'Тех режимы')

        # Перекраска в красный
        if not self.main_form.df_modes.empty:
            pr_data_level.setForeground(0, green_color)
        else:
            pr_data_level.setForeground(0, red_color)
        pr_data_level.setFont(0, font)

        pr_data_level.setText(5, str(NodeType.MODES.value))
        pr_data_level.setText(6, str('Тех режимы'))

        # Результат
        pr_result_item = QTreeWidgetItem(project_item)
        pr_result_item.setText(0, f'Результаты')
        pr_result_item.setText(5, str(NodeType.TAB_RES.value))

        pr_result_item.setFont(0, font)

        pr_result_level = QTreeWidgetItem(pr_result_item)
        pr_result_level.setText(0, 'Итоги расчета')

        # Перекраска в красный
        if not self.main_form.output_result_df.empty:
            pr_result_level.setForeground(0, green_color)
        else:
            pr_result_level.setForeground(0, red_color)
        pr_result_level.setFont(0, font)

        pr_result_level.setText(5, str(NodeType.RESULT.value))
        pr_result_level.setText(6, str('Итоги расчета'))

        self.tree.expandItem(pr_data_item)
        self.tree.expandItem(pr_result_item)

        self.tree.expandItem(project_item)


class MainForm(QMainWindow):
    """Главное окно программы"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(titleProgram)
        self.addProgressbar_main()
        self.set_values()

        # переменные проекта
        self.curr_thread = None

        # дерево проекта
        self.project_tree = ProjectView(self)
        self.ui.horizontalLayout.addWidget(self.project_tree)
        # графическая область
        self.right_widget = QWidget(self)
        self.ui.horizontalLayout.addWidget(self.right_widget)

        self.ui.take_data.triggered.connect(self.connect_data)
        self.ui.calc_action.triggered.connect(self.connect_settings_data)

    def set_values(self):
        # Текущий рабочий каталог
        self.current_directory = os.getcwd()
        self.all_files_uploaded = False

        self.tsk_path = None
        self.nag_dob_path = None
        self.ngt_path = None
        self.ku_path = None
        self.spectr_path = None
        self.opz_path = None
        self.vpp_path = None
        self.pvt_path = None
        self.nag_nag_path = None
        self.modes_path = None

        self.output_path = None
        self.plast_list = []

        self.nature_of_work = None
        self.state = None
        self.p_zak = None
        self.z1 = None
        self.z2 = None
        self.z3 = None
        self.x = None
        self.y1 = None
        self.y2 = None

        self.df_opz = pd.DataFrame()
        self.df_vpp = pd.DataFrame()
        self.df_ngt = pd.DataFrame()
        self.df_tsk = pd.DataFrame()
        self.df_spectr = pd.DataFrame()
        self.df_nag_dob = pd.DataFrame()
        self.df_nag_nag = pd.DataFrame()
        self.df_ku = pd.DataFrame()
        self.df_modes = pd.DataFrame()
        self.df_pvt = pd.DataFrame()

        self.output_result_df = pd.DataFrame()
        self.block1_result_df = pd.DataFrame()
        self.donors_recipients_result_df = pd.DataFrame()
        self.df_all_don_and_rec = pd.DataFrame()
        self.settings_df = pd.DataFrame()

    def addProgressbar_main(self):
        self.progress_bar = QProgressBar()
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('')
        self.progress_bar.setRange(1, 100)
        self.progress_bar.setGeometry(0, 0, 100, 20)
        self.progress_bar.setValue(0)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.setVisible(False)
        self.setAnimated(False)

    def set_progressbar_range_main(self, value):
        self.progress_bar.setRange(1, value)
        self.progress_bar.setVisible(True)
        self.progress_bar.update()

    def set_progressbar_value_main(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.update()

    def set_progress_bar_visible_main(self, value):
        self.progress_bar.setVisible(value)
        self.progress_bar.update()
    
    def set_status_bar_message_main(self, msg):
        self.status_bar.showMessage(self.tr(str(msg)))

    def reset_progress_bar_main(self):
        self.progress_bar.setValue(0)

    def writeLogs(self, typeErr="INFO", text="", e="", module=""):
        log_directory = "Logs"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        file_develop = self.replaceSlash("Logs\\log_for_develop.txt")
        with open(file_develop, "a", encoding="utf-8") as f:
            f.write("[" + str(datetime.datetime.now().replace(
                microsecond=0)) + "]" + " " + str(
                typeErr) + " " + module + ": " + str(text) + " " + str(
                e) + "\n")

    def replaceSlash(self, filePath):
        platform = sys.platform
        slashMap = {'win32': '\\',
                    'cygwin': '\\',
                    'darwin': '/',
                    'linux2': '/',
                    'linux': '/'}
        if platform not in slashMap.keys(): platform = 'linux2'
        return filePath.replace('/', slashMap[platform])
    
    def messageShowAfterImport(self, code:int):
        """Сообщение после импорта"""
        if code == 1:
            self.all_files_uploaded = True
            QMessageBox.information(
                self,
                "Информация",
                "Импорт данных завершён")
            return
        self.all_files_uploaded = False
        QMessageBox.information(
            self,
            "Информация",
            "Не найдены данные для импорта")

    def messageShowAfterCalc(self, code:int):
        """Сообщение после импорта"""
        if code == 1:
            QMessageBox.information(
                self,
                "Информация",
                "Расчет завершён")
            return
        QMessageBox.information(
            self,
            "Информация",
            "Ошибка при расчетах")

    def messageShowWithoutAllData(self):
        """Сообщение после импорта"""
        QMessageBox.information(
            self,
            "Информация",
            "Загрузите все необходимые файлы")

    def connect_data(self):
        """Функция, срабатывающая при нажатии на кнопку импорт данных"""
        create_json()
        self.dialog_input = InputDialog(self)  # Создаем экземпляр дочернего диалогового окна
        self.dialog_input.show()
        self.dialog_input.ok_button_clicked_input.connect(self.import_files)

        self.get_default_paths()

    def get_default_paths(self):
        """Считывание последних введенных данных(путей) из json"""
        try:
            with open('history.json', 'r', encoding='UTF-8') as f:
                conf = json.load(f)

            data_connect = conf["connect"]

            # считывание файла ОПЗ
            data_opz = data_connect["opz"]
            if data_opz["type"] == "table":
                self.opz_path = os.path.join(*data_opz["path"])
                self.dialog_input.ui_input.lineEdit_opz.setText(self.opz_path)

            # считывание файла ВПП
            data_vpp = data_connect["vpp"]
            if data_vpp["type"] == "table":
                self.vpp_path = os.path.join(*data_vpp["path"])
                self.dialog_input.ui_input.lineEdit_vpp.setText(self.vpp_path)

            # считывание файла Выгрузка из NGT
            data_ngt = data_connect["ngt"]
            if data_ngt["type"] == "table":
                self.ngt_path = os.path.join(*data_ngt["path"])
                self.dialog_input.ui_input.lineEdit_ngt.setText(self.ngt_path)

            # считывание файла ЦК
            data_tsk = data_connect["tsk"]
            if data_tsk["type"] == "table":
                self.tsk_path = os.path.join(*data_tsk["path"])
                self.dialog_input.ui_input.lineEdit_tsk.setText(self.tsk_path)

            # считывание файла СПЕКТР
            data_spectr = data_connect["spectr"]
            if data_spectr["type"] == "table":
                self.spectr_path = os.path.join(*data_spectr["path"])
                self.dialog_input.ui_input.lineEdit_spectr.setText(self.spectr_path)

            # считывание файла Окружение НАГ-ДОБ
            data_nag_dob = data_connect["nag_dob"]
            if data_nag_dob["type"] == "table":
                self.nag_dob_path = os.path.join(*data_nag_dob["path"])
                self.dialog_input.ui_input.lineEdit_nag_dob.setText(self.nag_dob_path)

            # считывание файла Окружение НАГ-НАГ
            data_nag_nag = data_connect["nag_nag"]
            if data_nag_nag["type"] == "table":
                self.nag_nag_path = os.path.join(*data_nag_nag["path"])
                self.dialog_input.ui_input.lineEdit_nag_nag.setText(self.nag_nag_path)

            # считывание файла КУ
            self.ku_path = []
            data_ku = data_connect["ku"]
            if data_ku["type"] == "table":
                for path_file in data_ku["path"]:
                    self.ku_path.append(os.path.join(*path_file))
                self.ku_path = f"[{', '.join(self.ku_path)}]"
                self.dialog_input.ui_input.lineEdit_ku.setText(self.ku_path)

            # считывание файла с тех. режимами
            data_modes = data_connect["modes"]
            if data_modes["type"] == "table":
                self.modes_path = os.path.join(*data_modes["path"])
                self.dialog_input.ui_input.lineEdit_modes.setText(self.modes_path)

            # считывание PVT
            data_pvt = data_connect["pvt"]
            if data_pvt["type"] == "table":
                self.pvt_path = os.path.join(*data_pvt["path"])
                self.dialog_input.ui_input.lineEdit_pvt.setText(self.pvt_path)

        except BaseException as e:
            msg = f"Ошибка при получении путей из json"
            self.writeLogs(
                "ERROR",
                msg,
                str(traceback.format_exc()),
                module="Получение путей из json"
            )
            logger.error(f"Ошибка при получении путей из json: {e}")

    def get_input_paths(self):
        """Сохранение путей в переменные класса"""
        self.tsk_path = self.dialog_input.ui_input.lineEdit_tsk.text()
        # self.tsk_path = os.path.relpath(self.tsk_path, current_directory)

        self.nag_dob_path = self.dialog_input.ui_input.lineEdit_nag_dob.text()
        # self.nag_dob_path = os.path.relpath(self.nag_dob_path, current_directory)

        self.ngt_path = self.dialog_input.ui_input.lineEdit_ngt.text()
        # self.ngt_path = os.path.relpath(self.ngt_path, current_directory)

        self.ku_path = self.dialog_input.ui_input.lineEdit_ku.text()
        # self.ku_path = os.path.relpath(self.ku_path, current_directory)

        self.spectr_path = self.dialog_input.ui_input.lineEdit_spectr.text()
        # self.spectr_path = os.path.relpath(self.spectr_path, current_directory)

        self.opz_path = self.dialog_input.ui_input.lineEdit_opz.text()
        # self.opz_path = os.path.relpath(self.opz_path, current_directory)

        self.vpp_path = self.dialog_input.ui_input.lineEdit_vpp.text()
        # self.vpp_path = os.path.relpath(self.vpp_path, current_directory)

        self.pvt_path = self.dialog_input.ui_input.lineEdit_pvt.text()
        # self.pvt_path = os.path.relpath(self.pvt_path, current_directory)

        self.nag_nag_path = self.dialog_input.ui_input.lineEdit_nag_nag.text()
        # self.nag_nag_path = os.path.relpath(self.nag_nag_path, current_directory)

        self.modes_path = self.dialog_input.ui_input.lineEdit_modes.text()
        # self.modes_path = os.path.relpath(self.modes_path, current_directory)

        self.dialog_input.close()

    def import_files(self):
        """Загрузка данных"""
        self.get_input_paths()
        self.save_hystory_paths_json()

        self.thread = QThread()
        self.worker = IMPORT(self)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.set_progress_bar_visible.connect(self.set_progress_bar_visible_main)
        self.worker.set_progressbar_range.connect(self.set_progressbar_range_main)
        self.worker.set_progressbar_value.connect(self.set_progressbar_value_main)
        self.worker.set_status_bar_message.connect(self.set_status_bar_message_main)
        self.worker.reset_progress_bar.connect(self.reset_progress_bar_main)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.show_results_after_import)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def show_results_after_import(self):
        self.project_tree.fill_tree()
        self.messageShowAfterImport(self.worker.res)

    def connect_settings_data(self):
        """Нажатие на кнопку начало расчета"""
        if self.all_files_uploaded:
            self.dialog_config = ConfigDialog(self)  # Создаем экземпляр дочернего диалогового окна
            self.dialog_config.show()
            self.get_default_values()
            self.dialog_config.ok_button_clicked_config.connect(self.ok_button_clicked_settings)
        else:
            self.messageShowWithoutAllData()

    def save_hystory_paths_json(self):
        """Функция, сохраняющая последние введенные пути в json"""
        # Открываем JSON-файл
        with open('history.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        try:
            data['connect']['tsk']['path'] = self.tsk_path.split("\\")
            data['connect']['nag_dob']['path'] = self.nag_dob_path.split("\\")
            data['connect']['ngt']['path'] = self.ngt_path.split("\\")

            # Удалить квадратные скобки и лишние пробелы из строки
            paths_str = self.ku_path.strip("[]").strip()

            # Разделить строку по запятым и удалить лишние пробелы
            paths_list = paths_str.split(",")
            paths_list = [path.strip() for path in paths_list]

            # Разбить каждый путь по обратному слэшу и создать список списков
            result = [path.split("\\") for path in paths_list]

            data['connect']['ku']['path'] = result
            data['connect']['spectr']['path'] = self.spectr_path.split("\\")
            data['connect']['opz']['path'] = self.opz_path.split("\\")
            data['connect']['vpp']['path'] = self.vpp_path.split("\\")
            data['connect']['pvt']['path'] = self.pvt_path.split("\\")
            data['connect']['nag_nag']['path'] = self.nag_nag_path.split("\\")
            data['connect']['modes']['path'] = self.modes_path.split("\\")
        except Exception as e:
            msg = f"Ошибка при сохранении путей в json"
            self.writeLogs(
                "ERROR",
                msg,
                str(traceback.format_exc()),
                module="Сохранение путей в json"
            )
            logger.error(f"Ошибка при сохранении путей в json: {e}")

        # Сохраняем обновленный JSON в кодировке UTF-8
        with open('history.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def ok_button_clicked_settings(self):
        """Функция, обрабатывающая настройки пользователя"""
        self.nature_of_work = self.dialog_config.ui_config.lineEdit_nature_of_work.text()
        self.state = self.dialog_config.ui_config.lineEdit_state.text()
        self.p_zak = self.dialog_config.ui_config.lineEdit_p_zak.text()
        self.z1 = self.dialog_config.ui_config.lineEdit_z1.text()
        self.z2 = self.dialog_config.ui_config.lineEdit_z2.text()
        self.z3 = self.dialog_config.ui_config.lineEdit_z3.text()
        self.x = self.dialog_config.ui_config.lineEdit_x.text()
        self.y1 = self.dialog_config.ui_config.lineEdit_y1.text()
        self.y2 = self.dialog_config.ui_config.lineEdit_y2.text()
        self.y3 = self.dialog_config.ui_config.lineEdit_y3_2.text()

        self.dialog_config.close()
        self.save_hystory_values_json()

        # закрыть старый виджет с тексовым редактором, если есть
        self.ui.horizontalLayout.removeWidget(self.right_widget)
        self.right_widget.deleteLater()

        self.right_widget = QWidget(self)
        self.ui.horizontalLayout.addWidget(self.right_widget)
        logger.clear()

        self.start_calc()

    def get_default_values(self):
        """Получение последних введенных данных настроек пользователя"""
        with open('history.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        try:
            # значения с конфига
            solver_data = data["solver"]
            # nature_of_work = '[' + ', '.join(solver_data["nature_of_work"]) + ']'
            self.dialog_config.ui_config.lineEdit_nature_of_work.setText(solver_data["nature_of_work"])

            # state = '[' + ', '.join(solver_data["state"]) + ']'
            self.dialog_config.ui_config.lineEdit_state.setText(solver_data["state"])

            self.dialog_config.ui_config.lineEdit_p_zak.setText(solver_data["P zac"])
            self.dialog_config.ui_config.lineEdit_z1.setText(solver_data["z1"])
            self.dialog_config.ui_config.lineEdit_z2.setText(solver_data["z2"])
            self.dialog_config.ui_config.lineEdit_z3.setText(solver_data["z3"])
            self.dialog_config.ui_config.lineEdit_x.setText(solver_data["x"])
            self.dialog_config.ui_config.lineEdit_y1.setText(solver_data["y1"])
            self.dialog_config.ui_config.lineEdit_y2.setText(solver_data["y2"])
            self.dialog_config.ui_config.lineEdit_y3_2.setText(solver_data["y3"])
        except Exception as e:
            msg = f"Ошибка при получении значений из json"
            self.writeLogs(
                "ERROR",
                msg,
                str(traceback.format_exc()),
                module="Получение значений из json"
            )
            logger.error(f"Ошибка при получении значений из json: {e}")

    def save_hystory_values_json(self):
        """Сохранение последних данных настроек пользователя в json"""
        # Открываем JSON-файл
        with open('history.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        try:
            data["solver"]["nature_of_work"] = self.nature_of_work.strip()
            data["solver"]["state"] = self.state.strip()
            data["solver"]["P zac"] = self.p_zak.strip()
            data["solver"]["z1"] = self.z1.strip()
            data["solver"]["z2"] = self.z2.strip()
            data["solver"]["z3"] = self.z3.strip()
            data["solver"]["x"] = self.x.strip()
            data["solver"]["y1"] = self.y1.strip()
            data["solver"]["y2"] = self.y2.strip()
            data["solver"]["y3"] = self.y3.strip()
        except Exception as e:
            msg = f"Ошибка при сохранении значений из json"
            self.writeLogs(
                "ERROR",
                msg,
                str(traceback.format_exc()),
                module="Сохранение значений из json"
            )
            logger.error(f"Ошибка при сохранении значений из json: {e}")

        # Сохраняем обновленный JSON в кодировке UTF-8
        with open('history.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def start_calc(self):
        self.thread = QThread()
        self.worker = CALCULATION(self)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.set_progress_bar_visible.connect(self.set_progress_bar_visible_main)
        self.worker.set_progressbar_range.connect(self.set_progressbar_range_main)
        self.worker.set_progressbar_value.connect(self.set_progressbar_value_main)
        self.worker.set_status_bar_message.connect(self.set_status_bar_message_main)
        self.worker.reset_progress_bar.connect(self.reset_progress_bar_main)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.show_log)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def show_log(self):
        """Вывод логов после окончания расчетов"""
        self.project_tree.fill_tree()
        self.messageShowAfterCalc(self.worker.res)
        # Удаление старого виджета справа
        # self.ui.horizontalLayout.removeWidget(self.right_widget)
        # self.right_widget.deleteLater()

        # Создание нового виджета QTextEdit
        self.right_widget = QTextEdit()
        self.right_widget.setReadOnly(True)  # Чтобы предотвратить редактирование текста
        self.ui.horizontalLayout.addWidget(self.right_widget)

        # Отображение логов в QTextEdit
        logs_text = '\n'.join(logger.logs)
        self.right_widget.setPlainText(logs_text)


class InputDialog(QDialog):
    """Класс для первого диалогового окна (импорт данных)"""
    ok_button_clicked_input = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui_input = Ui_Input()
        self.ui_input.setupUi(self)
        self.current_directory = os.getcwd()

        # обозреватель проводника для выбора файлов
        self.ui_input.toolButton_tsk.clicked.connect(lambda: self.on_tool_button_clicked(self.ui_input.lineEdit_tsk))
        self.ui_input.toolButton_nag_dob.clicked.connect(
            lambda: self.on_tool_button_clicked(self.ui_input.lineEdit_nag_dob))
        self.ui_input.toolButton_ngt.clicked.connect(lambda: self.on_tool_button_clicked(self.ui_input.lineEdit_ngt))
        self.selected_files_ku = []
        self.ui_input.toolButton_ku.clicked.connect(lambda: self.on_tool_button_clicked_ku(self.ui_input.lineEdit_ku))
        self.ui_input.toolButton_spectr.clicked.connect(
            lambda: self.on_tool_button_clicked(self.ui_input.lineEdit_spectr))
        self.ui_input.toolButton_opz.clicked.connect(lambda: self.on_tool_button_clicked(self.ui_input.lineEdit_opz))
        self.ui_input.toolButton_vpp.clicked.connect(lambda: self.on_tool_button_clicked(self.ui_input.lineEdit_vpp))
        self.ui_input.toolButton_pvt.clicked.connect(lambda: self.on_tool_button_clicked(self.ui_input.lineEdit_pvt))
        self.ui_input.toolButton_nag_nag.clicked.connect(lambda: self.on_tool_button_clicked(self.ui_input.lineEdit_nag_nag))
        self.ui_input.toolButton_modes.clicked.connect(lambda: self.on_tool_button_clicked(self.ui_input.lineEdit_modes))

        self.ui_input.buttonBox.accepted.connect(self.ok_button_clicked)

    def on_tool_button_clicked(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбор файла", '', 'Excel files (*.xlsx *.xls *.csv)')
        if file_path:
            # Получаем абсолютный путь к файлу
            file_path = os.path.abspath(file_path)
            if file_path.startswith(self.current_directory):
                file_path = os.path.relpath(file_path, self.current_directory)
            line_edit.setText(file_path)

    def on_tool_button_clicked_ku(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбор файла", '', 'Excel files (*.xlsx *.xls *.csv)')
        if file_path:
            # Получаем абсолютный путь к файлу
            file_path = os.path.abspath(file_path)
            if file_path.startswith(self.current_directory):
                file_path = os.path.relpath(file_path, self.current_directory)
            # Добавление выбранного файла в список
            self.selected_files_ku.append(file_path)
            list_ku = f"[{', '.join(self.selected_files_ku)}]"
            line_edit.setText(list_ku)

    def ok_button_clicked(self):
        # Обработка события нажатия кнопки "Ok"
        self.ok_button_clicked_input.emit()  # Отправляем сигнал при нажатии кнопки "Ok"


class ConfigDialog(QDialog):
    """Класс для второго диалогового окна (импорт данных)"""
    ok_button_clicked_config = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui_config = Ui_Config()
        self.ui_config.setupUi(self)

        self.ui_config.buttonBox.accepted.connect(self.ok_button_clicked)

    def ok_button_clicked(self):
        # Обработка события нажатия кнопки "Ok"
        self.ok_button_clicked_config.emit()  # Отправляем сигнал при нажатии кнопки "Ok"
