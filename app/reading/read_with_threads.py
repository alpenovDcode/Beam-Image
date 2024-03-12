from PyQt5.QtCore import pyqtSignal, QObject
import traceback
from app.reading.read_files import read_input_files
from app.start import run as run_calc


class IMPORT(QObject):
    set_progress_bar_visible = pyqtSignal(bool)
    set_progressbar_value = pyqtSignal(int)
    set_progressbar_range = pyqtSignal(int)
    set_status_bar_message = pyqtSignal(str)
    reset_progress_bar = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.main_form = parent
        self.progress_vall = 0
        self.res = 1

    def run(self):
        try:

            self.main_form.writeLogs(
                "INFO",
                "Запуск работы модуля",
                '',
                module="Запуск импорта данных"
            )
            self.set_progress_bar_visible.emit(True)
            self.main_form.df_opz, self.main_form.df_vpp, self.main_form.df_ngt, self.main_form.df_tsk, self.main_form.df_spectr, self.main_form.df_nag_dob, self.main_form.df_nag_nag, self.main_form.df_ku, self.main_form.df_modes, self.main_form.df_pvt = read_input_files(self.main_form, self)
            # self.main_form.addProgressbar_main()
            # self.main_form.output_result_df, self.main_form.block1_result_df = run_calc(self.main_form, self)

            self.main_form.writeLogs(
                "INFO",
                "Завершение работы модуля",
                '',
                module="Запуск импорта данных"
            )
            self.set_progress_bar_visible.emit(False)
            self.set_progressbar_value.emit(0)
            self.set_status_bar_message.emit('')
            self.res = 1
            self.finished.emit()
        except Exception as e:
            msg = f"Ошибка при считывании данных для расчета: {e}"
            self.main_form.writeLogs(
                "ERROR",
                msg,
                str(traceback.format_exc()),
                module="Считывание данных для расчета"
            )
            self.set_progress_bar_visible.emit(False)
            self.set_progressbar_value.emit(0)
            self.set_status_bar_message.emit('')
            self.finished.emit()
            self.res = 0


class CALCULATION(QObject):
    set_progress_bar_visible = pyqtSignal(bool)
    set_progressbar_value = pyqtSignal(int)
    set_progressbar_range = pyqtSignal(int)
    set_status_bar_message = pyqtSignal(str)
    reset_progress_bar = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.main_form = parent
        self.progress_vall = 0
        self.res = 1

    def run(self):
        try:

            self.main_form.writeLogs(
                "INFO",
                "Запуск работы модуля",
                '',
                module="Запуск расчетов"
            )
            self.set_progress_bar_visible.emit(True)
            # self.main_form.addProgressbar_main()
            self.main_form.output_result_df, self.main_form.block1_result_df, self.main_form.donors_recipients_result_df, self.main_form.df_all_don_and_rec, self.main_form.settings_df = run_calc(self.main_form, self)

            self.main_form.writeLogs(
                "INFO",
                "Завершение работы модуля",
                '',
                module="Запуск расчетов"
            )
            self.set_progress_bar_visible.emit(False)
            self.set_progressbar_value.emit(0)
            self.set_status_bar_message.emit('')
            self.res = 1
            self.finished.emit()
        except Exception as e:
            msg = f"Ошибка при расчетах: {e}"
            self.main_form.writeLogs(
                "ERROR",
                msg,
                str(traceback.format_exc()),
                module="Считывание данных для расчета"
            )
            self.set_progress_bar_visible.emit(False)
            self.set_progressbar_value.emit(0)
            self.set_status_bar_message.emit('')
            self.finished.emit()
            self.res = 0

