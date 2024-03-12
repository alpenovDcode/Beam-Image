import pandas as pd
from app.utils.logger import Logger as logger
from block1 import check_block1
from app.calculation.donors_and_receipents import search_for_donor
from time import time
from dateutil.relativedelta import relativedelta


class PrepareData:
    """
    Класс для подготовки данных.
    """

    def __init__(self, mainform, thread):
        self.mainform = mainform
        self.thread = thread
        self.get_settings_values()

        # датафрейм с настройками
        self.settings_df = pd.DataFrame({
            "Характер работы": [mainform.nature_of_work],
            "Состояние": [mainform.state],
            "P закачки": [self.p_zac_value],
            "z1": [self.z1],
            "z2": [self.z2],
            "z3": [self.z3],
            "x": [self.x],
            "y1": [self.y1],
            "y2": [self.y2],
            "y3": [self.y3]
        })
        self.time_df = pd.DataFrame(
            columns=['Полная отработка 1 нагн скв', 'Отработка блока 1', 'Предобработка файлов Ку и Окружение Наг Наг',
                     'Мерджинг файлов Ку и Цк', 'Проверка, что все скв работали z2 мес назад',
                     'Цикл по скважинам окружения и по ее пластам', 'Проверка условий'])

        thread.set_progressbar_range.emit(100)
        thread.progress_vall += 5
        thread.set_progressbar_value.emit(thread.progress_vall)
        thread.set_status_bar_message.emit('Подготовка данных')

        self.df_ngt = mainform.df_ngt.copy()
        self.df_ku = mainform.df_ku.copy()
        self.df_opz = mainform.df_opz.copy()
        self.df_vpp = mainform.df_vpp.copy()
        self.df_nag_dob = mainform.df_nag_dob.copy()
        self.df_nag_nag = mainform.df_nag_nag.copy()
        self.df_tsk = mainform.df_tsk.copy()
        self.df_pvt = mainform.df_pvt.copy()
        self.df_modes = mainform.df_modes.copy()
        self.df_spectr = mainform.df_spectr.copy()

        self.nature_of_work_list = [element.lower().strip() for element in self.nature_of_work_list]
        self.state_list = [element.lower().strip() for element in self.state_list]

        self.max_date = self.df_ngt['date'].max()
        self.date_z1 = self.max_date - relativedelta(months=self.z1)
        self.date_z2 = self.max_date - relativedelta(months=self.z2)
        self.date_z3 = self.max_date - relativedelta(months=self.z3)
        self.opz_months = 4
        self.vpp_months = 5
        self.date_start_opz = self.max_date - relativedelta(months=self.opz_months)
        self.date_start_vpp = self.max_date - relativedelta(months=self.vpp_months)

        min_date = min(self.date_z1, self.date_z2, self.date_z3, self.date_start_opz, self.date_start_vpp)
        # фильтрация Выгрузки из NGT по актуальным датам
        self.df_ngt = self.df_ngt[(self.df_ngt['date'].dt.year >= min_date.year) & (self.df_ngt['date'].dt.month >= min_date.month)]

        self.compare_dict = {}

    def get_settings_values(self):
        """
        Считывание значений настроек пользователя
        :param mainform: объект интерфейса
        :return:
        """
        self.nature_of_work_list = self.mainform.nature_of_work.strip("[]").split(",")
        self.state_list = self.mainform.state.strip("[]").split(",")
        self.p_zac_value = int(self.mainform.p_zak)
        self.z1 = int(self.mainform.z1)
        self.z2 = int(self.mainform.z2)
        self.z3 = int(self.mainform.z3)
        self.x = int(self.mainform.x)
        self.y1 = int(self.mainform.y1)
        self.y2 = int(self.mainform.y2)
        self.y3 = int(self.mainform.y3)

    def filter_intersect_ku_ngt(self):
        """
        Фильтрация файлов Ку и Выгрузка из NGT по общим пластам.
        Т.е. в этих файлах должны остаться данные по общим пластам
        """
        plasts_ngt = self.df_ngt['object'].unique()
        plasts_ku = self.df_ku['object'].unique()

        def translate_to_rus(text):
            # Замена букв английских "а" и "с" на русские "a" и "c"
            translated_text = text.replace("a", "а").replace("c", "с")
            return translated_text

        def translate_to_eng(text):
            # Замена букв русских "а" и "с" на английские "a" и "c"
            translated_text = text.replace("а", "a").replace("с", "c")
            return translated_text

        plasts_ngt = [translate_to_rus(string) for string in plasts_ngt]
        plasts_ku = [translate_to_rus(string) for string in plasts_ku]

        plasts_ngt = set(plasts_ngt)
        plasts_ku = set(plasts_ku)

        intersection_plasts = plasts_ngt & plasts_ku

        self.mainform.plast_list = list(intersection_plasts)

        intersection_plasts.add("all")

        self.df_ngt = self.df_ngt[self.df_ngt['object'].isin(intersection_plasts)]

        intersection_plasts = [translate_to_eng(string) for string in intersection_plasts]
        self.df_ku = self.df_ku[self.df_ku['object'].isin(intersection_plasts)]

    def create_additional_df(self):
        """
        Создание дополнительных датафреймов, которые понадобятся при расчетах
        """
        # df_ngt_to_iterate - список нагн скважин для расчета
        self.df_ngt_to_iterate = self.df_ngt[
            (self.df_ngt["nature_of_work"].isin(self.nature_of_work_list)) & (self.df_ngt["state"].isin(self.state_list))]
        self.df_ngt_to_iterate = self.df_ngt_to_iterate[
            (self.df_ngt_to_iterate['date'].dt.month == self.max_date.month) & (
                        self.df_ngt_to_iterate['date'].dt.year == self.max_date.year)]

        self.df_ngt_max_date = self.df_ngt[
            (self.df_ngt["date"].dt.month == self.max_date.month) & (self.df_ngt["date"].dt.year == self.max_date.year)]
        self.df_ngt_z1 = self.df_ngt[(self.df_ngt["date"].dt.month == self.date_z1.month) & (self.df_ngt["date"].dt.year == self.date_z1.year)]
        self.df_ngt_z2 = self.df_ngt[(self.df_ngt["date"].dt.month == self.date_z2.month) & (self.df_ngt["date"].dt.year == self.date_z2.year)]
        self.df_ngt_z3 = self.df_ngt[(self.df_ngt["date"].dt.month == self.date_z3.month) & (self.df_ngt["date"].dt.year == self.date_z3.year)]

        self.df_nag_dob = self.df_nag_dob[
            (self.df_nag_dob["date"].dt.month == self.max_date.month) & (
                    self.df_nag_dob["date"].dt.year == self.max_date.year)]

        self.df_ku = self.df_ku[
            (self.df_ku['date'].dt.month == self.max_date.month) & (self.df_ku['date'].dt.year == self.max_date.year)]

    def compare_ngt_nag_dob(self):
        """
        Создание словаря сопоставлений пластов из Окружение НАГ-ДОБ и Выгрузка из NGT
        :param ngt_plasts: уникальные пласты файла Выгрузка из NGT
        :param nag_dob_plasts: уникальные пласты файла Окружение НАГ-ДОБ
        :return: словарь сопоставлений пластов из Окружение НАГ-ДОБ и Выгрузка из NGT
        """
        unique_plasts_nag_dob = self.df_nag_dob['plast'].unique()
        unique_plasts_ngt = self.df_ngt['object'].unique()

        for plast_nag_dob in unique_plasts_nag_dob:
            plast_ngt_name = None
            for ngt_plast in unique_plasts_ngt:
                if plast_nag_dob.startswith(ngt_plast):
                    plast_ngt_name = ngt_plast
                    break

            self.compare_dict[plast_nag_dob] = plast_ngt_name

        return self.df_ngt_to_iterate

    def get_q_priemist(self, df) -> float:
        """
        Получение значения q текущее
        :param df: датафрейм Выгрузка из NGT по скважине на максимальную дату
        :return: приемистость по всем пластам данной скважины
        """

        # используется значение пласта ALL, если есть. Иначе суммируется по имеющимся пластам
        row_all = df[df['object'] == 'all']
        q_current = row_all['q_priemist'].values[0] if not row_all.empty else None
        if q_current is None:
            row_all = df[df['object'] != 'all']
            q_current = row_all['q_priemist'].sum()

        return q_current


