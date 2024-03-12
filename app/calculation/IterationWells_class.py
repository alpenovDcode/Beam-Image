import pandas as pd
from app.utils.logger import Logger as logger
from block1 import check_block1
from app.calculation.calc import check_factors
from app.calculation.donors_and_receipents import search_for_donor
from time import time
from dateutil.relativedelta import relativedelta


class IterationWells:
    """
    Класс для перебора нагнетательных скважин и расчеты по ним.
    """
    def __init__(self, id_name):
        self.not_worked_wells_df = None
        self.block1_df = None
        self.data_time_series = None
        self.ro_v_value = None
        self.ro_n_value = None
        self.vv_value = None
        self.vn_value = None
        self.compensation = None
        self.df_ku_slice_cell = None
        self.result_df = None
        self.q_target = None
        self.sum_debit_oil = None
        self.sum_debit_fluid = None
        self.not_worked_wells_list = None
        self.env_wells = None
        self.df_nag_dob_local = None
        self.df_output = None
        self.plast_list = None
        self.well_name = None
        self.mest = None
        self.q_current = None
        self.df_ngt_to_iterate_local = None
        self.id_name = id_name

    @classmethod
    def assign_values(cls, solver):
        cls.compare_dict = solver.compare_dict

        cls.df_ngt = solver.df_ngt
        cls.df_ku = solver.df_ku
        cls.df_opz = solver.df_opz
        cls.df_vpp = solver.df_vpp
        cls.df_nag_dob = solver.df_nag_dob
        cls.df_nag_nag = solver.df_nag_nag
        cls.df_tsk = solver.df_tsk
        cls.df_pvt = solver.df_pvt
        cls.df_modes = solver.df_modes
        cls.df_spectr = solver.df_spectr

        cls.max_date = solver.max_date
        cls.date_z1 = solver.date_z1
        cls.date_z2 = solver.date_z2
        cls.date_z3 = solver.date_z3
        cls.opz_months = solver.opz_months
        cls.vpp_months = solver.vpp_months
        cls.date_start_opz = solver.date_start_opz
        cls.date_start_vpp = solver.date_start_vpp

        cls.count_calc = 0
        cls.recipients_dict = {}
        cls.donors_dict = {}

        cls.p_zac_value = solver.p_zac_value
        cls.z1 = solver.z1
        cls.z2 = solver.z2
        cls.z3 = solver.z3
        cls.x = solver.x
        cls.y1 = solver.y1
        cls.y2 = solver.y2
        cls.y3 = solver.y3

        cls.df_ngt_z1 = solver.df_ngt_z1
        cls.df_ngt_z2 = solver.df_ngt_z2
        cls.df_ngt_z3 = solver.df_ngt_z3
        cls.df_ngt_to_iterate = solver.df_ngt_to_iterate
        cls.df_ngt_max_date = solver.df_ngt_max_date

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

    def check_worked_wells(self) -> list:
        """
        Проверка, что каждая скважина окружения работает z2 мес назад
        :param df_ngt_z2: датафрейм Выгрузка из NGT, отфильтрованный по дате z2
        :param df_nag_dob_local: датафрейм Окружение НАГ-ДОБ по скважине, отфильтрованный по макс дате
        :param env_wells: список окружения по центральной скважине
        :param compare_dict: словарь сопоставлений пластов из Окружение НАГ-ДОБ и Выгрузка из NGT
        :return: список неработающих скважин z2 мес назад (в данном случае указывается связка скв+мест+пласт
        """
        not_worked_wells_list = []
        # для каждой скважины окружения
        for well_env_id in self.env_wells:
            # находим датафрейм по этой скважине
            df_nag_dob_env_well = self.df_nag_dob_local[self.df_nag_dob_local["Id_name_environment"] == well_env_id]

            # проверка, что каждый пласт существует z2 мес назад
            for plast_nag_dob in df_nag_dob_env_well['plast'].unique():
                # датафрейм окружения с данной скважиной по пласту plast_nag_dob
                df_nag_dob_env_well_plast = df_nag_dob_env_well[df_nag_dob_env_well['plast'] == plast_nag_dob]
                # полное id скважины с учетом пласта
                id_name_full_nag_dob = df_nag_dob_env_well_plast['Id_name_environment_full'].max()
                # найти соответсвие пласта plast_nag_dob в выгрузке из ngt
                plast_ngt_name = self.compare_dict[plast_nag_dob]

                if plast_ngt_name:
                    df_ngt_well_env_z2_plast = self.df_ngt_z2[
                        (self.df_ngt_z2["Id_name"] == well_env_id) & (self.df_ngt_z2["object"] == plast_ngt_name)]

                    if df_ngt_well_env_z2_plast.empty:
                        not_worked_wells_list.append(id_name_full_nag_dob)

        return not_worked_wells_list

    def calc(self):
        """Обработка скважины. Нахождение q целевое, q расчетное"""
        result = {
            "isin_block1": None,
            "counted_successfully": False
        }

        time_all = time()
        check_worked_wells_time = ''
        time_cycle_wells = ''
        time_checking_conditions = ''
        tsk_ku_merging_time = ''

        # найдем данные по id_name
        self.df_ngt_to_iterate_local = self.df_ngt_to_iterate[self.df_ngt_to_iterate["Id_name"] == self.id_name]
        self.df_ngt_to_iterate_local = self.df_ngt_to_iterate_local.reset_index(drop=True)

        # получение значения q текущее
        self.q_current = self.get_q_priemist(self.df_ngt_to_iterate_local)

        time_check_block1 = time()
        # Блок 1
        res, self.block1_df = self.check_block1()
        if not res:

            result["isin_block1"] = True
            return result

        print(f'Время Блок 1 = {time() - time_check_block1}')
        time_check_block1 = time() - time_check_block1

        time_after_block1 = time()

        self.mest = self.df_ngt_to_iterate_local.at[0, 'mest']
        self.well_name = self.df_ngt_to_iterate_local.at[0, 'name_well']
        # id_name_full = df_ngt_to_iterate.loc[i, "Id_name_full"]

        # список рабочих пластов скважины
        plast_list = self.df_ngt_to_iterate_local['object'].unique()
        self.plast_list = [value for value in plast_list if value != "all"]

        # окружение
        self.df_nag_dob_local = self.df_nag_dob[self.df_nag_dob["Id_name_central"] == self.id_name]

        # по центральной скважине найдем ее окружение
        self.env_wells = self.df_nag_dob_local['Id_name_environment'].unique()

        self.df_ku_slice_cell = self.df_ku[self.df_ku['name_well'] == self.well_name]

        time_after_block1 = time() - time_after_block1

        if not self.df_ku_slice_cell.empty and not self.df_nag_dob_local.empty:
            # соединить с файлом ЦК
            tsk_ku_merging_time = time()
            self.merging_tsk_ku()
            tsk_ku_merging_time = time() - tsk_ku_merging_time

            # если целевая компенсация посчиталась
            if self.compensation:
                check_worked_wells_time = time()
                # проверка, что все скважины окружения работают z2 мес назад, иначе требуется не учитывать ее
                self.not_worked_wells_list = self.check_worked_wells()
                self.not_worked_wells_df = pd.DataFrame()

                check_worked_wells_time = time() - check_worked_wells_time

                self.init_output_df()

                # найдем qж* и qн*
                self.sum_debit_fluid = 0
                self.sum_debit_oil = 0

                # проверить, что для всех пластов окружения найдется соответствие в Выгрузке из NGT
                found_unrecognized_object = self.find_unrecognize_object()

                if not found_unrecognized_object:
                    time_cycle_wells = time()

                    # нахождение дебитов жидкости и нефти для всего окружения скважины
                    self.calc_environment_debits()

                    time_cycle_wells = time() - time_cycle_wells

                    # Подсчет усредненных костант из PVT
                    self.averaged_pvt_values()

                    # высчитываем q целевое
                    if self.sum_debit_fluid and self.sum_debit_oil:
                        q = (self.sum_debit_fluid - self.sum_debit_oil) * self.ro_v_value / self.vv_value + self.sum_debit_oil * self.ro_n_value / self.vn_value
                        self.q_target = self.compensation * q
                        self.df_output.at[0, 'q цел'] = self.q_target
                        self.df_output.at[0, 'q текущ'] = self.q_current
                        print(f"q целевое = {self.q_target} \n")
                        self.count_calc += 1
                        # датафрейм, где будут содержаться неработающие на определенном периоде
                        self.not_worked_wells_df = self.df_output[
                            self.df_output["Id_name_environment_full"].isin(self.not_worked_wells_list)]

                        # удалить из таблицы неработающие скважины
                        self.df_output = self.df_output[~self.df_output["Id_name_environment_full"].isin(self.not_worked_wells_list)]

                        time_checking_conditions = time()

                        self.df_output = check_factors(self.q_target, self.well_name, self.q_current, self.df_ngt_z1, self.df_ngt_z2,
                                                      self.df_ngt_z3, self.id_name, self.max_date,
                                                      self.plast_list, self.env_wells, self.sum_debit_fluid,
                                                      self.sum_debit_oil, self.p_zac_value, self.df_output, self.df_modes, self.df_spectr,
                                                      self.recipients_dict, self.donors_dict, self.mest, self.compare_dict, self.z1, self.z2, self.z3,
                                                      self.x, self.y1, self.y2, self.y3)
                        time_checking_conditions = time() - time_checking_conditions
                    else:
                        logger.warning(f"Для {self.well_name}: не посчитались суммарные дебиты ")

                result["counted_successfully"] = True

        else:
            logger.warning(f"Для {self.well_name}: не нашлось окружение НАГ_ДОБ или ячейка в КУ ")

        self.data_time_series = pd.DataFrame(
            {'Полная отработка 1 нагн скв': [time() - time_all], 'Отработка блока 1': [time_check_block1],
             'Предобработка файлов Ку и Окружение Наг Наг': [time_after_block1],
             'Мерджинг файлов Ку и Цк': [tsk_ku_merging_time],
             'Проверка, что все скв работали z2 мес назад': [check_worked_wells_time],
             'Цикл по скважинам окружения и по ее пластам': [time_cycle_wells],
             'Проверка условий': [time_checking_conditions]})

        return result

    def merging_tsk_ku(self):
        """
        Соединение файлов Ку и Цк по ячейке и получение целевой компенсации
        """
        self.result_df = pd.merge(self.df_ku_slice_cell, self.df_tsk, on='cell')
        # Фильтрация результатов по 'object_y' и 'weight'
        self.result_df = self.result_df[(self.result_df['object_y'] == "all") &
                              (self.result_df['weight'] == self.result_df['weight'].max())]

        if not self.result_df.empty:
            self.compensation = self.result_df['compensation'].max()
        else:
            logger.warning(f"Для скважины {self.well_name} не нашлась компенсация")
            self.compensation = None

    def init_output_df(self):
        """
        Формирование выходного файла
        """
        self.df_output = self.df_nag_dob_local.copy()
        self.df_output = self.df_output.reset_index(drop=True)
        self.df_output.at[0, 'Ячейка файл Ку'] = self.result_df['cell+object_x'].max()
        self.df_output.at[0, 'Ячейка'] = self.result_df['cell'].max()

    def find_unrecognize_object(self) -> bool:
        """
        Проверка, что все объекты, на которых работало окружение можно распознать
        """
        # булева переменная для случая, если найдется неопознанный пласт
        found_unrecognized_object = False

        for plast_nag_dob in self.df_nag_dob_local['plast'].unique():
            if not self.compare_dict[plast_nag_dob]:
                self.df_output.at[
                    0, f'Ошибка'] = f'Недостаточно данных, неизвестные пласты'
                found_unrecognized_object = True

        return found_unrecognized_object

    def calc_environment_debits(self):
        """
        Подсчет суммарного дебита нефти и жидкости всего окружения
        """
        # цикл по скважинам окружения и по ее пластам
        for well_env_id in self.env_wells:
            slice_well = self.df_nag_dob_local[(self.df_nag_dob_local['Id_name_environment'] == well_env_id)]
            for plast in slice_well['plast'].unique():
                slice_well_plast = slice_well[slice_well['plast'] == plast]

                slice_well_plast = slice_well_plast.reset_index(drop=True)

                if not slice_well_plast.empty:
                    k_part = slice_well_plast.at[0, 'K_part']
                    # найти дебит этой скважины
                    # найти соответствие пласта из файла Наг-Доб для пласта из выгрузки NGT
                    plast_ngt_name = self.compare_dict[plast]

                    # находим скважину в Выгрузке из NGT
                    df_ngt_slice = self.df_ngt_max_date[
                        (self.df_ngt_max_date['Id_name'] == well_env_id) & (
                                self.df_ngt_max_date['object'] == plast_ngt_name)]
                    # df_ngt_slice = df_ngt_slice[(df_ngt_slice['date'].dt.month == max_date.month) & (
                    #             df_ngt_slice['date'].dt.year == max_date.year)]

                    if not df_ngt_slice.empty:
                        index = self.df_output.loc[(self.df_output['Id_name_environment'] == well_env_id) & (
                                self.df_output['plast'] == plast)].index[0]

                        df_ngt_slice = df_ngt_slice.reset_index(drop=True)
                        # дебит жидкости
                        debit_fluid = df_ngt_slice.at[0, "q_water"]
                        # дебит нефти
                        debit_oil = df_ngt_slice.at[0, "q_oil"]

                        self.sum_debit_fluid += debit_fluid * k_part
                        self.sum_debit_oil += debit_oil * k_part

                        self.df_output.at[index, 'qж текущ'] = debit_fluid
                        self.df_output.at[index, 'qн текущ'] = debit_oil
                        self.df_output.at[index, 'qж* текущ'] = debit_fluid * k_part
                        self.df_output.at[index, 'qн* текущ'] = debit_oil * k_part

                        df_pvt_local = self.df_pvt[self.df_pvt['id_name'] == (self.mest + plast_ngt_name)]

                        if not df_pvt_local.empty:
                            vn_value = df_pvt_local["vn"].max()
                            vv_value = df_pvt_local["vv"].max()
                            ro_n_value = df_pvt_local["ro_n"].max()
                            ro_v_value = df_pvt_local["ro_v"].max()

                            # q = (debit_fluid - debit_oil) * ro_v_value / vv_value + debit_oil * ro_n_value / vn_value
                            # q_target_plast = compensation * q

                            self.df_output.at[index, 'Bн'] = vn_value
                            self.df_output.at[index, 'B в'] = vv_value
                            self.df_output.at[index, 'ρ н'] = ro_n_value
                            self.df_output.at[index, 'ρ в'] = ro_v_value
                            self.df_output.at[index, 'Цк'] = self.compensation

                            self.df_output.at[index, 'Bн*'] = vn_value * self.df_output.at[
                                index, 'qн* текущ']
                            self.df_output.at[index, 'B в*'] = self.df_output.at[index, 'B в'] * (
                                    self.df_output.at[index, 'qж* текущ'] - self.df_output.at[
                                index, 'qн* текущ'])

                            self.df_output.at[index, 'ρ н*'] = self.df_output.at[index, 'ρ н'] * self.df_output.at[
                                index, 'qн* текущ']
                            self.df_output.at[index, 'ρ в*'] = self.df_output.at[index, 'ρ в'] * (
                                    self.df_output.at[index, 'qж* текущ'] - self.df_output.at[
                                index, 'qн* текущ'])
                    else:
                        logger.warning(
                            f"Центральная скважина {self.well_name}: не нашлись дебиты в Выгрузке из NGT для окружающей скважина {well_env_id}")
                else:
                    logger.warning(
                        f"Центральная скважина {self.well_name}: не нашелся коэфф участия для скважины {well_env_id}")

    def averaged_pvt_values(self):
        """
        Подсчет усредненный костант из PVT
        """
        # суммарные значения дебитов
        self.df_output.at[0, 'qж текущ сумм'] = self.df_output['qж* текущ'].sum()
        self.df_output.at[0, 'qн текущ сумм'] = self.df_output['qн* текущ'].sum()

        # средние значения констант
        self.vn_value = self.df_output['Bн*'].sum() / self.df_output['qн* текущ'].sum()
        self.df_output.at[0, 'Bн средн'] = self.vn_value

        self.vv_value = self.df_output['B в*'].sum() / (
                self.df_output['qж* текущ'].sum() - self.df_output['qн* текущ'].sum())
        self.df_output.at[0, 'B в средн'] = self.vv_value

        self.ro_n_value = self.df_output['ρ н*'].sum() / self.df_output['qн* текущ'].sum()
        self.df_output.at[0, 'ρ н средн'] = self.ro_n_value

        self.ro_v_value = self.df_output['ρ в*'].sum() / (
                self.df_output['qж* текущ'].sum() - self.df_output['qн* текущ'].sum())
        self.df_output.at[0, 'ρ в средн'] = self.ro_v_value

        # удаляем промежуточные столбцы
        self.df_output = self.df_output.drop(['Bн', 'B в', 'ρ н', 'ρ в', 'Bн*', 'B в*', 'ρ н*', 'ρ в*'], axis=1)

    def check_block1(self):
        """
        Проверка на очаговость, Факт ОПЗ и ВПП
        :param df_ngt_to_iterate_local: датафрейм Выгрузка из NGT по скважине на максимальную дату
        :param id_name: id скважина+месторождение
        :param df_opz: файл ОПЗ`
        :param df_vpp: файл ВПП
        :param date_start_opz: дата начала проверки на факт ОПЗ
        :param date_start_vpp: дата начала проверки на факт ВПП
        :param max_date: максимальная дата
        :param q_current: премистость текущая
        :param df_ngt: Файл Выгрузка из NGT
        :return: кортеж из (t, block1_result_df), если t=True, то скважина выходит из цикла, block1_result_df - данные о скважинах, прошедших какую-либо из проверок
        """
        block1_result_df = pd.DataFrame(columns=["Скважина", "Комментарий", "Кскв", "q расч"])

        # Проверка на очаговость
        # «Проектное назначение скважины» стоит «НЕФТЯНЫЕ»
        # df_check_ojag = df[df["design purpose of the well"].str.startswith("неф")]
        check_condition = self.df_ngt_to_iterate_local["design purpose of the well"].str.startswith("неф").all()

        t = True

        if check_condition:
            data_series = pd.DataFrame(
                {"Скважина": [self.id_name], "Комментарий": ["Очаговая"], "Кскв": [''], "q расч": ["60-80"]})
            block1_result_df = pd.concat([block1_result_df, data_series], ignore_index=True)

            t = False
            return t, block1_result_df

        # проверка на опз
        df_opz_local = self.df_opz[self.df_opz['Id_name'] == self.id_name]
        # mask = (df_opz_local['date'].dt.month > date_start_opz.month) & (df_opz_local['date'].dt.month <= max_date.month) & (df_opz_local['date'].dt.year > date_start_opz.year) & (df_opz_local['date'].dt.year <= max_date.year)
        mask = (df_opz_local['date'] > self.date_start_opz) & (df_opz_local['date'] <= self.max_date)
        df_opz_in_interval = df_opz_local[mask]

        if not df_opz_in_interval.empty:
            data_series = pd.DataFrame(
                {"Скважина": [self.id_name], "Комментарий": ["Прошла проверку на факт ОПЗ"], "Кскв": [1],
                 "q расч": [self.q_current]})
            block1_result_df = pd.concat([block1_result_df, data_series], ignore_index=True)

            t = False
            return t, block1_result_df

        # проверка на впп
        df_vpp_local = self.df_vpp[self.df_vpp['Id_name'] == self.id_name]
        # mask = (df_vpp_local['date'].dt.month > date_start_vpp.month) & (df_vpp_local['date'].dt.month <= max_date.month) & (df_vpp_local['date'].dt.year > date_start_vpp.month) & (df_vpp_local['date'].dt.year <= max_date.year)
        mask = (df_vpp_local['date'] > self.date_start_vpp) & (df_vpp_local['date'] <= self.max_date)
        df_vpp_in_interval = df_vpp_local[mask]

        if not df_vpp_in_interval.empty:
            df_vpp = df_vpp_in_interval['date'].max()
            date_vpp_before = df_vpp - relativedelta(months=1)
            df_ngt_vpp_before = self.df_ngt[
                (self.df_ngt["Id_name"] == self.id_name) & (self.df_ngt['date'].dt.month == date_vpp_before.month) & (
                            self.df_ngt['date'].dt.year == date_vpp_before.year)]
            q_value = self.get_q_priemist(df_ngt_vpp_before)
            if not q_value:
                df_ngt_vpp = self.df_ngt[(self.df_ngt["Id_name"] == self.id_name) & (self.df_ngt['date'].dt.month == df_vpp.month) & (
                            self.df_ngt['date'].dt.year == df_vpp.year)]
                q_value = self.get_q_priemist(df_ngt_vpp)
            data_series = pd.DataFrame(
                {"Скважина": [self.id_name], "Комментарий": ["Прошла проверку на факт ВПП"], "Кскв": [''],
                 "q расч": [q_value]})
            block1_result_df = pd.concat([block1_result_df, data_series], ignore_index=True)

            t = False
            return t, block1_result_df

        return t, block1_result_df