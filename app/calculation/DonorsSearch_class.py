from app.utils.logger import Logger as logger
from app.calculation.calc import check_shtutser, check_opz
from time import time
import pandas as pd


class DonorsSearch:
    """
    Класс для поиска доноров для реципиентов
    """

    def __init__(self, output_result_df, class_iter_obj):
        self.output_result_df = output_result_df
        self.df_donors_and_recipients = pd.DataFrame()
        self.df_nag_nag = class_iter_obj.df_nag_nag
        self.df_modes = class_iter_obj.df_modes
        self.df_spectr = class_iter_obj.df_spectr
        self.recipients_dict = class_iter_obj.recipients_dict
        self.donors_dict = class_iter_obj.donors_dict
        self.max_date = class_iter_obj.max_date

    def search_q_current(self, well_name: str):
        '''
        Поиск текущей приемистости для скважины
        :param output_result_df: датафрейм с выходными данными
        :param well_name: имя скважины
        :return: значение текущ приемистости
        '''
        df_local = self.output_result_df[self.output_result_df['Id_name_central'] == well_name]
        q_current = df_local['q текущ'].max()
        return q_current

    def search_q_target(self, well_name: str):
        '''
        Поиск целевой приемистости для скважины
        :param output_result_df: датафрейм с выходными данными
        :param well_name: имя скважины
        :return: значение целевой приемистости
        '''
        df_local = self.output_result_df[self.output_result_df['Id_name_central'] == well_name]
        q_target = df_local['q цел'].max()
        return q_target

    def search_for_donor(self):
        # Создание множества из списка `donors_list`
        donors_id_name_list = self.donors_dict.keys()
        donors_set = set(donors_id_name_list)

        # Создание множества из списка `recipients_list`
        recipients_id_name_list = self.recipients_dict.keys()
        recipients_set = set(recipients_id_name_list)

        df_nag_nag = self.df_nag_nag[self.df_nag_nag['Id_name_central'].isin(recipients_set)]

        for recipient in recipients_set:
            df_nag_nag_slice = df_nag_nag[df_nag_nag['Id_name_central'] == recipient]

            # Создание множества из столбца `Id_name_environment`
            environment_set = set(df_nag_nag_slice["Id_name_environment"])

            # Проверка наличия общих элементов
            common_elements = donors_set.intersection(environment_set)
            note = ''
            # если найдутся "доноры"
            if common_elements:
                df_output = pd.DataFrame(
                    columns=['Номер скв Рецепиент', 'Месторождение Рецепиент', 'Доноры', 'Кскв', 'Примечание'])
                df_output.at[0, 'Номер скв Рецепиент'] = self.recipients_dict[recipient][0]
                df_output.at[0, 'Месторождение Рецепиент'] = self.recipients_dict[recipient][1]
                df_output.at[0, 'Доноры'] = ', '.join([self.donors_dict[element][0] for element in common_elements])
                # для каждого донора проверка условий
                for env_well in common_elements:
                    # df_output = df_nag_nag_slice[df_nag_nag_slice["Id_name_environment"] == env_well]
                    # проверка на штуцер
                    check1, df_output = check_shtutser(self.df_modes, self.donors_dict[env_well][0], self.max_date, df_output)

                    if check1:
                        q_current = self.search_q_current(env_well)
                        q_target = self.search_q_target(env_well)
                        note += " -> прошла проверку на штуцер"
                        df_output.at[0, 'Флаг'] = 'Регулирование (Кубышка)'

                        if q_current < q_target:
                            df_output.at[0, 'Кскв донор'] = 'Кскв10'
                        elif q_current > q_target:
                            df_output.at[0, 'Кскв донор'] = 'Кскв11'
                        else:
                            df_output.at[0, 'Кскв донор'] = 'Кскв12'

                        df_output.at[0, 'Примечание'] = note
                    else:
                        # проверка на факт ОПЗ
                        check2, df_output = check_opz(self.df_spectr, env_well, df_output)
                        note += " -> не прошла проверку на штуцер"

                        if check2:
                            q_current = self.search_q_current(env_well)
                            q_target = self.search_q_target(env_well)
                            note += " -> прошла проверку на факт ОПЗ"
                            df_output.at[0, 'Флаг'] = 'Кандидат на ОПЗ (Кубышка)'

                            if q_current < q_target:
                                df_output.at[0, 'Кскв донор'] = 'Кскв10'
                            elif q_current > q_target:
                                df_output.at[0, 'Кскв донор'] = 'Кскв11'
                            else:
                                df_output.at[0, 'Кскв донор'] = 'Кскв12'

                            df_output.at[0, 'Примечание'] = note
                        else:
                            note += " -> не прошла проверку на факт ОПЗ (Кубышка)"
                            df_output.at[0, 'Кскв донор'] = 'Кскв'
                    self.df_donors_and_recipients = pd.concat([self.df_donors_and_recipients, df_output], axis=0,
                                                         ignore_index=True)
            else:
                note += " -> не нашлись доноры"
                df_output = pd.DataFrame(
                    columns=['Номер скв Рецепиент', 'Месторождение Рецепиент', 'Доноры', 'Кскв', 'Примечание'])
                df_output.at[0, 'Номер скв Рецепиент'] = self.recipients_dict[recipient][0]
                df_output.at[0, 'Месторождение Рецепиент'] = self.recipients_dict[recipient][1]
                df_output.at[0, 'Доноры'] = 'Нет доноров'
                df_output.at[0, 'Примечание'] = note
                self.df_donors_and_recipients = pd.concat([self.df_donors_and_recipients, df_output], axis=0, ignore_index=True)

        if not self.df_donors_and_recipients.empty:
            self.df_donors_and_recipients['Кскв реципиент'] = 'Кскв'

        donors_name_list = [self.donors_dict[id_name][0] for id_name in donors_id_name_list]
        recipients_name_list = [self.recipients_dict[id_name][0] for id_name in recipients_id_name_list]
        # Определяем максимальную длину списков
        max_length = max(len(donors_name_list), len(recipients_name_list))

        # Заполняем списки до максимальной длины, используя пустую строку в качестве заполнителя
        donors_id_name_list_padded = donors_name_list + [''] * (max_length - len(donors_name_list))
        recipients_id_name_list_padded = recipients_name_list + [''] * (max_length - len(recipients_name_list))

        # Создаем DataFrame с пустыми столбцами
        all_donors_and_recipients_df = pd.DataFrame(columns=['Список доноров', 'Список рецепиентов'])

        # Присваиваем списки столбцам DataFrame
        all_donors_and_recipients_df['Список доноров'] = donors_id_name_list_padded
        all_donors_and_recipients_df['Список рецепиентов'] = recipients_id_name_list_padded

        return self.df_donors_and_recipients, all_donors_and_recipients_df


