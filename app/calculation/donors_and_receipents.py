import pandas as pd
import os
from app.utils.logger import Logger as logger
from app.calculation.calc import check_shtutser, check_opz
from time import time


def search_q_current(output_result_df: pd.DataFrame, well_name: str):
    '''
    Поиск текущей приемистости для скважины
    :param output_result_df: датафрейм с выходными данными
    :param well_name: имя скважины
    :return: значение текущ приемистости
    '''
    df_local = output_result_df[output_result_df['Id_name_central'] == well_name]
    q_current = df_local['q текущ'].max()
    return q_current


def search_q_target(output_result_df: pd.DataFrame, well_name: str):
    '''
    Поиск целевой приемистости для скважины
    :param output_result_df: датафрейм с выходными данными
    :param well_name: имя скважины
    :return: значение целевой приемистости
    '''
    df_local = output_result_df[output_result_df['Id_name_central'] == well_name]
    q_target = df_local['q цел'].max()
    return q_target


def search_for_donor(output_result_df: pd.DataFrame, df_donors_and_recipients: pd.DataFrame, df_nag_nag: pd.DataFrame, df_modes: pd.DataFrame, df_spectr: pd.DataFrame, recipients_dict: dict, donors_dict: dict, max_date):
    # Создание множества из списка `donors_list`
    donors_id_name_list = donors_dict.keys()
    donors_set = set(donors_id_name_list)

    # Создание множества из списка `recipients_list`
    recipients_id_name_list = recipients_dict.keys()
    recipients_set = set(recipients_id_name_list)

    df_nag_nag = df_nag_nag[df_nag_nag['Id_name_central'].isin(recipients_set)]

    for recipient in recipients_set:
        df_nag_nag_slice = df_nag_nag[df_nag_nag['Id_name_central'] == recipient]

        # Создание множества из столбца `Id_name_environment`
        environment_set = set(df_nag_nag_slice["Id_name_environment"])

        # Проверка наличия общих элементов
        common_elements = donors_set.intersection(environment_set)
        note = ''
        # если найдутся "доноры"
        if common_elements:
            df_output = pd.DataFrame(columns=['Номер скв Рецепиент', 'Месторождение Рецепиент', 'Доноры', 'Кскв', 'Примечание'])
            df_output.at[0, 'Номер скв Рецепиент'] = recipients_dict[recipient][0]
            df_output.at[0, 'Месторождение Рецепиент'] = recipients_dict[recipient][1]
            df_output.at[0, 'Доноры'] = ', '.join([donors_dict[element][0] for element in common_elements])
            # для каждого донора проверка условий
            for env_well in common_elements:
                # df_output = df_nag_nag_slice[df_nag_nag_slice["Id_name_environment"] == env_well]
                # проверка на штуцер
                check1, df_output = check_shtutser(df_modes, donors_dict[env_well][0], max_date, df_output)

                if check1:
                    q_current = search_q_current(output_result_df, env_well)
                    q_target = search_q_target(output_result_df, env_well)
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
                    check2, df_output = check_opz(df_spectr, env_well, df_output)
                    note += " -> не прошла проверку на штуцер"

                    if check2:
                        q_current = search_q_current(output_result_df, env_well)
                        q_target = search_q_target(output_result_df, env_well)
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
                df_donors_and_recipients = pd.concat([df_donors_and_recipients, df_output], axis=0, ignore_index=True)
        else:
            note += " -> не нашлись доноры"
            df_output = pd.DataFrame(columns=['Номер скв Рецепиент', 'Месторождение Рецепиент', 'Доноры', 'Кскв', 'Примечание'])
            df_output.at[0, 'Номер скв Рецепиент'] = recipients_dict[recipient][0]
            df_output.at[0, 'Месторождение Рецепиент'] = recipients_dict[recipient][1]
            df_output.at[0, 'Доноры'] = 'Нет доноров'
            df_output.at[0, 'Примечание'] = note
            df_donors_and_recipients = pd.concat([df_donors_and_recipients, df_output], axis=0, ignore_index=True)

    if not df_donors_and_recipients.empty:
        df_donors_and_recipients['Кскв реципиент'] = 'Кскв'

    donors_name_list = [donors_dict[id_name][0] for id_name in donors_id_name_list]
    recipients_name_list = [recipients_dict[id_name][0] for id_name in recipients_id_name_list]
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

    return df_donors_and_recipients, all_donors_and_recipients_df