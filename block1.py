import pandas as pd
from dateutil.relativedelta import relativedelta
import os


def get_q_priemist(df):
    '''
    Получение значения q текущее
    :param df: датафрейм Выгрузка из NGT по скважине на дату
    :return:
    '''
    # получение значения q текущее
    # используется значение пласта ALL, если есть. Иначе суммируется по имеющимся пластам
    row_all = df[df['object'] == 'all']
    q_current = row_all['q_priemist'].values[0] if not row_all.empty else None
    if q_current is None:
        row_all = df[df['object'] != 'all']
        q_current = row_all['q_priemist'].sum()

    return q_current


def check_block1(df: pd.DataFrame, id_name: str, df_opz: pd.DataFrame, df_vpp: pd.DataFrame, date_start_opz, date_start_vpp, max_date, q_current, df_ngt):
    '''
    Проверка на очаговость, Факт ОПЗ и ВПП
    :param df: датафрейм Выгрузка из NGT по скважине на максимальную дату
    :param id_name: id скважина+месторождение
    :param df_opz: файл ОПЗ
    :param df_vpp: файл ВПП
    :param date_start_opz: дата начала проверки на факт ОПЗ
    :param date_start_vpp: дата начала проверки на факт ВПП
    :param max_date: максимальная дата
    :param q_current: премистость текущая
    :param df_ngt: Файл Выгрузка из NGT
    :return: кортеж из (t, block1_result_df), если t=True, то скважина выходит из цикла, block1_result_df - данные о скважинах, прошедших какую-либо из проверок
    '''
    block1_result_df = pd.DataFrame(columns=["Скважина", "Комментарий", "Кскв", "q расч"])

    # Проверка на очаговость
    # «Проектное назначение скважины» стоит «НЕФТЯНЫЕ»
    # df_check_ojag = df[df["design purpose of the well"].str.startswith("неф")]
    check_condition = df["design purpose of the well"].str.startswith("неф").all()

    t = True

    if check_condition:
        data_series = pd.DataFrame({"Скважина": [id_name], "Комментарий": ["Очаговая"], "Кскв": [''], "q расч": ["60-80"]})
        block1_result_df = pd.concat([block1_result_df, data_series], ignore_index=True)

        t = False
        return t, block1_result_df

    # проверка на опз
    df_opz_local = df_opz[df_opz['Id_name'] == id_name]
    # mask = (df_opz_local['date'].dt.month > date_start_opz.month) & (df_opz_local['date'].dt.month <= max_date.month) & (df_opz_local['date'].dt.year > date_start_opz.year) & (df_opz_local['date'].dt.year <= max_date.year)
    mask = (df_opz_local['date'] > date_start_opz) & (df_opz_local['date'] <= max_date)
    df_opz_in_interval = df_opz_local[mask]

    if not df_opz_in_interval.empty:
        data_series = pd.DataFrame(
            {"Скважина": [id_name], "Комментарий": ["Прошла проверку на факт ОПЗ"], "Кскв": [1], "q расч": [q_current]})
        block1_result_df = pd.concat([block1_result_df, data_series], ignore_index=True)

        t = False
        return t, block1_result_df

    # проверка на впп
    df_vpp_local = df_vpp[df_vpp['Id_name'] == id_name]
    # mask = (df_vpp_local['date'].dt.month > date_start_vpp.month) & (df_vpp_local['date'].dt.month <= max_date.month) & (df_vpp_local['date'].dt.year > date_start_vpp.month) & (df_vpp_local['date'].dt.year <= max_date.year)
    mask = (df_vpp_local['date'] > date_start_vpp) & (df_vpp_local['date'] <= max_date)
    df_vpp_in_interval = df_vpp_local[mask]

    if not df_vpp_in_interval.empty:
        df_vpp = df_vpp_in_interval['date'].max()
        date_vpp_before = df_vpp - relativedelta(months=1)
        df_ngt_vpp_before = df_ngt[(df_ngt["Id_name"] == id_name) & (df_ngt['date'].dt.month == date_vpp_before.month) & (df_ngt['date'].dt.year == date_vpp_before.year)]
        q_value = get_q_priemist(df_ngt_vpp_before)
        if not q_value:
            df_ngt_vpp = df_ngt[(df_ngt["Id_name"] == id_name) & (df_ngt['date'].dt.month == df_vpp.month) & (df_ngt['date'].dt.year == df_vpp.year)]
            q_value = get_q_priemist(df_ngt_vpp)
        data_series = pd.DataFrame(
            {"Скважина": [id_name], "Комментарий": ["Прошла проверку на факт ВПП"], "Кскв": [''], "q расч": [q_value]})
        block1_result_df = pd.concat([block1_result_df, data_series], ignore_index=True)

        t = False
        return t, block1_result_df

    return t, block1_result_df







