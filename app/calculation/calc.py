import pandas as pd
from app.utils.logger import Logger as logger
from dateutil.relativedelta import relativedelta


def increase_qpr(df_well_z1, n_month: int, y: float, df_output, q_current: float, well_name: str) -> tuple:
    '''
    Проверка на увеличение приемистости центральной скважины на Х% Z1 меc. назад
    :param df_well_z1: датафрейм Выгрузка из NGT по скважине, отфильтрованный по дате z1
    :param n_month: z1 месяцев назад
    :param y: погрешность
    :param df_output: датафрейм с выходными данными
    :param q_current: текущая приемистость
    :param well_name: номер скважины (центральной)
    :return:
    '''
    if "all" in df_well_z1['object'].values:
        df_date = df_well_z1[df_well_z1['object'] == "all"]
        q_0 = df_date['q_priemist'].max()
    else:
        q_0 = df_well_z1['q_priemist'].sum()

    # на сколько процентов увеличилось
    if q_0:
        df_output.at[0, f'q0'] = q_0
        percentage_change = ((q_current - q_0) / q_0) * 100
        df_output.at[0, f'Увеличение qпр {n_month} мес назад, %'] = percentage_change
    else:
        logger.warning(f'Значение приемистости {n_month} назад для скважины {well_name} равно нулю')
        return df_output, None

    if percentage_change > y:
        result = True
    else:
        result = False

    return df_output, result


def increase_period_qpr(df_well_z1, df_well_z3, n_month: int, y2: float, df_output, z1: int) -> tuple:
    '''
    Проверка то, что прошло > Z3 мес после ↑ Qпр
    :param df_well_z1: датафрейм Выгрузка из NGT по скважине, отфильтрованный по дате z1
    :param df_well_z3: датафрейм Выгрузка из NGT по скважине, отфильтрованный по дате z3
    :param n_month: z3 месяцев назад
    :param y2: погрешность
    :param df_output: датафрейм с выходными данными
    :param z1: z1 месяцев назад
    :return: кортеж (df_output, result), result = True или False - ответ проверки
    '''
    # получение q приемистости z1 мес назад
    if "all" in df_well_z1['object'].values:
        df_date_z1 = df_well_z1[df_well_z1['object'] == "all"]
        q_value_z1 = df_date_z1['q_priemist'].max()
    else:
        q_value_z1 = df_well_z1['q_priemist'].sum()

    # получение q приемистости z3 мес назад
    if "all" in df_well_z3['object'].values:
        df_date_z3 = df_well_z3[df_well_z3['object'] == "all"]
        q_value_z3 = df_date_z3['q_priemist'].max()
    else:
        q_value_z3 = df_well_z3['q_priemist'].sum()

    if q_value_z1:
        result = ((q_value_z3 - q_value_z1) / q_value_z1) * 100
        df_output.at[0, f'Прошло больше {n_month} мес после ↑ q пр, (q_value_z3 - q_value_zero) / q_value_zero %'] = result
    else:
        result = 0
        df_output.at[0, f'Ошибка'] = f'q пр {z1} мес назад нулевая'

    if result > y2:
        result = True
    else:
        result = False

    return df_output, result


def calc_debits(df, n_month: int, df_output, env_wells: list, compare_dict) -> tuple:
    '''
    Подсчет суммарного дебита воды и нефти для скважин окружения n_month мес назад
    :param df: датафрейм Выгрузка из NGT по скважинам окружения, отфильтрованный по дате z2
    :param n_month: z2 месяцев назад
    :param df_output: датафрейм с выходными данными
    :param env_wells: список скважин окружения
    :param compare_dict: словарь сопоставлений пластов из Окружение НАГ-ДОБ и Выгрузка из NGT
    :return: кортеж (sum_debit_fluid, sum_debit_oil, df_output), sum_debit_fluid, sum_debit_oil - общая сумма дебитов жидкости и нефти для скважин окружения
    '''
    # найдем qж* и qн*
    sum_debit_fluid = 0
    sum_debit_oil = 0

    # цикл по скважинам окружения и по ее пластам
    for well_env_id in env_wells:
        slice_well = df_output[df_output['Id_name_environment'] == well_env_id]
        for plast in slice_well['plast'].unique():
            slice_well_plast = slice_well[slice_well['plast'] == plast]

            if not slice_well_plast.empty:
                k_part = slice_well_plast['K_part'].max()
                # найти дебит этой скважины
                # найти соответствие пласта из файла Наг-Доб для пласта из выгрузки NGT
                plast_ngt_name = compare_dict[plast]

                if plast_ngt_name:
                    # находим скважину в Выгрузке из NGT
                    df_ngt_slice = df[
                        (df['Id_name'] == well_env_id) & (df['object'] == plast_ngt_name)]
                    # df_ngt_slice = df_ngt_slice[(df_ngt_slice['date'].dt.month == date.month) & (df_ngt_slice['date'].dt.year == date.year)]

                    if not df_ngt_slice.empty:
                        index = df_output.loc[
                            (df_output['Id_name_environment'] == well_env_id) & (df_output['plast'] == plast)].index[0]

                        slice_debit_row = df_ngt_slice.reset_index(drop=True)
                        # дебит жидкости
                        debit_fluid = slice_debit_row.at[0, "q_water"]
                        # дебит нефти
                        debit_oil = slice_debit_row.at[0, "q_oil"]

                        sum_debit_fluid += debit_fluid * k_part
                        sum_debit_oil += debit_oil * k_part

                        df_output.at[index, f'qж_{n_month}'] = debit_fluid
                        df_output.at[index, f'qн_{n_month}'] = debit_oil
                        df_output.at[index, f'qж_{n_month}*'] = debit_fluid * k_part
                        df_output.at[index, f'qн_{n_month}*'] = debit_oil * k_part
                    else:
                        df_output.at[0, f'Ошибка'] = f'Не удалось найти скважину {well_env_id} в Выгрузке из NGT'
                else:
                    df_output.at[0, f'Ошибка'] = f'Не удалось сопоставить пласт скважины {well_env_id} в Выгрузке из NGT'

    df_output.at[0, f'qж_{n_month} сумм'] = df_output[f'qж_{n_month}*'].sum()
    df_output.at[0, f'qн_{n_month} сумм'] = df_output[f'qн_{n_month}*'].sum()

    return sum_debit_fluid, sum_debit_oil, df_output


def increase_obvodn(df, n_month: int, y1, sum_debit_fluid_curr, sum_debit_oil_curr, df_output, env_wells: list, well_name: str, compare_dict):
    '''
    Проверка на снижение по обводненности на Y1% Z2 месяцев назад
    :param df: датафрейм Выгрузка из NGT по скважинам окружения, отфильтрованный по дате z2
    :param n_month: z2 месяцев назад
    :param y1: погрешность
    :param sum_debit_fluid_curr: суммарный дебит жидкости окружения в текущую дату
    :param sum_debit_oil_curr: суммарный дебит нефти окружения в текущую дату
    :param df_output: датафрейм с выходными данными
    :param env_wells: список скважин окружения
    :param well_name: номер скважины (центральной)
    :param compare_dict: словарь сопоставлений пластов из Окружение НАГ-ДОБ и Выгрузка из NGT
    :return: кортеж (df_output, result), result = True или False - ответ проверки
    '''
    # найдем дебит нефти и обводненность ((qж – qн) / qж) в данный месяц (по всем пластам, умножая на коэфф участия)
    # start_date = date_curr - relativedelta(months=n_month)

    sum_debit_fluid_z2, sum_debit_oil_z2, df_output = calc_debits(df, n_month, df_output, env_wells, compare_dict)

    obvodn_z2 = (sum_debit_fluid_z2 - sum_debit_oil_z2) / sum_debit_fluid_z2
    obvodn_curr = (sum_debit_fluid_curr - sum_debit_oil_curr) / sum_debit_fluid_curr

    df_output.at[0, f'Обв_z2'] = obvodn_z2
    df_output.at[0, f'Обв текущ'] = obvodn_curr

    result = ((sum_debit_oil_curr + sum_debit_oil_z2) / 2) * (obvodn_curr - obvodn_z2)
    df_output.at[0, f'Δ по обв за {n_month} мес, т. / сут'] = result
    if sum_debit_oil_z2:
        result /= sum_debit_oil_z2
        result *= 100
        # logger.info(f'delta по обв-ти на {result} %')
        df_output.at[0, f'Δ по обв за {n_month} мес, %'] = result
    else:
        logger.warning(f'Не удалось получить значение дебита нефти для 0-го месяца для скв {well_name}')
        df_output.at[0, f'Ошибка'] = f'Не удалось получить значение дебита нефти для 0-го месяца для скв {well_name}'

    if result > y1:
        result = True
    else:
        result = False

    return df_output, result


def const_q_n(df, n_month: int, sum_debit_oil_curr, df_output, env_wells: list, y3, compare_dict):
    '''
    """Проверка на q нефти const за Z2 мес назад"""
    :param df: датафрейм Выгрузка из NGT по скважинам окружения, отфильтрованный по дате z2
    :param n_month: z2 месяцев назад
    :param sum_debit_oil_curr: суммарный дебит нефти окружения в текущую дату
    :param df_output: датафрейм с выходными данными
    :param env_wells: список скважин окружения
    :param y3: погрешность
    :param compare_dict: словарь сопоставлений пластов из Окружение НАГ-ДОБ и Выгрузка из NGT
    :return: кортеж (df_output, result), result = True или False - ответ проверки
    '''

    # найдем дебит нефти и обводненность ((qж – qн) / qж) в данный месяц (по всем пластам, умножая на коэфф участия?)
    # start_date = date_curr - relativedelta(months=n_month)

    sum_debit_fluid_z2, sum_debit_oil_z2, df_output = calc_debits(df, n_month, df_output, env_wells, compare_dict)

    if sum_debit_oil_z2 * (1 - y3 / 100) <= sum_debit_oil_curr <= sum_debit_oil_z2 * (1 + y3 / 100):
        result = True
    else:
        result = False

    return result


def increase_q_water(df, n_month: int, y2, sum_debit_fluid_curr, df_output, env_wells: list, compare_dict):
    '''
    Проверка на рост qж на Y2% за Z2 мес
    :param df: датафрейм Выгрузка из NGT по скважинам окружения, отфильтрованный по дате z2
    :param n_month: z2 месяцев назад
    :param y2: погрешность
    :param sum_debit_fluid_curr: суммарный дебит жидкости окружения в текущую дату
    :param df_output: датафрейм с выходными данными
    :param env_wells: список скважин окружения
    :param compare_dict: словарь сопоставлений пластов из Окружение НАГ-ДОБ и Выгрузка из NGT
    :return: кортеж (df_output, result), result = True или False - ответ проверки
    '''
    # найдем дебит нефти и обводненность ((qж – qн) / qж) в данный месяц (по всем пластам, умножая на коэфф участия?)
    # start_date = date_curr - relativedelta(months=n_month)

    sum_debit_fluid_z2, sum_debit_oil_z2, df_output = calc_debits(df, n_month, df_output, env_wells, compare_dict)

    # obvodn_start = (sum_debit_fluid_start - sum_debit_oil_start) / sum_debit_fluid_start
    # obvodn_end = (sum_debit_fluid_end - sum_debit_oil_end) / sum_debit_fluid_end

    # result = ((sum_debit_fluid_end - sum_debit_oil_start) / 2) * (1 - (obvodn_end + obvodn_start) / 200)
    result = (sum_debit_fluid_curr - sum_debit_fluid_z2) / sum_debit_fluid_z2
    result *= 100

    # logger.info(f'Изменение по qж {n_month} назад на {result} %')
    df_output.at[0, f'Изменение по qж {n_month} назад, %'] = result

    # eps = 10
    # if y2 - eps <= result <= y2 + eps:
    #     result = True
    # else:
    #     result = False

    if result > y2:
        result = True
    else:
        result = False

    return df_output, result


def check_shtutser(df_modes, name_well: str, max_date, df_output):
    '''
    Проверка на наличие штуцера
    :param df_modes: датафрейм с тех режимами
    :param name_well: номер скважины (центральной)
    :param max_date: максимальная дата (текущая)
    :param df_output: датафрейм с выходными данными
    :return: кортеж (df_output, result), result = True или False - ответ проверки
    '''
    df_modes_well = df_modes[df_modes['name_well'] == name_well]
    df_modes_well = df_modes_well[(df_modes_well['date'].dt.month == max_date.month) & (df_modes_well['date'].dt.year == max_date.year)]
    df_modes_well = df_modes_well.reset_index(drop=True)

    t = False

    if not df_modes_well.empty:
        p_bg = df_modes_well["p_bg"].max()
        df_output.at[0, f'Р бг'] = p_bg
        p_ust = df_modes_well["p_ust"].max()
        df_output.at[0, 'P уст'] = p_ust

        d_sht = df_modes_well.at[0, "d_sht"]

        if p_bg - p_ust > 10:
            df_output.at[0, 'Р бг - P уст'] = p_bg - p_ust
            df_output.at[0, 'Dшт'] = d_sht
            if d_sht < 18:
                df_output.at[0, f'Наличие штуцера'] = f'Есть, диаметр штуцера соответствует'
            else:
                df_output.at[0, f'Наличие штуцера'] = f'Есть, но диаметр штуцера > 18'
            t = True
        else:
            if d_sht < 18:
                df_output.at[0, f'Наличие штуцера'] = f'Разница давлений < 10, но диаметр штуцера меньше 18'
                logger.warning(f'Для скважины {name_well} разница давлений < 10, но диаметр штуцера меньше 18')
            else:
                # logger.info(f'Для скважины {name_well} штуцера нет')
                df_output.at[0, f'Наличие штуцера'] = 'Нет, d_sht > 18, разница давлений < 10 '
    else:
        df_output.at[0, f'Ошибка'] = f'Не удалось найти данную скважину в файлах с тех режимами'
        print(f'Для скв {name_well} не сопоставился файл с тех. режимами')

    return t, df_output


def check_opz(df_spectr, id_name: str, df_output):
    '''
    Проверка на кандидата ОПЗ
    :param df_spectr: датафрейм файла СПЕКТР
    :param id_name: id номер скважины (+месторождение)
    :param df_output: датафрейм с выходными данными
    :return: кортеж (df_output, result), result = True или False - ответ проверки
    '''
    df_spectr = df_spectr[df_spectr['Id_name'] == id_name]

    if not df_spectr.empty:
        t = True
        # logger.info(f'Скважина {id_name} является кандидатом на ОПЗ')
        df_output.at[0, f'Кандидат на ОПЗ'] = '+'
    else:
        t = False
        # logger.info(f'Скважина {id_name} не является кандидатом на ОПЗ')
        df_output.at[0, f'Кандидат на ОПЗ'] = '-'
    return t, df_output


def check_P_zac(df_modes, name_well: str, p_zac_value: float, df_output) -> pd.DataFrame:
    '''
    Проверка на увеличение Рзак
    :param df_modes: датафрейм с тех режимами
    :param name_well: номер скважины (центральной)
    :param p_zac_value: P закачки (указанный пользователем)
    :param df_output: датафрейм с выходными данными
    :return: кортеж (df_output, result), result = True или False - ответ проверки
    '''
    df_modes_well = df_modes[df_modes['name_well'] == name_well]
    df_modes_well = df_modes_well.reset_index(drop=True)

    if not df_modes_well.empty:
        p_ust = df_modes_well["p_ust"].max()
        p_ust_pr = df_modes_well["p_ust_pr"].max()
        df_output.at[0, 'P уст'] = p_ust
        df_output.at[0, 'Р уст. пр'] = p_ust_pr

        if p_ust - p_ust_pr > p_zac_value:
            print("Проверьте на увеличение P закачки")
            df_output.at[0, f'↑ Рзак'] = f'Проверьте на ↑ Рзак'
            # logger.info(f'Скважина {name_well}: Проверьте на увеличение P закачки')
        else:
            df_output.at[0, f'↑ Рзак'] = f'Не требуется'
    else:
        df_output.at[0, f'Ошибка'] = f'Не удалось найти данную скважину в файлах с тех режимами'

    return df_output


def check_factors(q_target: float, well_name: str, q_current: float,  df_ngt_z1, df_ngt_z2, df_ngt_z3, id_name, max_date, plast_list: list, env_wells: list, sum_debit_fluid, sum_debit_oil, p_zac_value, df_output, df_modes, df_spectr, recipients_dict, donors_dict: dict, mest: str, compare_dict, z1, z2, z3, x, y1, y2, y3):
    # проверка на увеличение q приемистости на x% z месяцев назад
    note = ''
    # z1, z2, z3 = 6, 2, 3
    # x, y1, y2 = 10, 20, 10

    df_well_z1 = df_ngt_z1[df_ngt_z1['Id_name'] == id_name]
    df_well_z3 = df_ngt_z3[df_ngt_z3['Id_name'] == id_name]
    df_output, check = increase_qpr(df_well_z1, z1, x, df_output, q_current, well_name)

    if check is None:
        df_output.at[0, 'Кскв'] = '1 (нулевая приемистость)'
        note += 'нулевая приемистость'
        df_output.at[0, 'Примечание'] = note
        return df_output

    # датафрейм по окружению
    df_env_wells = df_ngt_z2[df_ngt_z2['Id_name'].isin(env_wells)]

    # проверка на увеличение приемистости на X%
    if check:
        note += f"↑ q пр на {x}% за {z1} мес"
        df_output, check1 = increase_obvodn(df_env_wells, z2, y1, sum_debit_fluid, sum_debit_oil, df_output, env_wells, well_name, compare_dict)

        # проверка на снижение по обводненности на Y1%
        if check1:
            note += f" -> ↓ qн по по обв-ти на {y1}% за {z2} мес"
            # сравнение q текущ с q целевое
            df_output.at[0, 'Флаг'] = 'Штуцер'
            if q_current < q_target:
                k_skw = 0.9
                df_output.at[0, 'Кскв'] = 'Кскв1'
            elif q_current > q_target:
                k_skw = 0.8
                df_output.at[0, 'Кскв'] = 'Кскв3'
            else:
                k_skw = 0.85
                df_output.at[0, 'Кскв'] = 'Кскв2'

            df_output.at[0, 'Примечание'] = note
            return df_output

        # проверка на снижение по обводненности на Y1% не прошла
        else:
            note += f" -> не было ↓ qн по по обв-ти на {y1}% за {z2} мес"
            df_output, check12 = increase_q_water(df_env_wells, z2, y2, sum_debit_fluid, df_output, env_wells, compare_dict)

            # проверка на рост qж на Y2%
            if check12:
                note += f" -> рост qж  на {y2}% за {z2} мес"
                check121 = const_q_n(df_env_wells, z2, sum_debit_oil, df_output, env_wells, y3, compare_dict)

                # проверка на qн const за Z2 мес
                if check121:
                    note += f" -> qn const за {z2} мес"
                    # сравнение q текущ с q целевое
                    df_output.at[0, 'Флаг'] = 'Штуцер'
                    if q_current < q_target:
                        df_output.at[0, 'Кскв'] = 'Кскв4'
                    elif q_current > q_target:
                        df_output.at[0, 'Кскв'] = 'Кскв5'
                    else:
                        df_output.at[0, 'Кскв'] = 'Кскв6'

                    df_output.at[0, 'Примечание'] = note
                    return df_output

                    # проверка на qн const за Z2 мес не прошла
                else:
                    note += f" -> не было qn const за {z2} мес -> Кубышка «Доноры»"
                    # добавляем в кубышку
                    donors_dict[id_name] = [well_name, mest]
                    df_output.at[0, 'Примечание'] = note
                    return df_output
            else:
                note += f" -> нет роста qж на {y2}% за {z2} мес"
                df_output, check122 = increase_period_qpr(df_well_z1, df_well_z3, z3, y2, df_output, z1)

                if check122:
                    note += f" -> прошло > {z3} мес после ↑ qпр"
                    # проверка на штуцер
                    check1221, df_output = check_shtutser(df_modes, well_name, max_date, df_output)

                    # если штуцер есть
                    if check1221:
                        note += " -> штуцер есть"
                        # сравнение q текущ с q целевое
                        df_output.at[0, 'Флаг'] = 'Регулирование'
                        if q_current < q_target:
                            df_output.at[0, 'Кскв'] = 'Кскв13'
                        elif q_current > q_target:
                            df_output.at[0, 'Кскв'] = 'Кскв14'
                        else:
                            df_output.at[0, 'Кскв'] = 'Кскв15'

                        df_output.at[0, 'Примечание'] = note
                        return df_output
                    # если штуцера нет, смотрим Кандидат на ОПЗ
                    else:
                        note += " -> нет штуцера"
                        check12212, df_output = check_opz(df_spectr, id_name, df_output)

                        if check12212:
                            # Кандидат на ОПЗ
                            note += " -> является кандидатом на ОПЗ"
                            df_output.at[0, 'Флаг'] = 'ОПЗ'
                            if q_current < q_target:
                                df_output.at[0, 'Кскв'] = 'Кскв13'
                            elif q_current > q_target:
                                df_output.at[0, 'Кскв'] = 'Кскв14'
                            else:
                                df_output.at[0, 'Кскв'] = 'Кскв15'

                            df_output.at[0, 'Примечание'] = note
                            return df_output
                        # если не кандидат на ОПЗ
                        else:
                            note += " -> не кандидат на ОПЗ -> проверка на ↑ Рзак -> добавление в реципиенты"
                            # проверка на увеличение P закачки

                            df_output = check_P_zac(df_modes, well_name, p_zac_value, df_output)
                            df_output.at[0, 'Кскв'] = 'Добавление в список Реципиенты'
                            df_output.at[0, 'Примечание'] = note
                            # recipients_list.append([id_name, well_name, mest])
                            recipients_dict[id_name] = [well_name, mest]
                            return df_output
                            # # Добавление в Кубышку
                            # df_output.loc[0, 'Кскв'] = 'Добавление в Кубышку'
                            # return df_output
                else:
                    note += f" -> не прошло > {z3} мес после ↑ qпр"
                    df_output.at[0, 'Кскв'] = 'Кскв'

                    df_output.at[0, 'Примечание'] = note
                    return df_output

    else:
        note += f"не было ↑ q пр на {x}% за {z1} мес"
        df_output, check2 = increase_obvodn(df_env_wells, z2, y1, sum_debit_fluid, sum_debit_oil, df_output, env_wells, well_name, compare_dict)
        # проверка на снижение по обводненности на Y1%
        if check2:
            note += f" -> ↓ qн по по обв-ти на {y1}% за {z2} мес"
            df_output.at[0, 'Флаг'] = 'Регулирование'
            # сравнение q текущ с q целевое
            if q_current < q_target:
                df_output.at[0, 'Кскв'] = 'Кскв16'
            elif q_current > q_target:
                df_output.at[0, 'Кскв'] = ' Кскв17'
            else:
                df_output.at[0, 'Кскв'] = 'Кскв18'

            df_output.at[0, 'Примечание'] = note
            return df_output
        # проверка на снижение по обводненности на Y1% не прошла
        else:
            note += f" -> не было ↓ qн по по обв-ти на {y1}% за {z2} мес"
            df_output, check22 = increase_q_water(df_env_wells, z2, y2, sum_debit_fluid, df_output, env_wells, compare_dict)
            if check22:
                note += f" -> рост qж  на {y2}% за {z2} мес"
                df_output.at[0, 'Флаг'] = 'Контроль окружения'
                df_output.at[0, 'Кскв'] = 'Кскв'
                df_output.at[0, 'Примечание'] = note
                return df_output
            else:
                note += f" -> не было роста qж  на {y2}% за {z2} мес"
                # проверка на штуцер
                check222, df_output = check_shtutser(df_modes, well_name, max_date, df_output)

                if check222:
                    note += " -> штуцер есть"
                    # сравнение q текущ с q целевое
                    df_output.at[0, 'Флаг'] = 'Регулирование'
                    if q_current < q_target:
                        df_output.at[0, 'Кскв'] = 'Кскв19'
                    elif q_current > q_target:
                        df_output.at[0, 'Кскв'] = ' Кскв20'
                    else:
                        df_output.at[0, 'Кскв'] = 'Кскв21'

                    df_output.at[0, 'Примечание'] = note
                    return df_output
                else:
                    note += " -> штуцера нет"
                    check2222, df_output = check_opz(df_spectr, id_name, df_output)

                    if check2222:
                        note += " -> является кандидатом на ОПЗ"
                        df_output.at[0, 'Флаг'] = 'ОПЗ'
                        if q_current < q_target:
                            df_output.at[0, 'Кскв'] = 'Кскв19'
                        elif q_current > q_target:
                            df_output.at[0, 'Кскв'] = ' Кскв20'
                        else:
                            df_output.at[0, 'Кскв'] = 'Кскв21'

                        df_output.at[0, 'Примечание'] = note
                        return df_output

                    # проверка на увеличение P закачки
                    df_output = check_P_zac(df_modes, well_name, p_zac_value, df_output)
                    df_output.at[0, 'Кскв'] = 'Добавление в список Реципиенты'
                    note += " -> не кандидат на ОПЗ -> проверка на ↑ Рзак -> добавление в реципиенты"
                    df_output.at[0, 'Примечание'] = note

                    recipients_dict[id_name] = [well_name, mest]
                    return df_output


