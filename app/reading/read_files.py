import pandas as pd
import chardet
from app.utils.logger import Logger as logger


def read_opz(opz_path: str) -> pd.DataFrame:
    df_opz = pd.read_excel(opz_path, skiprows=4, nrows=300)
    df_opz = df_opz.drop(2)
    # выборка столбцов
    df_opz = df_opz.rename(columns=lambda x: x.strip())
    selected_columns = ["Дата  ОПЗ", "Месторождение", "Номер скважины"]
    df_opz = df_opz[selected_columns]
    # Переименовываем столбцы
    df_opz.rename(columns={"Дата  ОПЗ": "date", "Месторождение": "mest", "Номер скважины": 'name_well'}, inplace=True)
    # Приводим к строчному типу
    df_opz['mest'] = df_opz['mest'].astype(str)
    df_opz['name_well'] = df_opz['name_well'].astype(str)
    df_opz['Id_name'] = df_opz['name_well'].str.strip().str.lower() + df_opz['mest'].str.strip().str.lower()
    df_opz['date'] = pd.to_datetime(df_opz['date'])
    return df_opz


def read_vpp(vpp_path: str) -> pd.DataFrame:
    df_vpp = pd.read_excel(vpp_path)
    # выборка столбцов
    df_vpp = df_vpp.rename(columns=lambda x: x.strip())
    selected_columns = ["Дата обработки", "Месторождение", "Скважина"]
    df_vpp = df_vpp[selected_columns]
    # Переименовываем столбцы
    df_vpp.rename(columns={"Дата обработки": "date", "Месторождение": "mest", "Скважина": 'name_well'}, inplace=True)
    # Приводим к строчному типу
    df_vpp['mest'] = df_vpp['mest'].astype(str)
    df_vpp['name_well'] = df_vpp['name_well'].astype(str)
    df_vpp['Id_name'] = df_vpp['name_well'].str.strip().str.lower() + df_vpp['mest'].str.strip().str.lower()
    df_vpp['date'] = pd.to_datetime(df_vpp['date'])
    return df_vpp


def read_ngt(ngt_path) -> pd.DataFrame:
    df_ngt = pd.read_excel(ngt_path)
    # выборка столбцов
    df_ngt = df_ngt.rename(columns=lambda x: x.strip())
    selected_columns = ["Дата", "Месторождение", "№ скважины", "Дебит жидкости за последний месяц, т/сут",
                        "Дебит нефти за последний месяц, т/сут", "Приемистость за последний месяц, м3/сут", "Объект",
                        "Состояние", "Характер работы", "Проектное назначение скважины", "КНС (ТР)", "Куст"]
    df_ngt = df_ngt[selected_columns]
    # Переименовываем столбцы
    df_ngt.rename(columns={"Дата": "date", "Месторождение": "mest", "№ скважины": 'name_well',
                           "Дебит жидкости за последний месяц, т/сут": "q_water",
                           "Дебит нефти за последний месяц, т/сут": "q_oil",
                           "Приемистость за последний месяц, м3/сут": "q_priemist", "Объект": "object",
                           "Состояние": "state", "Характер работы": "nature_of_work",
                           "Проектное назначение скважины": "design purpose of the well", "КНС (ТР)": "kns",
                           "Куст": "bush"}, inplace=True)

    df_ngt["design purpose of the well"] = df_ngt['design purpose of the well'].astype(str).str.lower().str.strip()
    df_ngt["nature_of_work"] = df_ngt['nature_of_work'].astype(str).str.lower().str.strip()
    df_ngt['state'] = df_ngt['state'].astype(str).str.lower()
    df_ngt['object'] = df_ngt['object'].astype(str).str.lower().str.strip()
    # df_ngt = df_ngt[df_ngt["object"] == object_name]
    # df_ngt = df_ngt[df_ngt["q_priemist"].notna() & (df_ngt["q_priemist"] != 0)]

    df_ngt['mest'] = df_ngt['mest'].astype(str).str.lower()
    df_ngt['name_well'] = df_ngt['name_well'].astype(str).str.lower()
    df_ngt['Id_name_full'] = df_ngt['name_well'].str.strip() + df_ngt['mest'].str.strip() + df_ngt['object'].str.strip()
    df_ngt['Id_name'] = df_ngt['name_well'].str.strip() + df_ngt['mest'].str.strip()
    df_ngt['date'] = pd.to_datetime(df_ngt['date'], format='%d.%m.%Y', errors='coerce')
    return df_ngt


def read_tsk(tsk_path: str) -> pd.DataFrame:
    df_tsk_trial = pd.read_excel(tsk_path, sheet_name="compensation", skiprows=1, nrows=10)
    # выборка столбцов
    df_tsk_trial = df_tsk_trial.rename(columns=lambda x: x.strip())
    compensation_name = "Max отнош. НАГ/ДОБ"
    column_index = df_tsk_trial.columns.get_loc(compensation_name)

    df_tsk = pd.read_excel(tsk_path, sheet_name="compensation")
    df_tsk = df_tsk.drop(0)
    df_tsk = df_tsk.rename(columns=lambda x: x.strip())
    df_tsk = df_tsk.rename(columns={df_tsk.columns[column_index]: compensation_name})
    selected_columns = ["Месторождение", "Пласт", "Ячейка", "Max отнош. НАГ/ДОБ"]
    df_tsk = df_tsk[selected_columns]
    # Переименовываем столбцы
    df_tsk.rename(columns={"Месторождение": "mest", "Пласт": "object", "Ячейка": "cell+object",
                           "Max отнош. НАГ/ДОБ": "compensation"}, inplace=True)
    # Приводим к строчному типу
    df_tsk['mest'] = df_tsk['mest'].astype(str)
    df_tsk['cell+object'] = df_tsk['cell+object'].astype(str).str.lower()
    df_tsk['object'] = df_tsk['object'].astype(str).str.lower()

    # добавим ячейку

    def process_cell_object(row):
        # Разбиваем строку по разделителю "_"
        parts = row['cell+object'].split('_')

        # Если 'object' входит в 'cell+object', убираем его
        if "all" in row['cell+object']:
            parts.remove("all")

        result = '_'.join(parts)

        return result

    # Создаем новый столбец с результатами
    df_tsk['cell'] = df_tsk.apply(process_cell_object, axis=1)
    return df_tsk


def read_spectr(spectr_path: str) -> pd.DataFrame:
    df_spectr = pd.read_excel(spectr_path, skiprows=1)
    # выборка столбцов
    df_spectr = df_spectr.rename(columns=lambda x: x.strip())
    selected_columns = ["Месторождение", "Скважина"]
    df_spectr = df_spectr[selected_columns]
    # Переименовываем столбцы
    df_spectr.rename(columns={"Месторождение": "mest", "Скважина": "name_well"}, inplace=True)
    # Приводим к строчному типу
    df_spectr['mest'] = df_spectr['mest'].astype(str)
    df_spectr['name_well'] = df_spectr['name_well'].astype(str)
    df_spectr['Id_name'] = df_spectr['name_well'].str.strip().str.lower() + df_spectr['mest'].str.strip().str.lower()
    return df_spectr


def read_nag_dob(nag_dob_path: str) -> pd.DataFrame:
    df_nag_dob = pd.read_excel(nag_dob_path, sheet_name="Скважины с окружением")
    # выборка столбцов
    df_nag_dob = df_nag_dob.rename(columns=lambda x: x.strip())
    selected_columns = ["Дата", "Месторождение", "Скв. окружения", "Центр. скв.", "Пласт", "Коэффициент участия"]
    df_nag_dob = df_nag_dob[selected_columns]
    # Переименовываем столбцы
    df_nag_dob.rename(columns={"Дата": "date", "Месторождение": "mest", "Скв. окружения": "well_environment",
                               "Центр. скв.": "well_central", "Пласт": "plast", "Коэффициент участия": "K_part"},
                      inplace=True)
    # Приводим к строчному типу
    df_nag_dob['mest'] = df_nag_dob['mest'].astype(str)
    df_nag_dob['plast'] = df_nag_dob['plast'].astype(str)
    df_nag_dob['plast'] = df_nag_dob['plast'].str.strip().str.lower()
    df_nag_dob['well_environment'] = df_nag_dob['well_environment'].astype(str).str.lower().str.strip()
    df_nag_dob["well_central"] = df_nag_dob['well_central'].astype(str).str.lower().str.strip()
    df_nag_dob['Id_name_central'] = df_nag_dob['well_central'] + df_nag_dob[
        'mest'].str.strip().str.lower()
    df_nag_dob['Id_name_environment'] = df_nag_dob['well_environment'] + df_nag_dob[
        'mest'].str.strip().str.lower()
    df_nag_dob['Id_name_environment_full'] = df_nag_dob['Id_name_environment'] + df_nag_dob['plast']
    df_nag_dob['date'] = pd.to_datetime(df_nag_dob['date'], format='%d.%m.%Y', errors='coerce')
    # df_nag_dob['date'] = df_nag_dob['date'].dt.strftime("%d.%m.%Y")
    return df_nag_dob


def read_nag_nag(nag_nag_path) -> pd.DataFrame:
    df_nag_nag = pd.read_excel(nag_nag_path, sheet_name="Скважины с окружением")
    # выборка столбцов
    df_nag_nag = df_nag_nag.rename(columns=lambda x: x.strip())
    selected_columns = ["Дата", "Месторождение", "Скв. окружения", "Центр. скв.", "Пласт", "Коэффициент участия"]
    df_nag_nag = df_nag_nag[selected_columns]
    # Переименовываем столбцы
    df_nag_nag.rename(columns={"Дата": "date", "Месторождение": "mest", "Скв. окружения": "well_environment",
                               "Центр. скв.": "well_central", "Пласт": "plast", "Коэффициент участия": "K_part"},
                      inplace=True)
    # Приводим к строчному типу
    df_nag_nag['mest'] = df_nag_nag['mest'].astype(str).str.strip().str.lower()
    df_nag_nag['well_environment'] = df_nag_nag['well_environment'].astype(str).str.lower().str.strip()
    df_nag_nag["well_central"] = df_nag_nag['well_central'].astype(str).str.lower().str.strip()
    df_nag_nag['Id_name_central'] = df_nag_nag['well_central'] + df_nag_nag[
        'mest'].str.strip().str.lower()
    df_nag_nag['Id_name_environment'] = df_nag_nag['well_environment'] + df_nag_nag[
        'mest'].str.strip().str.lower()

    return df_nag_nag


def read_ku(ku_path: str) -> pd.DataFrame:
    # object_name = get_object_name(ku_path)
    # Определение кодировки
    with open(ku_path, 'rb') as f:
        result = chardet.detect(f.read(500))

    df_ku = pd.read_csv(ku_path, sep=';', encoding=result['encoding'])
    # выборка столбцов
    df_ku = df_ku.rename(columns=lambda x: x.strip())
    selected_columns = ["Ячейка", "скв", "Вес", "Дата"]
    df_ku = df_ku[selected_columns]
    # Переименовываем столбцы
    df_ku.rename(columns={"Ячейка": "cell+object", "скв": "name_well", "Вес": "weight", "Дата": "date"}, inplace=True)
    # Приводим к строчному типу
    df_ku['weight'] = df_ku['weight'].astype(float)
    df_ku['cell+object'] = df_ku['cell+object'].astype(str).str.lower().str.strip()
    df_ku['name_well'] = df_ku['name_well'].astype(str).str.lower().str.strip()
    # df_nag_dob['Id_name'] = df_nag_dob['name_well'].str.strip().str.lower() + df_nag_dob['mest'].str.strip().str.lower()
    df_ku['date'] = pd.to_datetime(df_ku['date'], format='%d.%m.%Y', errors='coerce')
    # df_ku['date'] = df_ku['date'].dt.strftime("%d.%m.%Y")
    # df_ku["object"] = object_name.lower()

    def get_object_name(row):
        parts = row['cell+object'].split('_')

        if parts:
            name_object = parts[1]
            return str(name_object).lower()

    def process_cell_object(row):
        # Разбиваем строку по разделителю "_"
        parts = row['cell+object'].split('_')

        # Если 'object' входит в 'cell+object', убираем его
        if row['object'] in row['cell+object']:
            parts.remove(row['object'])

        result = '_'.join(parts)

        return result

    # Создаем новый столбец с результатами
    df_ku["object"] = df_ku.apply(get_object_name, axis=1)
    df_ku['cell'] = df_ku.apply(process_cell_object, axis=1)
    return df_ku


def read_models(modes_path) -> pd.DataFrame:
    df_modes = pd.read_excel(modes_path, skiprows=1)
    # выборка столбцов
    df_modes = df_modes.rename(columns=lambda x: x.strip())
    selected_columns = ["Номер скважины", "Пласт", "Дата", "Dшт", "Р бг", "P уст", "Р уст. пр"]
    df_modes = df_modes[selected_columns]
    # Переименовываем столбцы
    df_modes.rename(
        columns={"Номер скважины": "name_well", "Пласт": "object", "Дата": "date", "Dшт": "d_sht", "Р бг": "p_bg",
                 "P уст": "p_ust", "Р уст. пр": "p_ust_pr"}, inplace=True)
    # Приводим к строчному типу
    df_modes['name_well'] = df_modes['name_well'].astype(str).str.lower().str.strip()
    df_modes['object'] = df_modes['object'].astype(str)
    df_modes['p_bg'] = df_modes['p_bg'].astype(float)
    df_modes['p_ust'] = df_modes['p_ust'].astype(float)
    df_modes['p_ust_pr'] = df_modes['p_ust_pr'].astype(float)
    df_modes['date'] = pd.to_datetime(df_modes['date'])
    return df_modes


def read_pvt(pvt_path) -> pd.DataFrame:
    df_pvt = pd.read_excel(pvt_path)
    # выборка столбцов
    df_pvt = df_pvt.rename(columns=lambda x: x.strip())
    selected_columns = ["месторождение", "объект", "Bн", "B в", "ρ н", "ρ в"]
    df_pvt = df_pvt[selected_columns]
    # Переименовываем столбцы
    df_pvt.rename(
        columns={"месторождение": "mest", "объект": "object", "Bн": "vn", "B в": "vv", "ρ н": "ro_n", "ρ в": "ro_v"},
        inplace=True)
    # Приводим к строчному типу
    df_pvt['mest'] = df_pvt['mest'].astype(str).str.lower()
    df_pvt['object'] = df_pvt['object'].astype(str).str.lower()
    df_pvt['id_name'] = df_pvt['mest'] + df_pvt['object']
    return df_pvt


def read_input_files(mainform, thread) -> tuple:
    # Определение датафреймов
    df_opz = pd.DataFrame()
    df_vpp = pd.DataFrame()
    df_ngt = pd.DataFrame()
    df_tsk = pd.DataFrame()
    df_spectr = pd.DataFrame()
    df_nag_dob = pd.DataFrame()
    df_nag_nag = pd.DataFrame()
    df_ku = pd.DataFrame()
    df_modes = pd.DataFrame()
    df_pvt = pd.DataFrame()

    thread.set_progressbar_range.emit(110)
    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)
    thread.set_status_bar_message.emit('Импорт ЦК')
    # считывание файла ЦК
    data_tsk = mainform.tsk_path
    try:
        if data_tsk:
            df_tsk = read_tsk(data_tsk)
    except Exception as e:
        logger.warning(e)
        logger.warning(f'Ошибка при импортировании файла ЦК')

    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)

    thread.set_status_bar_message.emit('Импорт Окружение НАГ-ДОБ')

    # считывание файла Окружение НАГ-ДОБ
    data_nag_dob = mainform.nag_dob_path
    try:
        if data_nag_dob:
            df_nag_dob = read_nag_dob(data_nag_dob)
    except Exception as e:
        logger.warning(e)
        logger.warning(f'Ошибка при импортировании файла НАГ-ДОБ')

    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)

    thread.set_status_bar_message.emit('Импорт Выгрузка из NGT')

    # считывание файла Выгрузка из NGT
    data_ngt = mainform.ngt_path
    try:
        if data_ngt:
            df_ngt = read_ngt(data_ngt)
    except Exception as e:
        logger.warning(e)
        logger.warning(f'Ошибка при импортировании файла Выгрузка из NGT')

    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)

    thread.set_status_bar_message.emit('Импорт КУ')

    # считывание файла КУ
    data_ku = mainform.ku_path

    # Уберем кавычки из строки
    data_ku_str = data_ku[1:-1]

    # Разделим строку на список путей
    data_ku_list = [path.strip() for path in data_ku_str.split(',')]

    try:
        if data_ku_list:
            df_ku = pd.DataFrame()
            for path_file in data_ku_list:
                df_ku_piece = read_ku(path_file)
                df_ku = pd.concat([df_ku, df_ku_piece], axis=0, ignore_index=True)
    except Exception as e:
        logger.warning(e)
        logger.warning(f'Ошибка при импортировании файла КУ')

    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)

    thread.set_status_bar_message.emit('Импорт СПЕКТР')

    # считывание файла СПЕКТР
    data_spectr = mainform.spectr_path
    try:
        if data_spectr:
            df_spectr = read_spectr(data_spectr)
    except Exception as e:
        logger.warning(e)
        logger.warning(f'Ошибка при импортировании файла СПЕКТР')

    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)

    thread.set_status_bar_message.emit('Импорт ОПЗ')

    # считывание файла ОПЗ
    data_opz = mainform.opz_path
    try:
        if data_opz:
            df_opz = read_opz(data_opz)
    except Exception as e:
        logger.warning(e)
        logger.warning(f'Ошибка при импортировании файла ОПЗ')

    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)

    thread.set_status_bar_message.emit('Импорт ВПП')

    # считывание файла ВПП
    data_vpp = mainform.vpp_path
    try:
        if data_vpp:
            df_vpp = read_vpp(data_vpp)
    except Exception as e:
        logger.warning(e)
        logger.warning(f'Ошибка при импортировании файла ВПП')

    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)

    thread.set_status_bar_message.emit('Импорт PVT')

    # считывание PVT
    data_pvt = mainform.pvt_path
    try:
        if data_pvt:
            df_pvt = read_pvt(data_pvt)
    except Exception as e:
        logger.warning(e)
        logger.warning(f'Ошибка при импортировании файла PVT')

    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)

    thread.set_status_bar_message.emit('Импорт Окружение НАГ-НАГ')

    # считывание файла Окружение НАГ-НАГ
    data_nag_nag = mainform.nag_nag_path
    try:
        if data_nag_nag:
            df_nag_nag = read_nag_nag(data_nag_nag)
    except Exception as e:
        logger.warning(e)
        logger.warning(f'Ошибка при импортировании файла Окружение НАГ-НАГ')

    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)

    thread.set_status_bar_message.emit('Импорт тех. режимы')

    # считывание файла с тех. режимами
    data_modes = mainform.modes_path
    try:
        if data_modes:
            df_modes = read_models(data_modes)
    except Exception as e:
        logger.warning(e)
        logger.warning(f'Ошибка при импортировании файла с тех. режимами')

    thread.progress_vall += 10
    thread.set_progressbar_value.emit(thread.progress_vall)

    return df_opz, df_vpp, df_ngt, df_tsk, df_spectr, df_nag_dob, df_nag_nag, df_ku, df_modes, df_pvt




