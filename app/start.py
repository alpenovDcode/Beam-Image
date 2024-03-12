import pandas as pd
from app.utils.logger import Logger as logger
from block1 import check_block1
from app.calculation.donors_and_receipents import search_for_donor
from app.calculation.IterationWells_class import IterationWells
from time import time
from dateutil.relativedelta import relativedelta
from app.calculation.PrepareData_class import PrepareData
from app.calculation.DonorsSearch_class import DonorsSearch


def run(mainform, thread):
    time_all = time()

    # подготовка данных
    solver = PrepareData(mainform, thread)
    solver.filter_intersect_ku_ngt()
    solver.create_additional_df()

    # присвоить атрибуты класса, общие для всех экземпляров
    IterationWells.assign_values(solver)

    time_df = pd.DataFrame(columns=['Полная отработка 1 нагн скв', 'Отработка блока 1', 'Предобработка файлов Ку и Окружение Наг Наг', 'Мерджинг файлов Ку и Цк', 'Проверка, что все скв работали z2 мес назад', 'Цикл по скважинам окружения и по ее пластам', 'Проверка условий'])
    block1_result_df = pd.DataFrame()
    output_result_df = pd.DataFrame()

    df_ngt_to_iterate = solver.compare_ngt_nag_dob()

    thread.progress_vall += 5
    thread.set_progressbar_value.emit(thread.progress_vall)
    thread.set_status_bar_message.emit('Запуск процедуры вычисления')

    piece_value = 80 / len(solver.df_ngt_to_iterate["Id_name"].unique())

    # цикл для всех нагн скважин
    for id_name in df_ngt_to_iterate["Id_name"].unique():
        thread.progress_vall += piece_value
        thread.set_progressbar_value.emit(thread.progress_vall)

        one_well_run = IterationWells(id_name)
        # расчеты по скважине
        result = one_well_run.calc()
        time_df = pd.concat([time_df, one_well_run.data_time_series], ignore_index=True)

        # обработка результатов после расчета
        if result["isin_block1"]:
            block1_result_df = pd.concat([block1_result_df, one_well_run.block1_df], axis=0, ignore_index=True)
        else:
            if result["counted_successfully"]:
                output_result_df = pd.concat([output_result_df, one_well_run.df_output, one_well_run.not_worked_wells_df], axis=0, ignore_index=True)

    time_df.to_excel('time.xlsx', index=False)
    # поиск доноров и реципиентов
    thread.set_status_bar_message.emit('Блок Кубышки')
    donors_finder = DonorsSearch(output_result_df, IterationWells)
    df_donors_and_recipients, df_all_don_and_rec = donors_finder.search_for_donor()

    thread.progress_vall += 5
    thread.set_progressbar_value.emit(thread.progress_vall)

    print(f"Время расчетов = {time() - time_all}")

    return output_result_df, block1_result_df, df_donors_and_recipients, df_all_don_and_rec, solver.settings_df




