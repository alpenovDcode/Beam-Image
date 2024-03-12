import json
import os


def create_json():
    file_name = 'history.json'
    if not os.path.exists(file_name):
        data = {
            "connect": {
                "opz": {
                    "type": "table",
                    "path": [
                        "Input",
                        "ОПЗ.xlsx"
                    ]
                },
                "vpp": {
                    "type": "table",
                    "path": [
                        "Input",
                        "ВПП.xlsx"
                    ]
                },
                "ngt": {
                    "type": "table",
                    "path": [
                        "Input",
                        "Выгрузка из NGT.xlsx"
                    ]
                },
                "tsk": {
                    "type": "table",
                    "path": [
                        "Input",
                        "ЦК.xlsx"
                    ]
                },
                "spectr": {
                    "type": "table",
                    "path": [
                        "Input",
                        "СПЕКТР.xlsx"
                    ]
                },
                "nag_dob": {
                    "type": "table",
                    "path": [
                        "Input",
                        "Окружение НАГ-ДОБ.xlsx"
                    ]
                },
                "nag_nag": {
                    "type": "table",
                    "path": [
                        "Input",
                        "Окружение НАГ-НАГ.xlsx"
                    ]
                },
                "ku": {
                    "type": "table",
                    "path": [
                        [
                            "Input",
                            "КУ.csv"
                        ],
                        [
                            "Input",
                            "КУ_AC10.0.csv"
                        ]
                    ]
                },
                "modes": {
                    "type": "table",
                    "path": [
                        "Input",
                        "ТР НАГ_All_2.xls"
                    ]
                },
                "pvt": {
                    "type": "table",
                    "path": [
                        "Input",
                        "PVT.xlsx"
                    ]
                }
            },
            "solver": {
                "nature_of_work": "[НАГ]",
                "state": "[РАБ., ПЬЕЗ, Б/Д ПР Л, ОСТ.]",
                "P zac": "10",
                "z1": "6",
                "z2": "2",
                "z3": "3",
                "x": "10",
                "y1": "20",
                "y2": "10",
                "y3": "10"
            }
        }

        # Сохраняем обновленный JSON в кодировке UTF-8
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)