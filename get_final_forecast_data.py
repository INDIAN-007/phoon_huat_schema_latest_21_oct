# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:15:19 2024

@author: MilanGowdaJP
"""
import pandas as pd
import os


def get_final_data_from_folder(list_H1, MONTH_YEAR_PATH, FINAL_FORECAST_FILE_NAME):
    final_data = pd.DataFrame()
    for H1 in list_H1:
        for chnl in os.listdir(MONTH_YEAR_PATH + H1):
            file_path = MONTH_YEAR_PATH + H1 + "/" + chnl + "/" + FINAL_FORECAST_FILE_NAME
            data = pd.read_csv(file_path)
            data['H1'] = H1
            data['CHNL_NAME'] = chnl
            final_data = pd.concat([final_data, data], axis=0)
    c = final_data['loop_number'] == 0
    c1 = final_data['loop_back_months_for_weights'] == final_data['forecast_month']
    return final_data[c & c1]
