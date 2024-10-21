import numpy as np
import pandas as pd
from tqdm import tqdm


def get_data_mapping_summary_product_sales(final_forecast_data):
    # Main datA Mapping
    data_mapping_1 = pd.DataFrame()
    categories = list(final_forecast_data['H1'].unique()) + ['ALL']
    channels = list(final_forecast_data['Channel'].unique()) + ['ALL']
    for i in categories:
        for j in channels:
            b = np.array([[1, 0], [0, 1]])
            a = np.array([[i] * 2, [j] * 2]).T
            combination = pd.DataFrame(np.hstack([a, b]), columns=['H1', 'Channel', "Units_Base_UOM", "Value"])
            data_mapping_1 = pd.concat([data_mapping_1, combination], axis=0)
    data_mapping_1.reset_index(drop=True, inplace=True)
    return data_mapping_1


def get_datamapping_pred_vs_last_year_demand(input_data):
    data_mapping = pd.DataFrame()
    categories = list(input_data['H1'].unique()) + ['ALL']
    channels = list(input_data['Channel'].unique()) + ['ALL']
    for i in categories:
        for j in channels:
            b = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]])
            a = np.array([[i] * 4, [j] * 4]).T
            combination = pd.DataFrame(np.hstack([a, b]),
                                       columns=['H1', 'Channel', 'Statistical Accuracy', 'Total Accuracy', "KG",
                                                "Value"])
            data_mapping = pd.concat([data_mapping, combination], axis=0)
    data_mapping.reset_index(drop=True, inplace=True)
    return data_mapping


def get_data_mapping_accuracy_segmentation(final_file_schema_1):
    # Main datA Mapping
    data_mapping_1 = pd.DataFrame()
    categories = list(final_file_schema_1['H1'].unique()) + ['ALL']
    channels = list(final_file_schema_1['Channel'].unique()) + ['ALL']
    for i in categories:
        for j in channels:
            b = np.array([[1, 0], [0, 1]])
            a = np.array([[i] * 2, [j] * 2]).T
            #         display(pd.DataFrame(np.hstack([a,b])))
            combination = pd.DataFrame(np.hstack([a, b]),
                                       columns=['H1', 'Channel', "Statistical Accuracy", "Total Accuracy"])
            data_mapping_1 = pd.concat([data_mapping_1, combination], axis=0)
    data_mapping_1.reset_index(drop=True, inplace=True)
    return data_mapping_1


def get_data_mapping_confusion_matrix(final_file_schema_1):
    # Main datA Mapping
    data_mapping_1 = pd.DataFrame()
    categories = list(final_file_schema_1['H1'].unique()) + ['ALL']
    channels = list(final_file_schema_1['Channel'].unique()) + ['ALL']
    for i in categories:
        for j in channels:
            #         print(i,j)
            b = np.array([[1, 0], [0, 1]])
            a = np.array([[i] * 2, [j] * 2]).T
            #         display(pd.DataFrame(np.hstack([a,b])))
            combination = pd.DataFrame(np.hstack([a, b]), columns=['H1', 'Channel', "Statistical", "Total"])
            data_mapping_1 = pd.concat([data_mapping_1, combination], axis=0)
    data_mapping_1.reset_index(drop=True, inplace=True)
    return data_mapping_1


def get_data_mapping_sku_demand_page_(final_file_schema_1):
    # Main datA Mapping
    data_mapping_1 = pd.DataFrame()
    categories = list(final_file_schema_1['H1'].unique())
    channels = list(final_file_schema_1['Channel'].unique())
    for i in tqdm(categories):
        for j in tqdm(channels):
            b = np.array([[1, 0], [0, 1]])
            a = np.array([[i] * 2, [j] * 2,]).T
            combination = pd.DataFrame(np.hstack([a, b]),
                                       columns=['H1', 'Channel', "Statistical", "Total"])
            data_mapping_1 = pd.concat([data_mapping_1, combination], axis=0)
    data_mapping_1.reset_index(drop=True, inplace=True)
    return data_mapping_1


def get_data_mapping_sku_demand_page(final_file_schema_1):
    # Main datA Mapping
    data_mapping_1 = pd.DataFrame()
    categories = list(final_file_schema_1['H1'].unique())
    # categories=['Dairy']
    # channels = list(final_file_schema_1['Channel'].unique()) + ['ALL']
    channels = list(final_file_schema_1['Channel'].unique())
    mrp_purchaser = list(final_file_schema_1['MrpPurchaser'].unique())
    h2_list = list(final_file_schema_1['H2'].unique())
    for i in tqdm(categories):
        for k in tqdm(mrp_purchaser):
            for j in tqdm(channels):
                for h2 in tqdm(h2_list):
                    #         print(i,j)
                    b = np.array([[1, 0], [0, 1]])
                    a = np.array([[i] * 2, [j] * 2, [k] * 2, [h2] * 2]).T
                    #         display(pd.DataFrame(np.hstack([a,b])))
                    combination = pd.DataFrame(np.hstack([a, b]),
                                               columns=['H1', 'Channel', 'MrpPurchaser', 'H2', "Statistical", "Total"])
                    data_mapping_1 = pd.concat([data_mapping_1, combination], axis=0)
    data_mapping_1.reset_index(drop=True, inplace=True)
    return data_mapping_1


def get_data_mapping_sku_demand_page_1(final_file_schema_1):
    # Main datA Mapping
    data_mapping_1 = pd.DataFrame()
    categories = list(final_file_schema_1['H1'].unique())
    channels = list(final_file_schema_1['Channel'].unique())
    mrp_purchaser = list(final_file_schema_1['MrpPurchaser'].unique())
    h2_list = list(final_file_schema_1['H2'].unique())
    for i in tqdm(categories):
        for k in tqdm(mrp_purchaser):
            for j in tqdm(channels):
                for h2 in tqdm(h2_list):
                    #         print(i,j)
                    b = np.array([[1, 0], [0, 1]])
                    a = np.array([[i] * 2, [j] * 2, [k] * 2, [h2] * 2]).T
                    #         display(pd.DataFrame(np.hstack([a,b])))
                    combination = pd.DataFrame(np.hstack([a, b]),
                                               columns=['H1', 'Channel', 'MrpPurchaser', 'H2', "Statistical", "Total"])
                    data_mapping_1 = pd.concat([data_mapping_1, combination], axis=0)
    data_mapping_1.reset_index(drop=True, inplace=True)
    # data_mapping_1.to_csv('./DATA/RESULTS/sku_demand_data_mapping.csv', index=False, sep=',')
    return data_mapping_1
