from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from schema_data_rules import get_data_from_rules, get_data_from_rules_1
from utility import generate_condition, load_filters, round_to_100, make_data_continous_


def generate_prediction_vs_last_year_demand(final_file_schema_1, data_mapping_prediction_vs_last_year_demand):
    schema_data_1 = pd.DataFrame()
    for i in range(data_mapping_prediction_vs_last_year_demand.shape[0]):
        columns_ = ['Statistical Accuracy', 'Total Accuracy', "KG", "Value"]
        filters = data_mapping_prediction_vs_last_year_demand.iloc[i, :]
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        # print(rules_values)
        final_data = get_data_from_rules(final_file_schema_1, rules_values,
                                         'Prediction_Vs_Last_Year_Demand').reset_index(drop=True)
        # print(rules_values)
        # display(final_data)
        # display(final_data)
        for j in ['H1', "Channel"]:
            # print(c.shape)
            # print(final_data.shape)
            # print(j,filters[j])
            c = generate_condition(final_data, j, filters[j])
            # print(c.shape)
            # print(final_data.shape)
            final_data = final_data[c].reset_index(drop=True)
        try:
            final_summary = final_data.groupby('year_month')[['Prediction', 'Last year Demand']].sum().reset_index()
        except:
            if final_data.shape[0] == 0:
                continue
            # display(final_data)
        final_summary['H1'] = filters['H1']
        final_summary['Channel'] = filters['Channel']
        for i in columns_:
            final_summary[i] = filters[i]
        schema_data_1 = pd.concat([schema_data_1, final_summary], axis=0)
        condition_list = pd.Series(final_data.shape[0] * [True])

    schema_data_1.reset_index(drop=True)

    schema_data_1['client_id'] = 1
    schema_data_1['last_updated_at'] = datetime.now().date()
    date = pd.to_datetime(schema_data_1['year_month'] + '-01')
    schema_data_1['year_month'] = date
    rename_cols = {
        "Prediction": "prediction",
        "Last year Demand": "last_year_demand",
        "Statistical Accuracy": "statistical_accuracy",
        "Total Accuracy": "total_accuracy",
        "KG": "kg",
        "Value": "value",
        "H1": "h1",
        "Channel": "channel"
    }
    schema_data_1.rename(columns=rename_cols, inplace=True)
    return schema_data_1


def generate_summary_product_sales(data, mapping):
    schema_data = pd.DataFrame()
    data_mapping = mapping.copy()
    columns_ = ["Units_Base_UOM", "Value"]
    for i in range(mapping.shape[0]):

        filters = mapping.iloc[i, :]
        #     print(filters)
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        # print(rules_values)
        final_data = get_data_from_rules(data, rules_values, 'summary_product_sales')
        #     print(final_data.shape)
        for j in ['H1', "Channel"]:
            # print(j,filters[j])
            c = generate_condition(final_data, j, filters[j])
            final_data = final_data[c].reset_index(drop=True)
            # display(final_data.shape)
        try:
            final_summary = final_data.groupby('year_month')[['Current Sales', 'Last Year Sales']].sum().reset_index()
        except:
            if final_data.shape[0] == 0:
                continue
            # print("ERROR Occurd")
            # print(filters['H1'])
            # print(j,filters[j])
        #         display(final_data)
        final_summary['H1'] = filters['H1']
        final_summary['Channel'] = filters['Channel']
        for i in columns_:
            final_summary[i] = filters[i]
        schema_data = pd.concat([schema_data, final_summary], axis=0)
        condition_list = pd.Series(final_data.shape[0] * [True])

    schema_data.reset_index(drop=True)
    return schema_data


def generate_summary_product_sales_1(data, mapping):
    schema_data = pd.DataFrame()
    data_mapping = mapping.copy()
    columns_ = ["Units_Base_UOM", "Value"]
    for i in range(mapping.shape[0]):

        filters = mapping.iloc[i, :]
        #     print(filters)
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        # print(rules_values)
        final_data = get_data_from_rules_1(data, rules_values, 'summary_product_sales')
        #     print(final_data.shape)
        for j in ['H1', "Channel"]:
            # print(j,filters[j])
            c = generate_condition(final_data, j, filters[j])
            final_data = final_data[c].reset_index(drop=True)
            # display(final_data.shape)
        try:
            final_summary = final_data.groupby('year_month')[['Current Sales', 'Last Year Sales']].sum().reset_index()
        except:
            if final_data.shape[0] == 0:
                continue
            # print("ERROR Occurd")
            # print(filters['H1'])
            # print(j,filters[j])
        #         display(final_data)
        final_summary['H1'] = filters['H1']
        final_summary['Channel'] = filters['Channel']
        for i in columns_:
            final_summary[i] = filters[i]
        schema_data = pd.concat([schema_data, final_summary], axis=0)
        condition_list = pd.Series(final_data.shape[0] * [True])

    schema_data.reset_index(drop=True)
    return schema_data


def get_accuracy_segmentation_schema(acc_final_data, data_mapping_accuracy_segmentation):
    accuracy_segmentation_schema = pd.DataFrame()
    for i in range(data_mapping_accuracy_segmentation.shape[0]):
        filters = data_mapping_accuracy_segmentation.iloc[i, :]
        columns_ = ["Statistical Accuracy", "Total Accuracy"]
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        print(rules_values)
        final_data = get_data_from_rules(acc_final_data, rules_values, 'segmentation_accuracy')
        for j in ['H1', "Channel"]:
            print(j, filters[j], final_data)
            c = generate_condition(final_data, j, filters[j])
            final_data = final_data[c].reset_index(drop=True)
            if final_data.shape[0] == 0:
                continue
        if final_data.shape[0] == 0:
            continue
        # print(final_data['H1'].unique())
        # print(final_data['Channel'].unique())
        print(final_data.shape)
        sales_sum = pd.pivot_table(final_data, values=['SALES_TOTAL'], aggfunc=['sum'], index=['year_month'],
                                   columns=["ACC_BUCKET"])
        sales_sum_cols = ['_'.join(i) for i in sales_sum.columns]
        sales_sum.columns = sales_sum_cols
        sales_sum_t = sales_sum.T
        sales_sum = (sales_sum_t / sales_sum_t.sum()).T
        # sales_sum
        sku_count = pd.pivot_table(final_data, values=['sku'], aggfunc=['count'], index=['year_month'],
                                   columns=["ACC_BUCKET"])
        sku_cols = ['_'.join(i) for i in sku_count.columns]
        sku_count.columns = sku_cols
        sku_count_t = sku_count.T
        # sku_count = (sku_count_t / sku_count_t.sum()).T
        sku_count = sku_count_t.T
        # sku_count
        data_ = pd.concat([sales_sum, sku_count], axis=1).reset_index()
        data_['Channel'] = filters['Channel']
        data_['H1'] = filters['H1']
        data_[columns_] = 0
        data_[rules_values] = 1
        accuracy_segmentation_schema = pd.concat([accuracy_segmentation_schema, data_], axis=0)
    return accuracy_segmentation_schema.reset_index(drop=True)


def get_forecast_accuracy_schema(acc_final_data, data_mapping_accuracy_segmentation):
    forecast_accuracy_schema = pd.DataFrame()
    for i in range(data_mapping_accuracy_segmentation.shape[0]):
        filters = data_mapping_accuracy_segmentation.iloc[i, :]
        columns_ = ["Statistical Accuracy", "Total Accuracy"]
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        print(rules_values)
        final_data = get_data_from_rules(acc_final_data, rules_values, 'forecast_accuracy')
        final_data['Revision_Sales_'] = final_data['Revision_Sales']
        final_data['actual_demand_'] = final_data['actual_demand']
        final_data['Predicted_'] = final_data['Predicted']
        final_data['Revision_Sales'] = final_data['Revision_Sales'] * final_data['NetWeight_calculated']
        final_data['actual_demand'] = final_data['actual_demand'] * final_data['NetWeight_calculated']
        final_data['Predicted'] = final_data['Predicted'] * final_data['NetWeight_calculated']

        for j in ['H1', "Channel"]:
            # print(j,filters[j])
            c = generate_condition(final_data, j, filters[j])
            final_data = final_data[c].reset_index(drop=True)
        if final_data.shape[0] == 0:
            continue
        # print(final_data.shape)
        temp_group = final_data.groupby(['year_month'])[['Predicted', 'actual_demand']].sum()
        temp_group["mape"] = (np.abs(temp_group['Predicted']
                                     - temp_group['actual_demand']) /
                              (temp_group['actual_demand']))
        # temp_group['mape']=temp_group['mape'].astype(int)
        temp_group.reset_index(inplace=True)
        temp_group["h1"] = filters['H1']
        temp_group["channel"] = filters['Channel']
        temp_group["statistical_accuracy"] = filters['Statistical Accuracy']
        temp_group["total_accuracy"] = filters['Total Accuracy']

        forecast_accuracy_schema = pd.concat([forecast_accuracy_schema, temp_group], axis=0)
    return forecast_accuracy_schema.reset_index(drop=True)


def get_confusion_matrix_schema(acc_final_data, data_mapping_accuracy_segmentation):
    confusion_matrix_schema = pd.DataFrame()
    for i in range(data_mapping_accuracy_segmentation.shape[0]):
        filters = data_mapping_accuracy_segmentation.iloc[i, :]
        columns_ = ["Statistical", "Total"]
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        print(rules_values)
        final_data = get_data_from_rules(acc_final_data, rules_values, 'confusion_matrix')
        # display(final_data)
        # limits = [10000, 100000, 200000]

        for j in ['H1', "Channel"]:
            # print(j,filters[j])
            c = generate_condition(final_data, j, filters[j])
            final_data = final_data[c].reset_index(drop=True)

        if final_data.shape[0] == 0:
            continue
        # final_data.to_clipboard(index=False)
        # return None

        # limits = [30, 120, 450]
        limits = [
            round_to_100(final_data[final_data['actual_sales'].fillna(0) > 0]['actual_sales'].quantile(.25)),
            round_to_100(final_data[final_data['actual_sales'].fillna(0) > 0]['actual_sales'].quantile(.5)),
            round_to_100(final_data[final_data['actual_sales'].fillna(0) > 0]['actual_sales'].quantile(.75))
        ]
        cols = "Predicted"
        final_data["FINAL_PRED_FLAG"], choice_pred = load_filters(limits, cols, final_data)
        cols = "actual_sales"
        final_data["ACTUAL_DEMAND_FLAG"], choice_actual = load_filters(limits, cols, final_data)

        data_collect_ = {
            "Actual Sales": [],
            "Predicted Sales": [],
            'year_month': []
        }
        # final_data.to_clipboard(index=False)
        # return None
        for dd in final_data['year_month'].unique():
            for ii in choice_actual:
                for jj in choice_pred:
                    data_collect_['Actual Sales'].append(ii)
                    data_collect_['Predicted Sales'].append(jj)
                    data_collect_['year_month'].append(dd)

        data_collect = {
            "Actual Sales": [],
            "Predicted Sales": [],
            "No of SKUS": [],
            'year_month': []
        }
        # print(final_data['ACTUAL_DEMAND_FLAG'].unique())
        # v=final_data['ACTUAL_DEMAND_FLAG']=="0"
        # display(final_data[v])
        for k in final_data['year_month'].unique():
            c = final_data['year_month'] == k
            t = final_data[c]
            for i in t['FINAL_PRED_FLAG'].unique():
                for j in t['ACTUAL_DEMAND_FLAG'].unique():
                    c1 = t['FINAL_PRED_FLAG'] == i
                    c2 = t['ACTUAL_DEMAND_FLAG'] == j
                    # print(i,j)
                    # print((c1&c2).sum())
                    data_collect['Actual Sales'].append(j)
                    data_collect['Predicted Sales'].append(i)
                    # data_collect['No of SKUS'].append((c1 & c2).sum())
                    data_collect['No of SKUS'].append(t[c1 & c2]['sku'].nunique())
                    data_collect['year_month'].append(k)
        d = pd.DataFrame(data_collect)
        # display(pd.DataFrame(data_collect_))
        # display(d)
        d = pd.merge(pd.DataFrame(data_collect_), d, on=['Actual Sales', 'Predicted Sales', 'year_month'], how='left')
        print(d.shape)
        d['No of SKUS'].fillna(0, inplace=True)
        d['H1'] = filters["H1"]
        d['Channel'] = filters["Channel"]
        d['Total'] = filters["Total"]
        d['Statistical'] = filters["Statistical"]
        # c = d['Actual Sales'] != 0
        # c1 = d['Predicted Sales'] != 0
        # d = d[c | c1]
        confusion_matrix_schema = pd.concat([confusion_matrix_schema, d], axis=0)

    return confusion_matrix_schema


def get_confusion_matrix_schema_old(acc_final_data, data_mapping_accuracy_segmentation):
    confusion_matrix_schema = pd.DataFrame()
    for i in range(data_mapping_accuracy_segmentation.shape[0]):
        filters = data_mapping_accuracy_segmentation.iloc[i, :]
        columns_ = ["Statistical", "Total"]
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        print(rules_values)
        final_data = get_data_from_rules(acc_final_data, rules_values, 'confusion_matrix')
        # display(final_data)
        # limits = [10000, 100000, 200000]
        limits = [30, 120, 450]
        cols = "Predicted"
        final_data["FINAL_PRED_FLAG"] = load_filters(limits, cols, final_data)
        cols = "actual_sales"
        final_data["ACTUAL_DEMAND_FLAG"] = load_filters(limits, cols, final_data)
        for j in ['H1', "Channel"]:
            # print(j,filters[j])
            c = generate_condition(final_data, j, filters[j])
            final_data = final_data[c].reset_index(drop=True)
        if final_data.shape[0] == 0:
            continue
        data_collect = {
            "Actual Sales": [],
            "Predicted Sales": [],
            "No of SKUS": [],
            'year_month': []
        }
        # print(final_data['ACTUAL_DEMAND_FLAG'].unique())
        # v=final_data['ACTUAL_DEMAND_FLAG']=="0"
        # display(final_data[v])
        for k in final_data['year_month'].unique():
            c = final_data['year_month'] == k
            t = final_data[c]
            for i in t['FINAL_PRED_FLAG'].unique():
                for j in t['ACTUAL_DEMAND_FLAG'].unique():
                    c1 = t['FINAL_PRED_FLAG'] == i
                    c2 = t['ACTUAL_DEMAND_FLAG'] == j
                    # print(i,j)
                    # print((c1&c2).sum())
                    data_collect['Actual Sales'].append(j)
                    data_collect['Predicted Sales'].append(i)
                    data_collect['No of SKUS'].append((c1 & c2).sum())
                    data_collect['year_month'].append(k)
        d = pd.DataFrame(data_collect)
        d['H1'] = filters["H1"]
        d['Channel'] = filters["Channel"]
        d['Total'] = filters["Total"]
        d['Statistical'] = filters["Statistical"]
        confusion_matrix_schema = pd.concat([confusion_matrix_schema, d], axis=0)

    return confusion_matrix_schema


def get_sku_demand_page_schema(acc_final_data, data_mapping_sku_demand_page, sku_details, input_data_c, current_date):
    sku_demand_page_schema = pd.DataFrame()
    for i in tqdm(range(data_mapping_sku_demand_page.shape[0])):

        filters = data_mapping_sku_demand_page.iloc[i, :]
        columns_ = ["Statistical", "Total"]
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        # print(filters)
        final_data = get_data_from_rules(acc_final_data, rules_values, 'sku_demand_page')
        for j in ['H1', "Channel", 'MrpPurchaser', 'H2']:
            print(j, filters[j])
            c = generate_condition(final_data, j, filters[j])
            final_data = final_data[c].reset_index(drop=True)
        if final_data.shape[0] == 0:
            continue

        for sku_ in tqdm(final_data['sku'].unique()):
            c = final_data['sku'] == sku_
            t = final_data[c]
            c2 = input_data_c['sku'] == sku_
            c1 = input_data_c['Channel'] == filters['Channel']
            input_ = input_data_c[c1 & c2]
            input_['mape'] = np.abs(input_['final_perdiction'] - input_['QTY_BASEUOM_SUM']) / input_['QTY_BASEUOM_SUM']
            input_['notinf'] = np.where(np.isinf(input_['mape']), 0, 1)
            input_['mape'] = np.where(np.isinf(input_['mape']), 0, input_['mape'])
            input_['3_mape'] = input_['mape'].rolling(3).sum()
            input_['3_notinf'] = input_['notinf'].rolling(3).sum()
            input_['3_months_acc'] = input_['3_mape'] / input_['3_notinf']
            top_df = pd.DataFrame()
            top_df['year_month'] = pd.date_range(current_date - pd.DateOffset(months=2), current_date,
                                                 freq="M") + pd.DateOffset(days=1)
            top_df['year_month'] = top_df['year_month'].astype(str).str[:-3]
            top_df['sku'] = sku_
            top_df['Channel'] = filters["Channel"]
            top_df = pd.merge(top_df, input_[
                ['sku', 'year_month', 'Channel', "last_year_demand", 'TOTAL_QTY_BASEUOM_SUM', 'QTY_BASEUOM_SUM',
                 'Safety Stock',
                 'Variability',
                 'final_perdiction', '3_months_acc', 'contract_sales']], on=['sku', 'year_month', 'Channel'],
                              how='left')
            top_df = pd.merge(top_df, sku_details, on=['sku'], how='left')
            top_df['Variability'] = t['Variability'].max()
            top_df = pd.concat([top_df, t], axis=0).reset_index(drop=True)
            top_df['Statistical'] = filters['Statistical']
            top_df['Total'] = filters['Total']
            top_df['Safety Stock'] = top_df['Safety Stock'].max()
            # display(top_df)

            top_df['Channel'] = filters['Channel']
            top_df['H1'] = filters['H1']
            top_df['H2'] = filters['H2']
            top_df['MrpPurchaser'] = filters['MrpPurchaser']

            sku_demand_page_schema = pd.concat([sku_demand_page_schema, top_df], axis=0)
            print(sku_demand_page_schema.shape)

    return sku_demand_page_schema


def get_sku_demand_page_schema_(acc_final_data, data_mapping_sku_demand_page, sku_details, input_data_c, current_date):
    sku_demand_page_schema = pd.DataFrame()
    for i in tqdm(range(data_mapping_sku_demand_page.shape[0])):

        filters = data_mapping_sku_demand_page.iloc[i, :]
        columns_ = ["Statistical", "Total"]
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        # print(filters)
        final_data = get_data_from_rules(acc_final_data, rules_values, 'sku_demand_page')
        for j in ['H1', "Channel", 'MrpPurchaser', 'H2']:
            print(j, filters[j])
            c = generate_condition(final_data, j, filters[j])
            final_data = final_data[c].reset_index(drop=True)
        if final_data.shape[0] == 0:
            continue

        for sku_ in tqdm(final_data['sku'].unique()):
            c = final_data['sku'] == sku_
            t = final_data[c]
            c2 = input_data_c['sku'] == sku_
            c1 = input_data_c['Channel'] == filters['Channel']
            input_ = input_data_c[c1 & c2]
            input_['mape'] = np.abs(input_['final_perdiction'] - input_['QTY_BASEUOM_SUM']) / input_['QTY_BASEUOM_SUM']
            input_['notinf'] = np.where(np.isinf(input_['mape']), 0, 1)
            input_['mape'] = np.where(np.isinf(input_['mape']), 0, input_['mape'])
            input_['3_mape'] = input_['mape'].rolling(3).sum()
            input_['3_notinf'] = input_['notinf'].rolling(3).sum()
            input_['3_months_acc'] = input_['3_mape'] / input_['3_notinf']
            top_df = pd.DataFrame()
            top_df['year_month'] = pd.date_range(current_date - pd.DateOffset(months=2), current_date,
                                                 freq="M") + pd.DateOffset(days=1)
            top_df['year_month'] = top_df['year_month'].astype(str).str[:-3]
            top_df['sku'] = sku_
            top_df['Channel'] = filters["Channel"]
            top_df = pd.merge(top_df, input_[
                ['sku', 'year_month', 'Channel', "last_year_demand", 'TOTAL_QTY_BASEUOM_SUM', 'QTY_BASEUOM_SUM',
                 'Safety Stock',
                 'Variability',
                 'final_perdiction', '3_months_acc', 'contract_sales']], on=['sku', 'year_month', 'Channel'],
                              how='left')
            top_df = pd.merge(top_df, sku_details, on=['sku'], how='left')
            top_df['Variability'] = t['Variability'].max()
            top_df = pd.concat([top_df, t], axis=0).reset_index(drop=True)
            top_df['Statistical'] = filters['Statistical']
            top_df['Total'] = filters['Total']
            top_df['Safety Stock'] = top_df['Safety Stock'].max()
            # display(top_df)

            top_df['Channel'] = filters['Channel']
            top_df['H1'] = filters['H1']
            top_df['H2'] = filters['H2']
            top_df['MrpPurchaser'] = filters['MrpPurchaser']

            sku_demand_page_schema = pd.concat([sku_demand_page_schema, top_df], axis=0)
            print(sku_demand_page_schema.shape)

    return sku_demand_page_schema


def generate_variability_index(input_data_c, current_date):
    dict_ = {
        "sku": [],
        "H1": [],
        "Channel": [],
        "Variability": [],
        "Standard_Dev": []
    }
    for cat in tqdm(list(input_data_c['H1'].unique()) + ['ALL']):
        #     print(cat)
        for chnl in tqdm(list(input_data_c['Channel'].unique()) + ['ALL']):
            print(cat, chnl)
            c = input_data_c['H1'] == cat
            c = generate_condition(input_data_c, "H1", cat)
            c1 = generate_condition(input_data_c, "Channel", chnl)
            input_data_ = input_data_c[c & c1]
            date_to_filter_data = current_date - pd.DateOffset(months=24)
            date_to_filter_data

            input_data_["date"] = pd.to_datetime(input_data_['year_month'] + "-01")
            c = input_data_['date'] >= date_to_filter_data
            c1 = input_data_['date'] <= current_date
            input_data_ = input_data_[c & c1]
            # groupby_col_index=["H1","Channel"]
            # groupby_column=[groupby_col_index[i] for i,j  in enumerate([cat,chnl]) if j.lower() !="all"]+['year_month','sku']
            # print(groupby_column)
            input_data_ = input_data_.groupby(['sku', 'year_month'])['TOTAL_QTY_BASEUOM_SUM'].sum().reset_index()
            if input_data_.shape[0] == 0:
                continue
            print("MAKING_DATA _CONTINOUS")
            input_data_['date']=pd.to_datetime(input_data_['year_month']+"-01")
            input_data_.sort_values(['sku','date'],inplace=True)
            # input_data = make_data_continous_(input_data_, 'sku', current_date, 6)
            input_data = input_data_.copy()
            input_data['TOTAL_QTY_BASEUOM_SUM'].fillna(0, inplace=True)
            for sku in tqdm(input_data_['sku'].unique()):
                dict_['sku'].append(sku)
                dict_['H1'].append(cat)
                dict_['Channel'].append(chnl)
                c = input_data_['sku'] == sku
                #             print(input_data[c].shape)
                variability = input_data_[c]['TOTAL_QTY_BASEUOM_SUM'].std() / input_data_[c][
                    'TOTAL_QTY_BASEUOM_SUM'].mean()
                # print(variability, 'printing variability')
                dict_['Variability'].append(variability)
                dict_['Standard_Dev'].append(input_data_[c]['TOTAL_QTY_BASEUOM_SUM'].std())
    return pd.DataFrame(dict_)


def generate_variability_schema(variability_data):
    variability_data["Variability"].fillna(0, inplace=True)
    variability_data['Variability'].isna().sum()

    c1 = variability_data['Variability'] <= 1
    c2 = (variability_data['Variability'] > 1) & (variability_data['Variability'] <= 1.5)
    c3 = (variability_data['Variability'] > 1.5) & (variability_data['Variability'] <= 2)
    c4 = (variability_data['Variability'] > 2) & (variability_data['Variability'] <= 2.5)
    c5 = variability_data['Variability'] > 2.5
    choice_list = ["Less than 1", "1-1.5", '1.5-2', '2-2.5', "More than 2.5"]
    condition_list = [c1, c2, c3, c4, c5]
    variability_data['Variability Bucket'] = np.select(condition_list, choice_list)
    variability_data

    variability_index = pd.DataFrame()
    for i in variability_data['H1'].unique():
        for j in variability_data['Channel'].unique():
            c1 = variability_data['H1'] == i
            c2 = variability_data['Channel'] == j
            temp = variability_data[c1 & c2]
            temp = temp['Variability Bucket'].value_counts(normalize=True).to_frame().reset_index()
            temp['proportion'] = np.round(temp['proportion'] * 100, 2).astype(str) + "%"
            temp.rename(columns={"proportion": '% Of SKUs'}, inplace=True)
            temp['Channel'] = j
            temp['H1'] = i
            temp['last_updated_at'] = datetime(2024, 8, 1)
            temp['client_id'] = 1
            variability_index = pd.concat([variability_index, temp], axis=0)

    return variability_index


def get_sku_demand_page_schema_1(acc_final_data, data_mapping_sku_demand_page, sku_details, input_data_c, current_date):
    sku_demand_page_schema = pd.DataFrame()
    for i in tqdm(range(data_mapping_sku_demand_page.shape[0])):

        filters = data_mapping_sku_demand_page.iloc[i, :]
        columns_ = ["Statistical", "Total"]
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        # print(filters)
        final_data = get_data_from_rules_1(acc_final_data, rules_values, 'sku_demand_page')
        for j in ['H1', "Channel", 'MrpPurchaser', 'H2']:
            print(j, filters[j])
            c = generate_condition(final_data, j, filters[j])
            final_data = final_data[c].reset_index(drop=True)
        if final_data.shape[0] == 0:
            continue

        for sku_ in tqdm(final_data['sku'].unique()):
            c = final_data['sku'] == sku_
            t = final_data[c]
            c2 = input_data_c['sku'] == sku_
            c1 = input_data_c['Channel'] == filters['Channel']
            input_ = input_data_c[c1 & c2]
            input_['mape'] = np.abs(input_['final_perdiction'] - input_['QTY_BASEUOM_SUM']) / input_['QTY_BASEUOM_SUM']
            input_['notinf'] = np.where(np.isinf(input_['mape']), 0, 1)
            input_['mape'] = np.where(np.isinf(input_['mape']), 0, input_['mape'])
            input_['3_mape'] = input_['mape'].rolling(3).sum()
            input_['3_notinf'] = input_['notinf'].rolling(3).sum()
            input_['3_months_acc'] = input_['3_mape'] / input_['3_notinf']
            top_df = pd.DataFrame()
            top_df['year_month'] = pd.date_range(current_date - pd.DateOffset(months=4), current_date,
                                                 freq="M") + pd.DateOffset(days=1)
            top_df['year_month'] = top_df['year_month'].astype(str).str[:-3]
            top_df['sku'] = sku_
            top_df['Channel'] = filters["Channel"]
            top_df = pd.merge(top_df, input_[
                ['sku', 'year_month', 'Channel', "last_year_demand", 'QTY_BASEUOM_SUM', 'Safety Stock', 'Variability',
                 'final_perdiction', '3_months_acc']], on=['sku', 'year_month', 'Channel'], how='left')
            top_df = pd.merge(top_df, sku_details, on=['sku'], how='left')
            top_df['Variability'] = t['Variability'].max()
            top_df = pd.concat([top_df, t], axis=0).reset_index(drop=True)
            top_df['Statistical'] = filters['Statistical']
            top_df['Total'] = filters['Total']
            top_df['Safety Stock'] = top_df['Safety Stock'].max()
            # display(top_df)

            top_df['Channel'] = filters['Channel']
            top_df['H1'] = filters['H1']
            top_df['H2'] = filters['H2']
            top_df['MrpPurchaser'] = filters['MrpPurchaser']

            sku_demand_page_schema = pd.concat([sku_demand_page_schema, top_df], axis=0)

    return sku_demand_page_schema


def get_confusion_matrix_schema_new(acc_final_data, data_mapping_accuracy_segmentation, current_date):
    confusion_matrix_schema = pd.DataFrame()
    for i in range(data_mapping_accuracy_segmentation.shape[0]):
        filters = data_mapping_accuracy_segmentation.iloc[i, :]
        columns_ = ["Statistical", "Total"]
        rules_ = filters.to_frame().loc[columns_, :]
        rules_index = rules_.index
        rules_values = rules_.iloc[:, 0].values
        rules_values = [rules_index[i] for i, j in enumerate(rules_values) if int(j) != 0]
        rules_values = '_'.join(rules_values)
        print(rules_values)
        final_data = get_data_from_rules(acc_final_data, rules_values, 'confusion_matrix')
        # display(final_data)
        limits = [10, 100, 200]
        cols = "Predicted"
        final_data["FINAL_PRED_FLAG"] = load_filters(limits, cols, final_data)
        cols = "actual_demand"
        final_data["ACTUAL_DEMAND_FLAG"] = load_filters(limits, cols, final_data)
        for j in ['H1', "Channel"]:
            # print(j,filters[j])
            c = generate_condition(final_data, j, filters[j])
            final_data = final_data[c].reset_index(drop=True)
        if final_data.shape[0] == 0:
            continue
        data_collect = {
            "Actual Sales": [],
            "Predicted Sales": [],
            "No of SKUS": [],
            'year_month': [],
        }
        # print(final_data['ACTUAL_DEMAND_FLAG'].unique())
        # v=final_data['ACTUAL_DEMAND_FLAG']=="0"
        # display(final_data[v])
        for k in final_data['year_month'].unique():
            c = final_data['year_month'] == k
            t = final_data[c]
            for i in t['FINAL_PRED_FLAG'].unique():
                for j in t['ACTUAL_DEMAND_FLAG'].unique():
                    for month in range(6):
                        date = current_date - pd.DateOffset(months=month)
                        date = str(date.date())[:-3]
                        print(date)
                        c1 = t['FINAL_PRED_FLAG'] == i
                        c2 = t['ACTUAL_DEMAND_FLAG'] == j
                        c3 = t['year_month'] == date
                    data_collect['Actual Sales'].append(j)
                    data_collect['Predicted Sales'].append(i)
                    data_collect['year_month'].append(date)
                    data_collect['No of SKUS'].append((c1 & c2 & c3).sum())
        d = pd.DataFrame(data_collect)
        d['H1'] = filters["H1"]
        d['Channel'] = filters["Channel"]
        d['Total'] = filters["Total"]
        d['Statistical'] = filters["Statistical"]
        confusion_matrix_schema = pd.concat([confusion_matrix_schema, d], axis=0)

    return confusion_matrix_schema
