import pandas as pd
import numpy as np
from datetime import datetime
import warnings

from get_data_mapping import get_datamapping_pred_vs_last_year_demand, get_data_mapping_summary_product_sales, \
    get_data_mapping_accuracy_segmentation, get_data_mapping_confusion_matrix, get_data_mapping_sku_demand_page, \
    get_data_mapping_sku_demand_page_1
from pred_vs_last_year_demand import generate_prediction_vs_last_year_demand, generate_summary_product_sales, \
    get_accuracy_segmentation_schema, get_forecast_accuracy_schema, get_confusion_matrix_schema, \
    get_sku_demand_page_schema, get_confusion_matrix_schema_new, get_sku_demand_page_schema_1, \
    generate_variability_index, generate_variability_schema
# from schemaPipeline import input_data_c
from utility import make_input_data_continous, collect_data

warnings.filterwarnings('ignore')

output_folder_ = './DATA/RESULTS/'
mape_file_name = 'OCT_24_MAR_25_SCHEMA_DATA_17_10'
# mape_file_name='MAPE_ARIMA_SES_EXP_REXP'
# sheet_name="APR_24"
sheet_name = "Schema_"

writer = pd.ExcelWriter(output_folder_ + mape_file_name + '.xlsx', engine='xlsxwriter')

# SCHEMA 1
final_forecast_path = 'DATA/MAY_24_FORECAST/combined_file_new_oct_24_mar_25.csv'
# forecast_history_path = '../SCHEMA_CODE/DATA/FORECAST_HISTORY/forecast_history.csv'
# forecast_history_path = './DATA/FORECAST_HISTORY/forecast_history.csv'
# forecast_history_path = './DATA/FORECAST_HISTORY/forecast_history_dec_23.csv'
forecast_history_path = './DATA/FORECAST_HISTORY/forecast_history_jul_sep.csv'
input_data_path = './DATA/INPUT_DATA/sales_groupby_31_09_24_final.csv'
final_forecast_data = pd.read_csv(final_forecast_path)
c = final_forecast_data['final_perdiction'].isna() == False
final_forecast_data = final_forecast_data[c]
forecast_history_data = pd.read_csv(forecast_history_path)
input_data = pd.read_csv(input_data_path)
print(final_forecast_data.shape)
print(forecast_history_data.shape)
print(input_data.shape)

categories_present=['Dairy', 'Bakery', 'Fruits', 'Beverage',
                    'Cheese', 'Chocolates', 'Fat & Oil', 'Filling & Jam',
                    'Grocery', 'Meat', 'Non Food', 'Flour, Grain & Flakes',
                    'Nuts, Seeds & Beans', 'PHD', 'Seafood']

import os

product_hierarchy_path = './DATA/Product_MASTER/'
product_hierarchy_files = os.listdir(product_hierarchy_path)
print(product_hierarchy_files)
col = ['Product', "NetWeight", 'H1_description', "H2_description",
       "H3_description", "H4_description", "H5_description", 'CrossPlantStatus', "LeadTime"
    , "ServiceLevel", 'MrpPurchaser', 'PdProductDescription', "ProductWeightUnit"]
sku_details = collect_data(product_hierarchy_path, product_hierarchy_files, col=col)
# master.rename(columns={'Product':"sku","CrossPlantStatus":"MSTAE"},inplace=True)
sku_details.rename(columns={'Product': "sku", 'H1_description': "H1", "H2_description": "H2"
    , "H3_description": 'H3', "H4_description": "H4", "H5_description": "H5"}, inplace=True)
sku_details['sku'] = sku_details['sku'].astype(int)

c=sku_details['H1'].isin(categories_present)
sku_details=sku_details[c]

sku_details['NetWeight'] = sku_details['NetWeight'].astype(float)

cond__ = [sku_details['ProductWeightUnit'] == "KG", sku_details['ProductWeightUnit'] == "G",
          sku_details['ProductWeightUnit'] == "MG"]

choices__ = [sku_details['NetWeight'], sku_details['NetWeight'] / 1000, sku_details['NetWeight'] / 1000000]

sku_details['NetWeight_calculated'] = np.select(cond__, choices__)

sku_details['ServiceLevel'] = sku_details['ServiceLevel'].astype(float)

stats_value_map = {
    .50: 0,
    .6: 0.7257,
    .80: 0.842,
    .85: 1.036,
    .90: 1.282,
    .95: 1.645,
    # .99: 2.57,
    # .999: 3.29,
}
sku_details['ServiceLevel_value'] = sku_details['ServiceLevel'].map(stats_value_map)

c = sku_details['CrossPlantStatus'] == "AC"
active_list = sku_details[c]['sku'].unique()

print(input_data.head())
c = input_data['MATERIAL'].isin(active_list)
input_data = input_data[c]

print(final_forecast_data.head())
c = final_forecast_data['sku'].isin(active_list)
c1 = forecast_history_data['sku'].isin(active_list)
forecast_history_data = forecast_history_data[c1]
print(forecast_history_data.shape, forecast_history_data['sku'].nunique())
final_forecast_data = final_forecast_data[c]

sku_details.head()
# print(final_forecast_data['year_month'].head())
final_forecast_data['forecast_month'] = final_forecast_data['year_month']
# print(final_forecast_data['forecast_month'].head()+"-01")
print(pd.to_datetime(final_forecast_data['year_month'] + "-01"))
current_date = pd.to_datetime(final_forecast_data['forecast_month'] + "-01").min() - pd.DateOffset(months=1)
print(current_date, "*" * 1000)
input_data['CHNL_NAME'] = np.where(input_data['CHNL_NAME'].isin(['Export', 'Re-Export']), 'Export_Re-Export',
                                   input_data['CHNL_NAME'])

input_data = input_data.groupby(['MATERIAL', 'year_month', 'CHNL_NAME'], as_index=False).sum()

input_data = pd.merge(input_data, sku_details[['sku', "H1"]],
                      left_on=['MATERIAL']
                      , right_on=['sku'], how='left')

c=input_data['H1'].isin(categories_present)
input_data=input_data[c]

input_data.drop(['sku'], axis=1, inplace=True)
print(input_data.head().T)
# c = input_data['H1'].isin(["Dairy"])
# input_data = input_data[c]
print(input_data['H1'].unique(), 'Printing_DATA')
print(final_forecast_data.head(), 'Printing_FF')

input_data_c = make_input_data_continous(input_data, current_date)

input_data_c['date'] = pd.to_datetime(input_data_c['year_month'] + "-01")

input_data_c.sort_values(['MATERIAL', 'H1', "CHNL_NAME", 'date'], inplace=True)
input_data_c.to_csv('./DATA/RESULTS/input_data_.csv', index=False)

# Calculating Last Year Demand for SKU H1 & CHNL_NAME
input_data_c['last_year_demand'] = input_data_c.groupby(['MATERIAL', 'H1', 'CHNL_NAME'])['TOTAL_QTY_BASEUOM_SUM'].shift(
    12)
input_data_c['last_year_demand_non_contract'] = input_data_c.groupby(['MATERIAL', 'H1', 'CHNL_NAME'])[
    'QTY_BASEUOM_SUM'].shift(12)
input_data_c['last_year_KG_SUM'] = input_data_c.groupby(['MATERIAL', 'H1', 'CHNL_NAME'])['KG_SUM'].shift(12)

final_forecast_data['year_month'] = final_forecast_data['forecast_month']

# c1=input_data['H1']=="Dairy"
# c2=input_data['CHNL_NAME']=="B2B"
# c3=input_data['year_month']=="2024-07"
# a=input_data[c1 & c2 & c3]['TOTAL_QTY_BASEUOM_SUM'].sum()
# print(a)

# ADDING UNIT COST START

input_d = input_data_c.copy()
c = input_d['TOTAL_QTY_BASEUOM_COUNT'] > 0
c1 = input_d['TOTAL_QTY_BASEUOM_SUM'] > 0
input_d = input_d[c & c1]
# input_d.shape
max_date_input = input_d.groupby(['MATERIAL', 'CHNL_NAME'], as_index=False)['date'].max()
max_date_input.rename(columns={"date": "max_date"}, inplace=True)
# max_date_input
input_d = pd.merge(input_d, max_date_input, on=['MATERIAL', 'CHNL_NAME'], how='left')
c = input_d['date'] == input_d['max_date']
input_d = input_d[c][['MATERIAL', 'CHNL_NAME', "TOTAL_SALES_SUM", 'TOTAL_QTY_BASEUOM_SUM']]

input_d["one_unit_cost"] = input_d['TOTAL_SALES_SUM'] / input_d['TOTAL_QTY_BASEUOM_SUM']
# input_d[['MATERIAL', 'CHNL_NAME', 'one_unit_cost']].to_clipboard(index=False)
input_data_c = pd.merge(input_data_c, input_d[['MATERIAL', 'CHNL_NAME', 'one_unit_cost']], on=['MATERIAL', 'CHNL_NAME'],
                        how='left')

# ADDING UNIT COST end

input_data_c.rename(columns={"MATERIAL": "sku", 'CHNL_NAME': "Channel"}, inplace=True)
# input_data_c.to_clipboard(index=False,sep=',')


input_data_c['TOTAL_QTY_BASEUOM_SUM'].fillna(0, inplace=True)
input_data_c['QTY_BASEUOM_SUM'].fillna(0, inplace=True)
input_data_c['contract_sales'] = input_data_c['TOTAL_QTY_BASEUOM_SUM'] - input_data_c['QTY_BASEUOM_SUM']

# SCHEMA PREDICTION VS LAST YEAR DEMAND - START
final_file_schema_1 = pd.merge(final_forecast_data,
                               input_data_c[
                                   ['sku', 'year_month', 'H1', 'Channel', 'last_year_demand_non_contract',
                                    'last_year_demand', 'one_unit_cost']],
                               on=['sku', 'H1', 'Channel', 'year_month'], how='left')

final_file_schema_1 = pd.merge(final_file_schema_1, sku_details[['sku', 'NetWeight']], on='sku', how='left')
final_file_schema_1['NetWeight'] = final_file_schema_1['NetWeight'].astype(float)

data_mapping_prediction_vs_last_year_demand = get_datamapping_pred_vs_last_year_demand(final_file_schema_1)

# Section to join revised sale
# For now attaching revised sales as 0 to all
final_file_schema_1['Revised_Sale'] = 0

# SCHEMA 1 PREDICTION VS LAST YEAR DEMAND
pred_vs_lyd_schema = generate_prediction_vs_last_year_demand(final_file_schema_1,
                                                             data_mapping_prediction_vs_last_year_demand)

pred_vs_lyd_schema.to_csv('./DATA/RESULTS/pred_vslast.csv', index=False, sep=',')

pred_vs_lyd_schema['prediction'] = np.ceil(pred_vs_lyd_schema['prediction'])

pred_vs_lyd_schema['last_year_demand'] = np.ceil(pred_vs_lyd_schema['last_year_demand'])

pred_vs_lyd_schema.to_excel(writer, sheet_name=sheet_name + "PRED_VS_LAST_YEAR_DEMAND", index=False)

# pred_vs_lyd_schema.to_clipboard(index=False,sep=',')

# SCHEMA PREDICTION VS LAST YEAR DEMAND - END

# SCHEMA PRODUCT SALES SUMMARY - START
c = input_data_c['date'] <= current_date
c2 = input_data_c['date'] > current_date - pd.DateOffset(months=6)
summary_product_sales_data = input_data_c[c & c2]

# c1=input_data_c['H1']=="Bakery"
# c2=input_data_c['Channel']=="B2B"
# c3=input_data_c['year_month']=='2024-05'
# ggg=input_data_c[c1 & c2 & c3]['TOTAL_QTY_BASEUOM_SUM'].sum()
# print(ggg)

summary_product_sales_data = pd.merge(summary_product_sales_data, sku_details[['sku', 'NetWeight']], on=['sku'],
                                      how='left')
print(summary_product_sales_data.head().T)
summary_product_sales_data['NetWeight'] = summary_product_sales_data['NetWeight'].astype(float)

data_mapping_summary_product_sales = get_data_mapping_summary_product_sales(input_data_c)

# SCHEMA 2 SUMMARY PRODUCT SALES
product_sales_schema_data = generate_summary_product_sales(summary_product_sales_data,
                                                           data_mapping_summary_product_sales)
product_sales_schema_data['client_id'] = 1

product_sales_schema_data['last_updated_at'] = datetime.now().date()

# product_sales_schema_data["Current Sales"] = np.ceil(product_sales_schema_data["Current Sales"])
product_sales_schema_data["Current Sales"] = product_sales_schema_data["Current Sales"].astype(int)
# product_sales_schema_data["Last Year Sales"] = np.ceil(product_sales_schema_data["Last Year Sales"])
product_sales_schema_data["Last Year Sales"] = product_sales_schema_data["Last Year Sales"].astype(int)

product_sales_schema_data.rename(columns={
    "Current Sales": "current_sales",
    "Last Year Sales": "last_year_sales",
    "H1": "h1",
    "Channel": "channel",
    "Units_Base_UOM": "units_base_uom",
    "Value": "value",
}, inplace=True)

product_sales_schema_data['year_month'] = product_sales_schema_data['year_month'] + "-01"

product_sales_schema_data.to_excel(writer, sheet_name=sheet_name + "PRODUCT_SALES", index=False)

product_sales_schema_data.to_csv('./DATA/RESULTS/prod_summary.csv', index=False, sep=',')
# product_sales_schema_data.to_clipboard(index=False,sep=',')

# SCHEMA PRODUCT SALES SUMMARY - END

# SCHEMA ACCURACY SEGMENTATION & Summary: Forecast Accuracy START
# current_date=datetime(2023,11,1)

# current_date=datetime(2023,11,1)
sku_channel_history = forecast_history_data[['sku', 'Channel']].drop_duplicates()
sku_channel_history_data = pd.DataFrame()
for i in range(sku_channel_history.shape[0]):
    # print(i)
    sku_details_ = sku_channel_history.iloc[i, :]
    # print(sku_details_)
    d = pd.DataFrame()
    d['year_month'] = pd.Series(
        pd.date_range(current_date - pd.DateOffset(months=6), current_date, freq="M")) + pd.DateOffset(days=1)
    d['year_month'] = d['year_month'].astype(str).str[:-3]
    d['sku'] = sku_details_['sku']
    d['Channel'] = sku_details_['Channel']
    # display(d)
    sku_channel_history_data = pd.concat([sku_channel_history_data, d], axis=0)

# print(sku_channel_history_data.head())
# print(forecast_history_data.head())
forecast_history_data = pd.merge(sku_channel_history_data, forecast_history_data, on=['sku', 'year_month', 'Channel'],
                                 how='left')

forecast_history_data_ = forecast_history_data[forecast_history_data['final_perdiction'].notnull()]
accuracy_segmentation = pd.merge(forecast_history_data_,
                                 input_data_c[['sku', 'year_month', 'Channel', 'TOTAL_QTY_BASEUOM_SUM',
                                               'TOTAL_SALES_SUM', 'contract_sales']],
                                 on=['sku', 'year_month', 'Channel'], how='left')

accuracy_segmentation['TOTAL_QTY_BASEUOM_SUM'].fillna(0, inplace=True)
accuracy_segmentation['TOTAL_SALES_SUM'].fillna(0, inplace=True)

print(accuracy_segmentation)

# Attachin
accuracy_segmentation['Revision_Sales'] = 0

acc_final_data = pd.merge(accuracy_segmentation, sku_details[['sku', 'H1', 'NetWeight_calculated']], on='sku',
                          how='left')
print(acc_final_data)

acc_final_data.rename(columns={"TOTAL_QTY_BASEUOM_SUM": "actual_demand"}, inplace=True)

acc_final_data['final_perdiction'] = acc_final_data['final_perdiction'] + acc_final_data["contract_sales"]

acc_final_data['mape'] = np.abs(acc_final_data['actual_demand'] - acc_final_data['final_perdiction']) / \
                         acc_final_data[
                             'actual_demand']
c1 = acc_final_data['mape'] <= .25
c2 = acc_final_data['mape'] > 0.5
c3 = (acc_final_data['mape'] > 0.25) & (acc_final_data['mape'] <= 0.5)
acc_final_data["ACC_BUCKET"] = np.select([c1, c2, c3], ["HIGH", "LOW", 'MEDIUM'], "NA")
acc_final_data["ACC_BUCKET"] = np.where(np.isinf(acc_final_data['mape']), 'NA', acc_final_data['ACC_BUCKET'])
print(acc_final_data['mape'].isna().sum())
acc_final_data["mape"] = np.where(np.isinf(acc_final_data['mape']), 'NA', acc_final_data['mape'])
acc_final_data["mape"] = np.where(acc_final_data['mape'].isna(), 'NA', acc_final_data['mape'])

data_mapping_accuracy_segmentation = get_data_mapping_accuracy_segmentation(input_data_c)
# SCHEMA 3 ACCURACY SEGMENTATION
acc_final_data_schema = get_accuracy_segmentation_schema(acc_final_data, data_mapping_accuracy_segmentation)
# CORRECTION ACC SEG COUNT << START
c = sku_details['CrossPlantStatus'] == "AC"
add_all_data = {
    "H1": ["ALL"],
    'sku_count': [c.sum()]
}
h1_count = sku_details[c].groupby('H1', as_index=False)["sku"].count()
h1_count.rename(columns={'sku': 'sku_count'}, inplace=True)
h1_count = pd.concat([pd.DataFrame(add_all_data), h1_count], axis=0)
h1_count
acc_final_data_schema = pd.merge(acc_final_data_schema, h1_count, on=['H1'], how='left')
# acc_final_data_schema['count_sku_NA'] = acc_final_data_schema["sku_count"] - (
#             acc_final_data_schema['count_sku_HIGH'] + acc_final_data_schema['count_sku_LOW'] + acc_final_data_schema[
#         'count_sku_MEDIUM'])
# CORRECTION ACC SEG COUNT << END

acc_final_data_schema["client_id"] = 1
acc_final_data_schema['last_updated_at'] = datetime.now().date()

rename_col_for_acc_seg = {
    "sum_SALES_TOTAL_HIGH": "high_percentage_value",
    "sum_SALES_TOTAL_MEDIUM": "medium_percentage_value",
    "sum_SALES_TOTAL_LOW": "low_percentage_value",
    "sum_SALES_TOTAL_NA": "others_percentage_value",
    "count_sku_HIGH": "high_sku_count",
    "count_sku_MEDIUM": "medium_sku_count",
    "count_sku_LOW": "low_sku_count",
    "count_sku_NA": "others_sku_count",
    "H1": "h1",
    "Channel": "channel",
    "Statistical Accuracy": "statistical_accuracy",
    "Total Accuracy": "total_accuracy",
}

acc_final_data_schema.rename(columns=rename_col_for_acc_seg, inplace=True)

columns = ['year_month', 'high_percentage_value', 'low_percentage_value',
           'medium_percentage_value', 'others_percentage_value', 'high_sku_count',
           'low_sku_count', 'medium_sku_count', 'others_sku_count', 'channel',
           'h1', 'statistical_accuracy', 'total_accuracy', 'client_id',
           'last_updated_at']
actual_columns = set(columns)

columns_present = set(acc_final_data_schema.columns)

for i in actual_columns - columns_present:
    acc_final_data_schema[i] = np.nan

# acc_final_data_schema['year_month'] = pd.to_datetime(acc_final_data_schema['year_month'] + "-01")
acc_final_data_schema['year_month'] = acc_final_data_schema['year_month'] + "-01"

acc_final_data_schema["high_percentage_value"] = np.round(acc_final_data_schema['high_percentage_value'], decimals=2)
acc_final_data_schema["medium_percentage_value"] = np.round(acc_final_data_schema['medium_percentage_value'],
                                                            decimals=2)
acc_final_data_schema["low_percentage_value"] = np.round(acc_final_data_schema['low_percentage_value'], decimals=2)

print("Acc final Data Schema")
acc_final_data_schema.to_csv("./DATA/RESULTS/acc_seg.csv")
acc_final_data_schema.to_excel(writer, sheet_name=sheet_name + "ACC_SEG", index=False)

# SCHEMA 4 FORECAST ACCURACY
acc_final_data.isna().sum()
# acc_final_data = pd.merge(acc_final_data, sku_details[["sku",
#                                                        "NetWeight_calculated"]], on=['sku'], how='left')
acc_final_data.to_csv("./DATA/acc_final.csv")
forecast_accuracy_schema = get_forecast_accuracy_schema(acc_final_data, data_mapping_accuracy_segmentation)
forecast_accuracy_schema.rename(columns={"Predicted": "predicted"}, inplace=True)

forecast_accuracy_schema['year_month'] = forecast_accuracy_schema['year_month'] + '-01'
forecast_accuracy_schema['mape'] = 1 - forecast_accuracy_schema['mape']
forecast_accuracy_schema['mape'] = np.where(forecast_accuracy_schema['mape'] < 0,
                                            0, forecast_accuracy_schema['mape'])

forecast_accuracy_schema['client_id'] = 1
forecast_accuracy_schema['last_updated_at'] = datetime.now().date()
print("Acc forecast accuracy schema")
forecast_accuracy_schema.to_csv("./DATA/RESULTS/fore_acc.csv")
forecast_accuracy_schema.to_excel(writer, sheet_name=sheet_name + "FORECAST_ACC", index=False)
# SCHEMA ACCURACY SEGMENTATION & Summary: Forecast Accuracy END


acc_final_data['TOTAL_SALES_SUM'].fillna(0, inplace=True)
acc_final_data['actual_demand'].fillna(0, inplace=True)

data_mapping_confusion_matrix = get_data_mapping_confusion_matrix(input_data_c)

c = data_mapping_confusion_matrix['Channel'] != "ALL"
data_mapping_confusion_matrix = data_mapping_confusion_matrix[c]

# SCHEMA 5 CONFUSION MATRIX
acc_final_data = pd.merge(acc_final_data,
                          input_data_c[['sku', 'Channel', 'one_unit_cost']].drop_duplicates(),
                          on=['sku', 'Channel'],
                          how='left')
acc_final_data["actual_sales"] = acc_final_data['actual_demand'] * acc_final_data['one_unit_cost']
acc_final_data.to_csv('acc_final.csv', sep=',')
confusion_matrix_schema = get_confusion_matrix_schema(acc_final_data, data_mapping_confusion_matrix)
# confusion_matrix_schema = get_confusion_matrix_schema_new(acc_final_data, data_mapping_confusion_matrix, current_date)
confusion_matrix_schema['client_id'] = 1
confusion_matrix_schema['last_updated_at'] = datetime.now().date()
confusion_matrix_schema['year_month'] = confusion_matrix_schema['year_month'] + "-01"
confusion_matrix_schema.rename(columns={
    "Actual Sales": "actual_sales",
    "Predicted Sales": "predicted_sales",
    "No of SKUS": "no_of_skus",
    "H1": "h1",
    "Channel": "channel",
    "Total": "total",
    "Statistical": "statistical",

}, inplace=True)

print("Acc confusion matrix schema")
confusion_matrix_schema.to_csv('./DATA/RESULTS/confusion_matrix.csv')
confusion_matrix_schema.to_excel(writer, sheet_name=sheet_name + "CONFUSION_MATRIX", index=False)

# SCHEMA 6 SKU DEMAND PAGE
sku_demand_page = pd.merge(final_forecast_data.drop('H1', axis=1), sku_details, on=['sku'], how='left')

print("data map started for sku demand page")
## CAUTION

# data_mapping_sku_demand_page = pd.DataFrame()
# data_mapping_sku_demand_page=pd.read_csv('./DATA/RESULTS/data_map_sku_demand_page.csv')
# data_mapping_sku_demand_page.to_csv('./DATA/RESULTS/data_map_sku_demand_page.csv',index=False)

variability_index = generate_variability_index(input_data_c, current_date)

variability_data = generate_variability_schema(variability_index)
variability_data.rename(columns={
    "Variability Bucket": "variability_bucket",
    "% Of SKUs": "percentage_of_skus",
    "Channel": "channel",
    "H1": "h1"
}, inplace=True)
variability_data['last_updated_at'] = datetime.now().date()
variability_data.to_csv('./DATA/RESULTS/variability_data.csv')
variability_data.to_excel(writer, sheet_name=sheet_name + "VARIABILITY_INDEX", index=False)

# writer.close()
data_mapping_sku_demand_page = get_data_mapping_sku_demand_page(sku_demand_page)
# variability_index.to_csv("./DATA/RESULTS/varia_index.csv")
print("data map over for sku demand page")
c = variability_index['H1'] != 'ALL'
c1 = variability_index['Channel'] == 'ALL'
# std_deviation_details = variability_index[c & c1][['sku', 'Standard_Dev', 'Variability']].drop_duplicates()
# VAIRA NEW CODE >>
input_data__c = input_data_c.copy()
input_data__c['TOTAL_QTY_BASEUOM_SUM'] = input_data__c['TOTAL_QTY_BASEUOM_SUM'].fillna(0)
all_channel_input_data = input_data__c.groupby(['sku', 'year_month'], as_index=False)[['TOTAL_QTY_BASEUOM_SUM']].sum()
all_channel_input_data["date"] = pd.to_datetime(all_channel_input_data['year_month'] + "-01")
c = all_channel_input_data['date'] <= current_date
c1 = all_channel_input_data['date'] > current_date - pd.DateOffset(months=24)
all_channel_input_data = all_channel_input_data[c & c1]

std__ = all_channel_input_data.groupby("sku", as_index=False).agg({'TOTAL_QTY_BASEUOM_SUM': ['mean', 'std']})
std__.columns = ['sku', 'Mean', 'Standard_Dev']
std__["Variability"] = std__["Standard_Dev"] / std__["Mean"]
std_deviation_details = std__.copy()
std_deviation_details.to_csv('./DATA/RESULTS/std_std.csv')

# VAIRA NEW CODE <<


sku_demand_page = pd.merge(sku_demand_page, std_deviation_details, on=['sku'], how='left')
sku_demand_page = pd.merge(sku_demand_page, input_data_c[['sku', 'year_month', 'Channel', "last_year_demand"]],
                           on=['sku', 'year_month', 'Channel'], how='left')

input_data_c = pd.merge(input_data_c, std_deviation_details, on=['sku'], how='left')

std_deviation_details = pd.merge(std_deviation_details, sku_details[['sku', 'LeadTime', 'ServiceLevel_value']]
                                 , on=['sku'], how='left')

input_data_c = pd.merge(input_data_c.drop('H1', axis=1), sku_details, on=['sku'], how='left')

# OCT 24 new changes avg of previous 12 x 1.2 instead of standard deviation >> Start

c1 = input_data_c['date'] > (current_date - pd.DateOffset(months=12))
c2 = input_data_c['date'] <= current_date
last_one_year_data = input_data_c[c1 & c2]
last_one_year_data = last_one_year_data.groupby(['sku', 'year_month'], as_index=False)['TOTAL_QTY_BASEUOM_SUM'].sum()
last_one_year_data = last_one_year_data.groupby('sku', as_index=False)['TOTAL_QTY_BASEUOM_SUM'].mean()
last_one_year_data['TOTAL_QTY_BASEUOM_SUM_AVG_1.2'] = last_one_year_data['TOTAL_QTY_BASEUOM_SUM'] * 1.2
last_one_year_data.drop('TOTAL_QTY_BASEUOM_SUM', axis=1, inplace=True)
input_data_c = pd.merge(input_data_c, last_one_year_data, on=['sku'], how='left')
std_deviation_details = pd.merge(std_deviation_details, last_one_year_data, on=['sku'], how='left')

# OLD method Std deviation calculation
std_deviation_details['Safety Stock'] = np.sqrt(np.ceil(std_deviation_details['LeadTime'].astype(float) / 30)) * \
                                        std_deviation_details[
                                            'ServiceLevel_value'].astype(float) * (
                                            std_deviation_details['Standard_Dev']) * 1.2

# NEW Method Calculation
# std_deviation_details['Safety Stock'] = np.sqrt(np.ceil(std_deviation_details['LeadTime'].astype(float) / 30)) * \
#                                         std_deviation_details[
#                                             'ServiceLevel_value'].astype(float) * (
#                                             std_deviation_details['TOTAL_QTY_BASEUOM_SUM_AVG_1.2'])


# OCT 24 new changes avg of previous 12 x 1.2 instead of standard deviation << END

# OLD METHOD SAFETY STOCK CALCULATION
input_data_c['Safety Stock'] = np.sqrt(np.ceil(sku_demand_page['LeadTime'].astype(float) / 30)) * input_data_c[
    'ServiceLevel_value'].astype(float) * (input_data_c['Standard_Dev']) * 1.2

# NEW METHOD SAFETY STOCK CALCULATION (OCT_24) avg of 12 month sales *1,3
# input_data_c['Safety Stock'] = np.sqrt(np.ceil(sku_demand_page['LeadTime'].astype(float) / 30)) * input_data_c[
#     'ServiceLevel_value'].astype(float) * (input_data_c['TOTAL_QTY_BASEUOM_SUM_AVG_1.2'])

input_data_c.head()

input_data_c = pd.merge(input_data_c, forecast_history_data, on=['year_month', 'sku', 'Channel'], how='left')

sku_demand_page['Revision_Sales'] = 0
sku_demand_page['Contract_Sales'] = 0
print("Schema Data started colected")
print(data_mapping_sku_demand_page.shape)
print(data_mapping_sku_demand_page.head())
c = data_mapping_sku_demand_page['Statistical'] == "1"
data_mapping_sku_demand_page = data_mapping_sku_demand_page[c]
print(data_mapping_sku_demand_page.shape, 'PRINT 1')
sku_details['sts'] = 1
data_mapping_sku_demand_page = pd.merge(data_mapping_sku_demand_page,
                                        sku_details[['H1', 'H2', 'MrpPurchaser', 'sts']].drop_duplicates(),
                                        on=['H1', 'H2', "MrpPurchaser"], how='left')
c = data_mapping_sku_demand_page['sts'].notnull()
data_mapping_sku_demand_page = data_mapping_sku_demand_page[c]
print(data_mapping_sku_demand_page.shape, "Print2")

data_mapping_sku_demand_page.drop('sts', axis=1, inplace=True)
print(data_mapping_sku_demand_page.shape, "PRINTING SHAPE OF DATA MAP")
#
# input_data_c['TOTAL_QTY_BASEUOM_SUM'].fillna(0, inplace=True)
# input_data_c['QTY_BASEUOM_SUM'].fillna(0, inplace=True)
# input_data_c['contract_sales'] = input_data_c['TOTAL_QTY_BASEUOM_SUM'] - input_data_c['QTY_BASEUOM_SUM']

sku_demand_schema = get_sku_demand_page_schema(sku_demand_page
                                               , data_mapping_sku_demand_page,
                                               sku_details,
                                               input_data_c,
                                               current_date)

sku_demand_schema['Revision_Sales'] = 0

try:
    sku_demand_schema.drop(['Variability', 'Safety Stock'], axis=1, inplace=True)
    print("REMOVED Variabilitu DATA & safety stock")
except:
    print("Error REMOVED Variabilitu DATA & safety stock")
    pass
input_data_c['Safety Stock'] = np.sqrt(np.ceil(sku_demand_page['LeadTime'].astype(float) / 30)) * input_data_c[
    'ServiceLevel_value'].astype(float) * (input_data_c['Standard_Dev']) * 1.2

sku_demand_schema = pd.merge(sku_demand_schema,
                             std_deviation_details[['sku', 'Variability', 'Safety Stock']].drop_duplicates(),
                             on=['sku'],
                             how='left')

# sku_demand_page['Safety Stock'] = np.sqrt(np.ceil(sku_demand_page['LeadTime'].astype(float) / 30)) * sku_demand_page[
#     'ServiceLevel_value'].astype(float) * (sku_demand_page['Standard_Dev'])

# sku_demand_schema['contract_sales'] = 0
sku_demand_schema.to_csv("./DATA/RESULTS/sku_demand.csv", index=False, sep=',')

mape = (np.abs(sku_demand_schema['QTY_BASEUOM_SUM'] - sku_demand_schema['final_perdiction']) / sku_demand_schema[
    'QTY_BASEUOM_SUM'])
c1 = np.isinf(mape)
c2 = mape.isna()
np.where(c1 | c2, "NULL", mape)
sku_demand_schema["mape"] = mape
# REquired
sku_demand_schema['year'] = pd.to_datetime(sku_demand_schema['year_month'] + "-01").dt.year
sku_demand_schema['month'] = pd.to_datetime(sku_demand_schema['year_month'] + "-01").dt.month

rename_dict = {
    # "year_month":"",
    "PdProductDescription": "product_name",
    "H1": "h1",
    "H2": "h2",
    "Channel": "channel",
    "MrpPurchaser": "mrp_purchaser",
    "ServiceLevel": "desired_service_level",
    "Variability": "variability",
    "3_months_acc": "three_month_accuracy",
    "Safety Stock": "safety_stock",
    "Revision_Sales": "revision",
    "final_perdiction": "statistical_forecast",
    "Predicted": "final_forecast",
    "TOTAL_QTY_BASEUOM_SUM": "actuals",
    "Statistical": "statistical",
    "Total": "total",
}
sku_demand_schema.rename(columns=rename_dict, inplace=True)

sku_demand_schema['client_id'] = 1
cols_req = ['sku', 'product_name', 'h1', 'h2', 'channel', 'mrp_purchaser',
            'desired_service_level', 'variability', 'three_month_accuracy', 'revision', 'contract_sales',
            'final_forecast', 'mape',
            'month', 'year', 'statistical_forecast', 'last_year_demand', 'actuals',
            'safety_stock', 'statistical', 'total', 'client_id']

# sku_demand_schema[cols_req].to_clipboard(index=False,sep=',')
# for i in ['three_month_accuracy', 'final_forecast', 'mape', 'actuals']:
#     sku_demand_schema[i].fillna("NULL", inplace=True)

sku_demand_schema['client_id'] = 1
cols_req = ['sku', 'product_name', 'h1', 'h2', 'channel', 'mrp_purchaser',
            'desired_service_level', 'variability', 'three_month_accuracy', 'revision', 'contract_sales',
            'final_forecast', 'mape',
            'month', 'year', 'statistical_forecast', 'last_year_demand', 'actuals',
            'safety_stock', 'statistical', 'total', 'client_id']

sku_demand_schema = sku_demand_schema[cols_req]

sku_demand_schema['statistical_forecast'] = np.round(sku_demand_schema['statistical_forecast'])
# .astype(int, errors='ignore'))
sku_demand_schema['safety_stock'] = np.ceil(sku_demand_schema['safety_stock'])
sku_demand_schema['mape'] = np.where(np.isinf(sku_demand_schema['mape']), np.nan, sku_demand_schema['mape'])
sku_demand_schema['final_forecast'] = sku_demand_schema['statistical_forecast'].fillna(0).astype(float).astype(int) + sku_demand_schema[
    'revision'].astype(int)
# .astype(int, errors='ignore'))
print(sku_demand_schema.head())
print(sku_demand_schema.dtypes)
print(sku_demand_schema['statistical'])
print(sku_demand_schema['statistical'].unique())
statistical_c = sku_demand_schema['statistical'] == '1'
sku_demand_schema['last_updated_at'] = datetime.now().date()

sku_demand_schema = sku_demand_schema[statistical_c]
print(sku_demand_schema.head())
print("Acc sku demand  schema")

# Adding Rank Section
sku_demand = sku_demand_schema.copy()
sku_demand['variability'] = np.round(sku_demand['variability'], 2)
filter_sku = pd.read_csv('./DATA/INPUT_DATA/filter_BIN_.csv')
c = filter_sku['Bin - Rev'].isin(['A', 'B'])

sku_list_ab_category = filter_sku[c]['MATERIAL'].unique()
a_b_sku_demand = sku_demand[sku_demand['sku'].isin(sku_list_ab_category)]
rank_ = pd.DataFrame()
rank_['sku'] = a_b_sku_demand['sku'].unique()
rank_ = rank_.sort_values('sku').reset_index(drop=True).reset_index()
rank_.rename(columns={"index": "rank"}, inplace=True)
rank_['rank'] = rank_['rank'] + 1
a_b_sku_demand = pd.merge(rank_, a_b_sku_demand, on=['sku'], how='left')

non_a_b_sku_demand = sku_demand[~sku_demand['sku'].isin(sku_list_ab_category)]

ranks = rank_.shape[0]

rank_ = pd.DataFrame()
rank_['sku'] = non_a_b_sku_demand['sku'].unique()
rank_ = rank_.sort_values('sku').reset_index(drop=True).reset_index()
rank_.rename(columns={"index": "rank"}, inplace=True)
rank_['rank'] = rank_['rank'] + (ranks + 1)

non_a_b_sku_demand = pd.merge(rank_, non_a_b_sku_demand, on=['sku'], how='left')

sku_demand_page = pd.concat([a_b_sku_demand, non_a_b_sku_demand], axis=0)

# Adding rank section end

# Changing accuracy >>
sku_demand_page['three_month_accuracy'] = 1 - sku_demand_page['three_month_accuracy']
sku_demand_page['three_month_accuracy'] = np.where(sku_demand_page['three_month_accuracy'] < 0,
                                                   0, sku_demand_page['three_month_accuracy'])

sku_demand_page['mape'] = 1 - sku_demand_page['mape']
sku_demand_page['mape'] = np.where(sku_demand_page['mape'] < 0,
                                   0, sku_demand_page['mape'])
sku_demand_page['mape'] = np.round(sku_demand_page['mape'] * 100)

# Changing accuracy <<

# Adding Extra zero sales sku >>

sample_sku_detail = sku_demand_page.iloc[0, :]
sku_ = sample_sku_detail['sku']
channel_ = sample_sku_detail['channel']
channel_ = sample_sku_detail['channel']
c = sku_demand_page['sku'] == sku_
c1 = sku_demand_page['channel'] == channel_
print(sku_, channel_)
sample_sku_schema = sku_demand_page[c & c1].drop_duplicates()

sample_sku_schema['three_month_accuracy'] = np.nan
sample_sku_schema['statistical'] = 0
sample_sku_schema['safety_stock'] = np.nan
sample_sku_schema['last_year_demand'] = 0
sample_sku_schema['contract_sales'] = 0
sample_sku_schema['revision'] = 0
sample_sku_schema['actuals'] = 0
sample_sku_schema['total'] = 0
sample_sku_schema['final_forecast'] = 0
sample_sku_schema['mape'] = np.nan
sample_sku_schema['variability'] = 0
sample_sku_schema['statistical_forecast'] = 0

a = set(sku_demand_page['sku'].unique())
b = set(sku_details[sku_details['CrossPlantStatus'] == "AC"]['sku'].unique())

extra_sku = pd.DataFrame()
for i in (b - a):
    c = sku_details['sku'] == i
    sku_details_ = sku_details[c]
    dummy_2 = sample_sku_schema.copy()
    # for j in sku_demand['channel'].unique():
    for j in ["B2B", "Ecomm", "Retail", "Export_Re-Export"]:
        dummy_2['sku'] = i
        dummy_2['channel'] = j
        dummy_2['product_name'] = sku_details_['PdProductDescription'].values[0]
        dummy_2['desired_service_level'] = sku_details_['ServiceLevel'].values[0]
        dummy_2['h1'] = sku_details_['H1'].values[0]
        dummy_2['h2'] = sku_details_['H2'].values[0]
    extra_sku = pd.concat([extra_sku, dummy_2], axis=0)

try:
    del extra_sku['rank']
except:
    pass

ranks = sku_demand_page['rank'].max() + 1
if len(b - a) != 0:
    rank_ = pd.DataFrame()
    rank_['sku'] = extra_sku['sku'].unique()
    rank_ = rank_.sort_values('sku').reset_index(drop=True).reset_index()
    rank_.rename(columns={"index": "rank"}, inplace=True)
    rank_['rank'] = rank_['rank'] + ranks
    extra_sku = pd.merge(rank_, extra_sku, on=['sku'], how='left')
    sku_demand_page = pd.concat([sku_demand_page, extra_sku], axis=0)

# Adding Extra zero sales sku <<


sku_other_details = pd.read_csv('./DATA/INPUT_DATA/PhoonHuat Service Level and Lead times 20240911.csv')
sku_other_details = sku_other_details[["MATERIAL", "SALES_BIN", "CUST_BIN", "SO_BIN"]].drop_duplicates(
    subset=['MATERIAL']
)
sku_other_details.rename(
    columns={"MATERIAL": "sku", "SALES_BIN": "sales_bin", "CUST_BIN": "cust_bin", "SO_BIN": "so_bin"}, inplace=True)
sku_other_details = sku_other_details[['sku', 'sales_bin', 'cust_bin', 'so_bin']]
sku_demand_page = pd.merge(sku_demand_page, sku_other_details, on=['sku'], how='left')

sku_demand_page["statistical"] = 1
sku_demand_page.to_excel(writer, sheet_name=sheet_name + "SKU_DEMAND_PAGE", index=False)

writer.close()
