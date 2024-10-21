import pandas as pd
import numpy as np
from datetime import datetime
import warnings

from get_data_mapping import get_datamapping_pred_vs_last_year_demand, get_data_mapping_accuracy_segmentation, \
    get_data_mapping_summary_product_sales, get_data_mapping_confusion_matrix
from pred_vs_last_year_demand import generate_prediction_vs_last_year_demand, get_confusion_matrix_schema, \
    get_forecast_accuracy_schema, get_accuracy_segmentation_schema, generate_summary_product_sales
from utility import make_input_data_continous, collect_data, make_input_data_continous_new

warnings.filterwarnings('ignore')

output_folder_ = './DATA/RESULTS/'
mape_file_name = 'OCT_24_MAR_25_SCHEMA_DATA_17_10'
# mape_file_name='MAPE_ARIMA_SES_EXP_REXP'
# sheet_name="APR_24"
sheet_name = "Schema_"

writer = pd.ExcelWriter(output_folder_ + mape_file_name + '.xlsx', engine='xlsxwriter')

# SCHEMA 1
final_forecast_path = 'DATA/MAY_24_FORECAST/combined_file_new_oct_24_mar_25.csv'
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
print(input_data['H1'].unique(), 'Printing_DATA')
print(final_forecast_data.head(), 'Printing_FF')


input_data_c=make_input_data_continous_new(input_data,current_date)

input_data_c.sort_values(['MATERIAL', 'H1', "CHNL_NAME", 'date'], inplace=True)
input_data_c.to_csv('./DATA/RESULTS/input_data_.csv', index=False)

# Calculating Last Year Demand for SKU H1 & CHNL_NAME
input_data_c['last_year_demand'] = input_data_c.groupby(['MATERIAL', 'H1', 'CHNL_NAME'])['TOTAL_QTY_BASEUOM_SUM'].shift(
    12)
input_data_c['last_year_demand_non_contract'] = input_data_c.groupby(['MATERIAL', 'H1', 'CHNL_NAME'])[
    'QTY_BASEUOM_SUM'].shift(12)
input_data_c['last_year_KG_SUM'] = input_data_c.groupby(['MATERIAL', 'H1', 'CHNL_NAME'])['KG_SUM'].shift(12)

final_forecast_data['year_month'] = final_forecast_data['forecast_month']


input_d = input_data_c.copy()
c = input_d['TOTAL_QTY_BASEUOM_COUNT'] > 0
c1 = input_d['TOTAL_QTY_BASEUOM_SUM'] > 0
input_d = input_d[c & c1]


max_date_input = input_d.groupby(['MATERIAL', 'CHNL_NAME'], as_index=False)['date'].max()
max_date_input.rename(columns={"date": "max_date"}, inplace=True)

input_d = pd.merge(input_d, max_date_input, on=['MATERIAL', 'CHNL_NAME'], how='left')

c = input_d['date'] == input_d['max_date']
input_d = input_d[c][['MATERIAL', 'CHNL_NAME', "TOTAL_SALES_SUM", 'TOTAL_QTY_BASEUOM_SUM']]


input_d["one_unit_cost"] = input_d['TOTAL_SALES_SUM'] / input_d['TOTAL_QTY_BASEUOM_SUM']
input_data_c = pd.merge(input_data_c, input_d[['MATERIAL', 'CHNL_NAME', 'one_unit_cost']], on=['MATERIAL', 'CHNL_NAME'],
                        how='left')




input_data_c.rename(columns={"MATERIAL": "sku", 'CHNL_NAME': "Channel"}, inplace=True)
# input_data_c.to_clipboard(index=False,sep=',')


input_data_c['TOTAL_QTY_BASEUOM_SUM'].fillna(0, inplace=True)
input_data_c['QTY_BASEUOM_SUM'].fillna(0, inplace=True)
input_data_c['contract_sales'] = input_data_c['TOTAL_QTY_BASEUOM_SUM'] - input_data_c['QTY_BASEUOM_SUM']



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


final_file_schema_1['Revised_Sale'] = 0

# SCHEMA 1 PREDICTION VS LAST YEAR DEMAND
pred_vs_lyd_schema = generate_prediction_vs_last_year_demand(final_file_schema_1,
                                                             data_mapping_prediction_vs_last_year_demand)

pred_vs_lyd_schema.to_csv('./DATA/RESULTS/pred_vslast.csv', index=False, sep=',')

pred_vs_lyd_schema['prediction'] = np.ceil(pred_vs_lyd_schema['prediction'])

pred_vs_lyd_schema['last_year_demand'] = np.ceil(pred_vs_lyd_schema['last_year_demand'])

pred_vs_lyd_schema.to_excel(writer, sheet_name=sheet_name + "PRED_VS_LAST_YEAR_DEMAND", index=False)

c = input_data_c['date'] <= current_date
c2 = input_data_c['date'] > current_date - pd.DateOffset(months=6)
summary_product_sales_data = input_data_c[c & c2]


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


sku_channel_history = forecast_history_data[['sku', 'Channel']].drop_duplicates()


history_date_range=(pd.date_range(current_date - pd.DateOffset(months=6), current_date, freq="M") + pd.DateOffset(days=1)).astype(str).str[:-3]
print(history_date_range)

sku_channel_history['grp']=1
sku_channel_history.head()

dummy_data=pd.DataFrame({'grp':[1],'year_month':[history_date_range]})
sku_channel_history_data_=pd.merge(sku_channel_history,dummy_data,on=['grp'],how='left')
sku_channel_history_data_=sku_channel_history_data_.explode('year_month')

forecast_history_data = pd.merge(sku_channel_history_data_, forecast_history_data, on=['sku', 'year_month', 'Channel'],
                                 how='left')


forecast_history_data_ = forecast_history_data[forecast_history_data['final_perdiction'].notnull()]

accuracy_segmentation = pd.merge(forecast_history_data_,
                                 input_data_c[['sku', 'year_month', 'Channel', 'TOTAL_QTY_BASEUOM_SUM',
                                               'TOTAL_SALES_SUM', 'contract_sales']],
                                 on=['sku', 'year_month', 'Channel'], how='left')

accuracy_segmentation['TOTAL_QTY_BASEUOM_SUM'].fillna(0, inplace=True)
accuracy_segmentation['TOTAL_SALES_SUM'].fillna(0, inplace=True)

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
acc_final_data_schema.drop('sku_count',axis=1,inplace=True)
acc_final_data_schema.to_csv("./DATA/RESULTS/acc_seg.csv")
acc_final_data_schema.to_excel(writer, sheet_name=sheet_name + "ACC_SEG", index=False)

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




# SKU DEMAND SECTION

forecast_to_attach=pd.concat([forecast_history_data_,final_forecast_data[['sku','year_month','Channel','final_perdiction']]])
input_data_c1=pd.merge(input_data_c,forecast_to_attach,on=['sku','year_month','Channel'],how='left')
input_data_c1['mape'] = np.abs(input_data_c1['final_perdiction'] - input_data_c1['QTY_BASEUOM_SUM']) / input_data_c1['QTY_BASEUOM_SUM']
input_data_c1['notinf'] = np.where(np.isinf(input_data_c1['mape']), 0, 1)
input_data_c1['mape'] = np.where(np.isinf(input_data_c1['mape']), 0, input_data_c1['mape'])

input_data_c1.sort_values(['sku','Channel','year_month'],inplace=True)
input_data_c1['3_mape']=input_data_c1.groupby(['sku','Channel'])['mape'].rolling(3).sum().values
input_data_c1['3_notinf']=input_data_c1.groupby(['sku','Channel'])['notinf'].rolling(3).sum().values
input_data_c1['3_months_acc'] = input_data_c1['3_mape'] / input_data_c1['3_notinf']
input_data_c1['3_months_acc']=np.where(input_data_c1['date']>current_date,np.nan,input_data_c1['3_months_acc'])
input_data_c1['TOTAL_QTY_BASEUOM_SUM']=np.where(input_data_c1['date']>current_date,np.nan,input_data_c1['TOTAL_QTY_BASEUOM_SUM'])


# Attach _revision here
input_data_c1['revision']=0

input_data_c1['statistical_forecast']=input_data_c1['final_perdiction']

input_data_c1['final_forecast'] = input_data_c1['statistical_forecast'].fillna(0).astype(float).astype(int) + input_data_c1['revision'].astype(int)

# print(sku_demand_schema['statistical'])
# print(sku_demand_schema['statistical'].unique())
input_data_c1['statistical'] = '1'
input_data_c1['last_updated_at'] = datetime.now().date()


input_data_c1=pd.merge(input_data_c1,sku_details[['sku','ServiceLevel','ServiceLevel_value',"LeadTime"]],on=['sku'],how='left')

c=input_data_c1['date']<=current_date
c1=input_data_c1['date']>(current_date-pd.DateOffset(months=24))
sku_wise_data=input_data_c[c & c1][['sku','year_month','TOTAL_QTY_BASEUOM_SUM']].groupby(['sku','year_month'],as_index=False)['TOTAL_QTY_BASEUOM_SUM'].sum()
std_dev=sku_wise_data.groupby(['sku'],as_index=False).agg({'TOTAL_QTY_BASEUOM_SUM':['std','var','mean']})
std_dev.columns=['sku','Standard_Dev','var','mean']
std_dev['Variability']=std_dev['Standard_Dev']/std_dev['mean']
input_data_c1=pd.merge(input_data_c1,std_dev,on=['sku'],how='left')

input_data_c1['Safety Stock'] = np.sqrt(np.ceil(input_data_c1['LeadTime'].astype(float) / 30)) * input_data_c1['ServiceLevel_value'].astype(float) * (input_data_c1['Standard_Dev']) * 1.2

c=input_data_c1['date']> (current_date-pd.DateOffset(months=2))
sku_demand_schema=input_data_c1[c]


sku_demand_schema['total']=0

sku_demand_schema=pd.merge(sku_demand_schema,sku_details[['sku','H2','MrpPurchaser','PdProductDescription']],on=['sku'],how='left')


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
    # "final_perdiction": "statistical_forecast",
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
sku_demand_schema=sku_demand_schema[cols_req]


sku_demand_schema['statistical_forecast'] = np.round(sku_demand_schema['statistical_forecast'])
# .astype(int, errors='ignore'))
sku_demand_schema['safety_stock'] = np.ceil(sku_demand_schema['safety_stock'])
sku_demand_schema['mape'] = np.where(np.isinf(sku_demand_schema['mape']), np.nan, sku_demand_schema['mape'])

sku_demand_schema['final_forecast'] = sku_demand_schema['statistical_forecast'].fillna(0).astype(float).astype(int) + sku_demand_schema[
    'revision'].astype(int)


statistical_c = sku_demand_schema['statistical'] == '1'
sku_demand_schema['last_updated_at'] = datetime.now().date()

# SKU _DEMAND PART 2

sku_demand_page = sku_demand_schema.copy()

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
a,b


from tqdm import tqdm
extra_sku = pd.DataFrame()
for i in tqdm(b - a):
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

sku_demand=pd.concat([sku_demand_page,extra_sku.reset_index(drop=True)],axis=0)

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

sku_demand_page_1 = pd.concat([a_b_sku_demand, non_a_b_sku_demand], axis=0)

# Adding rank section end

# Changing accuracy >>
sku_demand_page_1['three_month_accuracy'] = 1 - sku_demand_page_1['three_month_accuracy']
sku_demand_page_1['three_month_accuracy'] = np.where(sku_demand_page_1['three_month_accuracy'] < 0,
                                                   0, sku_demand_page_1['three_month_accuracy'])

sku_demand_page_1['mape'] = 1 - sku_demand_page_1['mape']
sku_demand_page_1['mape'] = np.where(sku_demand_page_1['mape'] < 0,
                                   0, sku_demand_page_1['mape'])
sku_demand_page_1['mape'] = np.round(sku_demand_page_1['mape'] * 100)

sku_other_details = pd.read_csv('./DATA/INPUT_DATA/PhoonHuat Service Level and Lead times 20240911.csv')
sku_other_details = sku_other_details[["MATERIAL", "SALES_BIN", "CUST_BIN", "SO_BIN"]].drop_duplicates(
    subset=['MATERIAL']
)

sku_other_details.rename(
    columns={"MATERIAL": "sku", "SALES_BIN": "sales_bin", "CUST_BIN": "cust_bin", "SO_BIN": "so_bin"}, inplace=True)
sku_other_details = sku_other_details[['sku', 'sales_bin', 'cust_bin', 'so_bin']]
sku_demand_page_1 = pd.merge(sku_demand_page_1, sku_other_details, on=['sku'], how='left')

# sku_demand_page["statistical"] = 1
sku_demand_page_1.to_excel(writer, sheet_name=sheet_name + "SKU_DEMAND_PAGE", index=False)

writer.close()
