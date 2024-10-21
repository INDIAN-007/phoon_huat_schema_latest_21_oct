import numpy as np
import pandas as pd


def get_data_from_rules(d, col_name, schema_name):
    df = d.copy()
    if schema_name == 'Prediction_Vs_Last_Year_Demand':
        rules_1 = {
            "Statistical Accuracy_KG": df['final_perdiction'].astype(float).astype(int) * df["NetWeight"],
            "Statistical Accuracy_Value": df['final_perdiction'].astype(float).astype(int) * df["one_unit_cost"],
            "Total Accuracy_KG": (df['final_perdiction'] + df['Revised_Sale']).astype(float).astype(int) * df[
                'NetWeight'],
            #         "Total Accuracy_Value":df['final_perdiction']+df['Revised_Sale'],
            "Total Accuracy_Value": pd.Series(np.ceil(df['final_perdiction'] + df['Revised_Sale'])).astype(
                float).astype(int) * df["one_unit_cost"],

        }

        rules_2 = {
            "Statistical Accuracy_KG": df['last_year_demand_non_contract'] * df["NetWeight"],
            "Statistical Accuracy_Value": df['last_year_demand_non_contract'] * df["one_unit_cost"],
            "Total Accuracy_KG": (df['last_year_demand_non_contract']) * df['NetWeight'],
            "Total Accuracy_Value": df['last_year_demand_non_contract'] * df["one_unit_cost"],
        }

        df['Prediction'] = rules_1[col_name]
        df['Last year Demand'] = rules_2[col_name]
        return df
    if schema_name == "summary_product_sales":
        rules_1 = {
            "Value": df['TOTAL_QTY_BASEUOM_SUM'] * df['one_unit_cost'],
            # "Units_Base_UOM": (df['TOTAL_QTY_BASEUOM_SUM']) * df['NetWeight']
            "Units_Base_UOM": df['KG_SUM']
        }
        rules_2 = {
            "Value": df['last_year_demand'] * df['one_unit_cost'],
            "Units_Base_UOM": df['last_year_KG_SUM']
        }

        df['Current Sales'] = rules_1[col_name]
        df['Last Year Sales'] = rules_2[col_name]
        return df

    if schema_name == "segmentation_accuracy":
        rules_1 = {
            "Statistical Accuracy": df['TOTAL_SALES_SUM'],
            "Total Accuracy": (df['TOTAL_SALES_SUM'] + df["Revision_Sales"])
        }

        df['SALES_TOTAL'] = rules_1[col_name]
        return df

    if schema_name == "forecast_accuracy":
        df['final_perdiction'].fillna(0, inplace=True)
        rules_1 = {

            "Statistical Accuracy": df['final_perdiction'].astype(float).astype(int),
            "Total Accuracy": (df['final_perdiction'].astype(float).astype(int) + df["Revision_Sales"])
        }

        df['Predicted'] = rules_1[col_name]
        return df

    if schema_name == "confusion_matrix":
        df['final_perdiction'].fillna(0, inplace=True)
        rules_1 = {
            "Statistical": df['final_perdiction'].astype(int) * df['one_unit_cost'],
            "Total": (df['final_perdiction'].astype(int) + df["Revision_Sales"]) * df['one_unit_cost']
        }

        df['Predicted'] = rules_1[col_name]
        return df
    if schema_name == "sku_demand_page":
        rules_1 = {
            "Statistical": df['final_perdiction'].astype(int),
            "Total": (df['final_perdiction'].astype(int) + df["Revision_Sales"])
        }

        df['Predicted'] = rules_1[col_name]
        return df


def get_data_from_rules_1(d, col_name, schema_name):
    df = d.copy()
    if schema_name == 'Prediction_Vs_Last_Year_Demand':
        rules_1 = {
            "Statistical Accuracy_KG": df['final_perdiction'].astype(float).astype(int) * df["NetWeight"],
            "Statistical Accuracy_Value": df['final_perdiction'].astype(float).astype(int) * df["one_unit_cost"],
            "Total Accuracy_KG": (df['final_perdiction'] + df['Revised_Sale']).astype(float).astype(int) * df[
                'NetWeight'],
            #         "Total Accuracy_Value":df['final_perdiction']+df['Revised_Sale'],
            "Total Accuracy_Value": pd.Series(np.ceil(df['final_perdiction'] + df['Revised_Sale'])).astype(
                float).astype(int) * df["one_unit_cost"],

        }

        rules_2 = {
            "Statistical Accuracy_KG": df['last_year_demand'] * df["NetWeight"],
            "Statistical Accuracy_Value": df['last_year_demand'] * df["one_unit_cost"],
            "Total Accuracy_KG": (df['last_year_demand']) * df['NetWeight'],
            "Total Accuracy_Value": df['last_year_demand'] * df["one_unit_cost"],
        }

        df['Prediction'] = rules_1[col_name]
        df['Last year Demand'] = rules_2[col_name]
        return df
    if schema_name == "summary_product_sales":
        rules_1 = {
            # "Value": df['TOTAL_QTY_BASEUOM_SUM'] * df['one_unit_cost'],
            # "Value": df['TOTAL_QTY_BASEUOM_SUM'] * df['one_unit_cost'],
            "Value": df['TOTAL_QTY_BASEUOM_SUM'],
            "Units_Base_UOM": (df['TOTAL_QTY_BASEUOM_SUM'])
            # "Units_Base_UOM": (df['TOTAL_QTY_BASEUOM_SUM']) * df['NetWeight']
        }
        rules_2 = {
            # "Value": df['last_year_demand'] * df['one_unit_cost'] ,
            "Value": df['last_year_demand'],
            "Units_Base_UOM": (df['last_year_demand'])
            # "Units_Base_UOM": (df['last_year_demand']) * df['NetWeight']
        }

        df['Current Sales'] = rules_1[col_name]
        df['Last Year Sales'] = rules_2[col_name]
        return df

    if schema_name == "segmentation_accuracy":
        rules_1 = {
            "Statistical Accuracy": df['SALES_SUM'],
            "Total Accuracy": (df['SALES_SUM'] + df["Revision_Sales"])
        }

        df['SALES_TOTAL'] = rules_1[col_name]
        return df

    if schema_name == "forecast_accuracy":
        df['final_perdiction'].fillna(0, inplace=True)
        rules_1 = {

            "Statistical Accuracy": df['final_perdiction'].astype(float).astype(int),
            "Total Accuracy": (df['final_perdiction'].astype(float).astype(int) + df["Revision_Sales"])
        }

        df['Predicted'] = rules_1[col_name]
        return df

    if schema_name == "confusion_matrix":
        df['final_perdiction'].fillna(0, inplace=True)
        rules_1 = {
            "Statistical": df['final_perdiction'].astype(int),
            "Total": (df['final_perdiction'].astype(int) + df["Revision_Sales"])
        }

        df['Predicted'] = rules_1[col_name]
        return df
    if schema_name == "sku_demand_page":
        rules_1 = {
            "Statistical": df['final_perdiction'].astype(int),
            "Total": (df['final_perdiction'].astype(int) + df["Revision_Sales"])
        }

        df['Predicted'] = rules_1[col_name]
        return df
