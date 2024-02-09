import pandas as pd
import numpy as np

filename = "Results/df_AMA_sol_based_0_Metrics_v2_all_days.xlsx"
df = pd.read_excel(filename)


days = ["LUN", "MAR", "MER", "GIO", "VEN", "SAB", "DOM"]

df_agg = pd.DataFrame(columns=[
                      'obj_val', 'total_km', 'total_time', 'active_facilities', 'AMA_km', 'AMA_time'])

df_agg_vehicles = pd.DataFrame(columns=[
    "number_of_service",	"LUN",	"MAR",	"MER",	"GIO",	"VEN",	"SAB",	"DOM"])
df_agg_vehicles["number_of_service"] = [1, 2, 3]

for day in days:
    df = pd.read_excel(filename, sheet_name=day)

    obj_val, total_km, total_time, active_facilities = df.iloc[0,
                                                               0], df.iloc[0, 3], df.iloc[0, 4], df.iloc[0, 5]

    AMA_km, AMA_time = df.iloc[2, 3], df.iloc[2, 4]

    df_agg.loc[day] = {"obj_val": obj_val,  "total_km": total_km, "total_time": total_time,
                       "active_facilities": active_facilities, "AMA_km": AMA_km, "AMA_time": AMA_time}

    df_agg_vehicles[day] = df.iloc[25:28, 1].to_list()


df_agg_vehicles.fillna(0, inplace=True)


writerResults = pd.ExcelWriter('Results/df_Metrics.xlsx', engine='xlsxwriter')
df_agg.to_excel(writerResults, sheet_name="All_Metrics",
                startcol=0, startrow=0)
df_agg_vehicles.to_excel(
    writerResults, sheet_name="All_Metrics", startcol=0, startrow=10)
writerResults.save()
