import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from pathlib import *
import pathlib
from support_functions import get_folium_map, get_folium_map_doublecircles
from support_functions import get_folium_map_circles, load_dictionaries

path_data = str(Path().resolve().parent) + '\\Data\\'


df_service  = pd.read_excel(path_data+'real_data.xlsx', engine='openpyxl', sheet_name='Service'  , converters={'service_id'     :str,'lat':float,'long':float})[['service_id','lat','long']]
df_depot    = pd.read_excel(path_data+'real_data.xlsx', engine='openpyxl', sheet_name='Depot'    , converters={'depot_id'       :str,'lat':float,'long':float})[['depot_id','lat','long']]
df_disposal = pd.read_excel(path_data+'real_data.xlsx', engine='openpyxl', sheet_name='Disposal' , converters={'disposal_id'    :str,'lat':float,'long':float})[['disposal_id','lat','long']]
df_tranship = pd.read_excel(path_data+'real_data.xlsx', engine='openpyxl', sheet_name='Transship', converters={'transhipment_id':str,'lat':float,'long':float})[['transhipment_id','lat','long']]

df_depots_sheet   = pd.read_excel(path_data+'real_data.xlsx', engine='openpyxl', sheet_name='Depot', index_col=0)
df_vehicles_sheet = pd.read_excel(path_data+'real_data.xlsx', engine='openpyxl', sheet_name='Vehicle', index_col=0)

df_service['node_type']  = 'service'
df_depot['node_type']    = 'depot'
df_disposal['node_type'] = 'disposal'
df_tranship['node_type'] = 'transshipment'

df_nodes = pd.DataFrame(np.concatenate((df_service.values, df_depot.values, df_disposal.values, df_tranship.values), axis=0))
df_nodes.columns = ['node_id', 'lat', 'long','node_type']
df_nodes = df_nodes.astype({"node_id": str,'lat':float,'long':float,'node_type':str}, errors='raise')

def get_days_results(days_list, PIPER_results_file, AMA_df_results):

    PIPER_df_results_list = []
    AMA_df_results_list = []
    for day in days_list:
        # PIPER daily results
        PIPER_df_day_results = pd.read_excel(PIPER_results_file, engine='openpyxl', sheet_name= day, index_col=0)
        PIPER_df_day_results = PIPER_df_day_results[PIPER_df_day_results['service_quantity(kg)'] > 0]

        daily_services = PIPER_df_day_results['service_id']
        PIPER_df_results_list.append(PIPER_df_day_results)

        # AMA daily results
        AMA_df_day_results = AMA_df_results[AMA_df_results['Itinerario cod'].isin(daily_services)]
        AMA_df_results_list.append(AMA_df_day_results)

    PIPER_df_results = pd.concat(PIPER_df_results_list, axis=0)
    AMA_df_results = pd.concat(AMA_df_results_list, axis=0)

    return PIPER_df_results, AMA_df_results

def get_solutions_delta_list(df_piper, df_ama):
    df_piper = df_piper[['service_id', 'depot','facility_id']].reset_index(drop=True)
    df_ama   = df_ama[['Itinerario cod', 'CDL Gestore','Destinazione Principale']].reset_index(drop=True)

    df_diffs = pd.merge(df_piper, df_ama, left_on='service_id', right_on='Itinerario cod')
    df_diffs.drop(['Itinerario cod'], axis=1, inplace=True)
    df_diffs = df_diffs[df_diffs['depot'] != df_diffs['CDL Gestore']]

    services_delta_list = df_diffs['service_id'].to_list()

    return df_diffs, services_delta_list

def get_disdur_diffs(df_piper, df_ama, dis_dict, dur_dict, facility_dict):

    df_diffs, _ = get_solutions_delta_list(df_piper, df_ama)

    diffs = []
    for index, row in df_diffs.iterrows():
        ama_facility_id = facility_dict[row['Destinazione Principale']]

        dis_piper = dis_dict[(row['depot'],row.service_id)]       + dis_dict[(row.service_id,row.facility_id)] + dis_dict[(row.facility_id, row['depot'])]
        dur_piper = dur_dict[(row['depot'],row.service_id)]       + dur_dict[(row.service_id,row.facility_id)] + dur_dict[(row.facility_id, row['depot'])]
        dis_ama   = dis_dict[(row['CDL Gestore'],row.service_id)] + dis_dict[(row.service_id,ama_facility_id)] + dis_dict[(ama_facility_id, row['CDL Gestore'])]
        dur_ama   = dur_dict[(row['CDL Gestore'],row.service_id)] + dur_dict[(row.service_id,ama_facility_id)] + dur_dict[(ama_facility_id, row['CDL Gestore'])]
        diff_dis  = dis_ama - dis_piper
        diff_dur  = dur_ama - dur_piper

        diffs.append([row.service_id, row['depot'], row['CDL Gestore'], dis_piper, dis_ama, diff_dis, dur_piper, dur_ama, diff_dur])
        df = pd.DataFrame(diffs, columns=['service_id', 'PIPER_depot', 'AMA_depot', 'dis_piper', 'dis_ama', 'diff_dis', 'dur_piper', 'dur_ama', 'diff_dur'])

    return df

#####################################################################################################################################
PIPER_results_path = str(Path().resolve().parent) + '\\Results\\'
constr_scenario = 'ope' # vtype
PIPER_results_file = PIPER_results_path + 'df_AMA_sol_based_' + constr_scenario + '_con_Result_v6_all_days.xlsx'
PIPER_metrics_file = PIPER_results_path + 'df_AMA_sol_based_' + constr_scenario + '_con_Metrics_v6_all_days.xlsx'
#####################################################################################################################################
AMA_path_results = str(Path().resolve().parent) + '\\Data\\'
AMA_df_results = pd.read_excel(AMA_path_results + 'R12_Anagrafica_Servizi_Settimanale_PIPER.xlsx', sheet_name='Anagrafica Servizi Settimanale', skiprows=[0,1], engine='openpyxl')
#####################################################################################################################################

#####################################################################################################################################
AMA_facilities = pd.read_excel(AMA_path_results + 'R12_Anagrafica_Servizi_Settimanale_PIPER.xlsx', sheet_name='Impianti', skiprows=[0], engine='openpyxl')
AMA_facilities = AMA_facilities[['COD_IMPIANTO', 'DESC_IMPIANTO']].rename(columns={"COD_IMPIANTO": "node_id", 'DESC_IMPIANTO': 'descrizione'})
facility_dict = AMA_facilities.set_index('descrizione').to_dict()['node_id']
#####################################################################################################################################
PIPER_materials_list = ['rifiuto indifferenziato', 'rifiuto rd carta', 'rifiuto rd multimateriale', 'rifiuto rd umido']
AMA_materials_list   = ['Rifiuto indifferenziato', 'Rifiuto rd carta', 'Rifiuto rd multimateriale', 'Rifiuto rd umido']
####################################################################################################################################

# days_list = ['LUN','MAR','MER','GIO','VEN','SAB','DOM']
days_list = ['MAR']
PIPER_df_results, AMA_df_results = get_days_results(days_list, PIPER_results_file, AMA_df_results)
if len(days_list) == 1:
    #if constr_scenario == 'vtype':
    # cols2skip = [i for i in range(5)]
    # cols = [i for i in range(13) if i not in cols2skip]
    # PIPER_df_day_metrics = pd.read_excel(PIPER_metrics_file, engine='openpyxl', sheet_name = days_list[0],
    #                                      skiprows = 4, nrows= 4+15, usecols=cols, index_col=0)
    # PIPER_df_day_metrics = PIPER_df_day_metrics.rename(columns={'depot.1': 'depot', 'shift.1': 'shift', '%.1': '%' }

    daily_depot_metrics = pd.read_excel(PIPER_metrics_file, engine='openpyxl', sheet_name = days_list[0],
                                         skiprows = 4, usecols=[0,1,2,3,4], nrows= 4+15)

    daily_facility_metrics = pd.read_excel(PIPER_metrics_file, engine='openpyxl', sheet_name = days_list[0],
                                         skiprows = 24, usecols=[0,1,2,3,4,5,6,7], nrows= 24+18)


##########################################################################
# dis_dict, dur_dict = load_dictionaries()
_, services_delta_list = get_solutions_delta_list(PIPER_df_results, AMA_df_results)
# df_diffs_results = get_disdur_diffs(PIPER_df_results, AMA_df_results, dis_dict, dur_dict, facility_dict)
# df_diffs_results.to_excel(PIPER_results_path+'df_diffs_results_days_'+' '.join(days_list)+'.xlsx')
###########################################################################

############ results plot for all different depots or facilities #########################
plot_all_differeces = True
if plot_all_differeces:
    for material in PIPER_materials_list[0:1]:
        ama_material = AMA_materials_list[PIPER_materials_list.index(material)]
        df = PIPER_df_results[PIPER_df_results['material'] == material]
        df = df[(df['depot'] != df['AMA_depot']) | (df['facility_id'] != df['AMA_facility_id'])]
        file_name = 'PIPER_v4_cap04_material=' + str(material) + ' '.join(days_list) + '_ONLY_DIFFS'
        key_node = 'depot'
        df_facility = daily_facility_metrics[(daily_facility_metrics['s_count'] !=0) & (daily_facility_metrics['material'] == material) & (daily_facility_metrics['total_kg'] > 0)]
        AMA_df_results = AMA_df_results[AMA_df_results['Tipologia Rifiuto'] == ama_material]
        net_cap_coeff = 0.7
        get_folium_map_doublecircles(df, df_nodes, key_node, file_name, df_depots_sheet, daily_depot_metrics, df_facility, AMA_df_results, net_cap_coeff)
############ results plot from ALL DEPOTS ####################
plot_ALL_depots_nets = False
if plot_ALL_depots_nets:

    sol = 'PIPER_sol'  # 'PIPER_sol' or 'AMA_sol'
    only_diffs = False

    if sol == 'PIPER_sol':
        materials_list = PIPER_materials_list
        depot_node_name = 'depot'
        service_node_name = 'service_id'
        facility_name = 'facility_id'
    else:
        materials_list = AMA_materials_list
        depot_node_name = 'CDL Gestore'
        service_node_name = 'Itinerario cod'
        facility_name = 'Destinazione Principale'

    for material in materials_list[:1]:
        if sol == 'PIPER_sol':
            df_results = PIPER_df_results[PIPER_df_results['material'] == material]
            if only_diffs:
                df_results = df_results[df_results['service_id'].isin(services_delta_list)]
            facilities_list = list(df_results['facility_id'].unique())
            if only_diffs:
                file_name = 'PIPER_services_from_ALL_depots_material=' + str(material) + ' '.join(days_list) + '_ONLY_DIFFS'
            else:
                file_name = 'PIPER_services_from_ALL_depots_material=' + str(material) + ' '.join(days_list)
        else:
            df_results = AMA_df_results[AMA_df_results['Tipologia Rifiuto'] == material]
            if only_diffs:
                df_results = df_results[df_results['Itinerario cod'].isin(services_delta_list)]
            facilities_description_list = list(df_results['Destinazione Principale'].unique())
            facilities_list = [facility_dict[x] for x in facilities_description_list if type(x) == str]
            if only_diffs:
                file_name = 'AMA_services_from_ALL_depots_material=' + str(material) + ' '.join(days_list) + '_ONLY_DIFFS'
            else:
                file_name = 'AMA_services_from_ALL_depots_material=' + str(material) + ' '.join(days_list)


        get_folium_map_circles(df_results, df_nodes, depot_node_name, service_node_name, facility_name, facilities_list, facility_dict, file_name)


############ results plot from depot ####################
plot_depots_nets = False
if plot_depots_nets:
    PIPER_sol = False
    if PIPER_sol:
        for depot in list(df_results['vehicle_autocenter'].unique()):
            services = df_results[df_results['vehicle_autocenter'] == depot]['service_id'].to_list()
            df_toplot = df_nodes[df_nodes['node_id'].isin(services+[depot])]
            get_folium_map(df_toplot, 'PIPER_sol_from_depot_' + depot)
    AMA_sol = False
    if AMA_sol:
        for depot in list(df_results['autocenter_arrival'].unique()):
            services = df_results[df_results['autocenter_arrival'] == depot]['service_id'].to_list()
            df_toplot = df_nodes[df_nodes['node_id'].isin(services + [depot])]
            get_folium_map(df_toplot, 'AMA_sol_from_depot_' + depot)


############ results plot from ALL FACILITIES ####################
plot_ALL_depots_nets = False
if plot_ALL_depots_nets:
    key_node = 'facility'
    file_name = 'circle_services_from_ALL_depots'
    get_folium_map_circles(df_results, df_nodes, key_node, file_name)


############### results plot from facility #######################
plot_facility_nets = False
if plot_facility_nets:
    df_facilities = pd.read_excel(path_data+'R12_Anagrafica_Servizi_Settimanale_PIPER.xlsx', sheet_name='Impianti', skiprows=1)[['COD_IMPIANTO', 'DESC_IMPIANTO']].rename(columns={"COD_IMPIANTO": "node_id", 'DESC_IMPIANTO': 'descrizione'})
    facility_dict = df_facilities.set_index('node_id').to_dict()['descrizione']

    for facility in list(df_results['facility_id'].unique()):
        services = df_results[df_results['facility_id'] == facility]['service_id'].to_list()
        df_toplot = df_nodes[df_nodes['node_id'].isin(services + [facility])]

        df_toplot.loc[df_toplot.node_id == facility, 'node_id'] = facility_dict[facility]
        prova = pd.merge(df_toplot, df_facilities, on = 'node_id', how='inner')

        get_folium_map(df_toplot, 'sol_from_facility_' + facility)
#######################################################

# Plot Full NetWork Instance
# print('start map creation')
# m = get_folium_map(df_nodes)
# print('map created in ' + str(time.time() - start))