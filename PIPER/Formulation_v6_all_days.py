#!/usr/bin/env python
# coding: utf-8

# # Formulazione matematica
#
# ## Questo gira sui dati reali

# ## Import packages
import numpy as np
import pandas as pd
from itertools import groupby

import gurobipy as gp
import yaml
from gurobipy import GRB

from support_functions import OD_dict_by_dict, load_dictionaries

with open("Config/config_file.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    day_of_the_week = cfg["day_of_the_week"]
    cost_euro_x_Km = cfg["cost_euro_x_Km"]
    facility_capacity_con = cfg['facility_capacity_con']
    increased_capacity = cfg['increased_capacity']
    vehicles_avail_rate = cfg["vehicles_avail_rate"]
    operator_avail_rate = cfg["operators_avail_rate"]
    operators_constraints = cfg["operators_constraints"]

# ## Solve and save for every day
file_name = '_'
if operators_constraints:
    file_name += 'ope_con'
else:
    file_name += 'vtype_con'
writerResults = pd.ExcelWriter('Results/df_' + facility_capacity_con + file_name +
                               '_Result_v6_all_days.xlsx', engine='xlsxwriter')
writerMetrics = pd.ExcelWriter('Results/df_' + facility_capacity_con + file_name +
                               '_Metrics_v6_all_days.xlsx', engine='xlsxwriter')

days = ["LUN", "MAR", "MER", "GIO", "VEN", "SAB", "DOM"]
# days = ["MAR"]

# ### Creating global objects
df_services_all = pd.read_excel(
    'Data/real_data.xlsx', sheet_name='Service', index_col=0)

df_depots = pd.read_excel('Data/real_data.xlsx',
                          sheet_name='Depot', index_col=0)
df_depots.fillna(0, inplace=True)
df_depots["M"] = df_depots["M"].apply(lambda x: int(operator_avail_rate * x))
df_depots["P"] = df_depots["P"].apply(lambda x: int(operator_avail_rate * x))
df_depots["N"] = df_depots["N"].apply(lambda x: int(operator_avail_rate * x))
df_depots["starting_t"] = pd.to_datetime(
    df_depots["starting_t"], format='%H:%M:%S').dt.time
df_depots["ending_t"] = pd.to_datetime(
    df_depots["ending_t"], format='%H:%M:%S').dt.time

df_disposals = pd.read_excel(
    'Data/real_data.xlsx', sheet_name='Disposal', index_col=0)
df_disposals.fillna(0, inplace=True)
df_disposals["starting_t"] = pd.to_datetime(
    df_disposals["starting_t"], format='%H:%M:%S').dt.time
df_disposals["ending_t"] = pd.to_datetime(
    df_disposals["ending_t"], format='%H:%M:%S').dt.time

# we consider only disposals of the following types
waste_types = ["rifiuto rd umido", "rifiuto rd carta",
               "rifiuto indifferenziato", "rifiuto rd multimateriale"]
df_disposals = df_disposals[df_disposals["material"].isin(waste_types)]

df_transships = pd.read_excel(
    'Data/real_data.xlsx', sheet_name='Transship', index_col=0)
df_transships.fillna(0, inplace=True)
df_transships["starting_t"] = pd.to_datetime(
    df_transships["starting_t"], format='%H:%M:%S').dt.time
df_transships["ending_t"] = pd.to_datetime(
    df_transships["ending_t"], format='%H:%M:%S').dt.time
df_transships = df_transships[df_transships["material"].isin(waste_types)]

df_vehicles = pd.read_excel(
    'Data/real_data.xlsx', sheet_name='Vehicle', index_col=0)
df_vehicles.fillna(0, inplace=True)

df_facilities = pd.concat([df_disposals, df_transships])
df_facilities.index.name = "facility_id"
df_facilities_desc = \
    pd.read_excel('Data/R12_Anagrafica_Servizi_Settimanale_PIPER.xlsx', sheet_name='Impianti', skiprows=1)[
        ["COD_IMPIANTO", "DESC_IMPIANTO"]].rename(
        columns={'COD_IMPIANTO': "facility_id", "DESC_IMPIANTO": "facility_desc"}).set_index("facility_id")
df_facilities = df_facilities_desc.merge(df_facilities, on='facility_id')

# # Create vehicle_sheet with the details of each vehicle.

# we consider vehicle of the following types
type_vehicle = ['COMP 2A', 'COMP 3A',
                'COMP SIDE LOA 2A', 'COMP SIDE LOA 3A']

# #### same columns of df_vehicles, with a column representing the depot. NB: I will have one row for each vehicle
df_vehicles_detailed = df_vehicles.copy()
df_vehicles_detailed = df_vehicles_detailed[0:0]
df_vehicles_detailed["depot_id"] = None
vehicles_per_depot_list = []

# retrieving vehicle details type per depot
vehicle_per_depot = df_depots[["vehicle"]]
for depot_id in vehicle_per_depot.index:
    depot_dic = eval(
        vehicle_per_depot[vehicle_per_depot.index == depot_id]["vehicle"][depot_id])
    for key in depot_dic.keys():
        template_row = df_vehicles[df_vehicles["type"] == key].copy()
        template_row.loc[:, "depot_id"] = depot_id

        new_df = pd.DataFrame(
            np.repeat(template_row.values, int(vehicles_avail_rate * depot_dic[key]), axis=0))
        new_df.columns = template_row.columns
        df_vehicles_detailed = pd.concat(
            (df_vehicles_detailed, new_df), axis=0)
        vehicles_per_depot_list.append(
            [depot_id, key, int(vehicles_avail_rate * depot_dic[key]), 3 * int(vehicles_avail_rate * depot_dic[key])])

df_vehicles_detailed.reset_index(drop=True, inplace=True)
df_vehicles_detailed.index.name = "vehicle_id"
df_vehicle_per_depot = pd.DataFrame(vehicles_per_depot_list,
                                    columns=["depot", "vehicle_type", "available_per_shift", "available_all_shift"])
df_vehicle_per_depot_dict = {key: {'dummy': 0} for key in df_depots.index}
for idx, vd in df_vehicle_per_depot.iterrows():
    df_vehicle_per_depot_dict[vd['depot']].update({vd['vehicle_type']: vd['available_per_shift']})

dis_dict, dur_dict = load_dictionaries()
df_services_result_all = pd.DataFrame(columns=["service_id", "material", "shift", "service_quantity(kg)",
                                               "vehicle_cap_weight", "overload", "facility_material", "facility_id",
                                               "facility_desc", "depot", "km", "time", "AMA_vehicle",
                                               "AMA_facility_id", "AMA_facility_desc", "AMA_depot", "AMA_km",
                                               "AMA_time", "Diff km (PIPER - AMA)", "Diff time (PIPER - AMA)", "day"])
df_metrics_all = pd.DataFrame(columns=["Objective_value", "MIPGap_sol", "GRB_status", "total_km", "total_time",
                                       "AMA_total_km", "AMA_total_time", "active_facility", "resolution_time", "n_diff",
                                       "n_diff_depot", "n_diff_facility", "n_diff%", "n_diff_depot%",
                                       "n_diff_facility%", "n_services_over_load", "services_over_load", "day"])
df_service_per_depot_tot = pd.DataFrame(columns=["depot", "n_services", "n_services_AMA"])

for day_of_the_week in days:

    print("**************")
    print('\n\nSolving ' + day_of_the_week)
    print("**************")

    df_services = df_services_all[df_services_all[day_of_the_week].notna()].copy(
    )
    df_services.drop(
        columns=[day for day in days if day != day_of_the_week], inplace=True)
    df_services.rename(columns={day_of_the_week: "shift"}, inplace=True)
    df_services["shift"].mask(df_services["shift"] == "SN", "N", inplace=True)
    df_services.fillna(0, inplace=True)
    # df_services = df_services[df_services['material']==material]
    df_services.set_index("service_id", inplace=True)
    over_load = df_services[df_services["over_load"]].index.to_list()

    # - $W$: set of all working shifts;
    # - $S$: set of all demand points (services); ($ S = \bigcup_{w \in W} S_w$)
    # - $A$: set of all depots (depots);
    # - $D$: set of all disposal points;
    # - $T$: set of all transshipment points;
    # - $V$: set of all vehicles.
    # - $F$: set of all possible facilities (equal to $D \cup T$).

    W = set(df_services["shift"].values)

    NA = len(df_depots)
    NS = len(df_services)
    ND = len(df_disposals)
    NT = len(df_transships)
    NV = len(df_vehicles_detailed)

    A = list(range(NA))
    df_depots['grb_id'] = A
    S = list(range(NA, NA + NS))
    df_services['grb_id'] = S
    D = list(range(NA + NS, NA + NS + ND))
    df_disposals['grb_id'] = D
    T = list(range(NA + NS + ND, NA + NS + ND + NT))
    df_transships['grb_id'] = T
    V = list(range(0, NV))
    df_vehicles_detailed['grb_vehicle_id'] = V
    F = D + T
    df_facilities['grb_id'] = F

    # creating two dictionaries with the association waste_type --> disposals/transhipment for that waste type
    facility_dictionary = {}
    for w_type in waste_types:
        facility_dictionary[w_type] = df_disposals[df_disposals["material"]
                                                   == w_type]["grb_id"].to_list()

    transhipment_dictionary = {}
    for w_type in waste_types:
        transhipment_dictionary[w_type] = df_transships[df_transships["material"]
                                                        == w_type]["grb_id"].to_list()

    for d_key in facility_dictionary.keys():
        facility_dictionary[d_key].extend(transhipment_dictionary[d_key])

    # creating two dictionaries with the association waste_type --> services
    material_services_dictionary = {}
    for w_type in waste_types:
        material_services_dictionary[w_type] = df_services[df_services["material"]
                                                           == w_type]["grb_id"].to_list()

    # creating one dictionary with the association shift --> services
    shift_dictionary = {}
    for w in W:
        shift_dictionary[w] = df_services[df_services["shift"]
                                          == w]["grb_id"].to_list()

    # creating one dictionary with the association service --> material
    service_material_dict = df_services[[
        "grb_id", "material"]].set_index("grb_id").to_dict()["material"]

    # creating one dictionary with the association facility --> material
    facility_material_dict = df_facilities[[
        "grb_id", "material"]].set_index("grb_id").to_dict()["material"]

    # create a list of all the nodes ID
    df = pd.DataFrame(np.concatenate(
        (df_depots.index.values, df_services.index.values,
         df_disposals.index.values, df_transships.index.values),
        axis=0))

    # create a compatibility_sv_dict, with the list of s that can be done by vehicle type v
    compatibility_sv_dict = {k: {} for k in W}
    for w in W:
        service_shift_vtype = df_services[df_services["shift"] == w][["grb_id", "AMA_vehicle"]]
        compatibility_sv_dict[w] = dict(zip(type_vehicle,
                                     map(lambda x: service_shift_vtype.loc[service_shift_vtype["AMA_vehicle"] == x][
                                         "grb_id"].to_list(), type_vehicle)))

    df.columns = ['node_id']
    nodes_list = df["node_id"].to_list()
    nodes_list_index = df.index.to_list()

    # # Parameters definition
    v_depot_dict = pd.merge(df_vehicles_detailed, df_depots.reset_index(
    ), how='left', on='depot_id')[["grb_id"]]
    v_depot_dict = v_depot_dict.to_dict()['grb_id']

    q = dict(zip(S, df_services['quantity(kg)']))  # quantity in kg
    nu = dict(zip(S, df_services['volume(m3)']))  # quantity in m^3

    E = {(ND + i): df_depots['starting_t'].values[i].hour * 60 + df_depots['starting_t'].values[i].minute for i in
         range(len(A))}
    E.update(
        {(ND + NA + i): df_facilities['starting_t'].values[i].hour * 60 + df_facilities['starting_t'].values[i].minute
         for i in range(len(F))})
    L = {(ND + i): df_depots['ending_t'].values[i].hour * 60 + df_depots['ending_t'].values[i].minute for i in
         range(len(A))}
    L.update(
        {(ND + NA + i): df_facilities['ending_t'].values[i].hour * 60 + df_facilities['ending_t'].values[i].minute for i
         in range(len(F))})

    # td = dict(zip(S, df_services['time']))
    # s = dict(zip(F, df_facilities['service_t']))
    s = dict(zip(S, df_services['time']))
    s.update(zip(F, df_facilities['service_t']))

    tau = dict(zip(V, df_vehicles_detailed['max_time']))
    # delta = dict(zip(V, df_vehicles['max_dist']))

    Inv = dict(zip(F, df_facilities['invest_cost']))
    K = dict(zip(F, df_facilities['capacity(kg)']))
    if facility_capacity_con == 'AMA_sol_based':
        # Version 2: facility capacity set to increased_capacity% more that AMA solution capacity
        df_AMA_capacity = pd.read_excel('Data/df_AMA_capacity_max_day.xlsx', sheet_name='Capacity', index_col=0)[
            'total_kg']
        df_AMA_capacity = np.round(
            df_AMA_capacity * (100 + increased_capacity) / 100, 0)
        K = pd.merge(left=df_facilities['grb_id'].reset_index(
        ), right=df_AMA_capacity.reset_index(), on='facility_id', how='left').fillna(0)
        df_facilities['capacity(kg)'] = K['total_kg'].to_list()
        K = K[['grb_id', 'total_kg']].set_index('grb_id').to_dict()['total_kg']

    # ### Graph data

    N = nodes_list_index
    n = len(N)

    d = OD_dict_by_dict(dis_dict, nodes_list, nodes_list_index)
    t = OD_dict_by_dict(dur_dict, nodes_list, nodes_list_index)

    M = 100000 ** 3

    # ### Objective Function
    #

    # approach 1:     #more indexes, easier to relax depots-vehicles association
    m = gp.Model('vrp')
    c = cost_euro_x_Km  # cost euro x Km

    x_asf = {(a, s, f): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, obj=(d[(a, s)] + d[(s, f)] + d[(f, a)]) * c,
                                 name='x[{},{},{}]'.format(a, s, f), column=None) for a in A for s in S
             for f in facility_dictionary[service_material_dict[s]]}
    y = {f: m.addVar(lb=0, ub=1, vtype=GRB.BINARY,
                     obj=Inv[f], name="y[{}]".format(f), column=None) for f in F}

    m.modelSense = GRB.MINIMIZE

    print("number of variables: ", len(x_asf) + len(y))

    # dictionaries that will be used in constraints definition and later
    depot_dict = df_depots.reset_index()[["grb_id", "depot_id"]].set_index(
        "grb_id").to_dict()["depot_id"]
    operator_dict = df_depots[["M", "P", "N"]].to_dict()
    vehicles_dict = df_vehicles_detailed[
        ["grb_vehicle_id", "type", "depot_id", "storage_cap_volume(m3)", "storage_cap_weight(kg)"]].set_index(
        "grb_vehicle_id").to_dict()
    a_vehicles_dict = {k: [j for j, _ in list(v)] for k, v in groupby(
        v_depot_dict.items(), lambda x: x[1])}
    facilities_dict = df_facilities.reset_index(
    )[["facility_id", "facility_desc", "grb_id", "material"]].set_index("grb_id").to_dict()

    # ### Constraints
    # note: F(s) is a function that links a service to the compatible facilities
    #
    # #### Flow-conservation constraints

    # constraint (1) - C1_at_most_k_operator_per_depot_per_shift or at_most_k_vehicles_per_depot_per_shift
    for a in A:
        for w in W:
            if operators_constraints:
                m.addConstr((gp.quicksum(x_asf[a, s, f] for s in shift_dictionary[w] for f in
                                         facility_dictionary[service_material_dict[s]])
                             <= operator_dict[w][depot_dict[a]]),
                            name='C1_at_most_k_operator_per_depot_per_shift_' + str(a) + "_" + str(w))
            else:
                s_CP_2A = compatibility_sv_dict[w]['COMP 2A'] + compatibility_sv_dict[w]['COMP 3A']
                m.addConstr((gp.quicksum(x_asf[a, s, f] for s in s_CP_2A
                                         for f in facility_dictionary[service_material_dict[s]])
                             <= df_vehicle_per_depot_dict[depot_dict[a]]['COMP 2A'] +
                             df_vehicle_per_depot_dict[depot_dict[a]]['COMP 3A']),
                            name='C1_at_most_k_vehicle_per_type_per_depot_per_shift_CP2A_' + str(a) + "_" + str(w))
                m.addConstr((gp.quicksum(x_asf[a, s, f] for s in compatibility_sv_dict[w]['COMP 3A']
                                         for f in facility_dictionary[service_material_dict[s]])
                             <= df_vehicle_per_depot_dict[depot_dict[a]]['COMP 3A']),
                            name='C1_at_most_k_vehicle_per_type_per_depot_per_shift_CP3A_' + str(a) + "_" + str(w))

                s_CSL_2A = compatibility_sv_dict[w]['COMP SIDE LOA 2A'] + compatibility_sv_dict[w]['COMP SIDE LOA 3A']
                m.addConstr((gp.quicksum(x_asf[a, s, f] for s in s_CSL_2A
                                         for f in facility_dictionary[service_material_dict[s]])
                             <= df_vehicle_per_depot_dict[depot_dict[a]]['COMP SIDE LOA 2A'] +
                             df_vehicle_per_depot_dict[depot_dict[a]]['COMP SIDE LOA 3A']),
                            name='C1_at_most_k_vehicle_per_type_per_depot_per_shift_CSL2A_' + str(a) + "_" + str(w))
                m.addConstr((gp.quicksum(x_asf[a, s, f] for s in compatibility_sv_dict[w]['COMP SIDE LOA 3A']
                                         for f in facility_dictionary[service_material_dict[s]])
                             <= df_vehicle_per_depot_dict[depot_dict[a]]['COMP SIDE LOA 3A']),
                            name='C1_at_most_k_vehicle_per_type_per_depot_per_shift_CSL3A_' + str(a) + "_" + str(w))


    # constraint (2) - C1_service_from_one_depot_only and to_one_compatible_facility_depot_only
    for s in S:
        m.addConstr((gp.quicksum(x_asf[a, s, f] for a in A for f
                                 in facility_dictionary[service_material_dict[s]]) == 1),
                    name='C2_service_from_one_depot_only' + str(s))

    # constraint (3) - C3_facility_capacity constraint:
    for f in F:
        m.addConstr((gp.quicksum(x_asf[a, s, f] * q[s] for a in A for s in
                                 material_services_dictionary[facility_material_dict[f]]) <= K[f] * y[f]),
                    name='C3_facility_capacity_' + str(f))

    # ## Solve

    m.printStats()
    m.update()
    # m.write('PIPER.lp')

    # compute optimal solution
    m.setParam('MIPGap', 0.05)
    m.optimize()
    print(m)

    # ### Analyze solution and collect data
    # here we create dictionaries that map the values of the variables at the optimal solution

    x_asf_opt = pd.Series(m.getAttr('X', x_asf)).reset_index()
    x_asf_opt.columns = ['a', 's', 'f', 'value']
    x_asf_opt = x_asf_opt[x_asf_opt["value"] > 0]
    x_asf_opt_dict = x_asf_opt[['a', 's', 'f']].set_index('s').to_dict()

    services_result_list = []
    service_dict = df_services.reset_index()[["grb_id", "material", "service_id", "shift", "quantity(kg)",
                                              "max_vehicle_cap(kg)", "over_load"]].set_index("grb_id").to_dict()

    for s in S:
        s_id = service_dict["service_id"][s]
        s_material = service_dict["material"][s]
        s_shift = service_dict["shift"][s]
        s_quantity = service_dict["quantity(kg)"][s]
        vehicle_capacity_weight = service_dict["max_vehicle_cap(kg)"][s]
        overload = service_dict["over_load"][s]

        depot = depot_dict[x_asf_opt_dict["a"].get(s, "service with no vehicle")]
        s_facility = facilities_dict["facility_id"][x_asf_opt_dict["f"].get(
            s, "service with no facility")]
        s_facility_desc = facilities_dict["facility_desc"][x_asf_opt_dict["f"].get(
            s, "service with no facility")]
        s_facility_material = facilities_dict["material"][x_asf_opt_dict["f"].get(
            s, "service with no facility")]

        AMA_vehicle = df_services.loc[df_services.index ==
                                      s_id]["AMA_vehicle"].values[0]
        AMA_facility_id = df_services.loc[df_services.index ==
                                          s_id]["AMA_facility_id"].values[0]
        AMA_facility_desc = df_services.loc[df_services.index ==
                                            s_id]["AMA_facility_desc"].values[0]
        AMA_depot = df_services.loc[df_services.index ==
                                    s_id]["AMA_depot"].values[0]
        AMA_km = dis_dict[AMA_depot, s_id] + dis_dict[s_id,
                                                      AMA_facility_id] + dis_dict[AMA_facility_id, AMA_depot]
        AMA_time = dur_dict[AMA_depot, s_id] + dur_dict[s_id,
                                                        AMA_facility_id] + dur_dict[AMA_facility_id, AMA_depot]

        km = dis_dict[depot, s_id] + dis_dict[s_id,
                                              s_facility] + dis_dict[s_facility, depot]
        time = dur_dict[depot, s_id] + dur_dict[s_id,
                                                s_facility] + dur_dict[s_facility, depot]

        diff_km = km - AMA_km
        diff_time = time - AMA_time

        services_result_list.append([s_id, s_material, s_shift, s_quantity, vehicle_capacity_weight, overload,
                                     s_facility_material, s_facility, s_facility_desc, depot, km, time, AMA_vehicle,
                                     AMA_facility_id, AMA_facility_desc, AMA_depot, AMA_km, AMA_time, diff_km,
                                     diff_time, day_of_the_week])

    df_services_result = pd.DataFrame(
        services_result_list, columns=df_services_result_all.columns)

    # ### Calculate metrics
    print("Calculating metrics\n")
    print("total_km: ", np.round(sum(df_services_result["km"]), 2))
    print("total_time: ", np.round(sum(df_services_result["time"]), 2))

    obj_value = 0
    if m.Status == GRB.OPTIMAL:
        obj = m.getObjective()
        obj_value = obj.getValue()
    df_metrics = pd.DataFrame(
        data={'Objective_value': [np.round(obj_value, 2)]})
    df_metrics['MIPGap_sol'] = round(m.MIPGap * 100, 2)
    df_metrics["GRB_status"] = m.Status
    df_metrics["total_km"] = np.round(sum(df_services_result["km"]), 2)
    df_metrics["total_time"] = np.round(sum(df_services_result["time"]), 2)
    df_metrics["AMA_total_km"] = np.round(sum(df_services_result["AMA_km"]), 2)
    df_metrics["AMA_total_time"] = np.round(
        sum(df_services_result["AMA_time"]), 2)
    df_metrics["active_facility"] = sum(m.getAttr('X', y))
    df_metrics["resolution_time"] = np.round(m.RunTime, 2)
    df_metrics["n_diff"] = len(
        df_services_result.loc[df_services_result["Diff km (PIPER - AMA)"] != 0])
    df_metrics["n_diff_depot"] = len(
        df_services_result.loc[df_services_result["depot"] != df_services_result["AMA_depot"]])
    df_metrics["n_diff_facility"] = len(
        df_services_result.loc[df_services_result["facility_id"] != df_services_result["AMA_facility_id"]])
    df_metrics["n_diff%"] = np.round(df_metrics["n_diff"] * 100 / NS, 2)
    df_metrics["n_diff_depot%"] = np.round(
        df_metrics["n_diff_depot"] * 100 / NS, 2)
    df_metrics["n_diff_facility%"] = np.round(
        df_metrics["n_diff_facility"] * 100 / NS, 2)
    df_metrics["n_services_over_load"] = len(over_load)
    df_metrics["services_over_load"] = str(over_load)
    df_metrics["day"] = day_of_the_week

    # ## Number of services per depot
    df_service_per_depot = df_services_result.groupby(["depot"], as_index=False).size().rename(
        columns={'size': "n_services"})
    df_service_per_depot_AMA = df_services_result.groupby(["AMA_depot"], as_index=False).size().rename(
        columns={'size': "n_services"})
    df_service_per_depot = df_service_per_depot.merge(df_service_per_depot_AMA, left_on="depot",
                                                          right_on="AMA_depot", how='outer',
                                                          suffixes=('', '_AMA')).drop(labels=['AMA_depot'], axis=1)
    df_ope_per_depot = df_services_result.groupby(["depot", "shift"], as_index=False).size().rename(
        columns={'size': "n_operators"})
    df_ope_per_depot["ope_available"] = [operator_dict[x["shift"]][x["depot"]]
                                         for i, x in df_ope_per_depot.iterrows()]
    df_ope_per_depot["%"] = np.round((df_ope_per_depot["n_operators"] / df_ope_per_depot["ope_available"]) * 100, 1)
    df_service_per_depot_tot = pd.concat([df_service_per_depot_tot, df_service_per_depot])

    # ## Vehicle used per depot per shift
    # df_vehicle_per_depot_per_shift_used: vehicle departed by each depot - shift detail
    df_vehicle_per_depot_per_shift_used = df_services_result.groupby(
        ["depot", "shift"], as_index=False).size().rename(columns={'size': "used"})
    df_vehicle_per_depot_per_shift_used = pd.merge(df_vehicle_per_depot_per_shift_used,
                                                   df_vehicle_per_depot.groupby("depot")[
                                                       ['available_per_shift']].agg(sum).reset_index()).fillna(0)
    df_vehicle_per_depot_per_shift_used["%"] = np.round((df_vehicle_per_depot_per_shift_used["used"] /
                                                         df_vehicle_per_depot_per_shift_used[
                                                             "available_per_shift"]) * 100, 1)
    df_vehicle_per_depot_per_shift_used["ope_available"] = [operator_dict[x["shift"]][x["depot"]]
                                                            for i, x in df_vehicle_per_depot_per_shift_used.iterrows()]
    df_vehicle_per_depot_per_shift_used["on_ope%"] = np.round((df_vehicle_per_depot_per_shift_used["used"] /
                                                               df_vehicle_per_depot_per_shift_used["ope_available"])
                                                              * 100, 1)
    df_vehicle_per_depot_per_shift_used = df_vehicle_per_depot_per_shift_used[
        ["depot", "shift", "used", "available_per_shift", "%", "ope_available", "on_ope%"]]

    # ## Vehicle usage for AMA solution
    print("\nChecking AMA vehicle")
    df_ama_solution = df_services_result[
        ["service_id", "material", "AMA_depot", "AMA_vehicle", "AMA_facility_desc", "AMA_facility_id",
         "service_quantity(kg)", "shift"]]

    # ## Number of service and total quantity per facility
    df_facility_results = df_services_result.groupby("facility_desc").agg(s_count=("service_id", "count"),
                                                                          mean_kg_per_s=(
                                                                              "service_quantity(kg)", "mean"),
                                                                          total_kg=(
                                                                              "service_quantity(kg)", "sum")).round(1)
    df_facility_summary = df_facilities[["facility_desc", "capacity(kg)", "material"]].join(df_facility_results,
                                                                                            on="facility_desc")[
        ["facility_desc", "material", "s_count", "mean_kg_per_s", "total_kg", "capacity(kg)"]].fillna(0)
    df_facility_summary["%"] = (df_facility_summary["total_kg"] / df_facility_summary["capacity(kg)"] * 100).round(
        2).fillna(0)

    # ### Saving results in a file

    # Write all daily results and metrics
    df_services_result.to_excel(writerResults, sheet_name=day_of_the_week)
    df_metrics.to_excel(writerMetrics, sheet_name=day_of_the_week, index=False)

    # ### Write each daily metrics in a different worksheet.
    df_ope_per_depot.to_excel(
        writerMetrics, sheet_name=day_of_the_week, startcol=0, startrow=4, index=False)
    df_vehicle_per_depot_per_shift_used.to_excel(
        writerMetrics, sheet_name=day_of_the_week, startcol=6, startrow=4, index=False)
    df_service_per_depot.to_excel(
        writerMetrics, sheet_name=day_of_the_week, startcol=18, startrow=4, index=False)
    df_facility_summary.to_excel(
        writerMetrics, sheet_name=day_of_the_week, startcol=0, startrow=24)

    # Collecting result of the day
    df_services_result_all = pd.concat(
        [df_services_result_all, df_services_result])
    df_metrics_all = pd.concat([df_metrics_all, df_metrics])

# ### Saving results and metrics of all days
df_services_result_all.to_excel(writerResults, sheet_name='All')
df_metrics_all.set_index('day', inplace=True)
df_metrics_all.to_excel(writerMetrics, sheet_name='All')

# ## Number of service and total quantity per depot
df_service_per_depot_agg = df_service_per_depot_tot.groupby("depot").agg(
    s_count=("n_services", "sum"), mean_s=("n_services", "mean"),
    s_count_AMA=("n_services_AMA", "sum"), mean_s_AMA=("n_services_AMA", "mean")).round(1)
df_service_per_depot_agg["vehicle_available"] = df_vehicle_per_depot.groupby("depot", as_index=False).sum()[
                                                    ['available_all_shift']]*7
df_service_per_depot_agg["ope_available"] = [7*(x['M']+x['P']+x['N']) for i, x in df_depots.iterrows()]
df_service_per_depot_agg.to_excel(
    writerMetrics, sheet_name='All', startcol=0, startrow=10)

# ## Number of service and total quantity per facility for all days
df_facility_results_all = df_services_result_all.groupby("facility_desc").agg(
    s_count=("service_id", "sum"), mean_kg_per_s=("service_quantity(kg)", "mean"),
    total_kg=("service_quantity(kg)", "sum")).round(1)
df_facilities["capacity(kg)"] = df_facilities["capacity(kg)"] * 7
df_facility_summary_all = df_facilities[["facility_desc", "capacity(kg)", "material"]].join(df_facility_results_all,
                                                                                            on="facility_desc")[
    ["facility_desc", "material", "s_count", "mean_kg_per_s", "total_kg", "capacity(kg)"]].fillna(0)
df_facility_summary_all["%"] = (
        df_facility_summary_all["total_kg"] / df_facility_summary_all["capacity(kg)"] * 100).round(
    2).fillna(0)
df_facility_summary_all.to_excel(
    writerMetrics, sheet_name='All', startcol=0, startrow=20)

writerResults.save()
writerMetrics.save()

print("\nProcess Ended")

# %%
