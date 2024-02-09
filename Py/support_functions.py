import random
import time
import json
from ast import literal_eval
import pandas as pd
import folium
import webbrowser
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import matplotlib.colors as matplotlibcolors
# from distinctipy import distinctipy
from folium.features import DivIcon


def OD_matrix_by_dict(spath_dict, nodes_list, nodes_list_index):
    start_time = time.time()
    OD_columns = []
    for node_from in nodes_list:
        column = []
        for node_to in nodes_list:
            if node_from != node_to:
                value = spath_dict[(node_from, node_to)]
            else:
                value = 0
            column.append(value)
        OD_columns.append(column)

    OD = pd.DataFrame(OD_columns, index=nodes_list_index,
                      columns=nodes_list_index)
    print("--- OD matrix (" + str(len(nodes_list)) + "x" + str(len(nodes_list)
                                                               ) + ") creation comp.time: %s seconds ---" % (
                  time.time() - start_time))
    return OD


def OD_dict_by_dict(spath_dict, nodes_list, nodes_list_index):
    start_time = time.time()
    OD_dict = {}
    for i in range(len(nodes_list)):
        node_from = nodes_list[i]
        for j in range(len(nodes_list)):
            node_to = nodes_list[j]
            if node_from != node_to:
                value = spath_dict[(node_from, node_to)]
            else:
                value = 0

            OD_dict[(nodes_list_index[i], nodes_list_index[j])] = value

    # OD = pd.DataFrame(OD_columns, index=nodes_list_index,
    #                   columns=nodes_list_index)
    # print("--- OD matrix (" + str(len(nodes_list)) + "x" + str(len(nodes_list)
    #                                                            ) + ") creation comp.time: %s seconds ---" % (time.time() - start_time))
    return OD_dict


def load_dictionaries():
    with open('Data/dis_dict.json', 'r') as json_file:
        dis_data = json.load(json_file)
    with open('Data/dur_dict.json', 'r') as json_file:
        dur_data = json.load(json_file)
    # 2) convert loaded keys from string back to tuple
    distance_dict = {literal_eval(k): v for k, v in dis_data.items()}
    duration_dict = {literal_eval(k): v for k, v in dur_data.items()}
    print('dis. & dur. dictionaries have been loaded')
    return distance_dict, duration_dict


def get_folium_map(nodes, file_name):
    # coordinate del centro mappa : una località al centro di Roma
    Lat_c = 41.94298561949368
    Long_c = 12.60683386876551

    tiles = None
    # creo la mappa centrata sul centro mappa
    m = folium.Map(location=[Lat_c, Long_c], tiles=tiles, zoom_start=10)

    # aggiungo un layer "piastrella" (tyle) come specificato di seguito
    tiles_url = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
    tile_layer = folium.TileLayer(
        tiles=tiles_url,
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        max_zoom=19,
        name='Positron',
        control=False,
        opacity=0.7
    )
    tile_layer.add_to(m)

    path = '/home/administrator/PycharmProjects/OptNetDesign/Data/'
    path = str(Path().resolve().parent) + '\\Data\\'
    geojson_filename = 'limits_IT_regions.geojson'
    with open(path + geojson_filename, 'r') as geojson_file:
        region_borders_layer = json.load(geojson_file)

    geojson_filename = 'limits_IT_provinces.geojson'
    with open(path + geojson_filename, 'r') as geojson_file:
        provinces_borders_layer = json.load(geojson_file)

    style = {'fillColor': '#00000000', 'linecolor': 'blue'}
    folium.GeoJson(region_borders_layer, style_function=lambda x: style).add_to(m)

    style = {'fillColor': '#00000000', 'linecolor': 'blue'}
    folium.GeoJson(provinces_borders_layer, style_function=lambda x: style).add_to(m)

    folium.LayerControl().add_to(m)

    colors = {'service': 'green', 'disposal': 'red', 'transshipment': 'orange', 'depot': 'blue'}
    icons = {'service': 'refresh', 'disposal': 'recycle', 'transshipment': 'arrows', 'depot': 'truck'}
    # icons from https://fontawesome.com/v4/icons/

    nodes_list = nodes.values.tolist()
    for node in nodes_list:
        node_name = node[0]
        lat = node[1]
        long = node[2]
        location = ([lat, long])
        node_type = node[3]
        node_color = colors[node_type]
        node_icon = icons[node_type]

        """ aggiungo i marker dei nodi"""
        folium.Marker(
            location=location,
            popup=node_name,
            icon=folium.Icon(color=node_color, icon=node_icon, prefix='fa'),
            draggable=True
        ).add_to(m)

    output_file = file_name + ".html"
    output_file_path = '/home/administrator/PycharmProjects/OptNetDesign/Plots/'
    output_file_path = str(Path().resolve().parent) + '\\Data\\'
    m.save(output_file_path + output_file)
    webbrowser.open(output_file_path + output_file, new=2)  # open in new tab

    return m._repr_html_()


def get_folium_map(nodes, file_name):
    # coordinate del centro mappa : una località al centro di Roma
    Lat_c = 41.94298561949368
    Long_c = 12.60683386876551

    tiles = None
    # creo la mappa centrata sul centro mappa
    m = folium.Map(location=[Lat_c, Long_c], tiles=tiles, zoom_start=10)

    # aggiungo un layer "piastrella" (tyle) come specificato di seguito
    tiles_url = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
    tile_layer = folium.TileLayer(
        tiles=tiles_url,
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        max_zoom=19,
        name='Positron',
        control=False,
        opacity=0.7
    )
    tile_layer.add_to(m)

    path = str(Path().resolve().parent) + '\\Data\\'
    geojson_filename = 'limits_IT_regions.geojson'
    with open(path + geojson_filename, 'r') as geojson_file:
        region_borders_layer = json.load(geojson_file)

    geojson_filename = 'limits_IT_provinces.geojson'
    with open(path + geojson_filename, 'r') as geojson_file:
        provinces_borders_layer = json.load(geojson_file)

    style = {'fillColor': '#00000000', 'linecolor': 'blue'}
    folium.GeoJson(region_borders_layer, style_function=lambda x: style).add_to(m)

    style = {'fillColor': '#00000000', 'linecolor': 'blue'}
    folium.GeoJson(provinces_borders_layer, style_function=lambda x: style).add_to(m)

    folium.LayerControl().add_to(m)

    icon_colors = {'service': 'green', 'disposal': 'red', 'transshipment': 'orange', 'depot': 'blue'}
    icons = {'service': 'refresh', 'disposal': 'recycle', 'transshipment': 'arrows', 'depot': 'truck'}
    # icons from https://fontawesome.com/v4/icons/

    nodes_list = nodes.values.tolist()
    for node in nodes_list:
        node_name = node[0]
        lat = node[1]
        long = node[2]
        location = ([lat, long])
        node_type = node[3]
        node_color = icon_colors[node_type]
        node_icon = icons[node_type]

        """ aggiungo i marker dei nodi"""
        folium.Marker(
            location=location,
            popup=node_name,
            icon=folium.Icon(color=node_color, icon=node_icon, prefix='fa'),
            draggable=True
        ).add_to(m)

    output_file = file_name + ".html"
    output_file_path = '/home/administrator/PycharmProjects/OptNetDesign/Plots/'
    m.save(output_file_path + output_file)
    webbrowser.open(output_file_path + output_file, new=2)  # open in new tab

    return m._repr_html_()


def get_folium_map_circles(df_results, df_nodes, depot_node_name, service_node_name, facility_name, facilities_list,
                           facility_dict, file_name):
    # coordinate del centro mappa : una località al centro di Roma
    Lat_c = 41.94298561949368
    Long_c = 12.60683386876551


    tiles = None
    # creo la mappa centrata sul centro mappa
    m = folium.Map(location=[Lat_c, Long_c], tiles=tiles, zoom_start=10)

    # aggiungo un layer "piastrella" (tyle) come specificato di seguito
    tiles_url = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
    tile_layer = folium.TileLayer(
        tiles=tiles_url,
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        max_zoom=19,
        name='Positron',
        control=False,
        opacity=0.7
    )
    tile_layer.add_to(m)

    path = str(Path().resolve().parent) + '\\Data\\'
    geojson_filename = 'limits_IT_regions.geojson'
    with open(path + geojson_filename, 'r') as geojson_file:
        region_borders_layer = json.load(geojson_file)

    geojson_filename = 'limits_IT_provinces.geojson'
    with open(path + geojson_filename, 'r') as geojson_file:
        provinces_borders_layer = json.load(geojson_file)

    style = {'fillColor': '#00000000', 'linecolor': 'blue'}
    folium.GeoJson(region_borders_layer, style_function=lambda x: style).add_to(m)

    style = {'fillColor': '#00000000', 'linecolor': 'blue'}
    folium.GeoJson(provinces_borders_layer, style_function=lambda x: style).add_to(m)

    folium.LayerControl().add_to(m)

    icons = {'service': 'refresh', 'disposal': 'recycle', 'transshipment': 'arrows', 'depot': 'truck'}

    # icons from https://fontawesome.com/v4/icons/
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % tuple([int(255 * i) for i in rgb])

    key_node = depot_node_name

    N = len(list(df_results[key_node].unique()))
    # set color scheme for the clusters
    x = np.arange(N)
    ys = [i + x + (i * x) ** 2 for i in range(N)]
    np.random.seed(11)
    colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
    # colors = [matplotlibcolors.rgb2hex(i) for i in colors_array]

    # colors = distinctipy.get_colors(N, rng = 533, pastel_factor=0, exclude_colors = None)  # colori pastello
    # colors = [rgb_to_hex(color) for color in colors]

    colors_dict = {'AUTO_AC': '#009900', 'AUTO_TP': '#B31111', 'AUTO_PM': '#0080FF', 'AUTO_SA': '#FF9933',
                   'AUTO_RC': '#CCCC00'}

    # N = len(list(df_results[key_node].unique()))
    # existing_colors = [(255, 255, 255), (0, 0, 0)]
    # colors = distinctipy.get_colors(N, rng = 533, pastel_factor=0, exclude_colors = None)  # colori pastello
    # # display the colours
    # # distinctipy.color_swatch(colors)
    # colors = [rgb_to_hex(color) for color in colors]

    for depot in ['AUTO_AC', 'AUTO_TP', 'AUTO_PM', 'AUTO_SA','AUTO_RC']:  # for depot in list(df_results[key_node].unique()):
        depot_services = df_results[df_results[key_node] == depot][service_node_name].to_list()
        if len(depot_services) == 0:
            continue
        depot_related_facilities_description = list(df_results[df_results[key_node] == depot][facility_name].unique())
        if depot_related_facilities_description[0].startswith('IMP'):
            depot_related_facilities = depot_related_facilities_description
        else:
            depot_related_facilities = [facility_dict[x] for x in depot_related_facilities_description if
                                        type(x) == str]

        nodes_toplot = df_nodes[df_nodes['node_id'].isin(depot_services + [depot] + depot_related_facilities)]

        depot_related_facilities = facilities_list

        nodes_list = nodes_toplot.values.tolist()
        for node in nodes_list:
            node_name = node[0]
            lat = node[1]
            long = node[2]
            location = ([lat, long])
            node_type = node[3]
            node_color = colors_dict[depot]

            if node_type == 'service':
                # aggiungo i marker dei servizi con i cerchi
                folium.CircleMarker(
                    location=location,
                    radius=7,
                    popup=node_name,
                    color=node_color,
                    fill=True,
                    fill_color=node_color,
                    fill_opacity=0.9
                ).add_to(m)
            elif node_type in ['transshipment', 'disposal']:
                folium.Marker(
                    location=location,
                    popup=node_name,
                    icon=folium.Icon(color='lightgray', icon_color='purple', icon=icons[node_type], prefix='fa'),
                    # icon=folium.Icon(color='lightgray', icon_color=node_color, icon=icons[node_type], prefix='fa'),
                    draggable=True
                ).add_to(m)
            else:  # i.e. the depot
                folium.Marker(
                    location=location,
                    popup=node_name,
                    icon=folium.Icon(color='black', icon_color=node_color, icon=icons[node_type], prefix='fa'),
                    draggable=True
                ).add_to(m)

    output_file = file_name + ".html"
    # output_file_path = '/home/administrator/PycharmProjects/OptNetDesign/Plots/'
    output_file_path = str(Path().resolve().parent) + '\\Plots\\'
    m.save(output_file_path + output_file)
    webbrowser.open(output_file_path + output_file, new=2)  # open in new tab

    return m._repr_html_()


def number_DivIcon(color, number):
    """ Create a 'numbered' icon
    """
    icon = DivIcon(
        icon_size=(150, 36),
        icon_anchor=(14, 40),
        #             html='<div style="font-size: 18pt; align:center, color : black">' + '{:02d}'.format(num+1) + '</div>',
        html="""<span class="fa-stack " style="font-size: 12pt" >>
                    <!-- The icon that will wrap the number -->
                    <span class="fa fa-circle-o fa-stack-2x" style="color : {:s}"></span>
                    <!-- a strong element with the custom content, in this case a number -->
                    <strong class="fa-stack-1x">
                         {:02d}  
                    </strong>
                </span>""".format(color, number)
    )
    return icon


def service_popup_html(df):
    service = df['service_id'].iloc[0]
    depot = df['depot'].iloc[0]
    depot_AMA = df['AMA_depot'].iloc[0]
    facility = df['facility_desc'].iloc[0]
    facility_AMA = df['AMA_facility_desc'].iloc[0]
    vehicle = df['AMA_vehicle'].iloc[0]
    # vehicle = df['vehicle_type'].iloc[0]
    vehicle_AMA = df['AMA_vehicle'].iloc[0]
    km = df['km'].iloc[0]
    km_AMA = df['AMA_km'].iloc[0]

    html = """
    <!DOCTYPE html>
    <html>
    <style>
    table, th, td {border:1px solid black;}
    </style>
    <td style="text-align: center">Differences Table</td>

    <body>
    <table style="width:100%">
      <tr>
        <th></th>
        <th style="text-align: center">PIPER</th>
        <th style="text-align: center">AMA</th>
      </tr>
      <tr>
        <th style="text-align: center"> service </th>
        <td style="text-align: center">""" + str(service) + """</td>
        <td style="text-align: center">""" + str(service) + """</td>
      </tr>
      <tr>
        <th style="text-align: center"> depot </th>
        <td style="text-align: center">""" + str(depot) + """</td>
        <td style="text-align: center">""" + str(depot_AMA) + """</td>
      </tr>
      <tr>
        <th style="text-align: center"> facility </th>
        <td style="text-align: center">""" + str(facility) + """</td>
        <td style="text-align: center">""" + str(facility_AMA) + """</td>
      </tr>
      <tr>
        <th style="text-align: center"> vehicle </th>
        <td style="text-align: center">""" + str(vehicle) + """</td>
        <td style="text-align: center">""" + str(vehicle_AMA) + """</td>
      </tr>
      <tr>
        <th style="text-align: center"> km </th>
        <td style="text-align: center">""" + str(km) + """</td>
        <td style="text-align: center">""" + str(km_AMA) + """</td>
      </tr>
    </table>
    </html>
    """
    return html

def depot_popup_html(df_vehicles, df_operators):


    depot_name = df_vehicles['depot'].iloc[0]

    if len(df_operators) == 2: #i.e. is sunday shift P is not present
        new_row = {'depot': depot_name, 'shift': 'P', 'n_operators': 0, 'ope_available': 0, '%': 0}
        df_operators = df_operators.append(new_row, ignore_index=True)

    vehicle_type = df_vehicles['vehicle_type']

    df_vehicles['PIPER_used_all_shift'] = df_vehicles['PIPER_used_all_shift'].astype(int)
    df_vehicles['AMA_used_all_shift'] = df_vehicles['AMA_used_all_shift'].astype(int)
    available = df_vehicles['available_all_shift']

    PIPER_used = df_vehicles['PIPER_used_all_shift']
    AMA_used   = df_vehicles['AMA_used_all_shift']

    PIPER_perc = df_vehicles['PIPER_%']
    AMA_perc   = df_vehicles['AMA_%']

    PIPER_total_used = sum(PIPER_used)
    AMA_total_used = sum(AMA_used)
    total_avaiable_vehicles = sum(available)
    PIPER_total_perc = round(100*PIPER_total_used/total_avaiable_vehicles,2)
    AMA_total_perc = round(100*AMA_total_used / total_avaiable_vehicles, 2)

    shift = df_operators['shift']
    operators = df_operators['n_operators']
    available_ope = df_operators['ope_available']
    operators_perc = df_operators['%']

    total_operatos = sum(operators)
    total_avaiable_ope = sum(available_ope)
    total_operators_perc = round(100*total_operatos / total_avaiable_ope,2)


    html = """
    <!DOCTYPE html>
    <html>
    <style>
    table, th, td {border:1px solid black;}
    </style>
    <th style="text-align: center"> <strong>""" + str(depot_name) + """ </strong> solution details with respect to <strong> vehicles </strong> </th>
    <p>
    <body>
    <table style="width:100%">
      <tr>
        <th style="text-align: center">Vehicle Type</th>
        <th style="text-align: center">PIPER used all shifts</th>
        <th style="text-align: center">AMA used all shifts</th>
        <th style="text-align: center">Available all shifts</th>
        <th style="text-align: center">PIPER cap. usage %</th>
        <th style="text-align: center">AMA cap. usage %</th>
      </tr>
      <tr>
        <td style="text-align: center">""" + str(vehicle_type.iloc[0]) + """</td>
        <td style="text-align: center">""" + str(PIPER_used.iloc[0]) + """</td>
        <td style="text-align: center">""" + str(AMA_used.iloc[0]) + """</td>
        <td style="text-align: center">""" + str(available.iloc[0]) + """</td>
        <td style="text-align: center">""" + str(PIPER_perc.iloc[0]) + """</td>
        <td style="text-align: center">""" + str(AMA_perc.iloc[0]) + """</td>
      </tr>
      <tr>
        <td style="text-align: center">""" + str(vehicle_type.iloc[1]) + """</td>
        <td style="text-align: center">""" + str(PIPER_used.iloc[1]) + """</td>
        <td style="text-align: center">""" + str(AMA_used.iloc[1]) + """</td>
        <td style="text-align: center">""" + str(available.iloc[1]) + """</td>
        <td style="text-align: center">""" + str(PIPER_perc.iloc[1]) + """</td>
        <td style="text-align: center">""" + str(AMA_perc.iloc[1]) + """</td>
      </tr>
      <tr>
        <td style="text-align: center">""" + str(vehicle_type.iloc[2]) + """</td>
        <td style="text-align: center">""" + str(PIPER_used.iloc[2]) + """</td>
        <td style="text-align: center">""" + str(AMA_used.iloc[2]) + """</td>
        <td style="text-align: center">""" + str(available.iloc[2]) + """</td>
        <td style="text-align: center">""" + str(PIPER_perc.iloc[2]) + """</td>
        <td style="text-align: center">""" + str(AMA_perc.iloc[2]) + """</td>
      </tr>
      <tr>
        <td style="text-align: center">""" + str(vehicle_type.iloc[3]) + """</td>
        <td style="text-align: center">""" + str(PIPER_used.iloc[3]) + """</td>
        <td style="text-align: center">""" + str(AMA_used.iloc[3]) + """</td>
        <td style="text-align: center">""" + str(available.iloc[3]) + """</td>
        <td style="text-align: center">""" + str(PIPER_perc.iloc[3]) + """</td>
        <td style="text-align: center">""" + str(AMA_perc.iloc[3]) + """</td>
      </tr>
      <tr>
        <th style="text-align: center">Total</th>
        <td style="text-align: center">""" + str(PIPER_total_used) + """</td>
        <td style="text-align: center">""" + str(AMA_total_used) + """</td>
        <td style="text-align: center">""" + str(total_avaiable_vehicles) + """</td>
        <td style="text-align: center">""" + str(PIPER_total_perc) + """</td>
        <td style="text-align: center">""" + str(AMA_total_perc) + """</td>
      </tr>
    </table>
    
    <p>
    
    <style>
    table, th, td {border:1px solid black;}
    </style>
    <th style="text-align: center"> <strong>""" + str(depot_name) + """ </strong> solution details with respect to <strong> operators </strong> engaged for <strong> all </strong> materials </th>
    <p>
    <body>
    <table style="width:100%">
      <tr>
        <th style="text-align: center">Shift</th>
        <th style="text-align: center">Engaged Operators</th>
        <th style="text-align: center">Available Operators</th>
        <th style="text-align: center"> % engaged</th>
      </tr>
      <tr>
        <td style="text-align: center">""" + str(shift.iloc[0]) + """</td>
        <td style="text-align: center">""" + str(operators.iloc[0]) + """</td>
        <td style="text-align: center">""" + str(available_ope.iloc[0]) + """</td>
        <td style="text-align: center">""" + str(operators_perc.iloc[0]) + """</td>
      </tr>
      <tr>
        <td style="text-align: center">""" + str(shift.iloc[1]) + """</td>
        <td style="text-align: center">""" + str(operators.iloc[1]) + """</td>
        <td style="text-align: center">""" + str(available_ope.iloc[1]) + """</td>
        <td style="text-align: center">""" + str(operators_perc.iloc[1]) + """</td>
      </tr>
      <tr>
        <td style="text-align: center">""" + str(shift.iloc[2]) + """</td>
        <td style="text-align: center">""" + str(operators.iloc[2]) + """</td>
        <td style="text-align: center">""" + str(available_ope.iloc[2]) + """</td>
        <td style="text-align: center">""" + str(operators_perc.iloc[2]) + """</td>
      </tr>
      <tr>
        <th style="text-align: center">Total</th>
        <td style="text-align: center">""" + str(total_operatos) + """</td>
        <td style="text-align: center">""" + str(total_avaiable_ope) + """</td>
        <td style="text-align: center">""" + str(total_operators_perc) + """</td>
      </tr>
    </table>
    
    </html>
    """
    return html


def facility_popup_html(df):

    facility_name = df['facility_desc'].iloc[0]

    html = """
    <!DOCTYPE html>
    <html>
    <style>
    table, th, td {border:1px solid black;}
    </style>
    <td style="text-align: center"> <strong>""" + str(facility_name) + """ </strong> solution details </td>

    <body>
    <table style="width:100%">
      <tr>
        <th></th>
        <th style="text-align: center">Services delivered</th>
        <th style="text-align: center">Total delivered weight</th>
        <th style="text-align: center">Used Facility Capacity [%]</th>
      </tr>
      <tr>
        <th style="text-align: center"> PIPER </th>
        <td style="text-align: center">""" + str(df['PIPER_s_count'].iloc[0]) + """</td>
        <td style="text-align: center">""" + str(df['PIPER_total_kg'].iloc[0]) + """</td>
        <td style="text-align: center">""" + str(df['PIPER_%'].iloc[0]) + """</td>
      </tr>
      <tr>
        <th style="text-align: center"> AMA </th>
        <td style="text-align: center">""" + str(df['AMA_s_count'].iloc[0]) + """</td>
        <td style="text-align: center">""" + str(df['AMA_total_kg'].iloc[0]) + """</td>
        <td style="text-align: center">""" + str(df['AMA_%'].iloc[0]) + """</td>
      </tr>
    </table>
    </html>
    """

    return html


def get_depot_results(df, df_depots, net_cap_coeff):

    vehicles_per_depot_list = []
    # retrieving vehicle details type per depot
    vehicle_per_depot = df_depots[["vehicle"]]
    for depot_id in vehicle_per_depot.index:
        depot_dic = eval(vehicle_per_depot[vehicle_per_depot.index == depot_id]["vehicle"][depot_id])
        for key in depot_dic.keys():
            vehicles_per_depot_list.append([depot_id, key, int(net_cap_coeff * depot_dic[key]), 3*int(net_cap_coeff * depot_dic[key])])

    df_vehicle_per_depot = pd.DataFrame(vehicles_per_depot_list, columns=["depot", "vehicle_type", "available_per_shift", "available_all_shift"])

    # Vehicle used per depot per shift
    # df_vehicle_per_depot_used: vehicle departed by each depot - vehicle_type detail
    df_vehicle_per_depot_used_PIPER = df.groupby(["depot", "AMA_vehicle"],as_index=False).size().rename(columns={'size': "PIPER_used_all_shift", 'AMA_vehicle': 'vehicle_type'})
    df_vehicle_per_depot_used_PIPER = df_vehicle_per_depot.merge(df_vehicle_per_depot_used_PIPER, on=["depot", "vehicle_type"], how='outer').fillna(0)
    df_vehicle_per_depot_used_PIPER["PIPER_%"] = np.round((df_vehicle_per_depot_used_PIPER["PIPER_used_all_shift"] / df_vehicle_per_depot_used_PIPER["available_all_shift"]) * 100, 1)

    df_vehicle_per_depot_used_AMA = df.groupby(["AMA_depot", "AMA_vehicle"],as_index=False).size().rename(columns={'size': "AMA_used_all_shift", 'AMA_depot': 'depot', 'AMA_vehicle': 'vehicle_type'})
    df_vehicle_per_depot_used_AMA = df_vehicle_per_depot.merge(df_vehicle_per_depot_used_AMA, on=["depot", "vehicle_type"], how='outer').fillna(0)
    df_vehicle_per_depot_used_AMA["AMA_%"] = np.round((100* df_vehicle_per_depot_used_AMA["AMA_used_all_shift"] / df_vehicle_per_depot_used_AMA["available_all_shift"]) , 2)

    df_vehicle_per_depot_used = df_vehicle_per_depot_used_PIPER.merge(df_vehicle_per_depot_used_AMA, on=["depot", "vehicle_type", 'available_per_shift', 'available_all_shift'])
    # df_vehicle_per_depot_per_shift_used: vehicle departed by each depot - shift detail
    # df_vehicle_per_depot_per_shift_used = df.groupby(["depot", "shift"], as_index=False).size().rename(columns={'size': "used"})
    # df_vehicle_per_depot_per_shift_used = pd.merge(df_vehicle_per_depot_per_shift_used, df_vehicle_per_depot.groupby("depot")[['available_per_shift']].agg(sum).reset_index()).fillna(0)
    # df_vehicle_per_depot_per_shift_used["%"] = np.round((df_vehicle_per_depot_per_shift_used["used"] / df_vehicle_per_depot_per_shift_used["available_per_shift"]) * 100, 1)
    # df_vehicle_per_depot_per_shift_used = df_vehicle_per_depot_per_shift_used[["depot", "shift", "used", "available_per_shift", "%"]]

    return df_vehicle_per_depot_used

def get_facility_results(df_facility, AMA_df_results):

    df_service_per_facility_AMA = AMA_df_results.groupby(["Destinazione Principale"], as_index=False).size().rename(columns={'size': "AMA_s_count", 'Destinazione Principale': 'facility_desc'})
    df_partial = df_facility.merge(df_service_per_facility_AMA, on=['facility_desc'])
    df_AMA_total_kg = AMA_df_results.groupby(["Destinazione Principale"], as_index=False)['PESO PRESUNTO\n(KG)'].sum()
    df_AMA_total_kg = df_AMA_total_kg.rename(columns={'PESO PRESUNTO\n(KG)': "AMA_total_kg", 'Destinazione Principale': 'facility_desc'})
    df = df_partial.merge(df_AMA_total_kg, on=['facility_desc'])
    df = df.rename(columns = {'s_count': 'PIPER_s_count', 'total_kg': 'PIPER_total_kg', '%' : 'PIPER_%'})
    df['AMA_%'] = np.round((100* df["AMA_total_kg"] / df['capacity(kg)']), 2)

    return df


def get_folium_map_doublecircles(df, df_nodes, key_node, file_name, df_depots, daily_depot_metrics, df_facility, AMA_df_results, net_cap_coeff):
    # coordinate del centro mappa : una località al centro di Roma
    Lat_c = 41.94298561949368
    Long_c = 12.60683386876551

    tiles = None
    # creo la mappa centrata sul centro mappa
    m = folium.Map(location=[Lat_c, Long_c], tiles=tiles, zoom_start=10)

    # aggiungo un layer "piastrella" (tyle) come specificato di seguito
    tiles_url = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
    tile_layer = folium.TileLayer(
        tiles=tiles_url,
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        max_zoom=19,
        name='Positron',
        control=False,
        opacity=0.7
    )
    tile_layer.add_to(m)

    path = '/home/administrator/PycharmProjects/OptNetDesign/Data/'
    path = str(Path().resolve().parent) + '\\Data\\'
    geojson_filename = 'limits_IT_regions.geojson'
    with open(path + geojson_filename, 'r') as geojson_file:
        region_borders_layer = json.load(geojson_file)

    geojson_filename = 'limits_IT_provinces.geojson'
    with open(path + geojson_filename, 'r') as geojson_file:
        provinces_borders_layer = json.load(geojson_file)

    style = {'fillColor': '#00000000', 'linecolor': 'blue'}
    folium.GeoJson(region_borders_layer, style_function=lambda x: style).add_to(m)

    style = {'fillColor': '#00000000', 'linecolor': 'blue'}
    folium.GeoJson(provinces_borders_layer, style_function=lambda x: style).add_to(m)

    folium.LayerControl().add_to(m)

    icons = {'service': 'refresh', 'disposal': 'recycle', 'transshipment': 'arrows', 'depot': 'truck'}
    # icons from https://fontawesome.com/v4/icons/

    colors_dict = {'AUTO_AC': '#009900', 'AUTO_TP': '#B31111', 'AUTO_PM': '#0080FF', 'AUTO_SA': '#FF9933',
                   'AUTO_RC': '#CCCC00'}

    df_vehicle_per_depot_used = get_depot_results(df, df_depots, net_cap_coeff)
    df_facility_used          = get_facility_results(df_facility, AMA_df_results)

    for depot in ['AUTO_AC', 'AUTO_TP', 'AUTO_PM', 'AUTO_SA', 'AUTO_RC']:

        df_depot = df[df['depot'] == depot]
        depot_services = df_depot['service_id'].to_list()
        if len(depot_services) == 0:
            depot_related_facilities = []
            nodes_toplot = df_nodes[df_nodes['node_id'].isin([depot] + depot_related_facilities)]
            depot_operators = daily_depot_metrics[daily_depot_metrics['depot'] == depot]
            depot_vehicles = df_vehicle_per_depot_used[df_vehicle_per_depot_used['depot'] == depot]
        else:
            depot_related_facilities = list(df[df[key_node] == depot]['facility_id'].unique())
            nodes_toplot = df_nodes[df_nodes['node_id'].isin(depot_services + [depot] + depot_related_facilities)]
            depot_operators = daily_depot_metrics[daily_depot_metrics['depot'] == depot]
            depot_vehicles  = df_vehicle_per_depot_used[df_vehicle_per_depot_used['depot'] == depot]

        nodes_list = nodes_toplot.values.tolist()
        for node in nodes_list:
            node_name = node[0]
            lat = node[1]
            long = node[2]
            location = ([lat, long])
            node_type = node[3]
            node_color = colors_dict[depot]

            if node_type == 'service':
                # aggiungo i marker circoali dei servizi PIPER
                facility = df[df['service_id'] == node_name]['facility_id'].iloc[0]
                html = service_popup_html(df[df['service_id'] == node_name])
                iframe = folium.IFrame(html, width=400, height=240)
                popup = folium.Popup(iframe, max_width=400)
                folium.CircleMarker(
                    location=location,
                    radius=7,
                    popup=popup,
                    color=node_color,
                    fill=True,
                    fill_color=node_color,
                    fill_opacity=0.9
                ).add_to(m)
                # aggiungo i marker circoali dei servizi AMA
                ama_depot = df[df['service_id'] == node_name]['AMA_depot'].iloc[0]
                ama_node_color = colors_dict[ama_depot]
                folium.CircleMarker(
                    location=location,
                    radius=12,
                    popup=node_name,
                    color=ama_node_color,
                    fill=False,
                ).add_to(m)
            elif node_type in ['transshipment', 'disposal']:
                html = facility_popup_html(df_facility_used[df_facility_used['facility_id'] == node_name])
                iframe = folium.IFrame(html, width=400, height=170)
                popup = folium.Popup(iframe, max_width=400)
                folium.Marker(
                    location=location,
                    popup=popup,
                    icon=folium.Icon(color='lightgray', icon_color='purple', icon=icons[node_type], prefix='fa'),
                    draggable=True
                ).add_to(m)
            else:  # i.e. the depot
                html = depot_popup_html(depot_vehicles, depot_operators)
                iframe = folium.IFrame(html, width=600, height=435)
                popup = folium.Popup(iframe, max_width=600)
                folium.Marker(
                    location=location,
                    popup=popup,
                    icon=folium.Icon(color='black', icon_color=node_color, icon=icons[node_type], prefix='fa'),
                    draggable=True
                ).add_to(m)

    output_file = file_name + ".html"
    # output_file_path = '/home/administrator/PycharmProjects/OptNetDesign/Plots/'
    output_file_path = str(Path().resolve().parent) + '\\Plots\\'
    m.save(output_file_path + output_file)
    webbrowser.open(output_file_path + output_file, new=2)  # open in new tab

    return m._repr_html_()




