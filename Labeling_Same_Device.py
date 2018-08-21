import pickle
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from geopy.distance import vincenty
from collections import Counter
import time
import numpy as np
import csv
import gmplot
import requests
import json
import datetime
a = ['sina', 'sahar', 'sana', 'sana', 'sina']
b = []
n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
from itertools import groupby
from operator import itemgetter
data = [ 1, 4,5,6, 10, 15,16,17,18, 22, 25,26,27,28]

def detect_int(n):
    first = 0
    list_ = []
    for x, y in zip(n, n[1:]):
        if (y-x) > 1:
            last = n.index(y)
            list_.append(n[first:last])
            first = last
    list_.append(n[first:])

    return list_


'''
sc = SparkContext()
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
'''
path = '../Codes/trips_labels_'
all_months = ['February1', 'February2', 'June', 'July', 'October1', 'October2']

def simple_route(coord_origin, coord_dest, coord_intermediate=None,
                 alternatives=False, steps=False, output="full",
                 geometry='polyline', overview="false", send_as_polyline=True, annotations='true'):
    """
    Function wrapping OSRM 'viaroute' function and returning the JSON reponse
    with the route_geometry decoded (in WKT or WKB) if needed.

    Parameters
    ----------

    coord_origin : list/tuple of two floats
        (x ,y) where x is longitude and y is latitude
    coord_dest : list/tuple of two floats
        (x ,y) where x is longitude and y is latitude
    coord_intermediate : list of 2-floats list/tuple
        [(x ,y), (x, y), ...] where x is longitude and y is latitude
    alternatives : bool, optional
        Query (and resolve geometry if asked) for alternatives routes
        (default: False)
    output : str, optional
        Define the type of output (full response or only route(s)), default : "full".
    geometry : str, optional
        Format in which decode the geometry, either "polyline" (ie. not decoded),
        "geojson", "WKT" or "WKB" (default: "polyline").
    overview : str, optional
        Query for the geometry overview, either "simplified", "full" or "false"
        (Default: "simplified")
    url_config : osrm.RequestConfig, optional
        Parameters regarding the host, version and profile to use

    annotations : str, optional ...
        parameters: true, false...

    Returns
    -------
    result : dict
        The result, parsed as a dict, with the geometry decoded in the format
        defined in `geometry`.
    """
    host = 'http://router.project-osrm.org/route/v1/driving/'
    url = [host, "{},{};{},{}".format(coord_origin[1], coord_origin[0], coord_dest[1], coord_dest[0]),
           "?overview=false&steps=false&alternatives=false&geometries=polyline"]
    rep = requests.get("".join(url))
    parsed_json = json.loads(rep.content.decode('utf-8'))
    print("".join(url))

    if "code" in parsed_json:
        if "Ok" in parsed_json["code"]:
            return parsed_json['routes'][0]['legs'][0]
        else:
            raise ValueError('Error - OSRM status - simple_route')
    else:
        raise ValueError('Error - OSRM status - simple_route')


def nearest(coord):
    """
    Useless function wrapping OSRM 'nearest' function,
    returning the reponse in JSON

    Parameters
    ----------
    coord : list/tuple of two floats
        (x ,y) where x is longitude and y is latitude
    url_config : osrm.RequestConfig, optional
        Parameters regarding the host, version and profile to use

    Returns
    -------
    result : dict
        The response from the osrm instance, parsed as a dict
    """
    host = 'http://router.project-osrm.org/nearest/v1/driving/'

    url = [host, "{},{}".format(coord[1], coord[0]), "?number=1"]
    rep = requests.get("".join(url))
    parsed_json = json.loads(rep.content.decode('utf-8'))

    if "code" in parsed_json:
        if "Ok" in parsed_json["code"]:
            return parsed_json['waypoints'][0]
        else:
            raise ValueError('Error - OSRM status - Nearest')
    else:
        raise ValueError('Error - OSRM status - Nearest')

def match(points, steps=False, overview="simplified", geometry="polyline",
          timestamps=None, radius=None):
    """
    Function wrapping OSRM 'match' function, returning the reponse in JSON

    Parameters
    ----------

    points : list of tuple/list of point
        A sequence of points as (x ,y) where x is longitude and y is latitude.
    steps : bool, optional
        Default is False.
    overview : str, optional
        Query for the geometry overview, either "simplified", "full" or "false"
        (Default: "simplified")
    geometry : str, optional
        Format in which decode the geometry, either "polyline" (ie. not decoded),
        "geojson", "WKT" or "WKB" (default: "polyline").
    timestamps : bool, optional
    radius : bool, optional
    url_config : osrm.RequestConfig, optional
        Parameters regarding the host, version and profile to use

    Returns
    -------
    dict
        The response from the osrm instance, parsed as a dict
    """
    host = 'http://router.project-osrm.org/match/v1/driving/'
    url = [host, ';'.join([','.join([str(coord[1]), str(coord[0])]) for coord in points]),
           "?overview=false&steps=false&geometries=polyline"]

    #if radius:
        #url.append(";".join([str(rad) for rad in radius]))
    #if timestamps:
        #url.append(";".join([str(timestamp) for timestamp in timestamps]))
    r = requests.get("".join(url))
    parsed_json = json.loads(r.content.decode('utf-8'))

    if "code" in parsed_json:
        if "Ok" in parsed_json["code"]:
            return parsed_json
        else:
            raise ValueError('Error - OSRM status - Match')
    else:
        raise ValueError('Error - OSRM status - Match')


def str_to_datetime(string):
    string = string.replace('T', ' ')
    string = string.replace('Z', '')
    return datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S.%f")


def time_difference(t1, t2):
    return abs((str_to_datetime(t1) - str_to_datetime(t2)).total_seconds())


def trip_dir(first_gps, last_gps, station_dir):
    if station_dir == 'S':
        return first_gps[0] > last_gps[0]
    if station_dir == 'E':
        return first_gps[1] < last_gps[1]
    if station_dir == 'N':
        return first_gps[0] < last_gps[0]
    if station_dir == 'W':
        return first_gps[1] > last_gps[1]


def match_time_range(before, after, num_points, station_gps, before_based, time_range):
    before_len = min(before.shape[0], num_points)
    before_points = before.values[-before_len:]
    after_len = min(after.shape[0], num_points)
    after_points = after.values[:after_len]
    merged_points = np.vstack((before_points, after_points))
    try:
        match_points = match(merged_points[:, 3:5])
        matched_gps = {index: tracepoint['location'][::-1] for index, tracepoint in
                       enumerate(match_points['tracepoints'])
                       if tracepoint is not None}
        condition = [before_len - 1 in matched_gps.keys(), before_len in matched_gps.keys()]
        if any(condition):
            time_station_osrm = []
            distance_station_osrm = []
            time_station_real = []
            indexes = list(matched_gps.keys())
            len_matched_points = len(indexes) - 1
            for i in range(len_matched_points):
                route = simple_route(matched_gps[indexes[i]], matched_gps[indexes[i + 1]])
                time_station_osrm.append(route['duration'])
                distance_station_osrm.append(route['distance'])
                time_station_real.append(
                    time_difference(merged_points[indexes[i], 2], merged_points[indexes[i + 1], 2]))

            speed = [x / y for x, y in zip(distance_station_osrm, time_station_real)]
            ave_speed = sum(speed) / len_matched_points * 2.23694  # convert m/s to mi/hr
            ave_error = sum(abs(time_station_real[i] - time_station_osrm[i]) for i in range(len_matched_points)) / \
                        len_matched_points

            if ave_error < time_range:
                ave_error = time_range

            if before_based:
                if condition[0]:
                    time_before = str_to_datetime(merged_points[before_len - 1, 2])
                    time_to_station = simple_route(matched_gps[before_len - 1], station_gps)['duration']
                    time_match = time_before + datetime.timedelta(seconds=time_to_station)
                    time_min = time_match - datetime.timedelta(seconds=ave_error)
                    time_max = time_match + datetime.timedelta(seconds=ave_error)
                    return time_min, time_match, time_max, ave_speed
                if not condition[0]:
                    time_after = str_to_datetime(merged_points[before_len, 2])
                    time_to_station = simple_route(station_gps, matched_gps[before_len])['duration']
                    time_match = time_after - datetime.timedelta(seconds=time_to_station)
                    time_min = time_match - datetime.timedelta(seconds=ave_error)
                    time_max = time_match + datetime.timedelta(seconds=ave_error)
                    return time_min, time_match, time_max, ave_speed
            if not before_based:
                if condition[1]:
                    time_after = str_to_datetime(merged_points[before_len, 2])
                    time_to_station = simple_route(station_gps, matched_gps[before_len])['duration']
                    time_match = time_after - datetime.timedelta(seconds=time_to_station)
                    time_min = time_match - datetime.timedelta(seconds=ave_error)
                    time_max = time_match + datetime.timedelta(seconds=ave_error)
                    return time_min, time_match, time_max, ave_speed
                if not condition[1]:
                    time_before = str_to_datetime(merged_points[before_len - 1, 2])
                    time_to_station = simple_route(matched_gps[before_len - 1], station_gps)['duration']
                    time_match = time_before + datetime.timedelta(seconds=time_to_station)
                    time_min = time_match - datetime.timedelta(seconds=ave_error)
                    time_max = time_match + datetime.timedelta(seconds=ave_error)
                    return time_min, time_match, time_max, ave_speed
        else:
            return []
    except ValueError:
        return []


def find_station(waypoints_gps, all_stations_GPS, available_station, threshold_dist, id):
    match_station = 0  # Use to see if we a trip crosses one of the stations
    stations_dir = {'1': 'S', '3': 'E', '4': 'S', '5': 'N', '7': 'W', '10': 'N', '11': 'E'}
    for station in available_station:
        dir = stations_dir[station]
        if dir == 'E':
            before = waypoints_gps.loc[waypoints_gps[4] < all_stations_GPS[station][1]]  # temporally before points
            if before.shape[0] == 0:  # Check if no before point exists.
                continue

            list_before = detect_int(before[1].tolist())
            min_list = []
            for index, list_ in enumerate(list_before):
                min_list.append((index, vincenty(before.loc[before[1] == list_[-1], [3, 4]].values[0],
                                                 all_stations_GPS[station]).miles))
            min_list = min(min_list, key=lambda x: x[1])
            before_after_distance = [min_list[1]]
            before_point_index = list_before[min_list[0]][-1]

            # Check if the after point exists
            if before_point_index + 1 >= waypoints_gps.shape[0]:
                continue

            # Double-check if the after point (the immediate point after the before point) is crossing the station
            after_point_value = waypoints_gps.loc[waypoints_gps[1] == before_point_index + 1].values[0]
            if after_point_value[4] < all_stations_GPS[station][1]:
                continue

            # Check if the distance of the before and after point to the station is less than a threshold
            before_after_distance.append(vincenty(after_point_value[3:5], all_stations_GPS[station]).miles)
            if min(before_after_distance) > threshold_dist:
                continue

            before = waypoints_gps.loc[waypoints_gps[1] <= before_point_index]
            after = waypoints_gps.loc[waypoints_gps[1] > before_point_index]
            before_based = before_after_distance[0] < before_after_distance[1]

            # Additional checks for station 3. If other stations with 'E' directions added, need to use if station=='3'
            num_match_points = min(before.shape[0], 15)  # no. of bef points for matching and route check
            lat_lon_10 = before.values[-num_match_points:, 3:5]
            try:
                match_json = match(lat_lon_10, steps=False, overview="simplified", geometry="polyline",
                                   timestamps=None, radius=None)
                trip_link_names = [tracepoint['name'] for tracepoint in match_json['tracepoints'] if
                                   tracepoint is not None and tracepoint['name'] != ""]
                waypoint_names = ['Benson Avenue', 'Caton Avenue', 'South Caton Avenue', 'Wilkens Avenue',
                                  'Caton Ave Exit', 'Caton Ave Exit 50A', 'Caton Ave-I95']
                intersect_names = list(set(trip_link_names).intersection(waypoint_names))
                if not intersect_names:
                    gmap = gmplot.GoogleMapPlotter(all_stations_GPS[station][0], all_stations_GPS[station][1], 16)
                    gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
                    filename = '../html/osm_not_correct_' + station + '_html/' + id + '.html'
                    gmap.draw(filename)
                    return []
                else:
                    gmap = gmplot.GoogleMapPlotter(all_stations_GPS[station][0], all_stations_GPS[station][1], 16)
                    gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
                    filename = '../html/osm_correct_' + station + '_html/' + id + '.html'
                    gmap.draw(filename)
                    return before, after, before_based, station
            except ValueError:
                gmap = gmplot.GoogleMapPlotter(all_stations_GPS[station][0], all_stations_GPS[station][1], 16)
                gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
                filename = '../html/osm_not_correct_' + station + '_html/' + id + '.html'
                gmap.draw(filename)
                return []

        elif dir == 'S':
            before = waypoints_gps.loc[waypoints_gps[3] > all_stations_GPS[station][0]]
            if before.shape[0] == 0:  # Check if no before point exists.
                gmap = gmplot.GoogleMapPlotter(all_stations_GPS[station][0], all_stations_GPS[station][1], 16)
                gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
                filename = '../html/osm_correct_' + station + '_html/WrongBefore_' + id + '.html'
                gmap.draw(filename)
                continue

            list_before = detect_int(before[1].tolist())
            min_list = []
            for index, list_ in enumerate(list_before):
                min_list.append((index, vincenty(before.loc[before[1] == list_[-1], [3, 4]].values[0],
                                                 all_stations_GPS[station]).miles))
            min_list = min(min_list, key=lambda x: x[1])
            before_after_distance = [min_list[1]]
            before_point_index = list_before[min_list[0]][-1]

            # Check if the after point exists
            if before_point_index + 1 >= waypoints_gps.shape[0]:
                gmap = gmplot.GoogleMapPlotter(all_stations_GPS[station][0], all_stations_GPS[station][1], 16)
                gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
                filename = '../html/osm_NoAfterPoint_' + station + '_html/' + id + '.html'
                gmap.draw(filename)
                continue

            # Double-check if the after point (the immediate point after the before point) is crossing the station
            after_point_value = waypoints_gps.loc[waypoints_gps[1] == before_point_index + 1].values[0]
            if after_point_value[3] > all_stations_GPS[station][0]:
                continue

            # Check if the distance of the before and after point to the station is less than a threshold
            # Distance here do not need to be accurate, just approximation is enough.
            before_after_distance.append(vincenty(after_point_value[3:5], all_stations_GPS[station]).miles)
            if min(before_after_distance) > threshold_dist:
                continue

            before = waypoints_gps.loc[waypoints_gps[1] <= before_point_index]
            after = waypoints_gps.loc[waypoints_gps[1] > before_point_index]
            before_based = before_after_distance[0] < before_after_distance[1]

            # Additional checks for stations 1 & 4
            if station == '1':
                gmap = gmplot.GoogleMapPlotter(all_stations_GPS[station][0], all_stations_GPS[station][1], 16)
                gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
                filename = '../html/osm_correct_' + station + '_html/' + id + '.html'
                gmap.draw(filename)
                return before, after, before_based, station

            elif station == '4':
                try:
                    nearest_before = nearest(before.iloc[-1, 3:5].values)['name']
                    nearest_after = nearest(after.iloc[0, 3:5].values)['name']

                    if 'Sykesville Road' in [nearest_before, nearest_after]:
                        gmap = gmplot.GoogleMapPlotter(all_stations_GPS[station][0], all_stations_GPS[station][1], 16)
                        gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
                        filename = '../html/osm_correct_' + station + '_html/' + id + '.html'
                        gmap.draw(filename)
                        return before, after, before_based, station
                    else:
                        gmap = gmplot.GoogleMapPlotter(all_stations_GPS[station][0], all_stations_GPS[station][1], 16)
                        gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
                        filename = '../html/osm_not_correct_' + station + '_html/' + id + '.html'
                        gmap.draw(filename)
                        return []
                except ValueError:
                    gmap = gmplot.GoogleMapPlotter(all_stations_GPS[station][0], all_stations_GPS[station][1], 16)
                    gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
                    filename = '../html/osm_not_correct_' + station + '_html/' + id + '.html'
                    gmap.draw(filename)
                    return []

        elif dir == 'N':
            before = waypoints_gps.loc[waypoints_gps[3] < all_stations_GPS[key][0]]
            if before.shape[0] == 0:  # Check if no before point exists.
                continue

            list_before = detect_int(before[1].tolist())
            min_list = []
            for index, list_ in enumerate(list_before):
                min_list.append((index, vincenty(before.loc[before[1] == list_[-1], [3, 4]].values[0],
                                                 all_stations_GPS[key]).miles))
            min_list = min(min_list, key=lambda x: x[1])
            before_after_distance = [min_list[1]]
            before_point_index = list_before[min_list[0]][-1]

            # check if the closest point is the last point in the list, if not, the trip direction is incorrect
            if min_list[1] > vincenty(before.loc[before[1] == list_before[min_list[0]][0], [3, 4]].values[0],
                                      all_stations_GPS[key]).miles:
                continue  # the direction is vice versa

            # Check if the after point exists
            if before_point_index + 1 >= waypoints_gps.shape[0]:
                continue

            # Double-check if the after point (the immediate point after the before point) is crossing the station
            after_point_value = waypoints_gps.loc[waypoints_gps[1] == before_point_index + 1].values[0]
            if after_point_value[3] < all_stations_GPS[key][0]:
                continue

            # Check if the distance of the before and after point to the station is less than a threshold
            before_after_distance.append(vincenty(after_point_value[3:5], all_stations_GPS[key]).miles)
            if min(before_after_distance) > threshold_dist:
                continue

            before = waypoints_gps.loc[waypoints_gps[1] <= before_point_index]
            after = waypoints_gps.loc[waypoints_gps[1] > before_point_index]
        elif dir == 'W':
            before = waypoints_gps.loc[waypoints_gps[4] > all_stations_GPS[key][1]]
            if before.shape[0] == 0:  # Check if no before point exists.
                continue

            list_before = detect_int(before[1].tolist())
            min_list = []
            for index, list_ in enumerate(list_before):
                min_list.append((index, vincenty(before.loc[before[1] == list_[-1], [3, 4]].values[0],
                                                 all_stations_GPS[key]).miles))
            min_list = min(min_list, key=lambda x: x[1])
            before_after_distance = [min_list[1]]
            before_point_index = list_before[min_list[0]][-1]

            # check if the closest point is the last point in the list, if not, the trip direction is incorrect
            if min_list[1] > vincenty(before.loc[before[1] == list_before[min_list[0]][0], [3, 4]].values[0],
                                      all_stations_GPS[key]).miles:
                continue  # the direction is vice versa

            # Check if the after point exists
            if before_point_index + 1 >= waypoints_gps.shape[0]:
                continue

            # Double-check if the after point (the immediate point after the before point) is crossing the station
            after_point_value = waypoints_gps.loc[waypoints_gps[1] == before_point_index + 1].values[0]
            if after_point_value[4] > all_stations_GPS[key][1]:
                continue

            # Check if the distance of the before and after point to the station is less than a threshold
            before_after_distance.append(vincenty(after_point_value[3:5], all_stations_GPS[key]).miles)
            if min(before_after_distance) > threshold_dist:
                continue

            before = waypoints_gps.loc[waypoints_gps[1] <= before_point_index]
            after = waypoints_gps.loc[waypoints_gps[1] > before_point_index]

    if not match_station:
        gmap = gmplot.GoogleMapPlotter(all_stations_GPS['3'][0], all_stations_GPS['3'][1], 16)
        gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
        filename = '../html/osm_no_station_html/' + id + '.html'
        gmap.draw(filename)
        return []


def trip_labeling_w_inrix_cls_(month, station):
    filename = 'trips_labels_' + month + '_N2_D2_T90.pickle'
    with open(filename, 'rb') as f:
        original_trip_device_class = pickle.load(
            f)  # dict with keys as class, and values as array[trip_id, device_id]

    original_trip_device = np.array(
        [[trip[0], trip[3], trip[2]] for trip in original_trip_device_class if trip[1] == 'Unique Class'])

    # check the stations of problem trips
    waypoints_path = '../Filtered_TripRecords/FilteredTripRecordsWaypoints' + month + '.csv'
    all_stations_GPS = {'1': (39.356984, -75.877640), '3': (39.266531, -76.664706), '4': (39.263491, -76.984693),
                        '5': (38.577882, -76.962717), '7': (38.983449, -76.335911), '10': (39.654049, -76.658056),
                        '11': (39.022491, -76.424581)}
    available_station = ['1', '3', '4']
    pd_waypoints = pd.read_csv(waypoints_path, header=None)
    trip_path = '../Filtered_TripRecords/FilteredTripRecords' + month + '.csv'
    pd_trip = pd.read_csv(trip_path, header=None)

    trip_label = []
    num_points = 1
    time_range = 0
    for index, id in enumerate(original_trip_device[:, 0]):
        waypoints_gps = pd_waypoints.loc[pd_waypoints[0] == id]
        return_ = find_station(waypoints_gps, all_stations_GPS, available_station, threshold_dist=2, id=id)
        if not return_:
            trip_label.append(('No_Station_Match', id))
            print('No_Station_Match')
            continue
        else:
            before = return_[0]
            after = return_[1]
            before_based = return_[2]
            station = return_[3]
        return_ = match_time_range(before, after, num_points=num_points, station_gps=all_stations_GPS[station],
                         before_based=before_based, time_range=time_range)
        if not return_:
            trip_label.append(('No_Time_Range_Found', id))
            print('No_Time_Range_Found')
            continue
        else:
            time_min = return_[0]
            time_match = return_[1]
            time_max = return_[2]
            speed_ave = return_[3]

        month = '0' + str(time_match.month) if time_match.month < 10 else str(time_match.month)
        day = '0' + str(time_match.day) if time_match.day < 10 else str(time_match.day)
        vws_path = '../VWS_2015_Feb_Jun_Jul_Oct/' + station + '/' + '2015' + month + day + '.csv'
        pd_stations = pd.read_csv(vws_path, header=0)
        date_time = pd.to_datetime(pd_stations['time'].apply(lambda x: x[:-3]),
                                   format="%Y-%m-%d %H:%M:%S", errors='coerce')
        date_time = date_time[date_time.notnull()]
        inrix_class = pd_trip.loc[pd_trip[0] == id, 17].values[0]
        if inrix_class == 1:
            pd_class = pd_stations.loc[
                (date_time > time_min) & (date_time < time_max) & (pd_stations['gross'] < 14000), ['vehicle_id', 'time',
                                                                                                   'class', 'speed']]
            unique_class = pd_class['class'].unique()
            length = len(unique_class)
            if length == 0:
                trip_label.append(('No_Time_Match_VWS', id))
                print('No_Time_Match_VWS')
            elif length >= 4:
                trip_label.append(('More than 3 classes', id))
                print('More than 3 classes')
            else:
                new_date_time = pd.to_datetime(pd_class['time'].apply(lambda x: x[:-3]),
                                               format="%Y-%m-%d %H:%M:%S")
                closest_time = pd_class.loc[
                    new_date_time == min(new_date_time, key=lambda d: abs(d - time_match)), ['class',
                                                                                             'vehicle_id']].values
                if length == 1:
                    device_id = pd_trip.loc[pd_trip[0] == id, 1].values[0]
                    trip_label.append(('Unique Class', id, device_id, unique_class[0], closest_time[0, 1]))
                    print('Unique Class', unique_class[0])
                elif length in [2, 3]:
                    closest_speed = pd_class.loc[
                        pd_class['speed'] == min(pd_class['speed'], key=lambda s: abs(s - speed_ave)), ['class',
                                                                                                        'vehicle_id']].values
                    if closest_time[0, 0] == closest_speed[0, 0]:
                        device_id = pd_trip.loc[pd_trip[0] == id, 1].values[0]
                        trip_label.append(('2-3 Classes, Time=Speed', id, device_id, closest_time[0, 0],
                                           closest_time[0, 1]))
                        print('2-3 Classes, Time=Speed', closest_speed[0, 0])
                    else:
                        trip_label.append(('2-3 Classes, Time != Speed', id))
                        print('2-3 Classes, Time!=Speed')

        elif inrix_class == 2:
            pd_class = pd_stations.loc[
                (date_time > time_min) & (date_time < time_max) & (pd_stations['gross'] > 14000) &
                (pd_stations['gross'] < 26000), ['vehicle_id', 'time', 'class', 'speed']]
            unique_class = pd_class['class'].unique()
            length = len(unique_class)
            if length == 0:
                trip_label.append(('No_Time_Match_VWS', id))
                print('No_Time_Match_VWS')
            elif length >= 4:
                trip_label.append(('More than 3 classes', id))
                print('More than 3 classes')
            else:
                new_date_time = pd.to_datetime(pd_class['time'].apply(lambda x: x[:-3]),
                                               format="%Y-%m-%d %H:%M:%S")
                closest_time = pd_class.loc[
                    new_date_time == min(new_date_time, key=lambda d: abs(d - time_match)), ['class',
                                                                                             'vehicle_id']].values
                if length == 1:
                    device_id = pd_trip.loc[pd_trip[0] == id, 1].values[0]
                    trip_label.append(('Unique Class', id, device_id, unique_class[0], closest_time[0, 1]))
                    print('Unique Class', unique_class[0])
                elif length in [2, 3]:
                    closest_speed = pd_class.loc[
                        pd_class['speed'] == min(pd_class['speed'], key=lambda s: abs(s - speed_ave)), ['class',
                                                                                                        'vehicle_id']].values
                    if closest_time[0, 0] == closest_speed[0, 0]:
                        device_id = pd_trip.loc[pd_trip[0] == id, 1].values[0]
                        trip_label.append(('2-3 Classes, Time=Speed', id, device_id, closest_time[0, 0],
                                           closest_time[0, 1]))
                        print('2-3 Classes, Time=Speed', closest_speed[0, 0])
                    else:
                        trip_label.append(('2-3 Classes, Time != Speed', id))
                        print('2-3 Classes, Time!=Speed')

        elif inrix_class == 3:
            pd_class = pd_stations.loc[
                (date_time > time_min) & (date_time < time_max) & (pd_stations['gross'] > 26000), ['vehicle_id', 'time',
                                                                                                   'class', 'speed']]
            unique_class = pd_class['class'].unique()
            length = len(unique_class)
            if length == 0:
                trip_label.append(('No_Time_Match_VWS', id))
                print('No_Time_Match_VWS')
            elif length >= 4:
                trip_label.append(('More than 3 classes', id))
                print('More than 3 classes')
            else:
                new_date_time = pd.to_datetime(pd_class['time'].apply(lambda x: x[:-3]),
                                               format="%Y-%m-%d %H:%M:%S")
                closest_time = pd_class.loc[
                    new_date_time == min(new_date_time, key=lambda d: abs(d - time_match)), ['class',
                                                                                             'vehicle_id']].values
                if length == 1:
                    device_id = pd_trip.loc[pd_trip[0] == id, 1].values[0]
                    trip_label.append(('Unique Class', id, device_id, unique_class[0], closest_time[0, 1]))
                    print('Unique Class', unique_class[0])
                elif length in [2, 3]:
                    closest_speed = pd_class.loc[
                        pd_class['speed'] == min(pd_class['speed'], key=lambda s: abs(s - speed_ave)), ['class',
                                                                                                        'vehicle_id']].values
                    if closest_time[0, 0] == closest_speed[0, 0]:
                        device_id = pd_trip.loc[pd_trip[0] == id, 1].values[0]
                        trip_label.append(('2-3 Classes, Time=Speed', id, device_id, closest_time[0, 0],
                                           closest_time[0, 1]))
                        print('2-3 Classes, Time=Speed', closest_speed[0, 0])
                    else:
                        trip_label.append(('2-3 Classes, Time != Speed', id))
                        print('2-3 Classes, Time!=Speed')


trip_labeling_w_inrix_cls_(month='February1', station='3')
# End of labeling process, which is used in the Labeling_Spark_OSRM. Here is used just for bugging and checking .
# ================================================================================================================


# Check and visualize if unique device ids has the same classes from original labeled data
# Check the summary of results after labeling trips for each month.
def check_device_ids_new(month, station):
    filename = 'osrm_trips_labels_' + month + '_N1_D2_T0.pickle'
    with open(filename, 'rb') as f:
        original_trip_device_class = pickle.load(f)  # dict with keys as class, and values as array[trip_id, device_id]
    print('Number of all trips: ', len(original_trip_device_class))
    print('Number of No_Station_Match: ', sum(1 for trip in original_trip_device_class if trip[0] == 'No_Station_Match'))
    print('Number of No_Time_Range_Found: ', sum(1 for trip in original_trip_device_class if trip[0] == 'No_Time_Range_Found'))
    print('Number of No_Time_Match_VWS: ', sum(1 for trip in original_trip_device_class if trip[0] == 'No_Time_Match_VWS'))
    print('Number of More than 3 classes: ', sum(1 for trip in original_trip_device_class if trip[0] == 'More than 3 classes'))
    print('Number of Unique Class: ', sum(1 for trip in original_trip_device_class if trip[0] == 'Unique Class'))
    print('Number of 2-3 Classes, Time=Speed: ', sum(1 for trip in original_trip_device_class if trip[0] == '2-3 Classes, Time=Speed'))
    print('Number of 2-3 Classes, Time != Speed: ', sum(1 for trip in original_trip_device_class if trip[0] == '2-3 Classes, Time != Speed'))

    original_trip_device = np.array(
        [[trip[1], trip[2], trip[3]] for trip in original_trip_device_class if trip[0] == 'Unique Class'])

    df_trip = pd.DataFrame(data=original_trip_device)
    print('Number of unique trips ids==number of trips: ', len(df_trip[0].unique()))
    print('Number of unique device: ', len(df_trip[1].unique()))
    print('Unique Class Set: ', df_trip[2].unique())

    devices_unique = list(set([trip[2] for trip in original_trip_device_class if trip[0] == 'Unique Class']))
    devices_2_3_classes = list(
        set([trip[2] for trip in original_trip_device_class if trip[0] == '2-3 Classes, Time=Speed']))
    for class_ in devices_unique:
        if class_ in devices_2_3_classes:
            print(class_)

    df_group = df_trip.groupby(df_trip[1])
    class_common = list(
        filter(lambda x: len(x[1][2].unique()) != 1, df_group))  # if the list of this is zero, then you good to go
    print('Number of devices with more than one class: ', len(class_common))

    # all_class_common = reduce(lambda x, y: pd.concat([x[1], y[1]]), class_common)
    all_class_common = class_common[0][1]
    for item in class_common[1:]:
        all_class_common = pd.concat([all_class_common, item[1]], ignore_index=True)
    print(all_class_common)

    # plot wrong trips: Trips with the same device ids but different classes
    waypoints_path = '../Filtered_TripRecords/FilteredTripRecordsWaypoints' + month + '.csv'
    pd_waypoints = pd.read_csv(waypoints_path, header=None)
    for item in class_common:
        trip_ids = item[1][[0, 2]].values
        device_id = item[1][1].unique()[0]
        all_stations_GPS = {'1': (39.356984, -75.877640), '3': (39.266531, -76.664706), '4': (39.263491, -76.984693),
                            '5': (38.577882, -76.962717), '7': (38.983449, -76.335911), '10': (39.654049, -76.658056),
                            '11': (39.022491, -76.424581)}
        trip_path = '../Filtered_TripRecords/FilteredTripRecords' + month + '.csv'
        for trip_id, class_ in trip_ids:
            waypoints_gps = pd_waypoints.loc[pd_waypoints[0] == trip_id]
            gmap = gmplot.GoogleMapPlotter(all_stations_GPS['3'][0], all_stations_GPS['3'][1], 16)
            gmap.heatmap(waypoints_gps[3].tolist(), waypoints_gps[4].tolist())
            filename = '../html/osm_problematic_html/' + device_id + '_' + class_ + '_' + trip_id + '.html'
            gmap.draw(filename)

# check_device_ids_new(month='February1', station='3')
# a = 1
# End of checking if unique device ids has the same classes from originial labeled data
# ===========================================================================================================

# Counting the number of originial classified trips in each month
def count_trips():
    """
    Counting the number of originial classified trips in each month
    :return:
    """
    months_trips = {'February1': [], 'February2': [], 'June': [], 'July': [], 'October1': [], 'October2': []}

    months = months_trips.keys()
    unique_month = []
    unique_all = []
    all = []
    for month in months:
        filename = path + month + '.pickle'
        with open(filename, 'rb') as f:
            trip_label = pickle.load(f)
        b = [trip[0] for trip in trip_label if trip[1] == 'Unique Class']
        all.extend(b)
        a = list(set([trip[0] for trip in trip_label if trip[1] == 'Unique Class']))
        unique_month.append(a)
        unique_all.extend(a)
    SUM_uni_month = sum(len(trip) for trip in unique_month)
    Sum_uni_all = len(set(unique_all))
    Sum_all = len(set(all))
    a = 1

        #months_trips[month].append(len(trip_label))

        #months_trips[month].append(len(list(filter(lambda x: x[1] == 'Unique Class', trip_label))))
        #months_trips[month].append(len(list(filter(lambda x: x[1] == '2-3 Classes, Time=Speed', trip_label))))
    print(months_trips)
    sum_unique = sum(item[1][1] for item in months_trips.items())
    sum_non_unique = sum(item[1][2] for item in months_trips.items())
    SUM = sum_non_unique + sum_unique
    return SUM, sum_unique, sum_non_unique, months_trips

#SUM, sum_unique, sum_non_unique, months_trips = count_trips()

# ==================================================================================================================

