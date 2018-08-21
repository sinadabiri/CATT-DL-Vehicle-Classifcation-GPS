import pickle
import pandas as pd
from pyspark import SparkContext
from geopy.distance import vincenty
import time
import numpy as np
import gmplot
import requests
import json
import datetime
import os
import keras
import random
from collections import Counter

def match(points):
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
           "?overview=false&steps=true&geometries=polyline"]
    r = requests.get("".join(url))
    parsed_json = json.loads(r.content.decode('utf-8'))

    if "code" in parsed_json:
        if "Ok" in parsed_json["code"]:
            return parsed_json
        else:
            raise ValueError('Error - OSRM status - Match')
    else:
        raise ValueError('Error - OSRM status - Match')


def simple_route(coord_origin, coord_dest):
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

    if "code" in parsed_json:
        if "Ok" in parsed_json["code"]:
            return parsed_json['routes'][0]['legs'][0]
        else:
            raise ValueError('Error - OSRM status - simple_route')
    else:
        raise ValueError('Error - OSRM status - simple_route')


def str_to_datetime(string):
    string = string.replace('T', ' ')
    string = string.replace('Z', '')
    return datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S.%f")


def time_difference(t1, t2):
    return abs((str_to_datetime(t1) - str_to_datetime(t2)).total_seconds())


def compute_gps_features(x, max_speed, max_acc, min_length, is_spark):
    # when using Spark
    # x in map partition function of spark is a generator. So, we need to first convert it to a list
    if is_spark:
        x = list(map(lambda y: y.split(','), x))
        pd_waypoints = pd.DataFrame.from_records(x)
        pd_waypoints[[3, 4]] = pd_waypoints[[3, 4]].astype(float)
        pd_waypoints[[1]] = pd_waypoints[[1]].astype(int)
    else:
        pd_waypoints = x  # When using only pandas, no spark
    # ============================================================================================

    trip_ids = iter(pd_waypoints[0].unique())
    all_trip_features = []
    for id in trip_ids:
        #id = '8ff1c17f30d06a73480c0447fb10e440'
        waypoints_gps = pd_waypoints.loc[pd_waypoints[0] == id]
        waypoints_gps = waypoints_gps[:min(100, waypoints_gps.shape[0])]  # match function does not work for a long trajectory.
        waypoints_coord = waypoints_gps[[3, 4]].values
        try:
            map_matched_points = match(waypoints_coord)
        except ValueError:
            continue

        matched_coord = [(index, tracepoint['location'][::-1]) for index, tracepoint in
                         enumerate(map_matched_points['tracepoints']) if
                         tracepoint is not None]

        relative_distance = []
        delta_time = []
        relative_speed = []
        speed = []
        acc = []
        bearing_rate = []
        road_bearing_rate = []
        num_intersection = []
        num_maneuver = []
        road_type = []
        index_num = 0
        for route in map_matched_points['matchings']:
            legs = route['legs']
            # for considering acceleration, we need to sacrifying one leg.
            len_legs = len(legs)
            if len_legs >= 2:  # need at least two legs for computing acceleration.
                for index in range(len_legs - 1):
                    steps = legs[index]['steps']

                    t1 = waypoints_gps.iloc[matched_coord[index + index_num][0], 2]
                    t2 = waypoints_gps.iloc[matched_coord[index + index_num + 1][0], 2]
                    t3 = waypoints_gps.iloc[matched_coord[index + index_num + 2][0], 2]

                    delta_time_1 = time_difference(t1, t2)
                    distance_1 = legs[index]['distance']
                    speed_1 = distance_1 / delta_time_1
                    delta_time_2 = time_difference(t2, t3)
                    distance_2 = legs[index+1]['distance']
                    speed_2 = distance_2 / delta_time_2
                    acc_1 = (speed_2 - speed_1) / delta_time_1
                    if any([speed_1 > max_speed, speed_2 > max_speed, abs(acc_1) > max_acc]):
                        continue
                    bearing_start = steps[0]['maneuver']['bearing_after']
                    bearing_end = steps[-1]['maneuver']['bearing_before']
                    bearing_rate_1 = abs(bearing_end - bearing_start)

                    relative_distance.append(distance_1)
                    delta_time.append(delta_time_1)
                    speed.append(speed_1)
                    acc.append(acc_1)
                    bearing_rate.append(bearing_rate_1)
                    relative_speed.append(abs((speed_2 - speed_1)) / speed_1 if speed_1 != 0 else 0)

                    # road features
                    num_intersection_1 = len(steps[0]['intersections']) - 1
                    all_bearing_rates = []
                    if len(steps) > 2:
                        for i in range(1, len(steps) - 1):
                            num_intersection_1 += len(steps[i]['intersections']) - 1
                            all_bearing_rates.append(
                                abs(steps[i]['maneuver']['bearing_after'] - steps[i]['maneuver']['bearing_before']))
                        road_bearing_rate_1 = sum(all_bearing_rates) / len(all_bearing_rates)
                    else:
                        road_bearing_rate_1 = 0
                    num_maneuver_1 = len(steps) - 2
                    road_type_1 = [legs[index]['summary'].split()[-1] if legs[index]['summary'].split() else None]

                    road_bearing_rate.append(road_bearing_rate_1)
                    num_intersection.append(num_intersection_1)
                    num_maneuver.append(num_maneuver_1)
                    road_type.append(road_type_1[0])

                index_num += len_legs + 1  # each leg is based on two points+scarifying due to acc + jumping to the next point
            else:
                 index_num += len_legs + 1  # each leg is based on two points+scarifying due to acc + jumping to the next point


        if len(speed) < min_length:  # remove trips with minimum number of final legs
            continue
        trip_motion_features = [relative_distance, delta_time, speed, acc, bearing_rate, relative_speed, road_bearing_rate,
                                    num_intersection, num_maneuver, road_type]
        all_trip_features.append((id, trip_motion_features))

    return all_trip_features

#compute_gps_features()
# +========================================================================================================


def trajectory_feature_array_per_class(class_set, max_speed, max_acc, min_length, is_spark):
    sc = SparkContext()
    for class_ in class_set:
        start_time = time.clock()
        waypoints_path = '../Final_Filtered_TripWaypoints/Final_Filtered_TripWaypoints_' + str(class_) + '.csv'

        # for spark
        if is_spark:
            rdd_waypoints = sc.textFile(waypoints_path)
            all_trip_features = rdd_waypoints.mapPartitions(lambda x: compute_gps_features(x, max_speed, max_acc, min_length, is_spark)).collect()
        else:
            # for pandas
            pd_waypoints = pd.read_csv(waypoints_path, header=None)
            all_trip_features = compute_gps_features(pd_waypoints, max_speed, max_acc, min_length, is_spark)

        pickle_file = '../GPS_Trajectory_Feature/GPS_Trajectory_Feature_' + str(class_) + '.pickle'
        with open(pickle_file, 'wb') as g:
            pickle.dump(all_trip_features, g)
        print('Computation time for waypoints of class {} is {}.'.format(class_, time.clock() - start_time))

# trajectory_feature_array_per_class(class_set=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], max_speed=54, max_acc=10, min_length=4, is_spark)
# +========================================================================================================


def check_length_road_type(class_set):
    trajec_len = {}
    all_road_type = {}
    trajec_duration = {}
    trajec_distance = {}
    for class_ in class_set:
        pickle_file = '../GPS_Trajectory_Feature/GPS_Trajectory_Feature_Clean_' + str(class_) + '.pickle'
        with open(pickle_file, 'rb') as g:
            all_trip_features = pickle.load(g)
        if class_ == 5:
            all_trip_features = all_trip_features[:100000]  # take out only the first 100k out of almost ~300k from class 5

        road_types = [road for trip in all_trip_features for road in trip[1][-1]]
        all_road_type[class_] = road_types

        # print(all_road_type[class_])
        '''
        length = [len(trip[1][0]) for trip in all_trip_features]
        trajec_len[class_] = length
        describe = pd.DataFrame(trajec_len[class_]).describe(percentiles=[i * 0.05 for i in range(20)])
        print('Stat summary for trajectory length of class {}: {}'.format(class_, describe))
        print('\n')
        
        duration = [sum(trip[1][1]) for trip in all_trip_features]
        trajec_duration[class_] = duration
        describe = pd.DataFrame(trajec_duration[class_]).describe(percentiles=[i * 0.05 for i in range(20)])
        print('Stat summary for trajectory duration of class {}: {}'.format(class_, describe))
        print('\n')
        
        distance = [sum(trip[1][0]) for trip in all_trip_features]
        trajec_distance[class_] = distance
        describe = pd.DataFrame(trajec_distance[class_]).describe(percentiles=[i * 0.05 for i in range(20)])
        print('Stat summary for trajectory distance of class {}: {}'.format(class_, describe))
        print('\n')
        
        '''
        length = [len(trip[1][0]) for trip in all_trip_features]
        trajec_len[class_] = length
        describe = pd.DataFrame(trajec_len[class_]).describe(percentiles=[i * 0.05 for i in range(20)])
        print('Stat summary for trajectory length of class {}: {}'.format(class_, describe))
        print('\n')



    '''
    duration_all = [item for list in trajec_duration.values() for item in list]
    describe = pd.DataFrame(duration_all).describe(percentiles=[i * 0.05 for i in range(20)])
    print('Stat summary for trajectory duration of class {}: {}'.format('ALL', describe))
    print('\n')
    
    distance_all = [item for list in trajec_distance.values() for item in list]
    describe = pd.DataFrame(distance_all).describe(percentiles=[i * 0.05 for i in range(20)])
    print('Stat summary for trajectory distance of class {}: {}'.format('ALL', describe))
    print('\n')
    
    '''
    length_all = [item for list in trajec_len.values() for item in list]
    describe = pd.DataFrame(length_all).describe(percentiles=[i * 0.05 for i in range(20)])
    print('Stat summary for trajectory length of class {}: {}'.format('ALL', describe))
    print('\n')

    road_type_all = [item for list in all_road_type.values() for item in list]
    road_type_count = Counter(road_type_all)
    print(road_type_count.most_common(200))
    # print('unique road tyepes: ', road_type_unique)

#check_length_road_type(class_set=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# +========================================================================================================
# Removing outliers + One hot encoding for road type

def road_one_hot_vector(trip):
    # trip: (id, trip)
    high_cap_types = ['arterial', 'expressway', 'trunk', 'pike', 'turnpike', 'autostrasse', 'parkway', 'gateway', 'gtwy']
    limited_acc_types = ['freeway', 'highway', 'motorway', 'beltway', 'thruway', 'bypass', 'throughway']

    trip_id = trip[0]
    trip = trip[1]

    low_cap = []
    high_cap = []
    limited_acc = []
    for road_type in trip[-1]:
        if road_type is None:
            low_cap.append(1)
            high_cap.append(0)
            limited_acc.append(0)
        elif road_type.lower() in high_cap_types:
            low_cap.append(0)
            high_cap.append(1)
            limited_acc.append(0)
        elif road_type.lower() in limited_acc_types:
            low_cap.append(0)
            high_cap.append(0)
            limited_acc.append(1)
        else:
            low_cap.append(1)
            high_cap.append(0)
            limited_acc.append(0)
    trip.pop()
    trip.append(low_cap)
    trip.append(high_cap)
    trip.append(limited_acc)
    return trip_id, trip


def trip_remove_outliers(class_set, min_distance, min_time):
    # Remove trip with less than a min-distance, less than a min trip time.
    for class_ in class_set:
        pickle_file = '../GPS_Trajectory_Feature/GPS_Trajectory_Feature_' + str(class_) + '.pickle'
        with open(pickle_file, 'rb') as g:
            all_trip_features = pickle.load(g)

        # remove trip with min distance and duration
        all_trip_features_clean = list(filter(lambda trip: sum(trip[1][0]) >= min_distance and sum(trip[1][1]) >= min_time, all_trip_features))

        # Apply one-hot encoding to all_trip_features
        all_trip_features_clean = list(map(lambda trip: road_one_hot_vector(trip), all_trip_features_clean))

        pickle_file = '../GPS_Trajectory_Feature/GPS_Trajectory_Feature_Clean_' + str(class_) + '.pickle'
        with open(pickle_file, 'wb') as g:
            pickle.dump(all_trip_features_clean, g)

trip_remove_outliers(class_set=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], min_distance=6000, min_time=600)
# +========================================================================================================