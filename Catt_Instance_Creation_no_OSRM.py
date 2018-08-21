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
import math

def str_to_datetime(string):
    string = string.replace('T', ' ')
    string = string.replace('Z', '')
    return datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S.%f")


def time_difference(t1, t2):
    return abs((str_to_datetime(t1) - str_to_datetime(t2)).total_seconds())


def compute_distance(gps1, gps2):
    return vincenty(gps1, gps2).meters


def compute_speed(distance, delta_time):
    return distance/delta_time


def compute_acceleration(speed1, speed2, delta_time):
    return (speed2 - speed1) / delta_time


def compute_jerk(acc1, acc2, delta_time):
    return (acc2 - acc1) / delta_time


def compute_bearing(p1, p2):
    y = math.sin(math.radians(p2[1]) - math.radians(p1[1])) * math.radians(math.cos(p2[0]))
    x = math.radians(math.cos(p1[0])) * math.radians(math.sin(p2[0])) - \
        math.radians(math.sin(p1[0])) * math.radians(math.cos(p2[0])) \
        * math.radians(math.cos(p2[1]) - math.radians(p1[1]))
    # Convert radian from -pi to pi to [0, 360] degree
    return (math.atan2(y, x) * 180. / math.pi + 360) % 360


def compute_bearing_rate(bearing1, bearing2):
    return abs(bearing1 - bearing2)

def compute_gps_features(x, min_length, min_distance, min_time, max_speed, max_acc, is_spark):
    """
    This function computes the motion features for every trip (i.e., a sequence of GPS points).
    There are four types of motion features: speed, acceleration, jerk, and bearing rate.
    :param trip: a sequence of GPS points
    :param data_type: is it related to a 'labeled' and 'unlabeled' data set.
    :return: A list with four sub-lists, where every sub-list is a motion feature.
    """
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
        # id = '8ff1c17f30d06a73480c0447fb10e440'
        waypoints_gps = pd_waypoints.loc[pd_waypoints[0] == id]
        waypoints_gps = waypoints_gps[
                        :min(100, waypoints_gps.shape[0])]  # match function does not work for a long trajectory.
        waypoints_coord = waypoints_gps[[3, 4]].values

        if len(waypoints_coord) >= 3:
            relative_distance = []
            delta_time = []
            relative_speed = []
            speed = []
            acc = []
            bearing_rate = []
            for i in range(len(waypoints_coord) - 2):
                t1 = waypoints_gps.iloc[i, 2]
                t2 = waypoints_gps.iloc[i+1, 2]
                t3 = waypoints_gps.iloc[i+2, 2]
                delta_time_1 = time_difference(t1, t2)
                distance_1 = compute_distance(waypoints_coord[i], waypoints_coord[i+1])
                speed1 = compute_speed(distance_1, delta_time_1)
                delta_time_2 = time_difference(t2, t3)
                distance_2 = compute_distance(waypoints_coord[i+1], waypoints_coord[i+2])
                speed2 = compute_speed(distance_2, delta_time_2)
                acc1 = compute_acceleration(speed1, speed2, delta_time_1)
                if any([speed1 > max_speed, speed2 > max_speed, abs(acc1) > max_acc]):
                    continue

                relative_distance.append(distance_1)
                delta_time.append(delta_time_1)
                relative_speed.append(abs((speed2 - speed1)) / speed1 if speed1 != 0 else 0)
                speed.append(speed1)
                acc.append(acc1)
                bearing_rate.append(compute_bearing_rate(compute_bearing(waypoints_coord[i], waypoints_coord[i + 1]),
                                                             compute_bearing(waypoints_coord[i + 1], waypoints_coord[i + 2])))

            # remove trips with minimum number of final legs
            if any([len(speed) < min_length, sum(relative_distance) < min_distance, sum(delta_time) < min_time]):
                continue

            trip_motion_features = [relative_distance, delta_time, speed, acc, bearing_rate, relative_speed]
            all_trip_features.append((id, trip_motion_features))

    return all_trip_features

# compute_gps_features()
# +========================================================================================================


def trajectory_feature_array_per_class(class_set, max_speed, max_acc, min_length, min_distance, min_time, is_spark):
    sc = SparkContext()
    for class_ in class_set:
        start_time = time.clock()
        waypoints_path = '../Final_Filtered_TripWaypoints/Final_Filtered_TripWaypoints_' + str(class_) + '.csv'

        # for spark
        if is_spark:
            rdd_waypoints = sc.textFile(waypoints_path)
            all_trip_features = rdd_waypoints.mapPartitions(lambda x: compute_gps_features(x, min_length, min_distance, min_time, max_speed, max_acc, is_spark)).collect()
        else:
            # for pandas
            pd_waypoints = pd.read_csv(waypoints_path, header=None)
            all_trip_features = compute_gps_features(pd_waypoints, min_length, min_distance, min_time, max_speed,
                                                     max_acc, is_spark)

        pickle_file = '../GPS_Trajectory_Feature/GPS_Trajectory_Feature_no_OSRM_' + str(class_) + '.pickle'
        with open(pickle_file, 'wb') as g:
            pickle.dump(all_trip_features, g)
        print('Computation time for waypoints of class {} is {}.'.format(class_, time.clock() - start_time))

#trajectory_feature_array_per_class(class_set=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], max_speed=54, max_acc=10, min_length=4, min_distance=6000, min_time=600, is_spark=1)
# +========================================================================================================


def check_length_road_type(class_set):
    trajec_len = {}
    all_road_type = {}
    trajec_duration = {}
    trajec_distance = {}
    for class_ in class_set:
        pickle_file = '../GPS_Trajectory_Feature/GPS_Trajectory_Feature_no_OSRM_' + str(class_) + '.pickle'
        with open(pickle_file, 'rb') as g:
            all_trip_features = pickle.load(g)
        if class_ == 5:
            all_trip_features = all_trip_features[
                                :100000]  # take out only the first 100k out of almost ~300k from class 5
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

check_length_road_type(class_set=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# +========================================================================================================
