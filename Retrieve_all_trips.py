import pickle
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
import time
from functools import reduce
import numpy as np
import os
import csv

# Extracting final unique device ids (excluding the erronous ones), which are then used for extracting trips with the same device ids.
def get_device_ids():
    """
    Extract unique and correct device ids from trip ids. The device ids are used for extracting the same-device trips.
    :return: A Dict per class, where each class has an array of all unique and correct device ids.
    """
    all_months = ['February1', 'February2', 'June', 'July', 'October1', 'October2']
    #all_months = ['February1']

    trip_device_per_class = {2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 15: []}
    class_set = set(trip_device_per_class.keys())
    for month in all_months:
        filename_label = '../Codes/osrm_trips_labels_' + month + '_N1_D2_T0.pickle'

        with open(filename_label, 'rb') as f:
            trip_label = pickle.load(f)

        original_trip_device = np.array([[trip[1], trip[2], trip[3]] for trip in trip_label if trip[0] == 'Unique Class'], dtype=object)

        pd_original_trip_device = pd.DataFrame(data=original_trip_device)

        # remove error device ids with more than one classes.
        error_device = 0
        for device_id in pd_original_trip_device[1].unique():
            all_class = pd_original_trip_device.loc[pd_original_trip_device[1] == device_id, 2].unique()
            if len(all_class) > 1:
                pd_original_trip_device = pd_original_trip_device.drop(
                    pd_original_trip_device.loc[pd_original_trip_device[1] == device_id].index)
                error_device += 1

        # double checking that erronous device ids are removed.
        # df_group = pd_original_trip_device.groupby(pd_original_trip_device[1])
        # class_common = list(filter(lambda x: len(x[1][2].unique()) != 1, df_group))  # if the list of this is zero, then you good to go

        # Summary after eliminating the erroneous trips with the same device ids
        print('Number of devices with more than one class: {}, for month {}'.format(error_device, month))
        print('Number of labeled trips after removing trips with the same device ids but different classes: {}, for month {}'.format(len(pd_original_trip_device[0].unique()), month))
        print('Number of unique device, excluding devices with more than one class: {}, for month {}'.format(len(pd_original_trip_device[1].unique()), month))
        print('Unique Class Set: {}, for month {}'.format(pd_original_trip_device[2].unique(), month))
        print('\n')
        # end of double checking and some basic information. Can be removed also

        for class_ in class_set:
            device_id = pd_original_trip_device.loc[pd_original_trip_device[2] == class_, 1].unique()
            trip_device_per_class[class_].append(device_id)

    for class_ in class_set:
        length = len(trip_device_per_class[class_])
        if length >= 2:
            trip_device_per_class[class_] = np.unique(np.hstack(tuple(trip_device_per_class[class_])))
        elif length == 1:
            if trip_device_per_class[class_][0].shape[0] != 0:
                trip_device_per_class[class_] = np.unique(trip_device_per_class[class_][0])
            else:
                trip_device_per_class.pop(class_, None)

    filename = '../Codes/osrm_trip_device_per_class_N1_D2_T0.pickle'
    with open(filename, 'wb') as g:
        pickle.dump(trip_device_per_class, g)

#get_device_ids()
a = 1
# =======================================================================================================================


def extract_trips_same_device():
    """
    This function extracts trips with the same device ids of original trips from the whole 20 m trips.
    :return: A Dict per month, where each month has a tuple per class. The tuple is the class and array of trip_device_ids
    """
    all_months = ['February1', 'February2', 'June', 'July', 'October1', 'October2']
    #all_months = ['February1']
    filename = '../Codes/osrm_trip_device_per_class.pickle'
    with open(filename, 'rb') as g:
        osrm_trip_device_per_class = pickle.load(g)

    SUM = sum(class_.shape[0] for class_ in osrm_trip_device_per_class.values())

    class_set = osrm_trip_device_per_class.keys()
    same_device_trip_per_class_per_month = {'February1': [], 'February2': [], 'June': [], 'July': [], 'October1': [], 'October2': []}
    for month in all_months:
        trip_path = '../Trip_Records/TripRecords' + month + '.csv'
        pd_trip = pd.read_csv(trip_path, header=None)
        for class_ in class_set:
            device_id = osrm_trip_device_per_class[class_]
            same_device_id = pd_trip.loc[(pd_trip[1].isin(device_id)), [0, 1]].values
            if same_device_id.shape[0] != 0:
                same_device_trip_per_class_per_month[month].append((class_, same_device_id))

    filename = '../Codes/osrm_same_device_trip_per_class_per_month.pickle'
    with open(filename, 'wb') as g:
        pickle.dump(same_device_trip_per_class_per_month, g)
    return same_device_trip_per_class_per_month


same_device_trip_per_class_per_month = extract_trips_same_device()
a = 1
# ========================================================================================================

# After extracting trips with the same device ids (i.e., all possible trips), Obtain the number of trips per class, month, etc.
def number_of_trips_per_class():
    with open('osrm_same_device_trip_per_class_per_month.pickle', 'rb') as f:
        same_device_trip_per_class_per_month = pickle.load(f)

    num_trips_per_class = {2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 15: []}

    for month, item in same_device_trip_per_class_per_month.items():
        for class_, trip_ids in item:
            num_trips_per_class[class_].append(len(trip_ids))

    total_trips = 0
    for class_, num_trips in num_trips_per_class.items():
        num = sum(num_trips)
        total_trips += num
        print('Class {}, Total number of trips: {}'.format(class_, num))
    print('Total number of trips: ', total_trips)

        # print('{} class {} number of trips {}'.format(month, item[0], len(item[1])))
        # id.append(item[1][:, 1])
    #all_ids = np.concatenate(tuple(id))
    #uniq = np.unique(all_ids)
    #a = 1
    # print('{} class {} number of trips {}'.format(month, item[0], len(item[1])))
    # sum_all_trips += len(item[1])

    # pd_sample = pd.read_csv('../Filtered_TripRecords/FilteredTripRecordsWaypointsFebruary1.csv', header=None)
    # print(pd_sample.columns)
    # a = pd_sample.groupby(by=pd_sample[0])
    # df1 = spark.read.load('../Filtered_TripRecords/FilteredTripRecordsFebruary1.csv', format='csv')
    # df2 = spark.read.load('../Filtered_TripRecords/FilteredTripRecordsFebruary2.csv', format='csv')
    # b = df2.count()
    # df = df1.union(df2)

#number_of_trips_per_class()
a = 1
# ============================================================================================================

def extract_waypoints():
    """
    extract waypoints of extracted trips from all months with the same device ids and save into csv file/folder per class
    :param trip_device_ids: a Dict with month as keys and a tuple(class, array[trip_id, device_id)) as the value
    :return:
    """
    start_time = time.clock()
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    with open('osrm_same_device_trip_per_class_per_month.pickle', 'rb') as f:
        same_device_trip_per_class_per_month = pickle.load(f)

    waypoints_per_class = {2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 15: []}

    for month in same_device_trip_per_class_per_month.keys():
        waypoints_path = '../Trip_Records/TripRecordsWaypoints' + month + '.csv'
        df = spark.read.load(waypoints_path, format='csv')

        for class_, trip_array in same_device_trip_per_class_per_month[month]:
            trip_ids = list(trip_array[:, 0])
            extracted_waypoints = df.filter(df['_c0'].isin(trip_ids))
            waypoints_per_class[class_].append(extracted_waypoints)

    for class_, waypoints_list in waypoints_per_class.items():
        #df = reduce(lambda x, y: x.union(y), waypoints_list)  # or the following loop
        df = waypoints_list[0]
        if len(waypoints_list) != 1:
            for new_df in waypoints_list[1:]:
                df = df.union(new_df)
        elif not waypoints_list:
            continue

        path_to_save = 'Final_Filtered_TripWaypoints_' + str(class_)
        df.write.save(path_to_save, format='csv')

    print('Computation Time: ', time.clock() - start_time)


# extract_waypoints()
# ================================================================================================================
# Combining the waypoints csv files per class, obtained from extracting waypoints of labeled trips


def combine_csv():
    class_set = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]
    for class_ in class_set:
        path_to_csv = 'Final_Filtered_TripWaypoints_' + str(class_)

        list_csv = filter(lambda x: x[-1] == 'v', os.listdir(path_to_csv))
        all_csv = [open(path_to_csv + '/' + csv_, mode='r', newline='') for csv_ in list_csv]

        path_to_save = '../Final_Filtered_TripWaypoints/Final_Filtered_TripWaypoints_' + str(class_) + '.csv'
        with open(path_to_save, 'w', newline='', encoding='utf-8') as g:
            writer = csv.writer(g)
            for csv_ in all_csv:
                for row in csv_:
                    writer.writerow(row.rstrip(',,\r\n').split(','))

#combine_csv()
# ==================================================================================================================

# Check the number of trips in waypoints be similar to the number of trips already found in each class
number_of_trips_per_class()
print('\n')
for class_ in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]:
    path = '../Final_Filtered_TripWaypoints/Final_Filtered_TripWaypoints_' + str(class_) + '.csv'
    pd_ = pd.read_csv(path, header=None)
    print('Waypoints data. Class {}, Total number of trips: {}'.format(class_, len(pd_[0].unique())))

