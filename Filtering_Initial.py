from pyspark import SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import SQLContext
import os
import csv
import time
start_time = time.time()
filename = '/home/fieldtest1/CATT_Intern/TripRecords/TripRecordsFebruary1.csv'
sc = SparkContext()

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

def combine_csv(path_to_csv, path_to_save):
    list_csv = os.listdir(path_to_csv)
    all_csv = []
    for csv_ in list_csv:
        if csv_[-1] == 'v':
            path = path_to_csv + '/' + csv_
            all_csv.append(open(path, mode='r', newline=''))
    with open(path_to_save, 'w', newline='', encoding='utf-8') as g:
        writer = csv.writer(g)
        for csv_ in all_csv:
            for row in csv_:
                writer.writerow(row.rstrip(',,\r\n').split(','))


def extract_crossed_trips(crossed_trips_path, trip_records_name, month):
    with open(crossed_trips_path, newline='', mode='r') as f:
        crossed_IDs = [row.split(',')[2] for row in f if row.split(',')[6][5:7] == month]

    trip_records_path = '../TripRecords/' + trip_records_name
    df = spark.read.load(trip_records_path, format='csv')
    filtered_df = df.filter(df['_c0'].isin(crossed_IDs))
    # print('Number of rows in the filtered file: ', filtered_df.count())
    path_to_save = 'Filtered' + trip_records_name
    filtered_df.write.save(path_to_save, format='csv')
    print(time.time() - start_time, 'Seconds')

extract_crossed_trips('../nikola_trips/nikola_trips.csv', 'TripRecordsJune.csv', month='06')
combine_csv('FilteredTripRecordsJune.csv', '../Filtered_TripRecords/FilteredTripRecordsJune.csv')

extract_crossed_trips('../nikola_trips/nikola_trips.csv', 'TripRecordsWaypointsJune.csv', month='06')
combine_csv('../Codes/FilteredTripRecordsWaypointsJune.csv', '../Filtered_TripRecords/FilteredTripRecordsWaypointsJune.csv')










# Connecting to mysql table
# sqlcontext = SQLContext(sparkContext=sc)
# dataframe_mysql = sqlcontext.read.format("jdbc").option("url", "jdbc:mysql://:3306/CATT_gps").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "test_trip").option("user", "root").option("password", "Sinajigo0ol!").load()
# dataframe_mysql.show()

# Writing sql programmings on the Dataframes
# df.createOrReplaceTempView("people")
# sqlDF = spark.sql("SELECT * FROM people where _c1 == '0'").show()