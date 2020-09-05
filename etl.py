import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, dayofweek, date_format
from pyspark.sql.types import TimestampType

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Creates a spark session if its not already initialized.

    Return:
        (obj) spark - Spark Session Object to interact with spark processing engine
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Fetch song data from S3 input bucket, process it to form songs and artists dimension,
    then saves the dimensions to S3 output bucket for future use.

    Args:
        (obj) spark - Spark Session Object to interact with spark processing engine.
        (str) input_data - input s3 path to read songs data 
        (str) output_data - output s3 path to write songs and artist dimension

    Return:
        None
    """
    # get filepath to song data file
    song_data = input_data + '/song_data/*/*/*/*.json'

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(['song_id', 'title', 'artist_id', 'year', 'duration']).dropDuplicates()

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode("overwrite").partitionBy("year").parquet(output_data + "song.parquet")

    # extract columns to create artists table
    artists_table = df.selectExpr(['artist_id',
                                   'artist_name as name',
                                   'artist_location as location',
                                   'artist_latitude as latitude',
                                   'artist_longitude as longitude'
                                   ]).dropDuplicates()

    # write artists table to parquet files
    artists_table.write.mode("overwrite").parquet(output_data + "artist.parquet")


def process_log_data(spark, input_data, output_data):
    """
    Fetch event logs data from S3 input bucket, process it to form time, users dimension,
    and songsplay fact then saves the fact and dimensions to S3 output bucket for future use.

    Args:
        (obj) spark - Spark Session Object to interact with spark processing engine.
        (str) input_data - input s3 path to read event logs data 
        (str) output_data - output s3 path to write users, time dimension and songsplay fact

    Return:
        None
    """

    # get filepath to log data file
    log_data = input_data + '/log_data/*/*/*.json'

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table    
    users_table = df.selectExpr(['userId as user_id',
                                 'firstName as first_name',
                                 'lastName as last_name',
                                 'gender',
                                 'level',
                                 'ts']) \
        .sort(['ts'], ascending=False) \
        .dropDuplicates(subset=['user_id']) \
        .drop('ts')

    # write users table to parquet files
    users_table.write.mode("overwrite").parquet(output_data + "users.parquet")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda ms: datetime.fromtimestamp(ms / 1000.0), TimestampType())
    df = df.withColumn('timestamp', get_timestamp('ts'))

    # create datetime column from original timestamp column
    get_datetime = udf()
    # df = ''

    time_data = df.withColumn('hour', hour('timestamp')) \
        .withColumn('day', dayofmonth('timestamp')) \
        .withColumn('week', weekofyear('timestamp')) \
        .withColumn('month', month('timestamp')) \
        .withColumn('year', year('timestamp')) \
        .withColumn('weekday', dayofweek('timestamp'))

    # extract columns to create time table
    time_table = time_data.selectExpr(['timestamp as start_time',
                                       'hour',
                                       'day',
                                       'week',
                                       'month',
                                       'year',
                                       'weekday'
                                       ])

    # write time table to parquet files partitioned by year and month
    time_table.write.mode("overwrite").partitionBy("year", "month").parquet(output_data + "time.parquet")

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data + 'song.parquet')

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = time_data.join(song_df.withColumnRenamed('year', 'song_year'),
                                     (time_data.song == song_df.title) \
                                     & (time_data.length == song_df.duration)) \
        .selectExpr(['ts as start_time',
                     'userId as user_id',
                     'level',
                     'song_id',
                     'artist_id',
                     'sessionId as session_id',
                     'location',
                     'userAgent as user_agent',
                     'month',
                     'year'])
    # Convert dataframe to temporary table to execute sql queries
    songplays_table.createOrReplaceTempView('songplays')

    # unique ids added to rows using sql window functions
    songplays_table = spark.sql('select row_number() over (order by "start_time") as songplay_id, * from songplays')

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode("overwrite").partitionBy("year", "month").parquet(output_data + "songplays.parquet")


def main():
    """
    Creates a spark session and executes ETL job to process songs and event data
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://vikasoutput/output/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
