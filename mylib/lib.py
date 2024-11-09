import os
import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType
)

LOG_FILE = "log_file.md"

def log_output(operation, output, query=None):
    """Add logs to markdown file"""
    with open(LOG_FILE, "a") as file:
        file.write(f"The operation is {operation}\n\n")
        if query:
            file.write(f"The query is {query}\n\n")
        file.write("The truncated output is:\n\n")
        file.write(output)
        file.write("\n\n")

def start_spark(appName="SpotifyApp"):
    """Initialize Spark session"""
    spark = SparkSession.builder.appName(appName).getOrCreate()
    return spark

def end_spark(spark):
    """Stop Spark session"""
    spark.stop()
    return "Stopped Spark session"

def extract(
    url="https://raw.githubusercontent.com/nogibjj/"
        "chris_moreira_week5_python_sql_db_project/main/"
        "data/Spotify_Most_Streamed_Songs.csv",
    file_path="data/Spotify_Most_Streamed_Songs.csv",
    directory="data"
):
    """Extract file from URL to local path"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    with requests.get(url) as r:
        with open(file_path, "wb") as f:
            f.write(r.content)
    return file_path

def load_data(spark, data="data/Spotify_Most_Streamed_Songs.csv"):
    """Load data with schema and log initial preview"""
    schema = StructType([
        StructField("track_name", StringType(), True),
        StructField("artist_name", StringType(), True),
        StructField("artist_count", IntegerType(), True),
        StructField("released_year", IntegerType(), True),
        StructField("released_month", IntegerType(), True),
        StructField("released_day", IntegerType(), True),
        StructField("in_spotify_playlists", IntegerType(), True),
        StructField("in_spotify_charts", IntegerType(), True),
        StructField("streams", IntegerType(), True),
        StructField("in_apple_playlists", IntegerType(), True),
        StructField("key", StringType(), True),
        StructField("mode", StringType(), True),
        StructField("danceability_percent", IntegerType(), True),
        StructField("valence_percent", IntegerType(), True),
        StructField("energy_percent", IntegerType(), True),
        StructField("acousticness_percent", IntegerType(), True),
        StructField("instrumentalness_percent", IntegerType(), True),
        StructField("liveness_percent", IntegerType(), True),
        StructField("speechiness_percent", IntegerType(), True),
        StructField("cover_url", StringType(), True)
    ])
    
    df = spark.read.option("header", "true").schema(schema).csv(data)
    log_output("load data", df.limit(10).toPandas().to_markdown())
    return df

def query(spark, df, query, name="SpotifyData"):
    """Executes Spark SQL query and logs the output"""
    df.createOrReplaceTempView(name)
    result_df = spark.sql(query)
    log_output(
        "query data",
        result_df.limit(10).toPandas().to_markdown(),
        query
    )
    return result_df.show()

def describe(df):
    """Generates descriptive statistics and logs the output"""
    summary_stats_str = df.describe().toPandas().to_markdown()
    log_output("describe data", summary_stats_str)
    return df.describe().show()

def example_transform(df):
    """Example transformation: categorizes popularity based on streams"""
    df = df.withColumn(
        "Popularity_Category",
        F.when(F.col("streams") > 1000000000, "Ultra Popular")
         .when(
             (F.col("streams") > 500000000)
             & (F.col("streams") <= 1000000000),
             "Very Popular"
         )
         .when(
             (F.col("streams") > 100000000)
             & (F.col("streams") <= 500000000),
             "Popular"
         )
         .otherwise("Less Popular")
    )
    
    log_output("transform data", df.limit(10).toPandas().to_markdown())
    return df.show()
