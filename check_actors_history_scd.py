

"""
Actors SCD (Slowly Changing Dimension) Transformation Script

This script performs an SCD transformation for actors, combining historical SCD data
with new/current year data. It calculates quality classes based on average film ratings,
updates start/end years, and handles missing actors using a backfill strategy.

Key Features:
1. Average rating calculation per actor for the new year.
2. Quality class assignment based on average rating:
   - >8 → 'star'
   - >7 → 'good'
   - >6 → 'average'
   - <=6 → 'bad'
3. Merge of historical and new data, updating SCD attributes correctly.
4. Backfill logic:
   - Actors missing from historical SCD are backfilled with start_year = lastyear,
     end_year = newyear-1, and inactive flags set to False.
   - Ensures no gaps in SCD for new actors or missing historical years.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, explode, avg, when, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType, BooleanType

def do_actors_scd_transformation(
    actors_df: DataFrame,
    actors_history_scd_df: DataFrame,
    lastyear: int,
    newyear: int
) -> DataFrame:
    """
    Perform SCD transformation for actors, merging historical and current data.

    Args:
        actors_df: DataFrame with columns (actorid, actor, currentyear, films)
        actors_history_scd_df: Historical SCD DataFrame
        lastyear: last year to consider for historical SCD
        newyear: year for new/current data

    Returns:
        DataFrame: final SCD table
    """

    # --- Step 1: Filter past records for the last year and rename columns to avoid ambiguity ---
    past_df = actors_history_scd_df.filter(
        (col("start_year") == lastyear) | (col("end_year") == lastyear)
    ).withColumnRenamed("actor", "history_actor") \
     .withColumnRenamed("currentyear", "history_currentyear") \
     .withColumnRenamed("start_quality_class", "history_start_quality_class") \
     .withColumnRenamed("end_quality_class", "history_end_quality_class") \
     .withColumnRenamed("start_isactive", "history_start_isactive") \
     .withColumnRenamed("end_isactive", "history_end_isactive")

    # --- Step 2: Unnest film ratings for the new year ---
    unnestedrating_df = actors_df.filter(col("currentyear") == newyear) \
        .withColumn("film", explode(col("films"))) \
        .withColumn("filmrating", col("film.rating")) \
        .select("actor", "actorid", "currentyear", "filmrating")

    # --- Step 3: Aggregate ratings to get average per actor ---
    presentaggdata_df = unnestedrating_df.groupBy("actor", "actorid", "currentyear") \
        .agg(avg("filmrating").alias("avg_rating"))

    # --- Step 4: Assign quality class based on average rating ---
    present_df = presentaggdata_df.withColumn(
        "quality_class",
        when(col("avg_rating") > 8, lit("star"))
        .when(col("avg_rating") > 7, lit("good"))
        .when(col("avg_rating") > 6, lit("average"))
        .otherwise(lit("bad"))
    )

    # --- Step 5: Merge historical and new data, updating SCD attributes ---
    newdata_df = present_df.join(
        past_df,
        on="actorid",
        how="fullouter"
    ).withColumn(
        "start_year",
        when(col("history_end_quality_class") == col("quality_class"), col("start_year")).otherwise(lit(newyear))
    ).withColumn(
        "end_year",
        when(col("history_end_quality_class") == col("quality_class"), lit(newyear)).otherwise(lit(newyear))
    ).withColumn(
        "start_quality_class",
        when(col("history_end_quality_class") == col("quality_class"), col("history_start_quality_class")).otherwise(col("quality_class"))
    ).withColumn(
        "end_quality_class",
        col("quality_class")
    ).withColumn(
        "start_isactive",
        when(col("currentyear") == newyear, lit(True)).otherwise(lit(False))
    ).withColumn(
        "end_isactive",
        lit(None)
    ).select(
        "actorid",
        when(col("actor").isNotNull(), col("actor")).otherwise(col("history_actor")).alias("actor"),
        "start_year", "end_year", "currentyear",
        "start_quality_class", "end_quality_class", "start_isactive", "end_isactive"
    )

    # --- Step 6: Backfill missing actors not present in historical SCD ---
    missing_actors_df = present_df.join(
        actors_history_scd_df,
        on="actorid",
        how="left_anti"  # Only actors not in history
    )

    backfill_df = missing_actors_df.withColumn("start_year", lit(lastyear)) \
        .withColumn("end_year", lit(newyear - 1)) \
        .withColumn("start_quality_class", col("quality_class")) \
        .withColumn("end_quality_class", col("quality_class")) \
        .withColumn("start_isactive", lit(False)) \
        .withColumn("end_isactive", lit(False)) \
        .select("actorid", "actor", "start_year", "end_year", "currentyear",
                "start_quality_class", "end_quality_class", "start_isactive", "end_isactive")

    # --- Step 7: Combine new data with backfilled rows to produce final SCD ---
    final_df = newdata_df.unionByName(backfill_df)

    return final_df


def main():
    """
    Main function to create Spark session, sample input data, and run the SCD transformation.
    """
    spark = SparkSession.builder \
        .appName("ActorsSCDExample") \
        .enableHiveSupport() \
        .getOrCreate()

    # --- Define actors input schema and data ---
    actors_schema = StructType([
        StructField("actorid", StringType(), True),
        StructField("actor", StringType(), True),
        StructField("currentyear", IntegerType(), True),
        StructField("films", ArrayType(
            StructType([StructField("rating", FloatType(), True)])
        ), True)
    ])

    actors_data = [
        ("A1", "Actor One", 1980, [{"rating": 8.5}, {"rating": 7.2}]),
        ("A2", "Actor Two", 1980, [{"rating": 6.0}, {"rating": 6.5}])
    ]
    actors_df = spark.createDataFrame(actors_data, schema=actors_schema)

    # --- Define historical SCD schema and data ---
    history_schema = StructType([
        StructField("actorid", StringType(), True),
        StructField("actor", StringType(), True),
        StructField("start_year", IntegerType(), True),
        StructField("end_year", IntegerType(), True),
        StructField("currentyear", IntegerType(), True),
        StructField("start_quality_class", StringType(), True),
        StructField("end_quality_class", StringType(), True),
        StructField("start_isactive", BooleanType(), True),
        StructField("end_isactive", BooleanType(), True)
    ])

    actors_history_scd_data = [
        ("A1", "Actor One", 1978, 1979, 1978, "good", "bad", True, None),
        ("A2", "Actor Two", 1978, 1979, 1978, "average", "average", True, None)
    ]
    actors_history_scd_df = spark.createDataFrame(actors_history_scd_data, schema=history_schema)

    # --- Run the SCD transformation ---
    final_df = do_actors_scd_transformation(
        actors_df=actors_df,
        actors_history_scd_df=actors_history_scd_df,
        lastyear=1979,
        newyear=1980
    )

    # --- Show results ---
    final_df.show(truncate=False)
    final_df.printSchema()


if __name__ == "__main__":
    main()
