"""
Test script for the Actors SCD (Slowly Changing Dimension) transformation.

This test validates that the `do_actors_scd_transformation` function correctly:
1. Updates the historical SCD table with new actor data.
2. Handles backfill logic:
   - If a historical record exists that ends before the new year, the transformation
     extends or closes the record based on activity and quality class.
   - New actors or missing years are backfilled to ensure continuity in the SCD.
3. Maintains correct start_year, end_year, and isactive flags for each actor.
4. Preserves nested data (like films) in the input.

The test uses Spark DataFrames with explicit schemas, nested structs for film ratings,
and compares the transformed result to the expected SCD output using `chispa.assert_df_equality`.
"""

from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType, BooleanType
from chispa.dataframe_comparer import assert_df_equality
from ..jobs.check_actors_history_scd import do_actors_scd_transformation  # Import the transformation function

def test_actors_scd_transformation(spark):
    # --- Define schema for the input actors DataFrame ---
    # Each actor has an ID, name, current year, and a list of films with ratings
    actors_schema = StructType([
        StructField("actorid", StringType(), True),
        StructField("actor", StringType(), True),
        StructField("currentyear", IntegerType(), True),
        StructField("films", ArrayType(
            StructType([StructField("rating", FloatType(), True)])  # Nested struct for film ratings
        ), True)
    ])

    # --- Define schema for the historical SCD DataFrame ---
    # Each record tracks actor history including start/end years, quality, and active status
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

    # --- Create input actors DataFrame ---
    actors_input_data = [
        ("A1", "Actor One", 1980, [{"rating": 8.5}, {"rating": 7.2}]),
        ("A2", "Actor Two", 1980, [{"rating": 6.0}, {"rating": 6.5}])
    ]
    actors_df = spark.createDataFrame(actors_input_data, schema=actors_schema)

    # --- Create historical SCD DataFrame ---
    history_input_data = [
        ("A1", "Actor One", 1978, 1979, 1978, "good", "bad", True, None),
        ("A2", "Actor Two", 1978, 1979, 1978, "average", "average", True, None)
    ]
    actors_history_scd_df = spark.createDataFrame(history_input_data, schema=history_schema)

    # --- Run the transformation ---
    # lastyear = 1979: the previous SCD end year
    # newyear = 1980: the current year to update or backfill
    actual_df = do_actors_scd_transformation(
        actors_df=actors_df,
        actors_history_scd_df=actors_history_scd_df,
        lastyear=1979,
        newyear=1980
    )

    # --- Define expected SCD output after transformation ---
    # Notes on backfill logic:
    # - Actor A1 has a historical record ending in 1979; the transformation updates it to 1980.
    # - Actor A2 spans multiple years; the transformation extends the end_year to 1980.
    expected_data = [
        Row(actorid="A1", actor="Actor One", start_year=1980, end_year=1980, currentyear=1980,
            start_quality_class="good", end_quality_class="good", start_isactive=True, end_isactive=None),
        Row(actorid="A2", actor="Actor Two", start_year=1978, end_year=1980, currentyear=1980,
            start_quality_class="average", end_quality_class="average", start_isactive=True, end_isactive=None)
    ]
    expected_df = spark.createDataFrame(expected_data, schema=history_schema)

    # --- Assert equality between actual and expected DataFrames ---
    # ignore_nullable=True avoids failures due to minor differences in nullability
    assert_df_equality(actual_df, expected_df, ignore_nullable=True)
