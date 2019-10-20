from datetime import date

from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from terradata_to_spark_migration import trigger_transformations

spark = SparkSession.builder.appName('company').getOrCreate()
sc = spark.sparkContext


def get_current_date():
    get_date = date.today()
    parsed_date = get_date.strftime("%Y-%m-%d")
    return parsed_date


def create_raw_insert(spark_session):
    raw_sample = [Row(reward_type_cd='B', reward_type_dscr='star enterprises',
                      active_date=get_current_date(), inactive_date='3001-01-01'),
                  Row(reward_type_cd='L', reward_type_dscr='yoda',
                      active_date=get_current_date(), inactive_date='3001-01-01'),
                  Row(reward_type_cd='U', reward_type_dscr='luke',
                      active_date=get_current_date(), inactive_date='3001-01-01'),
                  Row(reward_type_cd='A', reward_type_dscr='felt',
                      active_date=get_current_date(), inactive_date='3001-01-01'),
                  ]

    raw_schema = create_raw_schema()
    raw_df = spark_session.createDataFrame(raw_sample, raw_schema)
    return raw_df


def create_raw_schema():
    return StructType([StructField('reward_type_cd', StringType(), True),
                       StructField('reward_type_dscr', StringType(), True),
                       StructField('active_date', StringType(), True),
                       StructField('inactive_date', StringType(), True),
                       ])


def create_final_insert(spark_session):
    expected_sample = [Row(reward_type_cd='B', reward_type_dscr='star enterprises',
                           active_date=get_current_date(), inactive_date='3001-01-01'),
                       Row(reward_type_cd='L', reward_type_dscr='yoda',
                           active_date=get_current_date(), inactive_date='3001-01-01'),
                       Row(reward_type_cd='U', reward_type_dscr='luke',
                           active_date=get_current_date(), inactive_date='3001-01-01'),
                       Row(reward_type_cd='A', reward_type_dscr='felt',
                           active_date=get_current_date(), inactive_date='3001-01-01'),
                       ]

    expected_schema = create_reward_type_schema()
    expected_df = spark_session.createDataFrame(expected_sample, expected_schema)
    return expected_df


def create_reward_type_schema():
    return StructType([StructField('reward_type_cd', StringType(), True),
                       StructField('reward_type_dscr', StringType(), True),
                       StructField('active_date', StringType(), True),
                       StructField('inactive_date', StringType(), True)])


def create_history_insert(spark_session):
    reward_type_sample = []
    expected_schema = create_reward_type_schema()
    reward_type_sample_df = spark_session.createDataFrame(reward_type_sample, expected_schema)
    return reward_type_sample_df


def create_raw_update(spark_session):
    raw_sample = [Row(reward_type_cd='B', reward_type_dscr='star enterprises test',
                      active_date=get_current_date(), inactive_date='3001-01-01'),
                  Row(reward_type_cd='L', reward_type_dscr='yoda test',
                      active_date=get_current_date(), inactive_date='3001-01-01'),
                  Row(reward_type_cd='A', reward_type_dscr='han solo',
                      active_date=get_current_date(), inactive_date='3001-01-01'),
                  Row(reward_type_cd='Z', reward_type_dscr='admiral pett',
                      active_date=get_current_date(), inactive_date='3001-01-01'),
                  ]

    raw_schema = create_raw_schema()
    raw_df = spark_session.createDataFrame(raw_sample, raw_schema)
    return raw_df


def create_final_update(spark_session):
    expected_sample = [
        Row(reward_type_cd='B', reward_type_dscr='star enterprises',
            active_date=get_current_date(), inactive_date=get_current_date()),
        Row(reward_type_cd='L', reward_type_dscr='yoda',
            active_date=get_current_date(), inactive_date=get_current_date()),
        Row(reward_type_cd='U', reward_type_dscr='luke',
            active_date=get_current_date(), inactive_date='3001-01-01'),
        Row(reward_type_cd='A', reward_type_dscr='felt',
            active_date=get_current_date(), inactive_date=get_current_date()),
        Row(reward_type_cd='A', reward_type_dscr='han solo',
            active_date=get_current_date(), inactive_date='3001-01-01'),
        Row(reward_type_cd='B', reward_type_dscr='star enterprises test',
            active_date=get_current_date(), inactive_date='3001-01-01'),
        Row(reward_type_cd='L', reward_type_dscr='yoda test',
            active_date=get_current_date(), inactive_date='3001-01-01'),
        Row(reward_type_cd='Z', reward_type_dscr='admiral pett',
            active_date=get_current_date(), inactive_date='3001-01-01'), ]
    expected_schema = create_reward_type_schema()
    expected_df = spark_session.createDataFrame(expected_sample, expected_schema)
    return expected_df


def create_history_update(spark_session):
    reward_type_sample = [Row(reward_type_cd='B', reward_type_dscr='star enterprises',
                              active_date=get_current_date(), inactive_date='3001-01-01'),
                          Row(reward_type_cd='L', reward_type_dscr='yoda',
                              active_date=get_current_date(), inactive_date='3001-01-01'),
                          Row(reward_type_cd='U', reward_type_dscr='luke',
                              active_date=get_current_date(), inactive_date='3001-01-01'),
                          Row(reward_type_cd='A', reward_type_dscr='felt',
                              active_date=get_current_date(), inactive_date='3001-01-01'),
                          ]
    expected_schema = create_reward_type_schema()
    reward_type_df = spark_session.createDataFrame(reward_type_sample, expected_schema)
    return reward_type_df


# check if history is empty, then
#  current history should be equal to inserted data


def transformation_test_insert(spark_session):
    daily = create_raw_insert(spark_session)
    expected = create_final_insert(spark_session)
    empty_history = create_history_insert(spark_session)
    result_df = trigger_transformations(daily, empty_history)

    result_df = result_df.subtract(expected)
    result_df.show()
    assert result_df.count() == 0


# previous history + daily should be
# equal to expected current history


def transformation_test_update(spark_session):
    daily = create_raw_update(spark_session)
    expected = create_final_update(spark_session)
    history = create_history_update(spark_session)
    result_df = trigger_transformations(daily, history)
    result_df.printSchema()

    result_df.show()
    expected.show()
    result_df.subtract(expected).show()

    result_df = result_df.subtract(expected)
    assert result_df.count() == 0


transformation_test_insert(spark)
transformation_test_update(spark)
