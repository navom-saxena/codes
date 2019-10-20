import pyspark.sql.functions as func
from pyspark.sql.functions import lit
from pyspark.sql.functions import to_date
from pyspark.sql.functions import when, current_date
from transforms.api import transform, Input, incremental

inactive_date = '3001-01-01'
date = '2019-07-29'
is_history = False


@incremental(snapshot_inputs=[''])
@transform(
    history=Input(''),
    daily=Input('')
)
def pipeline(history, daily):
    history = history.dataframe()
    daily = daily.dataframe() \
        .withColumn('active_date', func.current_date()) \
        .withColumn('inactive_date', to_date(lit(inactive_date), "yyyy-MM-dd"))
    df = trigger_transformations(daily, history)
    return df


def trigger_transformations(daily, history):
    history_updated = transformations_1(daily, history)
    history_inserted_t2 = transformation_2(daily, history_updated)
    history_inserted_t3 = transformation_3(daily, history_inserted_t2)
    return history_inserted_t3


def transformations_1(daily, history):
    history_updated = history.join(daily, (history.reward_type_cd == daily.reward_type_cd), "left_outer") \
        .withColumn('inactive_date_temp', when(
        (history.reward_type_cd == daily.reward_type_cd)
        & (history.reward_type_dscr != daily.reward_type_dscr)
        & (history.inactive_date == func.lit(inactive_date)), current_date()).otherwise(history.inactive_date)) \
        .drop(daily.reward_type_cd) \
        .drop(daily.reward_type_dscr).withColumnRenamed('inactive_date_temp', 'inactive_date')
    return history_updated


def transformation_2(daily, history_updated):
    inner_joined_data = daily.join(history_updated, (history_updated.reward_type_cd == daily.reward_type_cd)
                                   & (history_updated.inactive_date == func.lit(date))) \
        .select(daily.reward_type_cd, daily.reward_type_dscr) \
        .withColumn('active_date', (func.lit(date))).withColumn('inactive_date', func.lit(inactive_date))
    history_inserted_t2 = history_updated.union(inner_joined_data)
    return history_inserted_t2


def transformation_3(daily, history_inserted):
    inner_query_df = history_inserted.filter(
        (history_inserted.active_date <= func.lit(date)) & (history_inserted.inactive_date > func.lit(date))) \
        .select('reward_type_cd')
    anti_join_df = daily.join(inner_query_df, 'reward_type_cd', 'left_anti') \
        .withColumn('active_date', func.current_date()) \
        .withColumn('inactive_date', func.lit(inactive_date))
    history_inserted_t3 = anti_join_df.union(history_inserted)
    return history_inserted_t3
