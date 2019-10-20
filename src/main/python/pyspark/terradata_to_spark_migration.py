import pyspark.sql.functions as func
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.functions import to_date
from pyspark.sql.functions import when, current_date

inactive_date = '3001-01-01'

spark = SparkSession.builder.appName('company').getOrCreate()
sc = spark.sparkContext

daily = spark.read.csv("file:///Users/navomsaxena/Downloads/daily_dataset.csv", header=True)
history = spark.read.csv("file:///Users/navomsaxena/Downloads/history_dataset.csv", header=True)

daily = daily.withColumn('active_date', func.current_date()).withColumn('inactive_date',
                                                                        to_date(lit(inactive_date), "yyyy-MM-dd"))
history = history.withColumn('active_date', to_date(history.active_date)).withColumn('inactive_date', to_date(
    to_date(history.inactive_date)))


def trigger_transformations(daily, history):
    print('daily.....count -', daily.count())
    print('history.....count -', history.count())
    history_updated = transformations_1(daily, history)
    history_inserted_t2 = transformation_2(daily, history_updated)
    history_inserted_t3 = transformation_3(daily, history_inserted_t2)
    return history_inserted_t3


# UPDATE mk_prod_db.reward_type_cd
# SET INACTIVE_DT = date
# WHERE mk_prod_db.reward_type_cd.reward_type_cd = MK_TEMP_DB.reward_type_cd.reward_type_Cd
#        AND mk_prod_db.reward_type_cd.INACTIVE_DT  = '3001-01-01'
#        AND mk_prod_db.reward_type_cd.reward_type_dscr <> mk_temp_db.reward_type_cd.reward_type_dscr;

def transformations_1(daily, history):
    history_updated = history.join(daily, (history.reward_type_cd == daily.reward_type_cd), "left_outer") \
        .withColumn('inactive_date_temp', when(
        (history.reward_type_cd == daily.reward_type_cd)
        & (history.reward_type_dscr != daily.reward_type_dscr)
        & (history.inactive_date == func.lit(inactive_date)), current_date()).otherwise(history.inactive_date)) \
        .drop(daily.reward_type_cd) \
        .drop(daily.inactive_date) \
        .drop(daily.active_date) \
        .drop(daily.reward_type_dscr) \
        .select(history.reward_type_cd, history.reward_type_dscr, history.active_date, 'inactive_date_temp') \
        .withColumnRenamed('inactive_date_temp', 'inactive_date')
    print('history_updated......count -', history_updated.count())
    history_updated.show()
    return history_updated


# locking table mk_prod_db.reward_type_cd for access
# insert into mk_prod_db.reward_type_cd
# (reward_type_cd
# ,reward_type_dscr
# ,active_dt
# ,inactive_dt)
#
# sel
# a.reward_type_cd
# ,a.reward_type_dscr
# ,date
# ,'3001-01-01'
# from mk_temp_db.reward_type_cd a
# inner join mk_prod_db.reward_type_cd b
# on a.reward_type_cd = b.reward_type_cd
# and b.inactive_dt = date;

def transformation_2(daily, history_updated):
    inner_joined_data = daily.join(history_updated, (history_updated.reward_type_cd == daily.reward_type_cd)
                                   & (history_updated.inactive_date == func.current_date())) \
        .select(daily.reward_type_cd, daily.reward_type_dscr) \
        .withColumn('active_date', (func.current_date())).withColumn('inactive_date', func.lit(inactive_date))
    history_inserted_t2 = history_updated.union(inner_joined_data)
    print('history_inserted_t2......count -', history_inserted_t2.count())
    history_inserted_t2.show()
    return history_inserted_t2


# locking table mk_prod_db.reward_type_cd for access
#  insert into mk_prod_db.reward_type_cd
# (reward_type_cd
# ,reward_type_dscr
# ,active_dt
# ,inactive_dt)
# sel
# a.reward_type_cd
# ,a.reward_type_dscr
# ,date
# ,'3001-01-01'
# from mk_temp_db.reward_type_cd a
# where reward_type_cd not in
# 	(select reward_type_cd
# 	from mk_prod_db.reward_type_cd
# 	where active_dt <= date
# 	and inactive_dt > date
# 	group by 1);

def transformation_3(daily, history_inserted):
    inner_query_df = history_inserted.filter(
        (history_inserted.active_date <= func.current_date()) & (history_inserted.inactive_date > func.current_date())) \
        .select('reward_type_cd')
    anti_join_df = daily.join(inner_query_df, 'reward_type_cd', 'left_anti') \
        .withColumn('active_date', func.current_date()) \
        .withColumn('inactive_date', func.lit(inactive_date))
    history_inserted_t3 = anti_join_df.union(history_inserted)
    print('history_inserted_t3......count -', history_inserted_t3.count())
    history_inserted_t3.show()
    return history_inserted_t3


df = trigger_transformations(daily, history)
print('final df......count -', df.count())
df.show()
