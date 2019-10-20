from itertools import islice

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import Row
from pyspark.sql.types import StringType

# reading poc2.csv
# DeviceId	Temperature	Timestamp
# sams761	43	6/24/2019 23:59
# sams762	42	6/24/2019 23:57
# sams763	40	6/24/2019 23:55
# sams764	26	6/24/2019 23:51
# sams765	21	6/23/2019 23:49
# sams766	18	6/23/2019 23:59
# sams767	34	6/22/2019 22:49
# sams768	23	6/22/2019 22:48
# sams769	19	6/22/2019 22:49
# sams770	24	6/21/2019 22:49
# sams771	43	6/21/2019 22:49
# sams772	42	6/21/2019 22:49
# sams773	40	6/21/2019 22:49
# sams774	26	6/21/2019 22:49
# sams775	18	6/20/2019 22:49
# sams776	34	6/20/2019 22:49
# sams777	23	6/20/2019 22:49
# sams778	22	6/19/2019 22:49
# sams779	35	6/19/2019 22:49

spark = SparkSession.builder.appName('company').getOrCreate()
sc = spark.sparkContext


# using rdd

def f(x):
    d = {}
    for i, j in enumerate(x):
        d[str(i)] = j
    return d


rdd1 = sc.textFile("file:///Users/navomsaxena/Downloads/poc2.csv")
rdd2 = rdd1.mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it)
rdd3 = rdd2.map(lambda x: x.split(" ")[0])
rdd4 = rdd3.map(lambda x: x.split(","))
rdd5 = rdd4.map(lambda x: Row(**f(x))).toDF()
# rdd5.write.partitionBy('2').mode('overwrite').format("com.databricks.spark.csv").save('file:///Users/navomsaxena/Downloads/outputRDD')
rdd5.show()

# rdd5 has timestamp column as date. Saving this hive hive partitioned will result in partitioned data in hive

# using dataframe

fn = lambda x: x.split(" ")[0]
fn_udf = udf(lambda z: fn(z), StringType())

df1 = spark.read.csv("file:///Users/navomsaxena/Downloads/poc2.csv", header=True)
df2 = df1.withColumn("Timestamp", fn_udf("Timestamp"))
# df2.write.partitionBy('Timestamp').mode('overwrite').format("com.databricks.spark.csv").save('file:///Users/navomsaxena/Downloads/output')
df2.show()
df2.registerTempTable("table")
spark.sql("select * from table").show()
spark.sql("select * from table where Temperature > 40").show()
spark.sql("select * from table where Timestamp = '6/24/2019'").show()
df2.groupby("Timestamp").agg({'Temperature': 'max'}).show()
df2.groupby("Timestamp").agg({'Temperature': 'min'}).show()
df2.groupby("Timestamp").agg({'Temperature': 'mean'}).show()
df2.groupby("Timestamp").count().show()


# df2 has timestamp column as date. Saving this hive hive partitioned will result in partitioned data in hive

def exceptionHandler(o):
    try:
        o()
    except Exception as e:
        print("exception is captured")


def o():
    print(10 / 1)


exceptionHandler(o)
