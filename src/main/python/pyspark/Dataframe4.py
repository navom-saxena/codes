from pyspark.sql import SparkSession
from pyspark.sql.functions import max, min


class Movies:
    if __name__ == '__main__':
        spark = SparkSession.builder.getOrCreate()
        df = spark.read.format('csv').option("header", "true").load("/Users/navom_saxena/Downloads/ratings1.csv");
        df.groupBy("userId").agg(max("rating"), min("rating")).show()
