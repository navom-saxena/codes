from pyspark.sql import SparkSession


class Movies:
    if __name__ == '__main__':
        spark = SparkSession.builder.getOrCreate()
        df = spark.read.format('csv').option("header", "true").load("/Users/navom_saxena/Downloads/ratings1.csv");
        df2 = df.groupBy("userId").count()
        df2.select("userId").where("count>0").show()
