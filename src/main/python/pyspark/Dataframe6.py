from pyspark.sql import SparkSession
from pyspark.sql.functions import max, min


class MoviesJoin:
    if __name__ == '__main__':
        spark = SparkSession.builder.getOrCreate()
        df = spark.read.format('csv').option("header", "true").load("/Users/navom_saxena/Downloads/ratings1.csv");

        df2 = spark.read.format('csv').option("header", "true").load("/Users/navom_saxena/Downloads/movies.csv");

        inner_join = df1.join(df2, df.movieId == df2.movieId).groupBy("title").agg(max("rating"), min("rating"),
                                                                                   avg("rating")).show()
