from pyspark.sql import SparkSession


class Movies:
    if __name__ == '__main__':
        spark = SparkSession.builder.getOrCreate()
        df1 = spark.read.format('csv').option("header", "true").load("/Users/navom_saxena/Downloads/ratings1.csv");
        ratingsdf = df1.groupBy("movieId").count()
        df2 = spark.read.format('csv').option("header", "true").load("/Users/navom_saxena/Downloads/movies.csv");
        moviesdf = df2.select("movieId", "title")
        df2.filter(lambda x: x.movieId < "5", moviesdf).show()
        combined_df = moviesdf.join(ratingsdf, "movieId")
        combined_df.select("title", "count").show()
