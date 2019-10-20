from pyspark.sql import SparkSession


class Movies:
    if __name__ == '__main__':
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext

        rating = sc.textFile("/Users/navom_saxena/Downloads/ratings1.csv")
        ratingrdd = rating.map(lambda t: t.split(",")[1])
        finalratingrdd = ratingrdd.map(lambda t: (t, 1)).reduceByKey(lambda x, y: x + y)
        movie = sc.textFile("/Users/navom_saxena/Downloads/movies.csv")
        movierdd = movie.map(lambda t: (t.split(",")[0], t.split(",")[1]))
        combined = movierdd.join(finalratingrdd)
        for item in combined.collect():
            for i in item:
                if (isinstance(i, tuple)):
                    print(i)

        for x, (k, v) in combined.collect():
            st = x + " " + k + " " + str(v)
            s = k + " " + str(v)
            print(s)
