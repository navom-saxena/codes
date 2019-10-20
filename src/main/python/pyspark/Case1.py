from pyspark.sql import SparkSession


class Movies:
    def userRating(self):
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
        rating_rdd = sc.textFile("/Users/navom_saxena/Downloads/ratings1.csv")
        words = rating_rdd.map(lambda s: (s.split(","))[0])
        wordspair = words.map(lambda s: (s, 1))
        # users with number of ratings..........
        wordscount = wordspair.reduceByKey(lambda x, y: x + y)
        print(wordscount.collect())
        return wordscount


if __name__ == '__main__':
    ob = Movies()
    ob.userRating()
