from pyspark.sql import SparkSession


# Following are movie Lens problem statements. DataframeN refers to solution of problem N and rdd analysis as CaseN
#
# 1. List all the movies and the number of ratings
# 2. List all the users and the number of ratings they have done for a movie
# 3. List all the Movie IDs which have been rated (Movie Id with at least one user rating it)
# 4. List all the Users who have rated the movies (Users who have rated at least one movie)
# 5. List of all the User with the max ,min ,average ratings they have given against any movie
# 6. List all the Movies with the max ,min, average ratings given by any user
# * */

class Movies:
    if __name__ == '__main__':
        spark = SparkSession.builder.getOrCreate()
        df = spark.read.format('csv').option("header", "true").load("/Users/navom_saxena/Downloads/ratings1.csv");

        df.show()
        df.groupBy("userId").count().show()
