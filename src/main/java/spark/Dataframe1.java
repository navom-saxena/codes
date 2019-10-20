package spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

/*
 * Following are movie Lens problem statements. DataframeN refers to solution of problem N
 * Corresponding solution in rdd ProblemStatementNx refers to solution of problem N sub-problem x
 * Solutions using pyspark can be found in python package of Dataframe analysis with name DataframeN and rdd as CaseN

 * 1. List all the movies and the number of ratings
 * 2. List all the users and the number of ratings they have done for a movie
 * 3. List all the Movie IDs which have been rated (Movie Id with at least one user rating it)
 * 4. List all the Users who have rated the movies (Users who have rated at least one movie)
 * 5. List of all the User with the max ,min ,average ratings they have given against any movie
 * 6. List all the Movies with the max ,min, average ratings given by any user
 */

public class Dataframe1 {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example").master("local")
                .getOrCreate();

        Dataset<Row> df2 = spark.read().format("csv").option("header", "true")
                .load("/Users/navom_saxena/Downloads/ratings.csv");

        df2.withColumn("add", functions.lit(123))
                .withColumnRenamed("rating", "myrating")
                .filter(functions.col("movieId").$less(6)).show();
    }
}
