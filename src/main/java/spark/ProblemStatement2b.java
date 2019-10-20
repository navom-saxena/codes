/*
Movie Lens Use Case using  SPARK SQL to List all the(only 10) users(user id) and the number of times  they have done movie rating
 */

package spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class ProblemStatement2b {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example").master("local")
                .getOrCreate();

        Dataset<Row> df2 = spark.read().format("csv").option("header", "true")
                .load("/Users/navom_saxena/Downloads/ratings.csv");

        df2.groupBy("userId").count().show();


    }
}

