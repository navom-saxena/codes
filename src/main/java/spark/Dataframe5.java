package spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;

public class Dataframe5 {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example").master("local")
                .getOrCreate();

        Dataset<Row> ratings = spark.read().format("csv").option("header", "true")
                .load("/Users/navom_saxena/Downloads/ratings.csv");

        ratings.groupBy("userId")
                .agg(max("rating"), min("rating"), avg("rating"))
                .show();

    }
}
