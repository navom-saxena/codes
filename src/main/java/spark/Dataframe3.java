package spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class Dataframe3 {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example").master("local")
                .getOrCreate();

        Dataset<Row> df2 = spark.read().format("csv").option("header", "true")
                .load("/Users/navom_saxena/Downloads/ratings.csv");
        df2.groupBy("userId").agg(functions.max("rating").as("maxrating"),
                functions.min("rating").as("minrating"),
                functions.avg("rating").as("avgrating")).show();
    }

}
