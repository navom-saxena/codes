package spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Dataframe4 {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example").master("local")
                .getOrCreate();

        Dataset<Row> df2 = spark.read().format("csv").option("header", "true")
                .load("/Users/navom_saxena/Downloads/ratings.csv");

        Dataset<Row> df3 = df2.groupBy("userId").count();
        df3.select("userId").where("count>0").show();

    }
}

