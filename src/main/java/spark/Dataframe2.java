package spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Dataframe2 {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example").master("local")
                .getOrCreate();

        Dataset<Row> df1 = spark.read().format("csv").option("header", "true")
                .load("/Users/navom_saxena/Downloads/movies.csv");

        Dataset<Row> df2 = spark.read().format("csv").option("header", "true")
                .load("/Users/navom_saxena/Downloads/ratings.csv");
        Dataset<Row> df3 = df1.select("movieId", "title");
        Dataset<Row> df4 = df2.groupBy("movieId").count();
        Dataset<Row> df5 = df3.join(df4, "movieId");
        df5.select("movieId", "title").show();
    }

}

