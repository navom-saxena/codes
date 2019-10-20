package spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Collections;

import static org.apache.spark.sql.functions.*;

public class Dataframe6 {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example").master("local")
                .getOrCreate();

        Dataset<Row> ratings = spark.read().format("csv").option("header", "true")
                .load("/Users/navom_saxena/Downloads/ratings.csv");

        Dataset<Row> movies = spark.read().format("csv").option("header", "true")
                .load("/Users/navom_saxena/Downloads/movies.csv");

        Dataset<Row> joinedDf = ratings.join(movies, scala.collection.JavaConverters.asScalaIteratorConverter(
                Collections.singletonList("movieId").iterator()
        ).asScala().toSeq(), "inner");

        joinedDf.groupBy("title")
                .agg(max("rating"), min("rating"), avg("rating"))
                .show();

    }
}
