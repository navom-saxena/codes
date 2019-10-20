package spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

public class ProblemStatement4a {
    public static void main(String[] str) {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("userrating");
        JavaSparkContext context = new JavaSparkContext(conf);
        JavaRDD<String> rdd1 = context.textFile("/Users/navom_saxena/Downloads/ratings.csv");
        JavaPairRDD<String, Integer> rdd2 = rdd1.mapToPair(s -> {
            String[] rowArr = s.split(",");
            return new Tuple2(rowArr[0], 1);
        });
        rdd2.reduceByKey((x, y) -> (int) x + y).filter(t -> t._2 > 0).collect();
    }
}
