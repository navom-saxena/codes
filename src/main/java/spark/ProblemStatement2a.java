/*
Movie Lens Use Case using Spark Core API to List all the(only 10) users(user id) and the number of times  they have done movie rating
 */

package spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.Arrays;

public class ProblemStatement2a {

    public static void main(String[] str) {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("userrating");
        JavaSparkContext context = new JavaSparkContext(conf);
        JavaRDD<String> rdd = context.textFile("/Users/navom_saxena/Downloads/ratings.csv");
        JavaRDD<String> wordsrdd1 = rdd.map(t -> Arrays.asList(t.split(",")).get(0)); // extracted user id by splitting on ,
        JavaPairRDD count = wordsrdd1.mapToPair(t -> new Tuple2(t, 1)).reduceByKey((x, y) -> (int) x + (int) y); //wordcount on user id
        System.out.println(count.take(10));
    }

}


