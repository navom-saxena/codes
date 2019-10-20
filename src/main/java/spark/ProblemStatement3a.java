/*
Movie Lens Use Case using Spark Core API to List all the Movie IDs which have been rated (Movie Id with at least one user rating it)
 */

package spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

public class ProblemStatement3a {
    public static void main(String[] str) {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("userrating");
        JavaSparkContext context = new JavaSparkContext(conf);
        JavaRDD<String> rdd1 = context.textFile("/Users/navom_saxena/Downloads/ratings.csv");
        //extract movie id and rating
        JavaRDD<String> ratingrdd = rdd1.map(s -> {
            String[] arr = s.split(",");
            return arr[1] + "," + arr[2];
        });
        JavaPairRDD<String, String> finalratingrdd = ratingrdd.mapToPair(t -> {   //add movie id and rating in a tuple
            String[] arr = t.split(",");
            return new Tuple2(arr[0], arr[1]);
        });
        JavaPairRDD<String, Iterable<String>> tuplerdd = finalratingrdd.groupByKey();
        tuplerdd.foreach(t -> {
            Iterable<String> getstr = t._2;

            int movieId = Integer.parseInt(t._1);
            int c = 0;
            for (String s : getstr) {

                c++;


            }
            if (c != 0)
                System.out.println(movieId);

        });

    }
}


