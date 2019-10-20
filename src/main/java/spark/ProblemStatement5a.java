/*
Movie Lens Use Case using Spark Core API to List of all the User with the max ,min ,average ratings they have given against any movie
 */


package spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

public class ProblemStatement5a {
    public static void main(String[] str) {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("userrating");
        JavaSparkContext context = new JavaSparkContext(conf);
        JavaRDD<String> rdd1 = context.textFile("/Users/navom_saxena/Downloads/ratings.csv");
        JavaRDD<String> ratingrdd = rdd1.map(s -> {
            String[] arr = s.split(",");
            return arr[0] + "," + arr[2];
        });
        JavaPairRDD<String, String> finalratingrdd = ratingrdd.mapToPair(t -> {
            String[] arr = t.split(",");
            return new Tuple2(arr[0], arr[1]);
        });
        JavaPairRDD<String, Iterable<String>> tuplerdd = finalratingrdd.groupByKey();
        tuplerdd.foreach(t -> {
            Iterable<String> getstr = t._2;


            int c = 0;
            float min = 0;
            float max = 0;
            float sum = 0;
            for (String s : getstr) {
                float rating = Float.parseFloat(s);
                if (max < rating)
                    max = rating;
                if (min > rating)
                    min = rating;
                c++;
                sum += rating;


            }
            System.out.println("min" + min);
            System.out.println("max" + max);
            float avg = sum / c;
            System.out.println("avg" + avg);

        });

    }
}



