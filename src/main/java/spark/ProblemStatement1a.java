/*
Movie Lens Use Case using Spark Core API to List all the movies and the number of ratings
 */

package spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import scala.Tuple2;

public class ProblemStatement1a {

    public static void main(String[] str) {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("userrating");
        JavaSparkContext context = new JavaSparkContext(conf);
        JavaRDD<String> rdd1 = context.textFile("/Users/navom_saxena/Downloads/ratings.csv");
        JavaRDD<String> rdd2 = context.textFile("/Users/navom_saxena/Downloads/movies.csv");
        JavaRDD<String> ratingrdd = rdd1.map(new Function<String, String>() {   /*split on , ratings file and takeout movie id */
                                                 public String call(String s) {
                                                     String[] arr = s.split(",");
                                                     return arr[1];
                                                 }
                                             }
        );
        JavaPairRDD finalratingrdd = ratingrdd.mapToPair(t -> new Tuple2(t, 1)).reduceByKey((x, y) -> (int) x + (int) y);  /*every movie id has a single movie name so we got total ratings corresponding to every movie id */
        JavaRDD<String> movierdd = rdd2.map(new Function<String, String>() {    /*split on , movies file and takeout movie id and title */
                                                public String call(String s) {
                                                    String[] arr = s.split(",");
                                                    return arr[0] + "," + arr[1];

                                                }
                                            }
        );
        JavaPairRDD<String, String> finalmovierdd = movierdd.mapToPair(t -> { /*created tuple for movie id and title */
            String[] arr = t.split(",");
            return new Tuple2(arr[0], arr[1]);

        });
        JavaPairRDD<Integer, Tuple2<String, Integer>> combined = finalmovierdd.join(finalratingrdd);  /*joined movie id, cumulative rating with movie id , title */

        combined.foreach(t -> {
            Tuple2<String, Integer> t1 = t._2;
            String title = t1._1;
            Integer count = t1._2;
            System.out.println(title + "" + count);
        });

    }
}


/*
    Movie Lens Use Case using
a.  Spark Core API and then
b.  SPARK SQL
PS1 List all the movies and the number of ratings
PS2 List all the users and the number of ratings they have done for a movie
PS3 List all the Movie IDs which have been rated (Movie Id with at least one user rating it)
PS4	List all the Users who have rated the movies (Users who have rated at least one movie)
PS5	List of all the User with the max ,min ,average ratings they have given against any movie
PS6	List all the Movies with the max ,min, average ratings given by any user



 */