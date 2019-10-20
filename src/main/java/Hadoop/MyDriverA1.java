package Hadoop;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Created by Navom on 11/19/2017.
 */

/*CODING PROBLEM
* The problem mentioned below revolves around movies dataset. The dataset contains 4 files which are follows,
*
* File Name
* Description / Schema
* movies.dat
* MovieID - Title - Genres
* ratings.dat
* UserID - MovieID - Rating - Timestamp
* users.dat
* UserID - Gender - Age - Occupation - ZipCode
* README
* Additional information / explanation about the above three files
* The dataset can be downloaded from the link : http://grouplens.org/datasets/movielens/1m/

* Please submit your code in
* Hadoop Map Reduce OR JAVA OR Spark scala ONLY
* DO NOT use HIVE QUERIES or SPARK SQL] to solve below questions:
* 1. Top ten most viewed movies with their movies Name (Ascending or Descending order)
* 2. Top twenty rated movies (Condition: The movie should be rated/viewed by at least 40 users)
* 3. We wish to know genres ranked by Average Rating , for each profession and age group.
* The age groups to be considered are: 18-35, 36-50 and 50+.
* We wish to know how have the genres ranked by Average Rating, for each profession and age
* You need to formulate results in following table:

* This is a table where groups refer to row. eg

                                 Genre Ranking by Avg. Rating
* Occupation      Age Group        Rank 1      Rank 2      Rank 3      Rank 4      Rank 5
* Programmer      18-35            Action      Suspense    Thriller    Romance     Horror
* Programmer      36-50            Action      Suspense    Thriller    Romance     Horror
* Programmer      50+              Action      Suspense    Thriller    Romance     Horror
* Farmer          18-35            Action      Suspense    Thriller    Romance     Horror
* Farmer          36-50            Action      Suspense    Thriller    Romance     Horror

* Note that values populated in following table are just representative, and will change with the actual data.
* The table should be output as a single CSV file, rows sorted by Occupation followed by Age Group.

* MyDriver, MyMapper, MyReducer A1, A2, A3 refer to answers of question 1,2,3

*/

public class MyDriverA1 {
    public static void main(String[] args) throws Exception {
        Configuration configuration = new Configuration();
        Job job = Job.getInstance(configuration);
        job.setJarByClass(MyDriverA1.class);
        job.setMapperClass(MyMapperA1.class);
        job.setReducerClass(MyReducerA1.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setNumReduceTasks(1);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        job.addCacheFile(new Path("/movies.dat").toUri());
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
