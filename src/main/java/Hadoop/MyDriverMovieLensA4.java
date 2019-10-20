package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Created by Navom on 6/18/2017.
 */
public class MyDriverMovieLensA4 {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Movie lens A4");
        job.setJarByClass(MyDriverMovieLensA4.class);
        job.setMapperClass(MyMapperMovieLensA4.class);
        job.setReducerClass(MyReducerMovieLensA4.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        job.setNumReduceTasks(1);
        FileInputFormat.addInputPath(job, new Path(args[0]));
//        MultipleInputs.addInputPath(job, new Path(args[0]),TextInputFormat.class, MyMapper.class);
//        MultipleInputs.addInputPath(job, new Path(args[1]),TextInputFormat.class, MyMapper1.class);
        job.addCacheFile(new Path("/dir1/agegroup.dat").toUri());

        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
