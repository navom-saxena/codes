package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Created by Navom on 6/18/2017.
 */
public class MyDriverMovieLensA2 {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Movie Lens A2");
        job.setJarByClass(MyDriverMovieLensA2.class);
        job.setMapperClass(MyMapperMyDriverMovieLensA2.class);
        job.setReducerClass(MyReducerMyDriverMovieLensA2.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);
        job.setNumReduceTasks(1);
        FileInputFormat.addInputPath(job, new Path(args[0]));
//        MultipleInputs.addInputPath(job, new Path(args[0]),TextInputFormat.class, MyMapper.class);
//        MultipleInputs.addInputPath(job, new Path(args[1]),TextInputFormat.class, MyMapper1.class);
//        try
//        {
//            DistributedCache.addCacheFile(new URI("/dir1/agegroup.dat"),job.getConfiguration());
//        }
//        catch (URISyntaxException e)
//        {
//            e.printStackTrace();
//        }
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
