package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Created by Navom on 6/18/2017.
 */
public class MyDriverReduceSideJoin {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Reduce Side Join");
        job.setJarByClass(MyDriverReduceSideJoin.class);
        job.setMapperClass(MyMapperReduceSideJoin.class);
        job.setPartitionerClass(MyPartitionerReduceSideJoin.class);
        job.setReducerClass(MyReducerReduceSideJoin.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        job.setNumReduceTasks(2);
        FileInputFormat.addInputPath(job, new Path(args[0]));
//        MultipleInputs.addInputPath(job, new Path(args[0]),TextInputFormat.class, MyMapper.class);
//        MultipleInputs.addInputPath(job, new Path(args[1]),TextInputFormat.class, MyMapper1.class);
        job.addCacheFile(new Path("/dir1/results.dat").toUri());

        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
