package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Created by Navom on 6/18/2017.
 */
public class MyDriverMapSideJoinWithCleanup {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Map Side Join with cleanup");
        job.setJarByClass(MyDriverMapSideJoinWithCleanup.class);
        job.setReducerClass(MyReducerMapSideJoinWithCleanup.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(1);
//        FileInputFormat.addInputPath(job,new Path(args[0]));
        MultipleInputs.addInputPath(job, new Path(args[0]), TextInputFormat.class, MyMapperMapSideJoinWithCleanup.class);
        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, MyMapper1MapSideJoinWithCleanup.class);
//        try
//        {
//            DistributedCache.addCacheFile(new URI("/dir1/agegroup.dat"),job.getConfiguration());
//        }
//        catch (URISyntaxException e)
//        {
//            e.printStackTrace();
//        }
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
