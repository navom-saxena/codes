package Hadoop;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by Navom on 7/13/2017.
 */
public class MyMapper1MapSideJoinWithCleanup extends Mapper<LongWritable, Text, Text, Text> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String record = value.toString();
        String[] line = record.split(",");
        String id = line[0];
        String name = line[1] + " " + line[2];
        context.write(new Text(id), new Text(name));
    }
}