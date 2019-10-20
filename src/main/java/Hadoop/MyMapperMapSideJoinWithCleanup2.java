package Hadoop;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyMapperMapSideJoinWithCleanup2 extends Mapper<LongWritable, Text, Text, Text> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String records = value.toString();
        String[] line = records.split(",");
        String id = line[2];
        String am = line[3];
        context.write(new Text(id), new Text(am));
    }
}

