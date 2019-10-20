package Hadoop;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyMapperReduceSideJoinWithPartitioner extends Mapper<LongWritable, Text, Text, Text> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String records = value.toString();
        String[] recordsarr = records.split(" ");
        String name = recordsarr[0];
        String ratings = recordsarr[1];
        context.write(new Text(ratings), new Text(name));
    }
}
