package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyMapperTransaction8 extends Mapper<LongWritable, Text, Text, IntWritable> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String records = value.toString();
        String[] lineParts = records.split(" ");
        context.write(new Text(lineParts[3]), new IntWritable(1));
    }
}

