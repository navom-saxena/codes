package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyMapperMovieLensA1 extends Mapper<LongWritable, Text, Text, IntWritable> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String records = value.toString();
        String[] lineparts = records.split(",");
        context.write(new Text(lineparts[1]), new IntWritable(1));
    }
}

