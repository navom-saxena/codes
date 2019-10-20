package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by Navom on 6/18/2017.
 */
public class MyMapperWordCountWithPartitioner extends Mapper<LongWritable, Text, Text, IntWritable> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] lineParts = line.split(" ");
        for (String SingleWord : lineParts) {
            if (SingleWord.equals("nikhil")) {
                Text OutKey = new Text(SingleWord);
                IntWritable One = new IntWritable(1);
                context.write(OutKey, One);
            }
        }
    }
}
