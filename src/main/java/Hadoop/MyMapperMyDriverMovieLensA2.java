package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyMapperMyDriverMovieLensA2 extends Mapper<LongWritable, Text, IntWritable, Text> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String records = value.toString();
        String[] lineparts = records.split(",");
        if (lineparts.length == 5) {
            String a = lineparts[1];
            if (!a.isEmpty()) {
                int year = Integer.parseInt(a);
                context.write(new IntWritable(year), new Text("a"));
            }
        }
    }
}

