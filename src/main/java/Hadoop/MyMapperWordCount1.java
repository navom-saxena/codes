package Hadoop;


import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by Navom on 6/26/2017.
 */
public class MyMapperWordCount1 extends Mapper<LongWritable, Text, Text, Text> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] wordarr = line.split(" ");
        String module = wordarr[2];
        String intermword = wordarr[3];
        String word = intermword.split("[^A-Za-z]")[1];
        context.write(new Text(module), new Text(word));
    }
}
