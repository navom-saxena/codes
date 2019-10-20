package Hadoop;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by Navom on 6/24/2017.
 */
public class MyMapperWordCount2 extends Mapper<LongWritable, Text, Text, DoubleWritable> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] linearr = line.split(",");
        String id = linearr[2];
        Double amount = Double.parseDouble(linearr[3]);
        context.write(new Text(id), new DoubleWritable(amount));
    }
}
