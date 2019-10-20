package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by Navom on 6/18/2017.
 */
public class MyReducerWordCountWithPartitioner extends Reducer<Text, IntWritable, Text, IntWritable> {
    public void reduce(Text key, Iterable<IntWritable> value, Context context) throws IOException, InterruptedException {
        int count = 0;
        for (IntWritable v : value) {
            count = count + v.get();
        }
        context.write(new Text(key), new IntWritable(count));
    }
}
