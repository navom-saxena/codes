package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyReducerMyDriverMovieLensA2 extends Reducer<IntWritable, Text, IntWritable, IntWritable> {
    public void reduce(IntWritable key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
        int count = 0;
        for (Text ignored : value) {
            count++;
        }
        context.write(key, new IntWritable(count));
    }
}
