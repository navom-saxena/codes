package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyReducerMovieLensA1 extends Reducer<Text, IntWritable, Text, IntWritable> {
    private int gcount = 0;

    public void reduce(Text key, Iterable<IntWritable> value, Context context) {
        int count = 0;
        for (IntWritable ignored : value) {
            count++;
        }
        gcount = gcount + count;
    }

    public void cleanup(Context context) throws IOException, InterruptedException {
        context.write(new Text("Number of movies -"), new IntWritable(gcount));
    }
}
