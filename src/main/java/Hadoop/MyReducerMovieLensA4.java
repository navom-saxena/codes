package Hadoop;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyReducerMovieLensA4 extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
    public void reduce(Text key, Iterable<DoubleWritable> value, Context context) throws IOException, InterruptedException {
        double sum = 0;
        int count = 0;
        for (DoubleWritable a : value) {
            sum = sum + a.get();
            count++;
        }
        double result = sum / count;
        context.write(new Text(key), new DoubleWritable(result));
    }
}
