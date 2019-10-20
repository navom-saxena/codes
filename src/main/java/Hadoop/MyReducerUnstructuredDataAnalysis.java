package Hadoop;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by Navom on 7/17/2017.
 */
public class MyReducerUnstructuredDataAnalysis extends Reducer<BytesWritable, Text, Text, NullWritable> {
    public void reduce(BytesWritable key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
        String names = "";
        for (Text v : value) {
            names = names + v.toString();
        }
        context.write(new Text(names), NullWritable.get());
    }
}
