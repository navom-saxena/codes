package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Navom on 11/19/2017.
 */
public class MyReducerA1 extends Reducer<Text, IntWritable, Text, IntWritable> {
    private TreeMap<Integer, String> tm = new TreeMap<>(Collections.reverseOrder());
    private TreeMap<String, Integer> ts = new TreeMap<>();

    public void reduce(Text key, Iterable<IntWritable> values, Context context) {
        String moviename = key.toString();
        int count = 0;
        for (IntWritable value : values) {
            count = count + value.get();
        }
        tm.put(count, moviename);
    }

    public void cleanup(Context context) throws IOException, InterruptedException {
        int exitpoint = 1;
        for (Map.Entry<Integer, String> me : tm.entrySet()) {
            if (exitpoint > 10) {
                break;
            }
            ts.put(me.getValue() + "    " + me.getKey(), exitpoint);
            exitpoint++;
        }
        for (Map.Entry<String, Integer> mee : ts.entrySet()) {
            context.write(new Text(mee.getKey()), new IntWritable(mee.getValue()));
        }
    }
}
