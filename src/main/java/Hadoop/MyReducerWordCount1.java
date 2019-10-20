package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Navom on 6/26/2017.
 */
public class MyReducerWordCount1 extends Reducer<Text, Text, Text, IntWritable> {
    private TreeMap<Integer, String> tm = new TreeMap<>(Collections.reverseOrder());

    public void reduce(Text key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
        String keystr = key.toString();
        int count = 0;
        for (Text v : value) {
            count++;
        }
        tm.put(count, keystr);
    }

    public void cleanup(Context context) throws IOException, InterruptedException {
        Integer key1 = null;
        String value1 = null;
        int count = 0;
        for (Map.Entry<Integer, String> me : tm.entrySet()) {
            key1 = me.getKey();
            value1 = me.getValue();
            if (count == 3) {
                break;
            }
            context.write(new Text(value1), new IntWritable(key1));
            count++;
        }
    }
}
