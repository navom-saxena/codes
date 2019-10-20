package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyReducerTransaction8 extends Reducer<Text, IntWritable, Text, IntWritable> {
    private TreeMap<Integer, String> tm = new TreeMap<>(Collections.reverseOrder());

    public void reduce(Text key, Iterable<IntWritable> value, Context context) {
        int count = 0;
        for (IntWritable ignored : value) {
            count++;
        }
        tm.put(count, key.toString());
    }

    public void cleanup(Context context) throws IOException, InterruptedException {
        Integer keytm;
        String valuetm = "";
        int c = 0;
        for (Map.Entry<Integer, String> me : tm.entrySet()) {
            keytm = me.getKey();
            valuetm = me.getValue();
            context.write(new Text(valuetm), new IntWritable(keytm));
            if (c == 3) {
                break;
            }
            c++;
        }
    }
}
