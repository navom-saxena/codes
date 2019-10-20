package Hadoop;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Navom on 6/24/2017.
 */
public class MyReducerWordCount2 extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
    private TreeMap<Double, String> tm = new TreeMap<>(Collections.reverseOrder());

    public void reduce(Text key, Iterable<DoubleWritable> value, Context context) throws IOException, InterruptedException {
        double sum = 0;
        for (DoubleWritable a : value) {
            sum = sum + a.get();
        }
        String keystr = key.toString();
        tm.put(sum, keystr);
    }

    public void cleanup(Context context) {
        try {
            Double key1 = 0.0;
            String value1 = null;
            int counter = 0;
            for (Map.Entry<Double, String> me : tm.entrySet()) {
                key1 = me.getKey();
                value1 = me.getValue();
                if (counter == 3) {
                    break;
                }
                context.write(new Text(value1), new DoubleWritable(key1));
                counter++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
