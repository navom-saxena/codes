package Hadoop;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyReducerMapSideJoinWithCleanup2 extends Reducer<Text, Text, Text, Text> {
    private TreeMap<Double, String> tm = new TreeMap<>(Collections.reverseOrder());

    public void reduce(Text key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
        String gkey = key.toString();
        String name = "";
        String op = "";
        double sum = 0;
        for (Text v : value) {
            String words = v.toString();
            if (words.contains(" ")) {
                name = words;
            } else {
                double s = Double.parseDouble(words);
                sum = sum + s;
            }
        }
        tm.put(sum, name);
    }

    public void cleanup(Context context) throws IOException, InterruptedException {
        String sumop = "";
        String nameop = "";
        for (Map.Entry<Double, String> me : tm.entrySet()) {
            sumop = me.getKey().toString();
            nameop = me.getValue();
            break;
        }
        String op = nameop + " " + sumop;
        context.write(new Text(nameop), new Text(sumop));
    }
}
