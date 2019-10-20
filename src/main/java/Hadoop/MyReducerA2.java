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
public class MyReducerA2 extends Reducer<Text, Text, IntWritable, Text> {
    private TreeMap<Double, String> tm = new TreeMap<>(Collections.reverseOrder());

    public void reduce(Text key, Iterable<Text> values, Context context) {
        String movieName = key.toString();
        int useridC = 0;
        double rating = 0;
        for (Text v : values) {
            String value = v.toString();
            String[] valuearr = value.split("-");
            String ratingS = valuearr[1];
            Double ratingI = Double.parseDouble(ratingS);
            rating += ratingI;
            useridC = useridC + 1;
        }
        if (useridC > 40) {
            double avg = rating / useridC;
            tm.put(avg, movieName);
        }
    }

    public void cleanup(Context context) throws IOException, InterruptedException {
        int exitNum = 1;
        for (Map.Entry<Double, String> me : tm.entrySet()) {
            if (exitNum > 20) {
                break;
            }
            String movID = me.getValue();
            String ratS = me.getKey().toString();
            String outputVal = movID + "    " + ratS;
            context.write(new IntWritable(exitNum), new Text(outputVal));
            exitNum++;
        }
    }
}
