package Hadoop;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyReducerMovieLensA3 extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
        int mcount = 0;
        int fcount = 0;

        for (Text a : value) {
            String astr = a.toString();
            if (astr.equals("Male")) {
                mcount++;
            } else {
                fcount++;
            }
        }
        String ratio = "male - " + mcount + " " + "female - " + fcount;
        context.write(new Text(key), new Text(ratio));
    }
}
