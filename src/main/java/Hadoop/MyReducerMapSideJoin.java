package Hadoop;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyReducerMapSideJoin extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        String rating = "";
        String name = "";
        for (Text singlevalue : values) {
            if (singlevalue.toString().endsWith("_rat")) {
                rating = singlevalue.toString().split("_")[0];
            } else if (singlevalue.toString().endsWith("_name")) {
                name = singlevalue.toString().split("_")[0];
            }
        }
        String OutputString = name + " " + rating;
        context.write(new Text(key), new Text(OutputString));
    }
}
