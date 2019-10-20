package Hadoop;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyMapperMapSideJoin extends Mapper<LongWritable, Text, Text, Text> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String number = value.toString().split("\\|\\|")[0];
        String rating = value.toString().split("\\|\\|")[1] + "_rat";
        context.write(new Text(number), new Text(rating));
    }
}
