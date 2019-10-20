package Hadoop;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.StringJoiner;

/**
 * Created by Navom on 8/23/2017.
 */
public class OrganisingData extends Mapper<LongWritable, Text, Text, NullWritable> {
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String input = value.toString();
        String inp1 = input.replace("\"", "");
        String inp2 = inp1.replace("{", "");
        String inp3 = inp2.replace("}", "");
        String inp4 = inp3.replace("  ", "").trim();
        String[] strarr = inp4.split(",");
        StringJoiner sj = new StringJoiner(",");
        for (String s : strarr) {
            String d1 = s.split(":")[1].trim();
            sj.add(d1);
        }
        String op = sj.toString();
        context.write(new Text(op), NullWritable.get());
    }
}
