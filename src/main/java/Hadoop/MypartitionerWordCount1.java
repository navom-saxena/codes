package Hadoop;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;


/**
 * Created by Navom on 6/26/2017.
 */
public class MypartitionerWordCount1 extends Partitioner<Text, Text> {
    public int getPartition(Text key, Text value, int NoRed) {
        String keystr = key.toString();
        if (keystr.equals("ERROR")) {
            return 0 % NoRed;
        }
        if (keystr.equals("DEBUG")) {
            return 1 % NoRed;
        }
        if (keystr.equals("TRACE")) {
            return 2 % NoRed;
        } else {
            return 3 % NoRed;
        }
    }
}
