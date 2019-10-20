package Hadoop;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;

/**
 * Created by Navom on 7/6/2017.
 */
public class MyPartitionerReduceSideJoin extends Partitioner<Text, Text> {
    public int getPartition(Text key, Text values, int NoRed) {
        String keystr = key.toString();
        if (keystr.equals("fail")) {
            return 0 % NoRed;
        } else {
            return 1 % NoRed;
        }
    }
}
