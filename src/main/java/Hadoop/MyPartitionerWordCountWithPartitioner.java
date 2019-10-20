package Hadoop;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;


/**
 * Created by Navom on 6/18/2017.
 */
public class MyPartitionerWordCountWithPartitioner extends Partitioner<Text, IntWritable> {
    public int getPartition(Text key, IntWritable value, int NoRed) {
        String inp = key.toString();
        if (inp.startsWith("a")) {
            return 0 % NoRed;
        }
        if (inp.startsWith("s")) {
            return 1 % NoRed;
        }
        if (inp.startsWith("n")) {
            return 2 % NoRed;
        } else {
            return 3 % NoRed;
        }
    }
}
