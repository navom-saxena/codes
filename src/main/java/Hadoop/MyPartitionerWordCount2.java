package Hadoop;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;

/**
 * Created by Navom on 6/24/2017.
 */
public class MyPartitionerWordCount2 extends Partitioner<Text, Text> {
    public int getPartition(Text key, Text value, int NoRed) {
        String keystr = key.toString();
        if (keystr.equals("01")) {
            return 0 % NoRed;
        }
        if (keystr.equals("02")) {
            return 1 % NoRed;
        }
        if (keystr.equals("03")) {
            return 2 % NoRed;
        }
        if (keystr.equals("04")) {
            return 3 % NoRed;
        }
        if (keystr.equals("05")) {
            return 4 % NoRed;
        }
        if (keystr.equals("06")) {
            return 5 % NoRed;
        }
        if (keystr.equals("07")) {
            return 6 % NoRed;
        }
        if (keystr.equals("08")) {
            return 7 % NoRed;
        }
        if (keystr.equals("09")) {
            return 8 % NoRed;
        }
        if (keystr.equals("10")) {
            return 9 % NoRed;
        }
        if (keystr.equals("11")) {
            return 10 % NoRed;
        } else {
            return 11 % NoRed;
        }
    }
}
