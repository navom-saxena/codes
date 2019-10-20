package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import java.io.IOException;
import java.net.URI;

/**
 * Created by Navom on 7/16/2017.
 */
public class SeqFileAnalysis4 {
    public static void main(String[] args) throws IOException {
        String[] abc = {"One, two, buckle my shoe", "Three, four, shut the door", "Five, six, pick up sticks", "Seven, eight, lay them straight", "Nine, ten, a big fat hen"};
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(args[0]), configuration);
        Path path = new Path(args[0]);
        LongWritable key = new LongWritable();
        Text value = new Text();
        SequenceFile.Writer writer = SequenceFile.createWriter(fs, configuration, path, key.getClass(), value.getClass());
        for (int i = 0; i < abc.length; i++) {
            key.set(i);
            value.set(abc[i]);
            System.out.println(writer.getLength() + " " + key + " " + value);
            writer.append(key, value);
        }
        writer.close();
    }
}
