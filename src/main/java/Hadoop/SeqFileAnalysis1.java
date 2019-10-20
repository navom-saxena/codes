package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;

/**
 * Created by Navom on 7/16/2017.
 */
public class SeqFileAnalysis1 {
    public static void main(String[] args) throws IOException {
        String[] abc = {"One, two, buckle my shoe", "Three, four, shut the door", "Five, six, pick up sticks", "Seven, eight, lay them straight", "Nine, ten, a big fat hen"};
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(args[0]), configuration);
        Path path = new Path(args[0]);
        Text key = new Text();
        BytesWritable value = new BytesWritable();
        SequenceFile.Writer writer = SequenceFile.createWriter(fs, configuration, path, key.getClass(), value.getClass());
        BufferedReader reader = new BufferedReader(new FileReader(new File(args[1])));
        int a;
        while ((a = reader.read()) != -1) {
            byte c = (byte) a;
            value = new BytesWritable(new byte[]{c});
            key = new Text(args[1]);
            writer.append(key, value);
        }
        writer.close();
    }
}
