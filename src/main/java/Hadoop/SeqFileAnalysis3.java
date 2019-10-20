package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;

import java.io.IOException;
import java.net.URI;

/**
 * Created by Navom on 7/16/2017.
 */
public class SeqFileAnalysis3 {
    public static void main(String[] args) throws IOException {
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(args[0]), configuration);
        Path path = new Path(args[0]);
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, configuration);
        Writable key = (Writable) ReflectionUtils.newInstance(reader.getKeyClass(), configuration);
        Writable value = (Writable) ReflectionUtils.newInstance(reader.getValueClass(), configuration);
        while (reader.next(key, value)) {
            System.out.print(key);
            System.out.print(" ");
            System.out.print(value);
            System.out.println();
        }
        reader.close();
    }
}
