package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URI;

/**
 * Created by Navom on 7/19/2017.
 */
public class SequenceFileReader {
    public static void main(String[] args) throws IOException {
        String inf = args[0];
        String opf = args[1];
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(inf), configuration);
        BufferedWriter out = new BufferedWriter(new FileWriter(new File(opf)));
        SequenceFile.Reader in = new SequenceFile.Reader(fs, new Path(inf), configuration);
        Writable key = (Writable) ReflectionUtils.newInstance(in.getKeyClass(), configuration);
        Writable value = (Writable) ReflectionUtils.newInstance(in.getValueClass(), configuration);
        while (in.next(key, value)) {
            out.write(key + " " + value);
        }
        IOUtils.closeStream(in);
        out.close();
    }
}
