package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;

/**
 * Created by Navom on 7/19/2017.
 */
public class SequenceFileReader2 {
    public static void main(String[] args) throws IOException {
        String inf = args[0];
        String opf = args[1];
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(opf), configuration);
        BufferedReader in = new BufferedReader(new FileReader(new File(inf)));
        LongWritable key = new LongWritable();
        Text value = new Text();
        SequenceFile.Writer writer = SequenceFile.createWriter(fs, configuration, new Path(opf), key.getClass(), value.getClass());
        String a;
        int i = 0;
        while ((a = in.readLine()) != null) {
            key.set(i);
            value.set(a);
            i++;
            writer.append(key, value);
        }
        IOUtils.closeStream(writer);
        in.close();
    }
}
