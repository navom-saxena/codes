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
public class SeqFileAnalysis2 {
    public static void main(String[] args) throws IOException {
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(args[0]), configuration);
        Path path = new Path(args[0]);
        Text key = new Text();
        BytesWritable value = new BytesWritable();
        SequenceFile.Writer writer = SequenceFile.createWriter(fs, configuration, path, key.getClass(), value.getClass());
        BufferedReader reader = new BufferedReader(new FileReader(new File(args[1])));
        int z;
        int count = 0;
        while ((z = reader.read()) != -1) {
            count++;
            System.out.println(count);
        }
        reader.close();
        BufferedReader reader1 = new BufferedReader(new FileReader(new File(args[1])));
        int a;
        int i = 0;
        byte[] e = new byte[count];
        System.out.println("final count - " + count);
        while ((a = reader1.read()) != -1) {
            System.out.println("hii");
            System.out.println("a - " + a);
            System.out.println("i -" + i);
            byte c = (byte) a;
            e[i] = c;
            i++;
        }
        value = new BytesWritable(e);
        key = new Text(args[1]);
        writer.append(key, value);
        writer.close();
    }
}
