package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.*;
import java.net.URI;

/**
 * Created by Navom on 7/17/2017.
 */
public class MyMapperUnstructuredDataAnalysis3 extends Mapper<Text, BytesWritable, BytesWritable, Text> {
    public void map(Text key, BytesWritable value, Context context) throws IOException, InterruptedException {
        String name = value.toString();
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(name), configuration);
        FSDataInputStream in = fs.open(new Path(name));
        BufferedReader reader = new BufferedReader(new FileReader(new File(name)));
        ByteArrayOutputStream bout = new ByteArrayOutputStream();
        byte[] b = new byte[1024 * 1024];
        int a;
        int i = 0;
        while ((a = reader.read()) != -1) {
            b[i] = (byte) a;
            i++;
        }
        IOUtils.copyBytes(in, bout, 1024 * 1024, true);
        bout.write(b);
        IOUtils.closeStream(in);
        while ((in.read(b, 0, b.length) >= 0)) {
            bout.write(b);
        }
        context.write(new BytesWritable(value.getBytes()), new Text(b));
        context.write(value, key);
    }
}
