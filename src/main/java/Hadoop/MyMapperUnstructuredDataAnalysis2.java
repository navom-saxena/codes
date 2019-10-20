package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URI;

/**
 * Created by Navom on 7/17/2017.
 */
public class MyMapperUnstructuredDataAnalysis2 extends Mapper<LongWritable, Text, BytesWritable, Text> {
    public void map(LongWritable key, Text value, Context context) throws FileNotFoundException, IOException, InterruptedException {
        String name = value.toString();
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(name), configuration);
        FSDataInputStream in = fs.open(new Path(name));
        ByteArrayOutputStream bout = new ByteArrayOutputStream();
        byte[] b = new byte[1024 * 1024];
        while ((in.read(b, 0, b.length) >= 0)) {
            bout.write(b);
        }
        context.write(new BytesWritable(b), new Text(value));
    }
}
