package Hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;

/**
 * Created by Navom on 7/19/2017.
 */
public class PrintContentsOfHdfsFile {
    public static void main(String[] args) throws IOException {
        String inf = args[0];
        String opf = args[1];
        Configuration configuration = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(opf), configuration);
//        OutputStream out = fs.create(new Path(opf),true);
        OutputStream out1 = fs.append(new Path(opf));
        FileInputStream in = new FileInputStream(new File(inf));
        IOUtils.copyBytes(in, out1, 4096, false);
        IOUtils.closeStream(out1);
        in.close();
    }
}
