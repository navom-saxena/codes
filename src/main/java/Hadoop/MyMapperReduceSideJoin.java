package Hadoop;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

/**
 * Created by Navom on 7/2/2017.
 */
public class MyMapperReduceSideJoin extends Mapper<LongWritable, Text, Text, Text> {
    HashMap<String, String> hm = new HashMap<String, String>();

    public void setup(Context context) throws IOException {
        Path[] files = DistributedCache.getLocalCacheFiles(context.getConfiguration());
        for (Path singlepath : files) {
            if (singlepath.getName().equals("results.dat")) {
                BufferedReader reader = new BufferedReader(new FileReader(singlepath.toString()));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] lineparts = line.split("\\|\\|");
                    hm.put(lineparts[0], lineparts[1]);
                }
                reader.close();
            }
        }
        if (hm.isEmpty()) {
            throw new IOException("Unable To Load Customer Data.");
        }
    }

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String records = value.toString();
        String[] recordparts = records.split("\\|\\|");
        String number = recordparts[1];
        String name = recordparts[0];
        context.write(new Text(hm.get(number)), new Text(name));
    }
}
