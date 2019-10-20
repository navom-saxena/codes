package Hadoop;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

/**
 * Created by Navom on 11/19/2017.
 */
public class MyMapperA1 extends Mapper<LongWritable, Text, Text, IntWritable> {
    private HashMap<Integer, String> hm = new HashMap<>();

    public void setup(Context context) throws IOException {
        Path[] files = DistributedCache.getLocalCacheFiles(context.getConfiguration());
        for (Path singlepath : files) {
            if (singlepath.getName().equals("movies.dat")) {
                BufferedReader br = new BufferedReader(new FileReader(singlepath.toString()));
                String line;
                while ((line = br.readLine()) != null) {
                    String[] lineparts = line.split("::");
                    Integer lineparts1 = Integer.parseInt(lineparts[0]);
                    hm.put(lineparts1, lineparts[1]);
                }
                if (hm.isEmpty()) {
                    throw new IOException("Hashmap empty");
                }
            }
        }
    }

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] linepar = line.split("::");
        String movieidS = linepar[1];
        Integer movieid = Integer.parseInt(movieidS);
        String moviename = hm.get(movieid);
        context.write(new Text(moviename), new IntWritable(1));
    }
}
