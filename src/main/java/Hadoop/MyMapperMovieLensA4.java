package Hadoop;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
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
public class MyMapperMovieLensA4 extends Mapper<LongWritable, Text, Text, DoubleWritable> {
    private HashMap<String, String> hm = new HashMap<>();

    public void setup(Context context) throws IOException {
        Path[] files = DistributedCache.getLocalCacheFiles(context.getConfiguration());
        for (Path singlePath : files) {
            if (singlePath.getName().equals("agegroup.dat")) {
                BufferedReader reader = new BufferedReader(new FileReader(singlePath.toString()));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] lineParts = line.split("[^A-Za-z0-9]");
                    hm.put(lineParts[0], lineParts[1]);
                }
                reader.close();
            }
        }
        if (hm.isEmpty()) {
            throw new IOException("Unable To Load Customer Data.");
        }
    }

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] lineparts = line.split(",");
        String[] fPart = lineparts[0].split("[^0-9]");
//            System.out.println(fPart[8]);
        String[] lPart = lineparts[5].split(" {2}");
//            System.out.println(lPart[1]);
        double d = Double.parseDouble(lPart[1]);
        context.write(new Text(hm.get(fPart[8])), new DoubleWritable(d));
    }
}

