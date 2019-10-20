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
import java.util.Map;

/**
 * Created by Navom on 11/19/2017.
 */
public class MyMapperA3 extends Mapper<LongWritable, Text, Text, Text> {
    private HashMap<Integer, String> hm1 = new HashMap<>();
    private HashMap<Integer, String> hm2 = new HashMap<>();
    private HashMap<Integer, Integer> hm3 = new HashMap<>();
    private HashMap<Integer, String> hm4 = new HashMap<>();

    public void setup(Context context) throws IOException {
        Path[] files = DistributedCache.getLocalCacheFiles(context.getConfiguration());
        for (Path singlepath : files) {
            if (singlepath.getName().equals("movies.dat")) {
                BufferedReader br = new BufferedReader(new FileReader(singlepath.toString()));
                String line;
                while ((line = br.readLine()) != null) {
                    String[] lineParts = line.split("::");
                    Integer movid = Integer.parseInt(lineParts[0]);
                    String genres = lineParts[2];
                    hm1.put(movid, genres);
                }
                if (hm1.isEmpty()) {
                    throw new IOException("Hashmap1 empty");
                }
            }
            if (singlepath.getName().equals("ratings.dat")) {
                BufferedReader br1 = new BufferedReader(new FileReader(singlepath.toString()));
                String line;
                while ((line = br1.readLine()) != null) {
                    String[] lineArr = line.split("::");
                    Integer movid = Integer.parseInt(lineArr[1]);
                    String ratin = lineArr[2];
                    hm2.put(movid, ratin);
                }
                if (hm2.isEmpty()) {
                    throw new IOException("hm2 is empty");
                }
            }
            if (singlepath.getName().equals("ratings.dat")) {
                BufferedReader br2 = new BufferedReader(new FileReader(singlepath.toString()));
                String line;
                while ((line = br2.readLine()) != null) {
                    String[] lineArr = line.split("::");
                    Integer movid = Integer.parseInt(lineArr[1]);
                    Integer uid = Integer.parseInt(lineArr[0]);
                    hm3.put(movid, uid);
                }
                if (hm3.isEmpty()) {
                    throw new IOException("hm2 is empty");
                }
            }
        }
        for (Map.Entry<Integer, Integer> me : hm3.entrySet()) {
            hm4.put(me.getValue(), hm1.get(me.getKey()) + ">" + hm2.get(me.getKey()));
        }
        if (hm4.isEmpty()) {
            throw new IOException("hm4 is empty");
        }
    }

    public void map(LongWritable key, Text values, Context context) throws NullPointerException {
        try {
            String v = values.toString();
            String[] vArr = v.split("::");
            Integer userid = Integer.parseInt(vArr[0]);
            String age = vArr[2];
            String occup = vArr[3];
            String agegrp = "";
            if (age.equals("18") || age.equals("25")) {
                agegrp = occup + ">" + "A1";
                String mapVal = hm4.get(userid);
                if (mapVal.isEmpty()) {
                    throw new NullPointerException("empty mapVAl");
                }
                context.write(new Text(agegrp), new Text(mapVal));
            }
            if (age.equals("35") || age.equals("45")) {
                agegrp = occup + ">" + "A2";
                String mapVal = hm4.get(userid);
                if (mapVal.isEmpty()) {
                    throw new NullPointerException("empty mapVAl");
                }
                context.write(new Text(agegrp), new Text(mapVal));
            }
            if (age.equals("50") || age.equals("56")) {
                agegrp = occup + ">" + "A3";
                String mapVal = hm4.get(userid);
                if (mapVal.isEmpty()) {
                    throw new NullPointerException("empty mapVAl");
                }
                context.write(new Text(agegrp), new Text(mapVal));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
