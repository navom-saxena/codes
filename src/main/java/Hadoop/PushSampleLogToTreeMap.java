package Hadoop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Navom on 7/12/2017.
 */
public class PushSampleLogToTreeMap {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(new File("C:\\Users\\Navom\\Documents\\Evaluation\\Server Log Data\\sample.log")));
        String l;
        TreeMap<Integer, String> tm = new TreeMap<>(Collections.reverseOrder());
        while ((l = reader.readLine()) != null) {
            String[] lineParts = l.split(" ");
            tm.put(1, lineParts[3]);
        }
        Integer keytm = 0;
        String valuetm = "";
        int c = 0;
        while (c != 1) {
            for (Map.Entry<Integer, String> me : tm.entrySet()) {
                keytm = me.getKey();
                valuetm = me.getValue();
            }
            c++;
        }
        System.out.println(keytm + " " + valuetm);
    }
}
