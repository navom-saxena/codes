package Hadoop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Navom on 7/7/2017.
 */
public class TestingTreeMap {
    public static void main(String[] args) {
        TreeMap<String, String> hm = new TreeMap<String, String>();
        try {

            File file = new File("C:\\Users\\Navom\\Documents\\Test.txt");
            FileReader fm = new FileReader(file);
            BufferedReader reader = new BufferedReader(fm);
            String line = reader.readLine();
            while (line != null) {
                String[] lineParts = line.split("\\|\\|");
                hm.put(lineParts[0], lineParts[1]);
                line = reader.readLine();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        for (Map.Entry<String, String> me : hm.entrySet()) {
            System.out.print(me.getKey());
            System.out.print(" ");
            System.out.print(me.getValue());
            System.out.println();
        }
    }
}
