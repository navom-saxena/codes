package Hadoop;

import java.io.*;

/**
 * Created by Navom on 7/16/2017.
 */
public class TestingImagesRead {
    public static void main(String[] args) throws IOException {
        byte b = 0;
        BufferedReader reader = new BufferedReader(new FileReader(new File("C:\\Users\\Navom\\Documents\\Images\\3.jpg")));
        BufferedWriter writer = new BufferedWriter(new FileWriter(new File("C:\\Users\\Navom\\Documents\\a.txt")));
        int a;
        while ((a = reader.read()) != -1) {
            byte d = (byte) a;
            b = d;
            writer.write(d);
        }
    }
}
