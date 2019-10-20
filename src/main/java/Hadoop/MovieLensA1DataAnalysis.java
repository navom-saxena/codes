package Hadoop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by Navom on 7/12/2017.
 */
public class MovieLensA1DataAnalysis {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(new File("C:\\Users\\Navom\\Documents\\Evaluation\\Movie Data\\MoviesDetails.dat")));
        String l;
        while ((l = reader.readLine()) != null) {
            String[] lineParts = l.split(",");
            if (lineParts.length == 5) {
                String a = lineParts[4];
                if ((!a.isEmpty())) {
                    Integer rating = Integer.parseInt(a);
                    if (rating > 5400) {
                        System.out.println(l.length());
                        System.out.println(rating);
                    }
                }
            }
        }
    }
}
