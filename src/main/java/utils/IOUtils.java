package utils;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.time.Instant;

/*
 * utility class to save string in a file
 * */

class IOUtils {

    static void saveStringToFile(String outputPath, String content) {
        try (PrintWriter outputWriter = new PrintWriter(outputPath + "\\file-" + Instant.now().toEpochMilli() + ".txt")) {
            outputWriter.println(content);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

}
