package utils;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDDocumentCatalog;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.File;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * parsing pdf and extracting 1st page from each chapter
 */

public class PdfContentsExtractor {

    public static void main(String[] args) throws IOException {

        String inputPath = args[0];
        String outputPath = args[1];
        PDDocument document = PDDocument.load(new File(inputPath));
        PDDocumentCatalog dc = document.getDocumentCatalog();
        int numPages = dc.getPages().getCount();
        System.out.println("num of pages: " + numPages);
        PDFTextStripper stripper = new PDFTextStripper();

        StringBuilder builder = new StringBuilder();

        for (int i = 1; i < 20; i++) {
            stripper.setStartPage(i);
            stripper.setEndPage(i);
            String content = stripper.getText(document);
            Pattern p = Pattern.compile("^\\w*\\s*((?m)Overview.*$)");
            Matcher m = p.matcher(content);
            while (m.find()) {
                Pattern digitsPattern = Pattern.compile("\\s+\\d+");
                Matcher digitsMatcher = digitsPattern.matcher(content);
                while (digitsMatcher.find()) {
                    String digitAsString = digitsMatcher.group();
                    if (!digitAsString.contains("\n")) {
                        System.out.println(digitAsString);
                        int digit = Integer.parseInt(digitAsString.trim());
                        stripper.setStartPage(digit);
                        stripper.setEndPage(digit);
                        String paragraph = stripper.getText(document);
                        String PARAGRAPH_SPLIT_REGEX = "(?!^)(?m)(?=^\\s{4})";
                        String firstPageOfChapter = paragraph.split(PARAGRAPH_SPLIT_REGEX)[0];
                        builder.append(firstPageOfChapter).append("\n");
                        System.out.println(firstPageOfChapter);
                    }
                }
            }
        }
        IOUtils.saveStringToFile(outputPath, builder.toString());
    }
}
