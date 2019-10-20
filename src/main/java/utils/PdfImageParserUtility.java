package utils;

//create an utility of PDF parsing. You can download Reliance and MRF annual report as
// sample PDF.Also, Please consider text extraction from an image in the file.

import org.apache.tika.config.TikaConfig;
import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.apache.tika.parser.ocr.TesseractOCRConfig;
import org.apache.tika.parser.pdf.PDFParserConfig;
import org.apache.tika.sax.BodyContentHandler;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;

// pre-requisite - Download ocr files and put it in a path. That path is 1st arg - ocrPath. Download stable version -
// https://www.google.com/url?q=https://github.com/tesseract-ocr/tesseract/wiki&sa=D&source=hangouts&ust=1568527890113000&usg=AFQjCNHICayJLjWRqC9dRQnt6Mjq_a76Gg

public class PdfImageParserUtility {
    public static void main(String[] args) throws IOException, SAXException, TikaException {
        scanAndSaveText(args[0], args[1], args[2]);
    }

    private static void scanAndSaveText(String ocrPath, String inputPath, String outputPath)
            throws IOException, TikaException, SAXException {
        InputStream pdf = Files.newInputStream(Paths.get(inputPath));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        TikaConfig config = TikaConfig.getDefaultConfig();
        BodyContentHandler handler = new BodyContentHandler(out);
        Parser parser = new AutoDetectParser(config);
        Metadata meta = new Metadata();
        ParseContext parsecontext = new ParseContext();
        PDFParserConfig pdfConfig = new PDFParserConfig();
        pdfConfig.setExtractInlineImages(true);
        TesseractOCRConfig tesserConfig = new TesseractOCRConfig();
        tesserConfig.setLanguage("eng");
        tesserConfig.setTesseractPath(ocrPath);
        parsecontext.set(Parser.class, parser);
        parsecontext.set(PDFParserConfig.class, pdfConfig);
        parsecontext.set(TesseractOCRConfig.class, tesserConfig);

        parser.parse(pdf, handler, meta, parsecontext);
        String outputTextString = new String(out.toByteArray(), Charset.defaultCharset());
        IOUtils.saveStringToFile(outputPath, outputTextString);
    }
}