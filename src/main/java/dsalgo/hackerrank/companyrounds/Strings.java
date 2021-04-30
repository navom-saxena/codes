package dsalgo.hackerrank.companyrounds;

public class Strings {

    public static void main(String[] args) {

        compressString("GGGGGrrrrrrt");
        encryptString("open","1234");
        decryptString("oppeeennnn", "1234");

    }

    public static void compressString(String inputString) {
        int length = inputString.length();
        int count = 1;
        System.out.println(length);
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < length - 1; i++) {
            if (inputString.charAt(i) == inputString.charAt(i + 1)) {
                count++;
            } else {
                b.append(inputString.charAt(i));
                b.append(count);
                count = 1;
            }
        }
        if (length > 0) {
            b.append(inputString.charAt(length - 1));
            b.append(count);
        }
        System.out.println(b.toString());
    }

    public static void encryptString(String message, String key) {
        String keyTrimmed = key.substring(0, Math.min(message.length(), key.length()));
        System.out.println(keyTrimmed);
        StringBuilder sb = new StringBuilder();
        int messageIndex = 0;
        for (int i = 0; i < keyTrimmed.length(); i++) {
            String keyAtIndex = Character.toString(keyTrimmed.charAt(i));
            int count = Integer.parseInt(keyAtIndex);
            while (count != 0) {
                sb.append(message.charAt(messageIndex));
                count--;
            }
            messageIndex++;
        }
        for (int j = 0; j < (message.length() - keyTrimmed.length()); j++) {
            sb.append(message.charAt(messageIndex));
            messageIndex++;
        }
        System.out.println(sb.toString());
    }

    public static void decryptString(String message, String key) {
        String keyTrimmed = key.substring(0, Math.min(message.length(), key.length()));
        StringBuilder sb = new StringBuilder();
        int messageIndex = 0;
        for (int i = 0; i < keyTrimmed.length(); i++) {
            if (messageIndex > message.length() -1) {
                break;
            }
            String keyAtIndex = Character.toString(keyTrimmed.charAt(i));
            int count = Integer.parseInt(keyAtIndex);
            sb.append(message.charAt(messageIndex));
            messageIndex += count;
        }
        int diff = message.length() - messageIndex;
        for (int j = 0; j < diff; j++) {
            sb.append(message.charAt(messageIndex));
            messageIndex++;
        }
        System.out.println(sb.toString());
    }

}
