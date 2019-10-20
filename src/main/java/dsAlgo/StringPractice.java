package dsAlgo;

public class StringPractice {

    public static void main(String[] args) {
        printPermutn("abc", "");
    }

    private static void printPermutn(String str, String ans) {

        // If string is empty
        if (str.length() == 0) {
            System.out.println(">>>>>>>>>>>>" + ans + " ");
            return;
        }

        for (int i = 0; i < str.length(); i++) {

            // ith character of str
            char ch = str.charAt(i);

            // Rest of the string after excluding
            // the ith character
            String ros = str.substring(0, i) +
                    str.substring(i + 1);

            // Recurvise call
            System.out.println(ros + " -- " + ans + " -- " + ch);
            printPermutn(ros, ans + ch);
        }
    }
}
