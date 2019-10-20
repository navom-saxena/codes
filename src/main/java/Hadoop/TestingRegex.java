package Hadoop;

import java.util.StringJoiner;

/**
 * Created by Navom on 8/23/2017.
 */
public class TestingRegex {
    public static void main(String[] args) {
        String input = "{\"Age\": 73,\"Education\": \" High school graduate\",\"MaritalStatus\": \" Widowed\",\"Gender\": \" Female\",\"TaxFilerStatus\": \" Nonfiler\",\"Income\":  1700.09,\"Parents\": \" Not in universe\",\"CountryOfBirth\": \" United-States\",\"Citizenship\": \" Native- Born in the United States\",\"WeeksWorked\":  0}";
        String inp1 = input.replace("\"", "");
        String inp2 = inp1.replace("{", "");
        String inp3 = inp2.replace("}", "");
        String inp4 = inp3.replace("  ", "").trim();
        String[] strarr = inp4.split(",");
        StringJoiner sj = new StringJoiner(",");
        for (String s : strarr) {
            String d1 = s.split(":")[1].trim();
            System.out.println(d1);
            sj.add(d1);
        }
        System.out.println(sj);
        System.out.println(sj.toString());
    }
}
