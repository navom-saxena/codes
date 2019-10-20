package Hadoop;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/**
 * Created by Navom on 11/19/2017.
 */
public class MyReducerA3 extends Reducer<Text, Text, Text, Text> {
    private TreeMap<Double, String> tm = new TreeMap<>(Collections.reverseOrder());

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        String k = key.toString();
        String[] kArr = k.split(">");
        String occ = kArr[0];
        String agegrp = "";
        String occupation = "";
        if (occ.equals("0")) {
            occupation = "other";
        }
        if (occ.equals("1")) {
            occupation = "academic/educator";
        }
        if (occ.equals("2")) {
            occupation = "artist";
        }
        if (occ.equals("3")) {
            occupation = "clerical/admin";
        }
        if (occ.equals("4")) {
            occupation = "college/grad";
        }
        if (occ.equals("5")) {
            occupation = "customer service";
        }
        if (occ.equals("6")) {
            occupation = "doctor/health care";
        }
        if (occ.equals("7")) {
            occupation = "executive/managerial";
        }
        if (occ.equals("8")) {
            occupation = "farmer";
        }
        if (occ.equals("9")) {
            occupation = "homemaker";
        }
        if (occ.equals("10")) {
            occupation = "K-12 student";
        }
        if (occ.equals("11")) {
            occupation = "lawyer";
        }
        if (occ.equals("12")) {
            occupation = "programmer";
        }
        if (occ.equals("13")) {
            occupation = "retired";
        }
        if (occ.equals("14")) {
            occupation = "sales/marketing";
        }
        if (occ.equals("15")) {
            occupation = "scientist";
        }
        if (occ.equals("16")) {
            occupation = "self-employed";
        }
        if (occ.equals("17")) {
            occupation = "technician/engineer";
        }
        if (occ.equals("18")) {
            occupation = "tradesman/craftsman";
        }
        if (occ.equals("19")) {
            occupation = "unemployed";
        }
        if (occ.equals("20")) {
            occupation = "writer";
        }
        if (kArr[1].equals("A1")) {
            agegrp = "18-35";
        }
        if (kArr[1].equals("A2")) {
            agegrp = "36-50";
        }
        if (kArr[1].equals("A3")) {
            agegrp = "50+";
        }
        int Action_count = 0;
        double Action_avg = 0;
        int Adventure_count = 0;
        double Adventure_avg = 0;
        int Animation_count = 0;
        double Animation_avg = 0;
        int Children_count = 0;
        double Children_avg = 0;
        int Comedy_count = 0;
        double Comedy_avg = 0;
        int Crime_count = 0;
        double Crime_avg = 0;
        int Documentary_count = 0;
        double Documentary_avg = 0;
        int Drama_count = 0;
        double Drama_avg = 0;
        int Fantasy_count = 0;
        double Fantasy_avg = 0;
        int noir_count = 0;
        double noir_avg = 0;
        int Horror_count = 0;
        double Horror_avg = 0;
        int Musical_count = 0;
        double Musical_avg = 0;
        int Mystery_count = 0;
        double Mystery_avg = 0;
        int Romance_count = 0;
        double Romance_avg = 0;
        int Sci_count = 0;
        double Sci_avg = 0;
        int Thriller_count = 0;
        double Thriller_avg = 0;
        int war_count = 0;
        double war_avg = 0;
        int western_count = 0;
        double western_avg = 0;
        for (Text v : values) {
            String value = v.toString();
            String[] valueArr = value.split(">");
            String genreCombined = valueArr[0];
            int rating = Integer.parseInt(valueArr[1]);
            String[] genreCombinedArr = genreCombined.split("\\|");
            for (String rating1 : genreCombinedArr) {
                if (rating1.equals("Action")) {
                    Action_avg = Action_avg + rating;
                    Action_count = Action_count + 1;
                }
                if (rating1.equals("Adventure")) {
                    Adventure_avg = Adventure_avg + rating;
                    Adventure_count = Adventure_count + 1;
                }
                if (rating1.equals("Animation")) {
                    Animation_avg = Animation_avg + rating;
                    Animation_avg = Animation_avg + 1;
                }
                if (rating1.equals("Children's")) {
                    Children_avg = Children_avg + rating;
                    Children_count = Children_count + 1;
                }
                if (rating1.equals("Comedy")) {
                    Comedy_avg = Comedy_avg + rating;
                    Comedy_count = Comedy_count + 1;
                }
                if (rating1.equals("Crime")) {
                    Crime_avg = Crime_avg + rating;
                    Crime_count = Crime_count + 1;
                }
                if (rating1.equals("Documentary")) {
                    Documentary_avg = Documentary_avg + rating;
                    Documentary_count = Documentary_count + 1;
                }
                if (rating1.equals("Drama")) {
                    Drama_avg = Drama_avg + rating;
                    Drama_count = Drama_count + 1;
                }
                if (rating1.equals("Fantasy")) {
                    Fantasy_avg = Fantasy_avg + rating;
                    Fantasy_count = Fantasy_count + 1;
                }
                if (rating1.equals("Film-Noir")) {
                    noir_avg = noir_avg + rating;
                    noir_count = noir_count + 1;
                }
                if (rating1.equals("Horror")) {
                    Horror_avg = Horror_avg + rating;
                    Horror_count = Horror_count + 1;
                }
                if (rating1.equals("Musical")) {
                    Musical_avg = Musical_avg + rating;
                    Musical_count = Musical_count + 1;
                }
                if (rating1.equals("Mystery")) {
                    Mystery_avg = Mystery_avg + rating;
                    Mystery_count = Mystery_count + 1;
                }
                if (rating1.equals("Romance")) {
                    Romance_avg = Romance_avg + rating;
                    Romance_count = Romance_count + 1;
                }
                if (rating1.equals("Sci-Fi")) {
                    Sci_avg = Sci_avg + rating;
                    Sci_count = Sci_count + 1;
                }
                if (rating1.equals("Thriller")) {
                    Thriller_avg = Thriller_avg + rating;
                    Thriller_count = Thriller_count + 1;
                }
                if (rating1.equals("War")) {
                    war_avg = war_avg + rating;
                    war_count = war_count + 1;
                }
                if (rating1.equals("Western")) {
                    western_avg = western_avg + rating;
                    western_count = western_count + 1;
                }
            }
        }
        double Action_r = Action_avg / Action_count;
        tm.put(Action_r, "Action");
        double Adventure_r = Adventure_avg / Adventure_count;
        tm.put(Adventure_r, "Adventure");
        double Animation_r = Animation_avg / Animation_count;
        tm.put(Animation_r, "Animation");
        double Children_r = Children_avg / Children_count;
        tm.put(Children_r, "Children");
        double Comedy_r = Comedy_avg / Comedy_count;
        tm.put(Comedy_r, "Comedy");
        double Crime_r = Crime_avg / Crime_count;
        tm.put(Crime_r, "Crime");
        double Docu_r = Documentary_avg / Documentary_count;
        tm.put(Docu_r, "Documentary");
        double Drama_r = Drama_avg / Drama_count;
        tm.put(Drama_r, "Drama");
        double Fantasy_r = Fantasy_avg / Fantasy_count;
        tm.put(Fantasy_r, "Fantasy");
        double noir_r = noir_avg / noir_count;
        tm.put(noir_r, "Film-Noir");
        double Horror_r = Horror_avg / Horror_count;
        tm.put(Horror_r, "Horror");
        double Musical_r = Musical_avg / Musical_count;
        tm.put(Musical_r, "Musical");
        double Mystery_r = Mystery_avg / Mystery_count;
        tm.put(Mystery_r, "Mystery");
        double Romance_r = Romance_avg / Romance_count;
        tm.put(Romance_r, "Romance");
        double sci_r = Sci_avg / Sci_count;
        tm.put(sci_r, "Sci-Fi");
        double Thriller_r = Thriller_avg / Thriller_count;
        tm.put(Thriller_r, "Thriller");
        double War_r = war_avg / war_count;
        tm.put(War_r, "War");
        double West_r = western_avg / western_count;
        tm.put(West_r, "Western");

        StringBuilder top5 = new StringBuilder();
        int exitnum = 1;
        for (Map.Entry me1 : tm.entrySet()) {
            if (exitnum > 5) {
                break;
            }
            top5.append("    ").append(me1.getValue());
            exitnum++;
        }
        String fullStrKey = occupation + "    " + agegrp;
        String fullStrVal = top5.toString();
        context.write(new Text(fullStrKey), new Text(fullStrVal));
    }

    public void cleanup(Context context) {

    }
}
