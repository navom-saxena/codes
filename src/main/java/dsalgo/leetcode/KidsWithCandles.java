package dsalgo.leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class KidsWithCandles {

    public static void main(String[] args) {
        kidsWithCandies(new int[]{12,1,12}, 10).forEach(System.out::println);
    }

    public static List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int max = Integer.MIN_VALUE;
        for (int candy: candies) {
            if (candy > max) {
                max = candy;
            }
        }
        List<Boolean> flagArr = new ArrayList<>();
        for (int i = 0; i < candies.length; i++) {
            flagArr.add(i, candies[i] + extraCandies >= max);
        }
        return flagArr;
    }

}
