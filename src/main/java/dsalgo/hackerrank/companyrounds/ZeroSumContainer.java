package dsalgo.hackerrank.companyrounds;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

// given a number n, create al list with size n when contains numbers whose sum is zero

public class ZeroSumContainer {

    public static void main(String[] args) {
        Arrays.stream(zeroSum(9)).forEach(System.out::println);
    }

    public static int [] zeroSum(int N) {
        List<Integer> arr = new ArrayList<>();
        int oddEven = N % 2;
        int half = N / 2;
        if (oddEven == 0) {
            int counter = 1;
            while (counter <= half) {
                arr.add(counter);
                counter++;
            }
            List<Integer> negativeArr = new ArrayList<>();
            for (int e: arr) {
                negativeArr.add(e * (-1));
            }
            arr.addAll(negativeArr);
        } else {
            if (N == 1) {
                arr.add(0);
            } else {
                int counter = 1;
                while (counter <= half) {
                    arr.add(counter);
                    counter++;
                }
                List<Integer> negativeArr = new ArrayList<>();
                for (int e: arr) {
                    negativeArr.add(e * (-1));
                }
                arr.addAll(negativeArr);
                arr.add(0);
            }
        }
        System.out.println(arr);
        int [] r = new int[arr.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = arr.get(i);
        }
        return r;
    }
}
