package dsalgo.hackerrank.companyrounds;

import java.util.*;

public class ArrayManipulation {

    public static void main(String[] args) {
        List<Integer> arr = new ArrayList<>();
        arr.add(1);
        arr.add(2);
        arr.add(1);
        arr.add(1);
        arr.add(2);
        arr.add(3);
        System.out.println(getDistanceMetrics(arr));
    }

    public static List<Long> getDistanceMetricsBrute(List<Integer> arr) {
        List<Long> output = new ArrayList<>();
        for (int i = 0; i < arr.size(); i++) {
            long sum = 0;
            for (int j = 0; j < arr.size(); j++) {
                if (i != j && arr.get(i).equals(arr.get(j))) {
                    sum += Math.abs(i - j);
                }
            }
            output.add(sum);
        }
        return output;
    }

    public static List<Long> getDistanceMetrics(List<Integer> arr) {
        Map<Integer, ArrayList<Integer>> m = new HashMap<>();
        for (int i = 0; i < arr.size(); i++) {
            int value = arr.get(i);
            ArrayList<Integer> indexes;
            if (m.containsKey(value)) {
                indexes = m.get(value);
            } else {
                indexes = new ArrayList<>();
            }
            indexes.add(i);
            m.put(value, indexes);
        }
        Map<Integer, Long> fm = new TreeMap<>();
        m.forEach((k, v) -> {
            for (Integer i : v) {
                long sum = 0L;
                for (Integer j : v) {
                    sum += Math.abs(i - j);
                }
                fm.put(i, sum);
            }
        });
        return new ArrayList<>(fm.values());
    }
}
