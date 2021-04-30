package dsalgo.hackerrank.companyrounds;

import java.util.*;
import java.util.stream.Collectors;

public class Array {

    public static void main(String[] args) {
//        List<Integer> arr = new ArrayList<>();
//        arr.add(1);
//        arr.add(2);
//        arr.add(1);
//        arr.add(1);
//        arr.add(2);
//        arr.add(3);
//        System.out.println(getDistanceMetrics(arr));
//        System.out.println(shortenString("AAAA"));
//        System.out.println(longestConsecutiveSequence(new int[]{0,3,7,2,5,8,4,6,0,1}));
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

//    ABABCABABCD to AB*C*D
    public static int shortenString(String input) {
        int updatedLength = input.length();
        int startingCounter = 0;
        int preIndex = -1;
        for (int i = 0; i < input.length(); i++) {
            if (i > 0 && input.charAt(i) == input.charAt(startingCounter)) {
                if (preIndex == -1) {
                    preIndex = i;
                }
                startingCounter++;
            } else {
                if (startingCounter == preIndex) {
                    updatedLength -= startingCounter + 1;
                }
                startingCounter = 0;
                preIndex = -1;
            }
        }
        return updatedLength;
    }

//    Given an array of integers, find the length of the longest consecutive elements sequence.
//
//    Input: nums = [100,4,200,1,3,2]
//    Output: 4
//
//    Input: nums = [0,3,7,2,5,8,4,6,0,1]
//    Output: 9

    public static int longestConsecutiveSequence(int [] arr) {
        Set<Integer> hs = Arrays.stream(arr).boxed().collect(Collectors.toCollection(HashSet::new));
        int maxLength = 0;
        int currValue = 0;
        for (int no : arr) {
            if (!hs.contains(no - 1)) {
                while (hs.contains(no)) {
                    no += 1;
                    currValue++;
                    maxLength = Math.max(maxLength, currValue);
                }
                currValue = 0;
            }
        }
        return maxLength;
    }

}
