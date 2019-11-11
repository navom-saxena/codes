package dsAlgo.hackerRank.datastructures;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Arrays {

    public static void main(String[] args) {
//        System.out.println(Arrays.toString(reverseArray(new int[]{1, 4, 3, 2})));
//
//        System.out.println(hourGlassSum(new int[][]{{-9, -9, -9, 1, 1, 1}, {0, -9, 0, 4, 3, 2},
//                {-9, -9, -9, 1, 2, 3}, {0, 0, 8, 6, 6, 0},
//                {0, 0, 0, -2, 0, 0}, {0, 0, 1, 2, 4, 0}}));
//        int[][] l = new int[][]{{1, 0, 5}, {1, 1, 7}, {1, 0, 3}, {2, 1, 0}, {2, 1, 1}};
//        List<List<Integer>> l1 =
//                Arrays.stream(l).map(x -> Arrays.stream(x).boxed().collect(Collectors.toList())).collect(Collectors.toList());
//        dynamicArray(2, l1);
//
//        int [] leftRotatedArr = leftRotation(4, new int[]{1,2,3,4,5});
//        Arrays.stream(leftRotatedArr).boxed().forEach(x -> System.out.print(x + " "));
//        System.out.println();

//        int[] frequencyArr = matchingStrings(new String[]{"def", "de", "fgh"}, new String[]{"de", "lmn", "fgh"});
//        Arrays.stream(frequencyArr).boxed().forEach(x -> System.out.print(x + " "));

//        System.out.println(arrayManipulation(5, new int[][]{{1, 2, 100}, {1, 5, 100}, {1, 4, 100}}));
    }

    private static int[] reverseArray(int[] a) {
        int length = a.length;
        int[] reversedArr = new int[length];
        for (int i = length - 1, j = 0; i >= 0 || j < length - 1; i--, j++) {
            reversedArr[j] = a[i];
        }
        return reversedArr;
    }

    private static int hourGlassSum(int[][] arr) {
        int length = arr.length;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < length - 2; i++) {
            for (int j = 0; j < length - 2; j++) {
                int sum = arr[i][j] + arr[i][j + 1] + arr[i][j + 2]
                        + arr[i + 1][j + 1]
                        + arr[i + 2][j] + arr[i + 2][j + 1] + arr[i + 2][j + 2];
                if (sum > max) {
                    max = sum;
                }
            }
        }
        return max;
    }

    private static List<Integer> dynamicArray(int n, List<List<Integer>> queries) {
        List<List<Integer>> seqList = new ArrayList<>();
        List<Integer> lastAnswerList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            seqList.add(new ArrayList<>());
        }
        int lastAnswer = 0;
        for (List<Integer> query : queries) {
            List<Integer> seq = seqList.get((query.get(1) ^ lastAnswer) % n);
            System.out.println(seq);
            if (query.get(0) == 1) {
                seq.add(query.get(2));
            } else if (query.get(0) == 2) {
                lastAnswer = seq.get(query.get(2) % seq.size());
                lastAnswerList.add(lastAnswer);
            }
        }
        return lastAnswerList;
    }

    private static int[] leftRotation(int d, int[] a) {
        int i = 0;
        int j = d - 1;
        while (i < j) {
            int temp = a[i];
            a[i] = a[j];
            a[j] = temp;
            i++;
            j--;
        }
        i = d;
        j = a.length - 1;
        while (i < j) {
            int temp = a[i];
            a[i] = a[j];
            a[j] = temp;
            i++;
            j--;
        }

        i = 0;
        j = a.length - 1;
        while (i < j) {
            int temp = a[i];
            a[i] = a[j];
            a[j] = temp;
            i++;
            j--;
        }
        return a;
    }

    private static int[] matchingStrings(String[] strings, String[] queries) {
        Map<String, Integer> hm = new HashMap<>();
        int[] frequencyArr = new int[queries.length];
        for (String string : strings) {
            hm.merge(string, 1, Integer::sum);
        }
        for (int i = 0; i < queries.length; i++) {
            frequencyArr[i] = hm.getOrDefault(queries[i], 0);
        }
        return frequencyArr;
    }

    private static long arrayManipulation(int n, int[][] queries) {
        long[] diff = new long[n + 1];
        long[] arr = new long[n];
        for (int[] query : queries) {
            diff[query[0] - 1] += query[2];
            diff[query[1]] += (-query[2]);
        }

        arr[0] = diff[0];
        long max = arr[0];
        for (int j = 1; j < arr.length; j++) {
            arr[j] = arr[j - 1] + diff[j];
            if (arr[j] > max) {
                max = arr[j];
            }
        }
        return max;
    }

}
