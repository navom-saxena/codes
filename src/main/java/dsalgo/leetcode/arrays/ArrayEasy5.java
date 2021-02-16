package dsalgo.leetcode.arrays;

import java.util.*;
import java.util.stream.Collectors;

public class ArrayEasy5 {

    public static void main(String[] args) {
//        OrderedStream os = new OrderedStream(5);
//        System.out.println(os.insert(3, "ccccc")); // Inserts (3, "ccccc"), returns [].
//        System.out.println(os.insert(1, "aaaaa")); // Inserts (1, "aaaaa"), returns ["aaaaa"].
//        System.out.println(os.insert(2, "bbbbb")); // Inserts (2, "bbbbb"), returns ["bbbbb", "ccccc"].
//        System.out.println(os.insert(5, "eeeee")); // Inserts (5, "eeeee"), returns [].
//        System.out.println(os.insert(4, "ddddd"));
//        System.out.println(Arrays.deepToString(highFive(new int[][]{{1,91},{1,92}
//        ,{2,93},{2,97},{1,60},{2,77},{1,65},{1,87},{1,100},{2,100},{2,76}})));
//        System.out.println(largestUniqueNumber(new int[]{9,9,8,8}));
//        System.out.println(Arrays.toString(largestSubarray(new int[]{2},2)));
//        System.out.println(shortestDistance(new String[] {"practice", "makes", "perfect", "coding", "makes"},
//        "makes","coding"));
//        System.out.println(twoSumLessThanK(new int[]{10,20,30},15));
//        System.out.println(countElements(new int[]{1,1,2}));
//        System.out.println(bSCielMajority(new int[]{1,1,1,1},0,3,1));
//        System.out.println(isMajorityElement(new int[]{10,100,101,101},101));
//        System.out.println(dietPlanPerformance(new int[]{1,2,3,4,5},1,3,3));
//        System.out.println(stringShift("wpdhhcj" ,new int[][]{{0,7},{1,7},{1,0},{1,3},{0,3},{0,6},{1,2}}));
//        System.out.println(transformArray(new int[]{1,6,3,4,3,5}));
//        System.out.println(findMissingRanges(new int[]{0,1,3,50,75},-2, 99));
    }

//    https://leetcode.com/problems/richest-customer-wealth/

    public static int maximumWealth(int[][] accounts) {
        int maxWealth = 0;
        for (int [] row : accounts) {
            int curr = 0;
            for (int value : row) {
                curr += value;
            }
            maxWealth = Math.max(maxWealth, curr);
        }
        return maxWealth;
    }

//    https://leetcode.com/problems/high-five/

    public static int[][] highFive(int[][] items) {
        int k = 5;
        Map<Integer,PriorityQueue<Integer>> hm = new HashMap<>();
        for (int [] item : items) {
            int id = item[0];
            int marks = item[1];
            PriorityQueue<Integer> minHeap;
            if (hm.containsKey(id)) {
                minHeap = hm.get(id);
            } else {
                minHeap = new PriorityQueue<>();
            }
            if (minHeap.size() < k) {
                minHeap.add(marks);
            } else if (minHeap.peek() < marks) {
                minHeap.remove();
                minHeap.add(marks);
            }
            hm.put(id, minHeap);
        }
        int [][] result = new int[hm.size()][];
        int i = 0;
        for (int key : hm.keySet()) {
            int sum = 0;
            PriorityQueue<Integer> minHeap = hm.get(key);
            while (!minHeap.isEmpty()) {
                sum += minHeap.remove();
            }
            result[i] = new int[]{key, sum / 5};
            i++;
        }
        Arrays.sort(result, Comparator.comparing(a -> a[0]));
        return result;
    }

//    https://leetcode.com/problems/sum-of-digits-in-the-minimum-number/

    public static int sumOfDigits(int[] a) {
        int minValue = Integer.MAX_VALUE;
        for (int num : a) {
            minValue = Math.min(minValue,num);
        }
        int sum = 0;
        while (minValue != 0) {
            int lastDigit = minValue % 10;
            sum += lastDigit;
            minValue /= 10;
        }
        return sum % 2 == 0 ? 1 : 0;
    }

//    https://leetcode.com/problems/largest-unique-number/

    public static int largestUniqueNumber(int[] a) {
        Map<Integer,Integer> hm = new HashMap<>();
        for (int num : a) {
            hm.merge(num, 1, Integer::sum);
        }
        int result = Integer.MIN_VALUE;
        for (int key : hm.keySet()) {
            if (key > result && hm.get(key) == 1) {
                result = key;
            }
        }
        return result == Integer.MIN_VALUE ? -1 : result;
    }

//    https://leetcode.com/problems/fixed-point/

    public static int fixedPoint(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == i) {
                return i;
            }
        }
        return -1;
    }

//    https://leetcode.com/problems/largest-subarray-length-k/

    public static int[] largestSubarray(int[] nums, int k) {
        int n = nums.length;
        int maxStart = 0;
        int maxNoIndex = -1;
        for (int i = 0; i <= n - k; i++) {
            if (maxStart < nums[i]) {
                maxStart = nums[i];
                maxNoIndex = i;
            }
        }
        int [] result = new int[k];
        for (int i = 0; i < k; i++) {
            result[i] = nums[maxNoIndex];
            maxNoIndex++;
        }
        return result;
    }

//    https://leetcode.com/problems/shortest-word-distance/

    public static int bSFloorShortestDistance(List<Integer> arr, int low, int high, int x) {
        if (x < arr.get(low)) {
            return -1;
        } else if (x > arr.get(high)) {
            return arr.get(high);
        }
        int mid = low + (high - low) / 2;
        if (arr.get(mid + 1) > x && arr.get(mid) < x) {
            return arr.get(mid);
        } else if (arr.get(mid) > x) {
            return bSFloorShortestDistance(arr, low, mid - 1, x);
        } else {
            return bSFloorShortestDistance(arr, mid + 1, high, x);
        }
    }

    public static int bSCielShortestDistance(List<Integer> arr, int low, int high, int x) {
        if (x < arr.get(low)) {
            return arr.get(low);
        } else if (x > arr.get(high)) {
            return -1;
        }
        int mid = low + (high - low) / 2;
        if (arr.get(mid + 1) > x && arr.get(mid) < x) {
            return arr.get(mid + 1);
        } else if (low < mid && arr.get(mid) > x && arr.get(mid - 1) < x) {
            return arr.get(mid);
        }
        else if (arr.get(mid) > x) {
            return bSCielShortestDistance(arr, low, mid - 1, x);
        } else {
            return bSCielShortestDistance(arr, mid + 1, high, x);
        }
    }

    public static int shortestDistance(String[] words, String word1, String word2) {
//        List<Integer> arr1 = new ArrayList<>();
//        List<Integer> arr2 = new ArrayList<>();
//        for (int i = 0; i < words.length; i++) {
//            String word = words[i];
//            if (word.equals(word1)) {
//                arr1.add(i);
//            }
//            if (word.equals(word2)) {
//                arr2.add(i);
//            }
//        }
//        int shortestDistance = Integer.MAX_VALUE;
//        for (int index : arr1) {
//            int floor = bSFloorShortestDistance(arr2, 0, arr2.size() - 1, index);
//            int ciel = bSCielShortestDistance(arr2, 0, arr2.size() - 1, index);
//            int floorCielMin = Integer.MAX_VALUE;
//            if (floor != -1 && ciel != -1) {
//                floorCielMin  = Math.min(index - floor, ciel - index);
//            } else if (floor != -1) {
//                floorCielMin = index - floor;
//            } else if (ciel != -1) {
//                floorCielMin = ciel - index;
//            }
//            shortestDistance = Math.min(shortestDistance, floorCielMin);
//        }
//        return shortestDistance;
        int a = Integer.MAX_VALUE;
        int b = Integer.MAX_VALUE;
        int minDist = Integer.MAX_VALUE;
        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1)) {
                a = i;
            }
            if (words[i].equals(word2)) {
                b = i;
            }
            if (a != Integer.MAX_VALUE && b != Integer.MAX_VALUE) {
                minDist = Math.min(minDist, Math.abs(a - b));
            }
        }
        return minDist;
    }

//    https://leetcode.com/problems/two-sum-less-than-k/

    public static int twoSumLessThanK(int[] nums, int k) {
        Arrays.sort(nums);
        int i = 0;
        int j = nums.length - 1;
        int closestSum = -1;
        int diff = Integer.MAX_VALUE;
        while (i < j) {
            int sum = nums[i] + nums[j];
            int d = Math.abs(k - sum);
            if (sum < k) {
                if (d < diff) {
                    diff = d;
                    closestSum = sum;
                }
                i++;
            } else if (sum > k) {
                j--;
            } else {
                i++;
                j--;
            }
        }
        return closestSum;
    }

//    https://leetcode.com/problems/counting-elements/

    public static int countElements(int[] arr) {
        Set<Integer> hm = new HashSet<>();
        int count = 0;
        for (int num : arr) {
            hm.add(num);
        }
        for (int num : arr) {
            if (hm.contains(num + 1)) {
                count++;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/check-if-a-number-is-majority-element-in-a-sorted-array/

    public static int bSFloorMajority(int [] arr, int low, int high, int x) {
        if (x <= arr[low]) {
            return -1;
        } else if (x > arr[high]) {
            return high;
        }
        int mid = low + (high - low) / 2;
        if (mid < high && arr[mid + 1] >= x && arr[mid] < x) {
            return mid;
        } else if (arr[mid] >= x) {
            return bSFloorMajority(arr, low, mid - 1, x);
        } else {
            return bSFloorMajority(arr, mid + 1, high, x);
        }
    }

    public static int bSCielMajority(int [] arr, int low, int high, int x) {
        if (x < arr[low]) {
            return low;
        } else if (x >= arr[high]) {
            return -1;
        }
        int mid = low + (high - low) / 2;
        if (mid < high && arr[mid + 1] > x && arr[mid] <= x) {
            return mid + 1;
        } else if (low < mid && arr[mid] > x && arr[mid - 1] <= x) {
            return mid;
        }
        else if (arr[mid] > x) {
            return bSCielMajority(arr, low, mid - 1, x);
        } else {
            return bSCielMajority(arr, mid + 1, high, x);
        }
    }

    public static boolean isMajorityElement(int[] nums, int target) {
        int floor = bSFloorMajority(nums, 0, nums.length - 1, target);
        int ciel = bSCielMajority(nums, 0, nums.length - 1, target);
        if (floor != -1 && ciel != -1) {
            return ciel - floor > nums.length / 2;
        } else if (floor != -1) {
            return (nums.length - 1) - floor > nums.length / 2;
        } else if (ciel != -1) {
            return ciel > nums.length / 2;
        } else {
            return true;
        }
    }

//    https://leetcode.com/problems/diet-plan-performance/

    public static int dietPlanPerformance(int[] calories, int k, int lower, int upper) {
        int result = 0;
        int runningSum = 0;
        int i = 0;
        while (i < k) {
            runningSum += calories[i];
            i++;
        }
        if (runningSum < lower) {
            result -= 1;
        } else if (runningSum > upper) {
            result += 1;
        }
        while (i < calories.length) {
            runningSum = runningSum - calories[i - k] + calories[i];
            if (runningSum < lower) {
                result -= 1;
            } else if (runningSum > upper) {
                result += 1;
            }
            i++;
        }
        return result;
    }

//    https://leetcode.com/problems/perform-string-shifts/

    public static void swap(char [] arr, int i, int j) {
        char temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    public static String rotate(char [] arr, int k) {
        int i = 0;
        int j = k < 0 ? (k * -1) - 1 : arr.length - 1 - k;
        int j1 = j;
        while (i < j) {
            swap(arr, i, j);
            i++;
            j--;
        }
        i = j1 + 1;
        j = arr.length - 1;
        while (i < j) {
            swap(arr, i, j);
            i++;
            j--;
        }
        i = 0;
        j = arr.length - 1;
        while (i < j) {
            swap(arr, i, j);
            i++;
            j--;
        }
        return String.valueOf(arr);
    }

    public static String stringShift(String s, int[][] shift) {
        int rotationDegree = 0;
        for (int [] value : shift) {
            int pos = value[0]  == 0 ? -1 : 1;
            rotationDegree += value[1] * pos;
        }
        return rotate(s.toCharArray(), rotationDegree % s.length());
    }

//    https://leetcode.com/problems/array-transformation/

    public static List<Integer> transformArray(int[] arr) {
        while (true) {
            boolean flag = true;
            int temp = arr[0];
            for (int i = 1; i < arr.length - 1; i++) {
                int temp1 = arr[i];
                if (arr[i] > temp && arr[i] > arr[i + 1]) {
                    arr[i]--;
                    flag = false;
                } else if (arr[i] < temp && arr[i] < arr[i + 1]) {
                    arr[i]++;
                    flag = false;
                }
                temp = temp1;
            }
            if (flag) {
                break;
            }
        }
        return Arrays.stream(arr).boxed().collect(Collectors.toList());
    }

//    https://leetcode.com/problems/missing-ranges/

    public static List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> result = new ArrayList<>();
        if (nums.length == 0) {
            if (upper == lower) {
                result.add(upper + "");
            } else {
                result.add(lower + "->" + upper);
            }
            return result;
        }
        if (nums[0] - lower > 0) {
            if (nums[0] - lower == 1) {
                result.add(lower + "");
            } else {
                result.add(lower + "->" + (nums[0] - 1));
            }
        }
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i + 1] - nums[i] > 1) {
                int a = nums[i] + 1;
                int b = nums[i + 1] - 1;
                if (a == b) {
                    result.add(a + "");
                } else {
                    result.add(a + "->" + b);
                }
            }
        }
        if (upper - nums[nums.length - 1] > 0) {
            if (upper - nums[nums.length - 1] == 1) {
                result.add("" + upper);
            } else {
                result.add((nums[nums.length - 1] + 1) + "->" + upper);
            }
        }
        return result;
    }

//    https://leetcode.com/problems/dot-product-of-two-sparse-vectors/

    static class SparseVector {

        int [] obj;

        SparseVector(int[] nums) {
            obj = nums;
        }

        // Return the dotProduct of two sparse vectors
        public int dotProduct(SparseVector vec) {
            int sum = 0;
            for (int i = 0; i < obj.length; i++) {
                sum += obj[i] * vec.obj[i];
            }
            return sum;
        }

    }

//    https://leetcode.com/problems/design-an-ordered-stream/

    static class OrderedStream {

        String [] stream;
        int pointer;

        public OrderedStream(int n) {
            stream = new String[n + 1];
            pointer = 0;
        }

        public List<String> insert(int id, String value) {
            List<String> result = new ArrayList<>();
            stream[id - 1] = value;
            while (stream[pointer] != null) {
                result.add(stream[pointer]);
                pointer++;
            }
            return result;
        }

    }

}
