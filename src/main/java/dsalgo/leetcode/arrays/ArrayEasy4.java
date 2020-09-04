package dsalgo.leetcode.arrays;

import java.util.*;

public class ArrayEasy4 {

    public static void main(String[] args) {
//        System.out.println(binarySearchFirstNegative(new int[]{2,1},0, 1));
//        System.out.println(countNegatives(new int[][]{{-1}}));
//        System.out.println(Arrays.toString(smallerNumbersThanCurrent(new int[]{8,1,2,2,3})));
//        System.out.println(findTheDistanceValue(new int[]{4, 5, 8}, new int[]{10, 9, 1, 8}, 2));
//        System.out.println(luckyNumbers(new int[][]{{7,8},{1,2}}));
//        System.out.println(countLargestGroup(15));
//        System.out.println(Arrays.toString(createTargetArray(new int[]{0,1,2,3,4},new int[]{0,1,2,2,1})));
//        System.out.println(findLucky(new int[]{2,2,2,3,3}));
//        System.out.println(minStartValue(new int[]{2,3,5,-5,-1}));
//        System.out.println(canBeEqual(new int[]{1,1,1,1}, new int[]{1,1,1,1}));
//        System.out.println(busyStudent(new int[]{9,8,7,6,5,4,3,2,1}, new int[]{10,10,10,10,10,10,10,10,10},5));
        System.out.println(Arrays.toString(finalPrices(new int[]{10,1,1,6})));
    }

//    https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix/

    public static int[] kWeakestRows(int[][] mat, int k) {
        Map<Integer, List<Integer>> tm = new TreeMap<>();
        for (int i = 0; i < mat.length; i++) {
            int onesCount = 0;
            for (int j = 0; j < mat[i].length; j++) {
                if (mat[i][j] == 1) {
                    onesCount++;
                }
            }
            List<Integer> arr;
            if (tm.get(onesCount) == null) {
                arr = new ArrayList<>();
            } else {
                arr = tm.get(onesCount);
            }
            arr.add(i);
            tm.put(onesCount, arr);
        }
        int counter = 0;
        int[] result = new int[k];
        for (int key : tm.keySet()) {
            List<Integer> arr = tm.get(key);
            for (int value : arr) {
                if (counter < k) {
                    result[counter] = value;
                    counter++;
                } else {
                    break;
                }
            }
        }
        return result;
    }

//    https://leetcode.com/problems/check-if-n-and-its-double-exist/

    public static boolean checkIfExist(int[] arr) {
        Set<Integer> hs = new HashSet<>();
        for (int num : arr) {
            if (num % 2 == 0 && hs.contains(num / 2) || hs.contains(num * 2)) {
                return true;
            }
            hs.add(num);
        }
        return false;
    }

//    https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/

    public static int countNegatives(int[][] grid) {
        int count = 0;
        for (int[] row : grid) {
            int index = binarySearchFirstNegative(row, 0, row.length - 1);
            if (index != -1) {
                count += row.length - index;
            }
        }
        return count;
    }

    public static int binarySearchFirstNegative(int[] arr, int low, int high) {
        if (arr[low] < 0) {
            return low;
        }
        if (arr[high] >= 0) {
            return -1;
        }
        int mid = low + (high - low) / 2;
        if (low < mid && arr[mid] < 0 && arr[mid - 1] >= 0) {
            return mid;
        } else if (arr[mid] < 0) {
            return binarySearchFirstNegative(arr, low, mid);
        } else if (mid < high && arr[mid] >= 0 && arr[mid + 1] < 0) {
            return mid + 1;
        } else {
            return binarySearchFirstNegative(arr, mid, high);
        }
    }

//    https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/

    public static int[] smallerNumbersThanCurrent(int[] nums) {
        int[] countArr = new int[101];
        for (int num : nums) {
            countArr[num]++;
        }
        int[] lessThanArr = new int[101];
        int sum = 0;
        for (int i = 0; i < 101; i++) {
            lessThanArr[i] = sum;
            sum += countArr[i];
        }
        int[] result = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            result[i] = lessThanArr[nums[i]];
        }
        return result;
    }

//    https://leetcode.com/problems/find-the-distance-value-between-two-arrays/

    public static int findTheDistanceValue(int[] arr1, int[] arr2, int d) {
        int result = arr1.length;
        for (int num : arr1) {
            boolean flag = false;
            for (int n : arr2) {
                if (Math.abs(num - n) <= d) {
                    flag = true;
                    break;
                }
            }
            if (flag) {
                result--;
            }
        }
        return result;
    }

//    https://leetcode.com/problems/lucky-numbers-in-a-matrix/

    public static List<Integer> luckyNumbers(int[][] matrix) {
        int[] rowsArr = new int[matrix[0].length];
        int[] columnsArr = new int[matrix[0].length];
        for (int[] ints : matrix) {
            int min = Integer.MAX_VALUE;
            int column = 0;
            for (int j = 0; j < matrix[0].length; j++) {
                columnsArr[j] = Math.max(columnsArr[j], ints[j]);
                if (ints[j] < min) {
                    min = ints[j];
                    column = j;
                }
            }
            if (rowsArr[column] < min) {
                rowsArr[column] = min;
            }
        }
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < rowsArr.length; i++) {
            if (rowsArr[i] == columnsArr[i]) {
                result.add(rowsArr[i]);
            }
        }
        return result;
    }

//    https://leetcode.com/problems/count-largest-group/

    public static int countLargestGroup(int n) {
        int[] countsArr = new int[37];
        int maxSumCount = 0;
        int maxSum = -1;
        for (int i = 1; i <= n; i++) {
            int sum = 0;
            int j = i;
            while (j != 0) {
                sum += j % 10;
                j /= 10;
            }
            countsArr[sum]++;
            if (countsArr[sum] > maxSum) {
                maxSum = countsArr[sum];
                maxSumCount = 1;
            } else if (countsArr[sum] == maxSum) {
                maxSumCount++;
            }
        }
        return maxSumCount;
    }

//    https://leetcode.com/problems/create-target-array-in-the-given-order/

    public static int[] createTargetArray(int[] nums, int[] index) {
        List<Integer> targetList = new ArrayList<>();
        for (int i = 0; i < index.length; i++) {
            targetList.add(index[i],nums[i]);
        }
        int [] result = new int[targetList.size()];
        for (int i = 0; i < targetList.size(); i++) {
            result[i] = targetList.get(i);
        }
        return result;
    }

//    https://leetcode.com/problems/find-lucky-integer-in-an-array/

    public static int findLucky(int[] arr) {
        int [] frequency = new int[501];
        for (int num : arr) {
            frequency[num]++;
        }
        int magicNo = -1;
        for (int i = 1; i < frequency.length; i++) {
            if (frequency[i] == i) {
                magicNo = i;
            }
        }
        return magicNo;
    }

//    https://leetcode.com/problems/minimum-value-to-get-positive-step-by-step-sum/

    public static int minStartValue(int[] nums) {
        int sum = 0;
        int diff = 0;
        for (int num : nums) {
            sum += num;
            if (sum < 1) {
                int d = 1 - sum;
                diff += d;
                sum += d;
            }
        }
        return diff == 0 ? 1 : diff;
    }

//    https://leetcode.com/problems/make-two-arrays-equal-by-reversing-sub-arrays/

    public static boolean canBeEqual(int[] target, int[] arr) {
       int [] countArr = new int[1001];
        for (int num : target) {
            countArr[num]++;
        }
        for (int num : arr) {
            countArr[num]--;
        }
        for (int num : countArr) {
            if (num != 0) {
                return false;
            }
        }
        return true;
    }

//    https://leetcode.com/problems/number-of-students-doing-homework-at-a-given-time/

    public static int busyStudent(int[] startTime, int[] endTime, int queryTime) {
        int count = 0;
        for (int i = 0; i < startTime.length; i++) {
            if (startTime[i] <= queryTime && endTime[i] >= queryTime) {
                count++;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/

    public static int[] finalPrices(int[] prices) {
        int [] result = new int[prices.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = prices.length - 1; i >= 0; i--) {
            if (stack.isEmpty()) {
                result[i] = prices[i];
            } else {
                if (stack.peek() <= prices[i]) {
                    result[i] = prices[i] - stack.peek();
                } else {
                    while (!stack.isEmpty() && stack.peek() > prices[i]) {
                        stack.pop();
                    }
                    if (stack.isEmpty()) {
                        result[i] = prices[i];
                    } else {
                        result[i] = prices[i] - stack.peek();
                    }
                }
            }
            stack.push(prices[i]);
        }
        return result;
    }
}
