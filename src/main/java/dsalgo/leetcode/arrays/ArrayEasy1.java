package dsalgo.leetcode.arrays;

import java.util.*;
import java.util.stream.Collectors;

public class ArrayEasy1 {

    public static void main(String[] args) {
//        System.out.println(containerWithMostWater(new int[]{1,8,6,2,5,4,8,3,7}));
//        kidsWithCandies(new int[]{12, 1, 12}, 10).forEach(System.out::println);
//        System.out.println(trappingRainWater(new int[]{0,1,0,2,1,0,1,3,2,1,2,1}));
//        Arrays.stream(twoSum(new int[]{2, 7, 11, 15}, 9)).forEach(System.out::println);
//        System.out.println(threeSumClosest(new int[]{-1,2,1,-4}, -1));
//        System.out.println(fourSum(new int[]{0,0,0,0}, 0));
//        System.out.println(removeDuplicates(new int[]{1, 1, 1, 1}));
//        System.out.println(removeDuplicatesTwo(new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8}));
//        System.out.println(removeElement(new int[]{3,2,2,3},3));
//        System.out.println(searchInsert(new int[]{1,3,5,6}, 2));
//        System.out.println(maxSubArray(new int[]{-1, -2}));
//        System.out.println(Arrays.stream(plusOne(new int[]{9,9,8}))
//                .mapToObj(Objects::toString).collect(Collectors.joining(",")));
//        merge(new int[]{1,2,3,0,0,0}, 3, new int[]{2,5,6}, 3);
//        System.out.println(pascalTriangle(7));
//        System.out.println(pascalTriangleTwo(30));
//        System.out.println(twoSumSquares(new int[]{0,0,4,2,2,3}));
//        System.out.println(maxProfit(new int[]{7, 1, 5, 3, 6, 4}));
//        System.out.println(maxProfitTwo(new int[]{1,2,3,4,5}));
//        System.out.println(Arrays.stream(twoSumPointerApproach(new int[]{0,0,3,4},0))
//                .mapToObj(Objects::toString).collect(Collectors.joining(",")));
//        rotate(new int[]{1, 2}, 2);
//        System.out.println(containsDuplicate(new int[]{1,1,1,3,3,4,3,2,4,2}));
//        System.out.println(containsNearbyDuplicate(new int[]{99,99},2));
//        System.out.println(missingNumber(new int[]{9,6,4,2,3,5,7,0,1}));
//        moveZeroes(new int[]{0,1,0,3,12,0});
//        System.out.println(thirdMax(new int[]{1,2,-2147483648}));
//        System.out.println(findDisappearedNumbers(new int[]{4,3,2,7,8,2,3,1}));
//        System.out.println(findMaxConsecutiveOnes(new int[]{1,1,0,0,1,1}));
//        System.out.println(arrayPairSum(new int[]{1,4,3,2}));
//        System.out.println(findPairs(new int[]{1, 1, 1, 1, 1}, 0));
    }

//    https://leetcode.com/problems/container-with-most-water/

    public static int containerWithMostWater(int[] height) {
        int i = 0;
        int j = height.length - 1;
        int max = Integer.MIN_VALUE;
        while (i < j) {
            int minHeight = Math.min(height[i], height[j]);
            max = Math.max(max, minHeight * (j - i));
            if (height[i] < height[j]) {
                i++;
            } else {
                j--;
            }
        }
        return max;
    }

//    https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/

    public static List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int max = Integer.MIN_VALUE;
        for (int candy : candies) {
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

//    https://leetcode.com/problems/3sum/

    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for (int a = 0; a < nums.length - 2; a++) {
            if (a > 0 && nums[a] == nums[a - 1]) {
                continue;
            }
            int x = nums[a];
            int i = a + 1;
            int j = nums.length - 1;
            while (i < j) {
                int sum = x + nums[i] + nums[j];
                if (sum == 0) {
                    List<Integer> arr = new ArrayList<>();
                    arr.add(x);
                    arr.add(nums[i]);
                    arr.add(nums[j]);
                    result.add(arr);
                    while (nums[i] == nums[i + 1]) {
                        i++;
                    }
                    while (nums[j] == nums[j - 1]) {
                        j--;
                    }
                    i++;
                    j--;
                } else if (sum < 0) {
                    i++;
                } else {
                    j--;
                }
            }
        }
        return result;
    }

//    https://leetcode.com/problems/trapping-rain-water/

    public static int trappingRainWater(int[] height) {
        int i = 0;
        int j = height.length - 1;
        if (height.length == 0) {
            return 0;
        }
        int lMax = height[0];
        int rMax = height[j];
        int sum = 0;
        while (i <= j) {
            lMax = Math.max(lMax, height[i]);
            rMax = Math.max(rMax, height[j]);
            if (lMax <= rMax) {
                sum += lMax - height[i];
                i++;
            } else {
                sum += rMax - height[j];
                j--;
            }
        }
        return sum;
    }

//    https://leetcode.com/problems/two-sum/

    public static int[] twoSum(int[] nums, int target) {
        int[] outputArr = new int[2];
        Map<Integer, Integer> hm = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int value = nums[i];
            int diffValue = target - value;
            if (hm.get(diffValue) != null) {
                outputArr[0] = hm.get(diffValue);
                outputArr[1] = i;
            } else {
                hm.put(value, i);
            }
        }
        return outputArr;
    }

//    https://leetcode.com/problems/3sum-closest/

    public static int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        long minSum = Integer.MAX_VALUE;
        for (int a = 0; a < nums.length - 2; a++) {
            int x = nums[a];
            int i = a + 1;
            int j = nums.length - 1;
            while (i < j) {
                int sum = x + nums[i] + nums[j];
                if (Math.abs(target - sum) <= Math.abs(target - minSum)) {
                    minSum = sum;
                }
                if (sum < target) {
                    i++;
                } else {
                    j--;
                }
            }
        }
        return (int) minSum;
    }

//    https://leetcode.com/problems/4sum/

    public static List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums == null || nums.length < 4) {
            return result;
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            for (int j = i + 1; j < nums.length - 2; j++) {
                if (j != i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                int low = j + 1;
                int high = nums.length - 1;
                while (low < high) {
                    int sum = nums[i] + nums[j] + nums[low] + nums[high];
                    if (sum == target) {
                        List<Integer> arr = new ArrayList<>();
                        arr.add(nums[i]);
                        arr.add(nums[j]);
                        arr.add(nums[low]);
                        arr.add(nums[high]);
                        result.add(arr);
                        low++;
                        high--;
                        while (low < high && nums[low] == nums[low - 1]) {
                            low++;
                        }
                        while (low < high && nums[high] == nums[high + 1]) {
                            high--;
                        }
                    } else if (sum < target) {
                        low++;
                    } else {
                        high--;
                    }
                }
            }
        }
        return result;
    }

//    https://leetcode.com/problems/remove-duplicates-from-sorted-array/

    public static int removeDuplicates(int[] nums) {
        int nonDuplicatePointer = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[nonDuplicatePointer] = nums[i];
                nonDuplicatePointer++;
            }
        }
        return nonDuplicatePointer;
    }

//    https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/

    public static int removeDuplicatesTwo(int[] nums) {
        int nonDuplicatePointer = 1;
        int repeatCount = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1] && repeatCount >= 2) {
                continue;
            }
            if (nums[i] == nums[i - 1]) {
                repeatCount++;
            } else {
                repeatCount = 1;
            }
            nums[nonDuplicatePointer] = nums[i];
            nonDuplicatePointer++;
        }
//        System.out.println(Arrays.stream(nums).mapToObj(Objects::toString).collect(Collectors.joining(",")));
        return nonDuplicatePointer;
    }

//    https://leetcode.com/problems/remove-element/

    public static int removeElement(int[] nums, int val) {
        int j = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != val) {
                nums[j] = nums[i];
                j++;
            }
        }
//        System.out.println(Arrays.stream(nums).mapToObj(Objects::toString).collect(Collectors.joining(",")));
        return j;
    }

//    https://leetcode.com/problems/search-insert-position/

    public static int searchInsert(int[] nums, int target) {
        int low = 0;
        int high = nums.length - 1;
        while (low <= high) {
            if (target <= nums[low]) {
                return low;
            }
            if (target > nums[high]) {
                return high + 1;
            }
            int mid = low + (high - low) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return -1;
    }

//    https://leetcode.com/problems/maximum-subarray/

    public static int maxSubArray(int[] nums) {
        int maxSum = Integer.MIN_VALUE;
        int subArrSum = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > subArrSum + nums[i]) {
                subArrSum = nums[i];
            } else {
                subArrSum += nums[i];
            }
            maxSum = Math.max(maxSum, subArrSum);
        }
        return Math.max(maxSum, nums[0]);
    }

//    https://leetcode.com/problems/plus-one/

    public static int[] plusOne(int[] digits) {
        if (digits[digits.length - 1] != 9) {
            digits[digits.length - 1] = digits[digits.length - 1] + 1;
        } else {
            int carryOver = 1;
            int i = digits.length - 1;
            while (i >= 0) {
                int value = digits[i] + carryOver;
                digits[i] = value % 10;
                carryOver = value / 10;
                i--;
            }
            if (carryOver > 0) {
                int[] newDigits = new int[digits.length + 1];
                newDigits[0] = carryOver;
                System.arraycopy(digits, 0, newDigits, 1, newDigits.length - 1);
                return newDigits;
            }
        }
        return digits;
    }

//    https://leetcode.com/problems/merge-sorted-array/

    public static void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int placeCounter = m + n - 1;
        while (i >= 0 && j >= 0) {
            if (nums1[i] > nums2[j]) {
                nums1[placeCounter] = nums1[i];
                i--;
            } else {
                nums1[placeCounter] = nums2[j];
                j--;
            }
            placeCounter--;
        }
        while (i >= 0) {
            nums1[placeCounter] = nums1[i];
            i--;
            placeCounter--;
        }
        while (j >= 0) {
            nums1[placeCounter] = nums2[j];
            j--;
            placeCounter--;
        }
        System.out.println(Arrays.stream(nums1).mapToObj(Objects::toString).collect(Collectors.joining(",")));
    }

//    https://leetcode.com/problems/pascals-triangle/

    public static List<List<Integer>> pascalTriangle(int numRows) {
        List<List<Integer>> result = new ArrayList<>();
        if (numRows == 0) {
            return result;
        }
        List<Integer> arr0 = new ArrayList<>();
        arr0.add(1);
        result.add(arr0);
        for (int i = 1; i < numRows; i++) {
            List<Integer> arr = new ArrayList<>();
            List<Integer> prevArr = result.get(i - 1);
            int prevSum = 0;
            for (int j = 0; j < prevArr.size(); j++) {
                if (j < 2) {
                    prevSum += prevArr.get(j);
                    arr.add(prevSum);
                } else {
                    prevSum = prevSum - prevArr.get(j - 2) + prevArr.get(j);
                    arr.add(prevSum);
                }
            }
            arr.add(1);
            result.add(arr);
        }
        return result;
    }

//    https://leetcode.com/problems/pascals-triangle-ii/

    public static List<Integer> pascalTriangleTwo(int rowIndex) {
        List<Integer> arr = new ArrayList<>();
        int prev = 1;
        arr.add(prev);
        for (int i = 1; i <= rowIndex; i++) {
            int value = (int) (((long) prev * (rowIndex - i + 1)) / i);
            arr.add(value);
            prev = value;
        }
        return arr;
    }

//    expedia sd1 question - sum of 2 squares equal to third

    static int twoSumSquares(int[] arr) {
        Arrays.sort(arr);
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            int j = i + 1;
            int k = arr.length - 1;
            while (j < k) {
                if (arr[i] * arr[i] + arr[j] * arr[j] == arr[k] * arr[k]) {
                    System.out.println(arr[i] + " " + arr[j] + " " + arr[k]);
                    count++;
                    j++;
                    k--;
                } else if (arr[i] * arr[i] + arr[j] * arr[j] < arr[k] * arr[k]) {
                    k--;
                } else {
                    j++;
                }
            }
        }
        return count;
    }

//    https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

    static int maxProfit(int[] prices) {
        int maxDiff = Integer.MIN_VALUE;
        if (prices.length < 2) {
            return 0;
        }
        int max = prices[prices.length - 1];
        for (int i = prices.length - 2; i >= 0; i--) {
            if (prices[i] > max) {
                max = prices[i];
            } else if (max - prices[i] > maxDiff) {
                maxDiff = max - prices[i];
            }
        }
        return Math.max(maxDiff, 0);
    }

//    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

    public static int maxProfitTwo(int[] prices) {
        int i = 0;
        int diff = 0;
        while (i < prices.length - 1) {
            while (i < prices.length - 1 && prices[i] > prices[i + 1]) {
                i++;
            }
            int valley = prices[i];
            while (i < prices.length - 1 && prices[i] < prices[i + 1]) {
                i++;
            }
            int peak = prices[i];
            diff += peak - valley;
        }
//        2nd way, instead of calculating peak n valley, find consecutive difference if p[i] < p[i-1]
//        as a + b + c = difference b/w peak n valley
//        https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/solution/
//        int diff = 0;
//        for (int i = 1; i < prices.length; i++) {
//            if (prices[i] > prices[i - 1]) {
//                diff += prices[i] -prices[i - 1];
//            }
//        }
        return diff;
    }

//    https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/

    public static int[] twoSumPointerApproach(int[] numbers, int target) {
        int[] result = new int[2];
        if (numbers.length < 2) {
            return result;
        }
        int i = 0;
        int j = numbers.length - 1;
        while (i < j) {
            if (numbers[i] + numbers[j] == target) {
                result[0] = i + 1;
                result[1] = j + 1;
                i++;
                j--;
                while (i > 0 && numbers[i] == numbers[i - 1]) {
                    i++;
                }
                while (j < numbers.length - 1 && numbers[j] == numbers[j + 1]) {
                    j--;
                }
            } else if (numbers[i] + numbers[j] < target) {
                i++;
            } else {
                j--;
            }
        }
        return result;
    }

//    https://leetcode.com/problems/majority-element/

    public static int majorityElement(int[] nums) {
        int elemIndex = 0;
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[elemIndex]) {
                count++;
            } else {
                count--;
                if (count == 0) {
                    elemIndex = i;
                    count = 1;
                }
            }
        }
        if (count > nums.length / 2) {
            return nums[elemIndex];
        } else {
            count = 0;
            for (int num : nums) {
                if (num == nums[elemIndex]) {
                    count++;
                }
            }
            if (count > nums.length / 2) {
                return nums[elemIndex];
            }
            return -1;
        }
    }

//    https://leetcode.com/problems/rotate-array/

    public static void rotate(int[] nums, int k) {
        k %= nums.length;
        int i = nums.length - k;
        int j = nums.length - 1;
        while (i < j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            i++;
            j--;
        }
        i = 0;
        j = nums.length - 1 - k;
        while (i < j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            i++;
            j--;
        }
        i = 0;
        j = nums.length - 1;
        while (i < j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            i++;
            j--;
        }
        System.out.println(Arrays.stream(nums).mapToObj(Objects::toString).collect(Collectors.joining(",")));
    }

//    https://leetcode.com/problems/contains-duplicate/

    public static boolean containsDuplicate(int[] nums) {
        Set<Integer> hs = new HashSet<>();
        for (int num : nums) {
            if (hs.contains(num)) {
                return true;
            } else {
                hs.add(num);
            }
        }
        return false;
    }

//    https://leetcode.com/problems/contains-duplicate-ii/

    public static boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> hm = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (hm.containsKey(nums[i])) {
                int index = hm.get(nums[i]);
                if (Math.abs(i - index) <= k) {
                    return true;
                }
            }
            hm.put(nums[i], i);
        }
        return false;
    }

//    https://leetcode.com/problems/missing-number/

    public static int missingNumber(int[] nums) {
        int n = nums.length;
        int totalSum = (n * (n + 1)) / 2;
        int sum = 0;
        for (int number : nums) {
            sum += number;
        }
        return totalSum - sum;
    }

//    https://leetcode.com/problems/move-zeroes/

    public static void moveZeroes(int[] nums) {
        int zeroPointer = 0;
        int nonZeroPointer = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[zeroPointer] = nums[nonZeroPointer];
                zeroPointer++;
            }
            nonZeroPointer++;
        }
        for (int i = zeroPointer; i < nums.length; i++) {
            nums[i] = 0;
        }
        System.out.println(Arrays.stream(nums).mapToObj(Objects::toString).collect(Collectors.joining(",")));
    }

//    https://leetcode.com/problems/third-maximum-number/

    public static int thirdMax(int[] nums) {
//        long firstMax = Long.MIN_VALUE;
//        long secondMax = Long.MIN_VALUE;
//        long thirdMax = Long.MIN_VALUE;
//        boolean nullFlag = false;
//        boolean secondFlag = false;
//        for (int number: nums) {
//            if (number == Integer.MIN_VALUE) {
//                nullFlag = true;
//            }
//            if (number > thirdMax && number > secondMax && number > firstMax) {
//                thirdMax = secondMax;
//                secondMax = firstMax;
//                firstMax = number;
//            } else if (number < firstMax && number > thirdMax && number > secondMax) {
//                secondFlag = true;
//                thirdMax = secondMax;
//                secondMax = number;
//            } else if (number < firstMax && number < secondMax && number > thirdMax) {
//                secondFlag = true;
//                thirdMax = number;
//            }
//        }
//        if (thirdMax == Integer.MIN_VALUE && nullFlag && secondFlag) {
//            return (int) thirdMax;
//        } else if (thirdMax == Long.MIN_VALUE) {
//            return (int) firstMax;
//        } else {
//            return (int) thirdMax;
//        }
//        another better approach
        Integer firstMax = null;
        Integer secondMax = null;
        Integer thirdMax = null;
        for (Integer number : nums) {
            if (number.equals(firstMax) || number.equals(secondMax) || number.equals(thirdMax)) continue;
            if (firstMax == null || number > firstMax) {
                thirdMax = secondMax;
                secondMax = firstMax;
                firstMax = number;
            } else if (secondMax == null || number > secondMax) {
                thirdMax = secondMax;
                secondMax = number;
            } else if (thirdMax == null || number > thirdMax) {
                thirdMax = number;
            }
        }
        return thirdMax == null ? firstMax : thirdMax;
    }

//    https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/

    public static List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> result = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int number = Math.abs(nums[i]) % n;
            if (nums[number] > 0) {
                nums[number] = -nums[number];
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] > 0) {
                if (i == 0) {
                    result.add(n);
                } else {
                    result.add(i);
                }
            }
        }
        return result;
    }

//    https://leetcode.com/problems/max-consecutive-ones/

    public static int findMaxConsecutiveOnes(int[] nums) {
        int maxCount = 0;
        int count = 0;
        for (int num : nums) {
            if (num == 1) {
                count++;
                maxCount = Math.max(count, maxCount);
            } else {
                count = 0;
            }
        }
        return maxCount;
    }

//    https://leetcode.com/problems/array-partition-i/

    public static int arrayPairSum(int[] nums) {
        Arrays.sort(nums);
        int sum = 0;
        for (int i = 0; i < nums.length; i = i + 2) {
            sum += Math.min(nums[i], nums[i + 1]);
        }
        return sum;
    }

//    https://leetcode.com/problems/k-diff-pairs-in-an-array/

    public static int findPairs(int[] nums, int k) {
        if (k < 0 || nums == null) {
            return 0;
        }
        int count = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int n : nums) {
            map.put(n, map.getOrDefault(n, 0) + 1);
        }
        for (int key : map.keySet()) {
            if (k == 0) {
                if (map.get(key) > 1) {
                    count++;
                }
            } else {
                if (map.containsKey(key + k)) {
                    count++;
                }
            }
        }
        return count;
    }

}