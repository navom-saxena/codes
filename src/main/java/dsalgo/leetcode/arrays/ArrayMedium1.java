package dsalgo.leetcode.arrays;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class ArrayMedium1 {

    public static void main(String[] args) {
//        System.out.println(containerWithMostWater(new int[]{1,8,6,2,5,4,8,3,7}));
//        System.out.println(threeSumClosest(new int[]{-1,2,1,-4}, -1));
//        System.out.println(fourSum(new int[]{0,0,0,0}, 0));
//        nextPermutation(new int[]{1,5,1});
//        System.out.println(getPivot(new int[]{1}, 0 , 0));
//        System.out.println(binarySearch(new int[]{1,2,3},0,2,3));
//        System.out.println(search(new int[]{1}, 0));
//        System.out.println(excludedFloorSearch(new int[]{6,6,6,6}, 0, 3,6));
//        System.out.println(excludedCeilSearch(new int[]{6,6,6,6}, 0,3,6));
//        System.out.println(Arrays.toString(searchRange(new int[]{1, 2, 3}, 2)));
//        printSubSequence(new int[]{1,2}, new int[]{0,0},0,0);
//        printSubSequenceForLoop(new int[]{1,2,3}, new ArrayList<>(), -1);
//        System.out.println(combinationSum2(new int[]{2,2,2},4));
//        System.out.println(spiralOrder(new int[][]{}));
//        System.out.println(canJump(new int[]{2, 3, 1, 1, 4}));
//        System.out.println(Arrays.deepToString(merge(new int[][]{{0, 0}, {1, 2}, {5, 5}, {2, 4}, {3, 3}, {5, 6}, {5, 6}, {4, 6}, {0, 0}, {1, 2}, {0, 2}, {4, 5}})));
//        System.out.println(Arrays.deepToString(insert(new int[][]{{1,2},{3,5},{6,7},{8,10},{12,16}},new int[]{4,8})));
//        System.out.println(Arrays.deepToString(generateMatrix(4)));
//        System.out.println(uniquePaths(3,3));
//        System.out.println(uniquePathsWithObstacles(new int[][]{{0}, {1}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}, {0}, {0}, {0}, {0}, {1}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1}, {0}, {0}, {0}, {0}, {1}}));
//        setZeroes(new int[][]{{1,1,1},{1,0,1},{1,1,1}});
//        System.out.println(searchMatrix(new int[][]{{1}}, 0));
//        System.out.println(minPathSum(new int[][]{{1,3,1},{1,5,1},{4,2,1}}));
//        System.out.println(subsets(new int[]{1, 2, 3}));
//        System.out.println(getPivotInDuplicate(new int[]{2,5,6,0,0,1,2}, 0, 6));
//        System.out.println(searchPivotDuplicate(new int[]{1,3}, 1));
//        System.out.println(subsetsWithDup(new int[]{2, 2}));
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

//    https://leetcode.com/problems/next-permutation/

    public static void nextPermutation(int[] nums) {
        if (nums.length > 1 && nums[nums.length - 1] > nums[nums.length - 2]) {
            int temp = nums[nums.length - 2];
            nums[nums.length - 2] = nums[nums.length - 1];
            nums[nums.length - 1] = temp;
            System.out.println(Arrays.toString(nums));
            return;
        }
        boolean seqBreak = false;
        int breakIndex = nums.length - 1;
        for (int i = nums.length - 2; i >= 0; i--) {
            if (nums[i] < nums[i + 1]) {
                breakIndex = i;
                seqBreak = true;
                break;
            }
        }
        if (seqBreak) {
            int i = nums.length - 1;
            while (nums[i] <= nums[breakIndex]) {
                i--;
            }
            int temp = nums[breakIndex];
            nums[breakIndex] = nums[i];
            nums[i] = temp;
            i = breakIndex + 1;
            int j = nums.length - 1;
            while (i <= j) {
                int temp1 = nums[i];
                nums[i] = nums[j];
                nums[j] = temp1;
                i++;
                j--;
            }
        } else {
            int i = 0;
            int j = nums.length - 1;
            while (i <= j) {
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
                i++;
                j--;
            }
        }
        System.out.println(Arrays.toString(nums));
    }

//    https://leetcode.com/problems/search-in-rotated-sorted-array/

    public static int getPivot(int[] arr, int low, int high) {
        if (arr[low] <= arr[high]) {
            return -1;
        }
        int mid = low + (high - low) / 2;
        if (low < mid && arr[mid] < arr[mid - 1]) {
            return mid;
        } else if (mid < high && arr[mid] > arr[mid + 1]) {
            return mid + 1;
        } else if (arr[low] <= arr[mid]) {
            return getPivot(arr, mid, high);
        } else {
            return getPivot(arr, low, mid);
        }
    }

    public static int binarySearch(int[] arr, int low, int high, int value) {
        if (arr[low] > value) {
            return -1;
        }
        if (arr[high] < value) {
            return -1;
        }
        int mid = low + (high - low) / 2;
        if (arr[mid] == value) {
            return mid;
        } else if (arr[mid] > value) {
            return binarySearch(arr, low, mid - 1, value);
        } else {
            return binarySearch(arr, mid + 1, high, value);
        }
    }

    public static int search(int[] nums, int target) {
        int pivot = getPivot(nums, 0, nums.length - 1);
        if (pivot != -1) {
            int leftSearch = binarySearch(nums, 0, pivot - 1, target);
            if (leftSearch != -1) {
                return leftSearch;
            }
            return binarySearch(nums, pivot, nums.length - 1, target);
        } else {
            return binarySearch(nums, 0, nums.length - 1, target);
        }
    }

//    https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/

    public static int excludedFloorSearch(int[] arr, int low, int high, int value) {
        if (arr.length == 0 || value < arr[low] || value > arr[high]) {
            return -1;
        }
        if (value == arr[low]) {
            return -2;
        }
        int mid = low + (high - low) / 2;
        if (mid < high && arr[mid] < value && arr[mid + 1] == value) {
            return mid;
        } else if (arr[mid] >= value) {
            return excludedFloorSearch(arr, low, mid, value);
        } else {
            return excludedFloorSearch(arr, mid + 1, high, value);
        }
    }

    public static int excludedCeilSearch(int[] arr, int low, int high, int value) {
        if (arr.length == 0 || value < arr[low] || value > arr[high]) {
            return -1;
        }
        if (value == arr[high]) {
            return arr.length;
        }
        int mid = low + (high - low) / 2;
        if (mid < high && arr[mid] == value && arr[mid + 1] > value) {
            return mid + 1;
        } else if (arr[mid] <= value) {
            return excludedCeilSearch(arr, mid + 1, high, value);
        } else {
            return excludedCeilSearch(arr, low, mid, value);
        }
    }

    public static int[] searchRange(int[] nums, int target) {
        int floor = excludedFloorSearch(nums, 0, nums.length - 1, target);
        int ceil = excludedCeilSearch(nums, 0, nums.length - 1, target);
        int firstIndex = floor + 1;
        int lastIndex = ceil - 1;
        if (floor == -1) {
            firstIndex = -1;
        } else if (floor == -2) {
            firstIndex = 0;
        }
        if (ceil == -1) {
            lastIndex = -1;
        }
        return new int[]{firstIndex, lastIndex};
    }

//    print subsequence

    public static void printSubSequence(int[] arr, int[] subsequence, int arrIndex, int subSeqIndex) {
        if (arrIndex == arr.length) {
            System.out.println(Arrays.toString(subsequence));
            return;
        }
        printSubSequence(arr, subsequence, arrIndex + 1, subSeqIndex);
        subsequence[subSeqIndex] = arr[arrIndex];
        printSubSequence(arr, subsequence, arrIndex + 1, subSeqIndex + 1);
    }

//    print subsequence using for loop and recursion

    public static void printSubSequenceForLoop(int[] arr, List<Integer> subsequence, int index) {
        if (index == arr.length) {
            return;
        }
        System.out.println(subsequence);
        for (int i = index + 1; i < arr.length; i++) {
            subsequence.add(arr[i]);
            printSubSequenceForLoop(arr, subsequence, i);
            subsequence.remove(subsequence.size() - 1);
        }
    }

//    https://leetcode.com/problems/combination-sum/

    public static void generateCombinationSum(int[] candidates, int index, int target,
                                              List<List<Integer>> result, List<Integer> values) {
        if (target == 0) {
            result.add(new ArrayList<>(values));
        }
        for (int i = index; i < candidates.length; i++) {
            if (target - candidates[i] < 0) {
                break;
            }
            values.add(candidates[i]);
            generateCombinationSum(candidates, i, target - candidates[i], result, values);
            values.remove(values.size() - 1);
        }
    }

    public static List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        int index = 0;
        Arrays.sort(candidates);
        List<Integer> values = new ArrayList<>();
        generateCombinationSum(candidates, index, target, result, values);
        return result;
    }

//    https://leetcode.com/problems/combination-sum-ii/

    public static void generateCombinationSum2(int[] candidates, int index, int target,
                                               List<List<Integer>> result, List<Integer> values) {
        if (target == 0) {
            result.add(new ArrayList<>(values));
        }
        for (int i = index; i < candidates.length; i++) {
            if (i != index && candidates[i] == candidates[i - 1]) {
                continue;
            }
            if (target < candidates[i]) {
                break;
            }
            values.add(candidates[i]);
            generateCombinationSum2(candidates, i + 1, target - candidates[i], result, values);
            values.remove(values.size() - 1);
        }
    }

    public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        int index = 0;
        Arrays.sort(candidates);
        List<Integer> values = new ArrayList<>();
        generateCombinationSum2(candidates, index, target, result, values);
        return new ArrayList<>(result);
    }

//    https://leetcode.com/problems/rotate-image/

    public static void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < Math.ceil(((double) n) / 2.); j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = temp;
            }
        }
    }

//    https://leetcode.com/problems/spiral-matrix/

    public static List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        if (matrix.length == 0) {
            return result;
        }
        int k = 0;
        int rows = matrix.length;
        int columns = matrix[0].length;
        int count = matrix.length * matrix[0].length;
        while (count > 0) {
            int i = k;
            int j = k;
            while (j < columns - k) {
                result.add(matrix[i][j]);
                count--;
                j++;
            }
            j = j - 1;
            i = i + 1;
            while (i < rows - k) {
                result.add(matrix[i][j]);
                count--;
                i++;
            }
            j = j - 1;
            i = i - 1;
            if (count == 0) {
                break;
            }
            while (j >= k) {
                result.add(matrix[i][j]);
                count--;
                j--;
            }
            i = i - 1;
            j = j + 1;
            if (count == 0) {
                break;
            }
            while (i > k) {
                result.add(matrix[i][j]);
                count--;
                i--;
            }

            k++;
        }
        return result;
    }

//    https://leetcode.com/problems/jump-game/

    public static boolean canJump(int[] nums) {
        int maxJumpSoFar = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > maxJumpSoFar) {
                return false;
            }
            maxJumpSoFar = Math.max(maxJumpSoFar, i + nums[i]);
        }
        return true;
    }

//    https://leetcode.com/problems/merge-intervals/

    public static int[][] merge(int[][] intervals) {
        if (intervals.length == 0) return intervals;
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        System.out.println(Arrays.deepToString(intervals));
        List<int[]> result = new ArrayList<>();
        result.add(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            int[] currentInterval = intervals[i];
            int[] lastInterval = result.get(result.size() - 1);
            if (lastInterval[1] >= currentInterval[0]) {
                lastInterval[1] = Math.max(lastInterval[1], currentInterval[1]);
            } else {
                result.add(currentInterval);
            }
        }
        return result.toArray(new int[result.size()][]);
    }

//    https://leetcode.com/problems/insert-interval/

    public static int floorBinarySearchInterval(int[][] interval, int[] newInterval, int low, int high) {
        if (interval[low][0] > newInterval[1]) {
            return -1;
        } else if (interval[high][0] < newInterval[0]) {
            return high;
        }
        int mid = interval[low][0] + (interval[high][0] - interval[low][0]) / 2;
        if (mid < high && interval[mid][0] <= newInterval[0] && interval[mid + 1][0] > newInterval[0]) {
            return mid;
        } else if (interval[mid][0] >= newInterval[0]) {
            return floorBinarySearchInterval(interval, newInterval, low, mid);
        } else {
            return floorBinarySearchInterval(interval, newInterval, mid + 1, high);
        }
    }

    public static int[][] insert(int[][] intervals, int[] newInterval) {
        if (intervals.length == 0) {
            return new int[][]{newInterval};
        }
        List<int[]> result = new ArrayList<>();
        int i = 0;
        while (i < intervals.length && intervals[i][1] < newInterval[0]) {
            result.add(intervals[i]);
            i++;
        }
        if (i < intervals.length && intervals[i][0] <= newInterval[0]) {
            result.add(new int[]{Math.min(intervals[i][0], newInterval[0]), Math.max(intervals[i][1], newInterval[1])});
        } else {
            result.add(newInterval);
        }
        while (i < intervals.length) {
            int[] last = result.get(result.size() - 1);
            if (last[1] < intervals[i][0]) {
                result.add(intervals[i]);
            } else {
                last[1] = Math.max(last[1], intervals[i][1]);
            }
            i++;
        }
        return result.toArray(new int[result.size()][]);
    }

//    https://leetcode.com/problems/spiral-matrix-ii/

    public static int[][] generateMatrix(int n) {
        int[][] result = new int[n][n];
        if (n == 0) {
            return result;
        }
        int k = 0;
        int rows = result.length;
        int columns = result[0].length;
        int currentNo = 1;
        int numberRange = result.length * result[0].length;
        while (currentNo <= numberRange) {
            int i = k;
            int j = k;
            while (j < columns - k) {
                result[i][j] = currentNo;
                currentNo++;
                j++;
            }
            j = j - 1;
            i = i + 1;
            while (i < rows - k) {
                result[i][j] = currentNo;
                currentNo++;
                i++;
            }
            j = j - 1;
            i = i - 1;
            if (currentNo > numberRange) {
                break;
            }
            while (j >= k) {
                result[i][j] = currentNo;
                currentNo++;
                j--;
            }
            i = i - 1;
            j = j + 1;
            if (currentNo > numberRange) {
                break;
            }
            while (i > k) {
                result[i][j] = currentNo;
                currentNo++;
                i--;
            }
            k++;
        }
        return result;
    }

//    https://leetcode.com/problems/unique-paths/

    public static int uniquePaths(int m, int n) {
        int[][] pathMatrix = new int[m][n];
        pathMatrix[0][0] = 0;
        for (int j = 1; j < n; j++) {
            pathMatrix[0][j] = 1;
        }
        for (int i = 0; i < m; i++) {
            pathMatrix[i][0] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                pathMatrix[i][j] = pathMatrix[i][j - 1] + pathMatrix[i - 1][j];
            }
        }
        return pathMatrix[m - 1][n - 1];
    }

//    https://leetcode.com/problems/unique-paths-ii/

    public static int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        if (m == 1 && n == 1 && obstacleGrid[0][0] == 0) {
            return 1;
        }
        if (obstacleGrid[0][0] == 1) {
            return 0;
        }
        obstacleGrid[0][0] = 0;
        boolean blockedPath = false;
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[0][j] == 1 || blockedPath) {
                obstacleGrid[0][j] = 0;
                blockedPath = true;
            } else {
                obstacleGrid[0][j] = 1;
            }
        }
        blockedPath = false;
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 1 || blockedPath) {
                obstacleGrid[i][0] = 0;
                blockedPath = true;
            } else {
                obstacleGrid[i][0] = 1;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    obstacleGrid[i][j] = 0;
                } else {
                    obstacleGrid[i][j] = obstacleGrid[i][j - 1] + obstacleGrid[i - 1][j];
                }
            }
        }
        return obstacleGrid[m - 1][n - 1];
    }

//    https://leetcode.com/problems/set-matrix-zeroes/

    public static void setZeroes(int[][] matrix) {
        boolean isFirstRowZero = false;
        for (int j = 0; j < matrix[0].length; j++) {
            if (matrix[0][j] == 0) {
                isFirstRowZero = true;
                break;
            }
        }
        for (int i = 1; i < matrix.length; i++) {
            boolean isZero = false;
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    isZero = true;
                }
            }
            if (isZero) {
                Arrays.fill(matrix[i], 0);
            }
        }
        for (int i = 1; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (isFirstRowZero) {
            Arrays.fill(matrix[0], 0);
        }
        System.out.println(Arrays.deepToString(matrix));
    }

    public static int findMatrixFloor(int[][] matrix, int target, int low, int high) {
        if (target < matrix[low][0]) {
            return -1;
        } else if (target >= matrix[high][0]) {
            return high;
        } else if (target == matrix[low][0]) {
            return low;
        }
        int mid = low + (high - low) / 2;
        if (mid < high && matrix[mid][0] <= target && matrix[mid + 1][0] > target) {
            return mid;
        } else if (matrix[mid][0] > target) {
            return findMatrixFloor(matrix, target, low, mid - 1);
        } else {
            return findMatrixFloor(matrix, target, mid, high);
        }
    }

    public static int findMatrixCeil(int[][] matrix, int target, int low, int high) {
        if (target <= matrix[low][matrix[0].length - 1]) {
            return low;
        } else if (target > matrix[high][matrix[0].length - 1]) {
            return high;
        }
        int mid = low + (high - low) / 2;
        if (mid < high && matrix[mid][matrix[0].length - 1] < target && matrix[mid + 1][matrix[0].length - 1] >= target) {
            return mid + 1;
        } else if (matrix[mid][matrix[0].length - 1] < target) {
            return findMatrixCeil(matrix, target, mid + 1, high);
        } else {
            return findMatrixCeil(matrix, target, low, mid);
        }
    }

//    https://leetcode.com/problems/search-a-2d-matrix/

    public static boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int floorRow = findMatrixFloor(matrix, target, 0, matrix.length - 1);
        return floorRow >= 0 && Arrays.binarySearch(matrix[floorRow], target) >= 0;
    }

//    https://leetcode.com/problems/sort-colors/

    public static void sortColors(int[] nums) {
        int l = 0;
        int m = 0;
        int h = nums.length - 1;
        while (m <= h) {
            if (nums[m] == 0) {
                int temp = nums[m];
                nums[m] = nums[l];
                nums[l] = temp;
                l++;
                m++;
            } else if (nums[m] == 1) {
                m++;
            } else {
                int temp = nums[m];
                nums[m] = nums[h];
                nums[h] = temp;
                h--;
            }
        }
    }

//    https://leetcode.com/problems/minimum-path-sum/

    public static int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int sum = 0;
        for (int i = 0; i < m; i++) {
            sum += grid[i][0];
            grid[i][0] = sum;
        }
        sum = 0;
        for (int j = 0; j < n; j++) {
            sum += grid[0][j];
            grid[0][j] = sum;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[m - 1][n - 1];
    }

//    https://leetcode.com/problems/subsets/

    public static void generateSubSets(int[] nums, int i, List<Integer> currList, List<List<Integer>> result) {
        result.add(new ArrayList<>(currList));
        for (int index = i; index < nums.length; index++) {
            currList.add(nums[index]);
            generateSubSets(nums, index + 1, currList, result);
            currList.remove(currList.size() - 1);
        }
    }

    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> currList = new ArrayList<>();
        generateSubSets(nums, 0, currList, result);
        return result;
    }

//    https://leetcode.com/problems/word-search/

    public static boolean wordSearchDFS(char[][] board, int i, int j, int count, String word) {
        if (count == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || word.charAt(count) != board[i][j]) {
            return false;
        }
        char temp = board[i][j];
        board[i][j] = ' ';
        boolean found = wordSearchDFS(board, i + 1, j, count + 1, word) ||
                wordSearchDFS(board, i - 1, j, count + 1, word) ||
                wordSearchDFS(board, i, j + 1, count + 1, word) ||
                wordSearchDFS(board, i, j - 1, count + 1, word);
        board[i][j] = temp;
        return found;
    }

    public static boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (word.charAt(0) == board[i][j] && wordSearchDFS(board, i, j, 0, word)) {
                    return true;
                }
            }
        }
        return false;
    }

//    https://leetcode.com/problems/search-in-rotated-sorted-array-ii/

    public static int getPivotInDuplicate(int[] nums, int low, int high) {
        if (low >= high) {
            return -1;
        }
        if (nums[low] < nums[high]) {
            return low;
        }
        int mid = low + (high - low) / 2;
        if (nums[low] == nums[high] && nums[low] == nums[mid]) {
            int right = getPivotInDuplicate(nums, mid + 1, high);
            if (right == -1) {
                return getPivotInDuplicate(nums, low, mid);
            }
            return right;
        }
        if (mid < high && nums[mid] > nums[mid + 1]) {
            return mid + 1;
        } else if (nums[low] <= nums[mid]) {
            return getPivotInDuplicate(nums, mid + 1, high);
        } else {
            return getPivotInDuplicate(nums, low, mid);
        }
    }


    public static boolean searchPivotDuplicate(int[] nums, int target) {
        if (nums.length == 0) {
            return false;
        }
        int pivot = getPivotInDuplicate(nums, 0, nums.length - 1);
        if (pivot == -1 || pivot == 0) {
            return binarySearch(nums, 0, nums.length - 1, target) >= 0;
        }
        int index = binarySearch(nums, 0, pivot - 1, target);
        if (index >= 0) {
            return true;
        }
        return binarySearch(nums, pivot, nums.length - 1, target) >= 0;
    }

//    https://leetcode.com/problems/subsets-ii/

    public static void generateSubsetsWithoutDup(int[] nums, int i, List<Integer> curr, List<List<Integer>> result) {
        result.add(new ArrayList<>(curr));
        for (int index = i; index < nums.length; index++) {
            if (i != index && nums[index] == nums[index - 1]) {
                continue;
            }
            curr.add(nums[index]);
            generateSubsetsWithoutDup(nums, index + 1, curr, result);
            curr.remove(curr.size() - 1);
        }
    }

    public static List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> current = new ArrayList<>();
        generateSubsetsWithoutDup(nums, 0, current, result);
        return result;
    }

}