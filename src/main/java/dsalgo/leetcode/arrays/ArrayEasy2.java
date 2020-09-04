package dsalgo.leetcode.arrays;

import java.util.*;

public class ArrayEasy2 {

    static double toDivide = 9;

//    https://leetcode.com/problems/reshape-the-matrix/

    public static void main(String[] args) {
//        System.out.println(Arrays.stream(matrixReshape(new int[][]{{1,2},{3,4}},2,4))
//        .map(row -> Arrays.stream(row).mapToObj(String::valueOf).collect(Collectors.joining(" ")))
//        .collect(Collectors.joining(",")));
//        System.out.println(findUnsortedSubarray(new int[]{1,2,3,4}));
//        System.out.println(canPlaceFlowers(new int[]{0}, 0));
//        System.out.println(maximumProduct(new int[]{-12, 0,2,93,44}));
//        System.out.println(findMaxAverage(new int[]{0,1,1,3,3},4));
//        System.out.println(Arrays.deepToString(imageSmoother(new int[][]{{1,1,1},{1,0,1},{1,1,1}})));
//        System.out.println(checkPossibility(new int[]{4,2,1}));
//        System.out.println(findLengthOfLCIS(new int[]{2,2,2,2,2}));
//        System.out.println(findShortestSubArray(new int[]{1,2,2,3,1}));
//        System.out.println(isOneBitCharacter(new int[]{0}));
//        System.out.println(pivotIndex(new int[]{-1,-1,1,1,0,0}));
//        System.out.println(minCostClimbingStairs(new int[]{10, 15, 20}));
//        System.out.println(dominantIndex(new int[]{1,1,1,3}));
//        System.out.println(isToeplitzMatrix(new int[][]{{1,2,3,4}}));
//        System.out.println(largeGroupPositions("aaa"));
//        System.out.println(Arrays.stream(flipAndInvertImage(new int[][]{{1,2},{3,4}}))
//        .map(row -> Arrays.stream(row).mapToObj(String::valueOf).collect(Collectors.joining(" ")))
//        .collect(Collectors.joining(",")));
//        System.out.println(numMagicSquaresInside(new int[][]{{5, 5, 5}, {5, 5, 5}, {5, 5, 5}}));
//        System.out.println(maxDistToClosest(new int[]{0,1,1,1,0,0,1,0,0}));
//        System.out.println(Arrays.stream(transpose(new int[][]{{1, 2, 3}, {4, 5, 6}}))
//                .map(row -> Arrays.stream(row).mapToObj(String::valueOf).collect(Collectors.joining(" ")))
//                .collect(Collectors.joining(",")));
//        System.out.println(Arrays.toString(fairCandySwap(new int[]{1, 2, 5}, new int[]{2, 4})));
//        System.out.println(isMonotonic(new int[]{1,1,1}));
//        System.out.println(Arrays.toString(sortArrayByParity(new int[]{3, 1, 2, 4})));
//        System.out.println(hasGroupsSizeX(new int[]{0,0,0,0,0,0,0,0,0,1,1,1,2,2,2,3,3,3}));
//        System.out.println(findHCF(15,3));
//        System.out.println(Arrays.toString(sortArrayByParityII(new int[]{4, 2, 5, 7})));
//        System.out.println(validMountainArray(new int[]{3,4}));
//        System.out.println(fib(4));
//        System.out.println(Arrays.toString(sortedSquares(new int[]{-4, -1, 0, 3, 10})));
        System.out.println(Arrays.toString(
                sumEvenAfterQueries(new int[]{1, 2, 3, 4},new int[][]{{1,0},{-3,1},{-4,0},{2,3}})));
    }

//    https://leetcode.com/problems/shortest-unsorted-continuous-subarray/

    public static int[][] matrixReshape(int[][] nums, int r, int c) {
        if (nums == null || nums.length == 0) {
            return nums;
        }
        int rowsLength = nums.length;
        int columnsLength = nums[0].length;
        if (rowsLength * columnsLength != r * c) {
            return nums;
        }
        int[] rowWiseTraversal = new int[r * c];
        int k = 0;
        for (int[] row : nums) {
            for (int value : row) {
                rowWiseTraversal[k] = value;
                k++;
            }
        }
        k = 0;
        int[][] newArr = new int[r][c];
        for (int i = 0; i < newArr.length; i++) {
            for (int j = 0; j < newArr[i].length; j++) {
                newArr[i][j] = rowWiseTraversal[k];
                k++;
            }
        }
        return newArr;
    }

//    https://leetcode.com/problems/can-place-flowers/

    public static int findUnsortedSubarray(int[] nums) {
        Integer startingIndex = null;
        Integer endingIndex = null;
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] > nums[i + 1] && startingIndex == null) {
                startingIndex = i;
                int j = i;
                while (j > 0 && nums[j] == nums[j - 1]) {
                    j--;
                    startingIndex = j;
                }
            }
        }
        for (int i = nums.length - 2; i >= 0; i--) {
            if (nums[i] > nums[i + 1] && endingIndex == null) {
                endingIndex = i + 1;
                int j = i + 1;
                while (j < nums.length - 1 && nums[j] == nums[j + 1]) {
                    j++;
                    endingIndex = j;
                }
            }
        }
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        Integer i = startingIndex;
        while (i != null && endingIndex != null && i <= endingIndex) {
            max = Math.max(max, nums[i]);
            min = Math.min(min, nums[i]);
            i++;
        }
        i = startingIndex;
        while (i != null && i >= 0 && nums[i] > min) {
            i--;
        }
        if (i != null) {
            startingIndex = i + 1;
        }
        i = endingIndex;
        while (i != null && i < nums.length && nums[i] < max) {
            i++;
        }
        if (i != null) {
            endingIndex = i - 1;
        }
        if (endingIndex != null) {
            return endingIndex - startingIndex + 1;
        } else {
            return 0;
        }
    }

//    https://leetcode.com/problems/maximum-product-of-three-numbers/

    public static boolean canPlaceFlowers(int[] flowerbed, int n) {
        if (flowerbed.length == 1) {
            if (flowerbed[0] == 0) {
                return n <= 1;
            } else {
                return n == 0;
            }
        }
        for (int i = 0; i < flowerbed.length; i++) {
            if (n <= 0) {
                return true;
            }
            if (i == 0) {
                if ((flowerbed[i] == 0 && flowerbed[i + 1] == 0)) {
                    flowerbed[i] = 1;
                    n--;
                }
            } else if (i == flowerbed.length - 1) {
                if (flowerbed[i] == 0 && flowerbed[i - 1] == 0) {
                    flowerbed[i] = 1;
                    n--;
                }
            } else {
                if (flowerbed[i] == 1) {
                    if (!(flowerbed[i - 1] == 0 && flowerbed[i + 1] == 0)) {
                        return false;
                    }
                } else {
                    if (flowerbed[i - 1] == 0 && flowerbed[i + 1] == 0) {
                        flowerbed[i] = 1;
                        n--;
                    }
                }
            }
        }
        return n <= 0;
    }

//    https://leetcode.com/problems/maximum-average-subarray-i/

    public static int maximumProduct(int[] nums) {
        int maxA = Integer.MIN_VALUE;
        int maxB = Integer.MIN_VALUE;
        int maxC = Integer.MIN_VALUE;
        int minA = Integer.MAX_VALUE;
        int minB = Integer.MAX_VALUE;
        for (int number : nums) {
            if (number > maxA) {
                maxC = maxB;
                maxB = maxA;
                maxA = number;
            } else if (number > maxB) {
                maxC = maxB;
                maxB = number;
            } else if (number > maxC) {
                maxC = number;
            }
            if (number < minA) {
                minB = minA;
                minA = number;
            } else if (number < minB) {
                minB = number;
            }
        }
        return Math.max(maxA * maxB * maxC, maxA * minA * minB);
    }

//    https://leetcode.com/problems/image-smoother/

    public static double findMaxAverage(int[] nums, int k) {
        int subArraySum = 0;
        double maxSubArraySum = Integer.MIN_VALUE;
        for (int i = 0; i < k; i++) {
            subArraySum += nums[i];
        }
        maxSubArraySum = subArraySum;
        for (int i = k; i < nums.length; i++) {
            subArraySum = subArraySum - nums[i - k] + nums[i];
            maxSubArraySum = Math.max(maxSubArraySum, subArraySum);
        }
        return maxSubArraySum / k;
    }

    public static int[][] imageSmoother(int[][] M) {
        int[][] smoothMatrix = new int[M.length][M[0].length];
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[i].length; j++) {
                smoothMatrix[i][j] = (int) ((
                        getValueOrZero(M, i, j) +
                                getValueOrZero(M, i, j - 1) + getValueOrZero(M, i, j + 1) +
                                getValueOrZero(M, i - 1, j) + getValueOrZero(M, i - 1, j - 1) +
                                getValueOrZero(M, i - 1, j + 1) + getValueOrZero(M, i + 1, j) +
                                getValueOrZero(M, i + 1, j - 1) + getValueOrZero(M, i + 1, j + 1)) / toDivide);
                toDivide = 9;
            }
        }
        return smoothMatrix;
    }

    static int getValueOrZero(int[][] matrix, int i, int j) {
        try {
            return matrix[i][j];
        } catch (IndexOutOfBoundsException e) {
            toDivide--;
            return 0;
        }
    }

//    https://leetcode.com/problems/non-decreasing-array/

    public static boolean checkPossibility(int[] nums) {
        if (nums.length <= 1) {
            return true;
        }
        int count = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] < nums[i - 1]) {
                count++;
                if (i == 1 || nums[i - 2] <= nums[i]) {
                    nums[i - 1] = nums[i];
                } else {
                    nums[i] = nums[i - 1];
                }
            }
        }
        return count <= 1;
    }

//    https://leetcode.com/problems/longest-continuous-increasing-subsequence/

    public static int findLengthOfLCIS(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int maxLength = 1;
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1]) {
                count++;
            } else {
                count = 1;
            }
            maxLength = Math.max(maxLength, count);
        }
        return maxLength;
    }

//    https://leetcode.com/problems/degree-of-an-array/

    public static int findShortestSubArray(int[] nums) {
        Map<Integer, Integer> left = new HashMap<>(), right = new HashMap<>(), count = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            int number = nums[i];
            left.putIfAbsent(number, i);
            right.put(number, i);
            count.merge(number, 1, Integer::sum);
        }
        int minLength = Integer.MAX_VALUE;
        int degree = Collections.max(count.values());
        for (int number : count.keySet()) {
            int frequency = count.get(number);
            if (frequency == degree) {
                minLength = Math.min(minLength, right.get(number) - left.get(number) + 1);
            }
        }
        return minLength;
    }

//    https://leetcode.com/problems/1-bit-and-2-bit-characters/

    public static boolean isOneBitCharacter(int[] bits) {
        int i = 0;
        while (i < bits.length) {
            if (i == bits.length - 1) {
                return true;
            }
            if (bits[i] == 1) {
                i = i + 2;
            } else {
                i = i + 1;
            }
        }
        return false;
    }

//    https://leetcode.com/problems/find-pivot-index/

    public static int pivotIndex(int[] nums) {
        int sum = 0;
        for (int number : nums) {
            sum += number;
        }
        int preFixSum = 0;
        for (int i = 0; i < nums.length; i++) {
            if (preFixSum == sum - nums[i] - preFixSum) {
                return i;
            }
            preFixSum += nums[i];
        }
        return -1;
    }

//    https://leetcode.com/problems/min-cost-climbing-stairs/

    public static int minCostClimbingStairs(int[] cost) {
        int i = 2;
        while (i < cost.length) {
            cost[i] += Math.min(cost[i - 2], cost[i - 1]);
            i = i + 1;
        }
        return Math.min(cost[i - 2], cost[i - 1]);
    }

//    https://leetcode.com/problems/largest-number-at-least-twice-of-others/

    public static int dominantIndex(int[] nums) {
        int max = Integer.MIN_VALUE;
        int maxIndex = -1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > max) {
                max = nums[i];
                maxIndex = i;
            }
        }
        for (int number : nums) {
            if (number * 2 > max && number != max) {
                return -1;
            }
        }
        return maxIndex;
    }

//    https://leetcode.com/problems/toeplitz-matrix/

    public static boolean isToeplitzMatrix(int[][] matrix) {
        int i = 0;
        while (i < matrix.length) {
            int j = 0;
            if (i == 0) {
                while (j < matrix[i].length - 1) {
                    int firstValue = matrix[i][j];
                    int k = i;
                    int l = j;
                    while (k < matrix.length && l < matrix[k].length) {
                        if (matrix[k][l] != firstValue) {
                            return false;
                        }
                        k++;
                        l++;
                    }
                    j++;
                }
            } else {
                int firstValue = matrix[i][j];
                int k = i;
                int l = j;
                while (k < matrix.length && l < matrix[k].length) {
                    if (matrix[k][l] != firstValue) {
                        return false;
                    }
                    k++;
                    l++;
                }
            }
            i++;
        }
        return true;
    }

//    https://leetcode.com/problems/positions-of-large-groups/

    public static List<List<Integer>> largeGroupPositions(String s) {
        List<List<Integer>> result = new ArrayList<>();
        int secondPointer = 0;
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i - 1) != s.charAt(i)) {
                if (i - secondPointer >= 3) {
                    List<Integer> arr = new ArrayList<>();
                    arr.add(secondPointer);
                    arr.add(i - 1);
                    result.add(arr);
                }
                secondPointer = i;
            } else {
                if (i == s.length() - 1 && i - secondPointer + 1 >= 3) {
                    List<Integer> arr = new ArrayList<>();
                    arr.add(secondPointer);
                    arr.add(i);
                    result.add(arr);
                }
            }
        }
        return result;
    }

//    https://leetcode.com/problems/flipping-an-image/

    public static int[][] flipAndInvertImage(int[][] matrix) {
        for (int[] row : matrix) {
            int i = 0;
            int j = row.length - 1;
            while (i <= j) {
                if (row[i] == 0 && row[j] == 0) {
                    row[i] = 1;
                    row[j] = 1;
                } else if (row[i] == 1 && row[j] == 0) {
                    row[i] = 1;
                    row[j] = 0;
                } else if (row[i] == 0 && row[j] == 1) {
                    row[i] = 0;
                    row[j] = 1;
                } else {
                    row[i] = 0;
                    row[j] = 0;
                }
                i++;
                j--;
            }
        }
        return matrix;
    }

//    https://leetcode.com/problems/magic-squares-in-grid/

    public static int numMagicSquaresInside(int[][] grid) {
        int magicSqCount = 0;
        for (int i = 0; i < grid.length - 2; i++) {
            for (int j = 0; j < grid[i].length - 2; j++) {
                int magicNo = grid[i][j] + grid[i][j + 1] + grid[i][j + 2];
                boolean flag = true;
                int[] distinctArr = new int[10];
                if (grid[i][j] > 0 && grid[i][j] < 10 && distinctArr[grid[i][j]] == 0) {
                    distinctArr[grid[i][j]] = 1;
                } else {
                    flag = false;
                }
                if (grid[i][j + 1] > 0 && grid[i][j + 1] < 10 && distinctArr[grid[i][j + 1]] == 0) {
                    distinctArr[grid[i][j + 1]] = 1;
                } else {
                    flag = false;
                }
                if (grid[i][j + 2] > 0 && grid[i][j + 2] < 10 && distinctArr[grid[i][j + 2]] == 0) {
                    distinctArr[grid[i][j + 2]] = 1;
                } else {
                    flag = false;
                }
                if (grid[i + 1][j] > 0 && grid[i + 1][j] < 10 && distinctArr[grid[i + 1][j]] == 0) {
                    distinctArr[grid[i + 1][j]] = 1;
                } else {
                    flag = false;
                }
                if (grid[i + 1][j + 1] > 0 && grid[i + 1][j + 1] < 10 && distinctArr[grid[i + 1][j + 1]] == 0) {
                    distinctArr[grid[i + 1][j + 1]] = 1;
                } else {
                    flag = false;
                }
                if (grid[i + 1][j + 2] > 0 && grid[i + 1][j + 2] < 10 && distinctArr[grid[i + 1][j + 2]] == 0) {
                    distinctArr[grid[i + 1][j + 2]] = 1;
                } else {
                    flag = false;
                }
                if (grid[i + 2][j] > 0 && grid[i + 2][j] < 10 && distinctArr[grid[i + 2][j]] == 0) {
                    distinctArr[grid[i + 2][j]] = 1;
                } else {
                    flag = false;
                }
                if (grid[i + 2][j + 1] > 0 && grid[i + 2][j + 1] < 10 && distinctArr[grid[i + 2][j + 1]] == 0) {
                    distinctArr[grid[i + 2][j + 1]] = 1;
                } else {
                    flag = false;
                }
                if (grid[i + 2][j + 2] > 0 && grid[i + 2][j + 2] < 10 && distinctArr[grid[i + 2][j + 2]] == 0) {
                    distinctArr[grid[i + 2][j + 2]] = 1;
                } else {
                    flag = false;
                }
                if (flag && grid[i + 1][j] + grid[i + 1][j + 1] + grid[i + 1][j + 2] == magicNo
                        && grid[i + 2][j] + grid[i + 2][j + 1] + grid[i + 2][j + 2] == magicNo
                        && grid[i][j] + grid[i + 1][j] + grid[i + 2][j] == magicNo
                        && grid[i][j + 1] + grid[i + 1][j + 1] + grid[i + 2][j + 1] == magicNo
                        && grid[i][j + 2] + grid[i + 1][j + 2] + grid[i + 2][j + 2] == magicNo
                        && grid[i][j] + grid[i + 1][j + 1] + grid[i + 2][j + 2] == magicNo
                        && grid[i][j + 2] + grid[i + 1][j + 1] + grid[i + 2][j] == magicNo
                ) {
                    magicSqCount++;
                }
            }
        }
        return magicSqCount;
    }

//    https://leetcode.com/problems/maximize-distance-to-closest-person/

    public static int maxDistToClosest(int[] seats) {
        int start = 0;
        int mostSpacedValue = -1;
        for (int i = 0; i < seats.length; i++) {
            if (seats[i] == 1) {
                if (start == 0 && seats[start] == 0) {
                    mostSpacedValue = i;
                } else if ((i - start) / 2 > mostSpacedValue) {
                    mostSpacedValue = (i - start) / 2;
                }
                start = i;
            }
        }
        if ((seats.length - start) / 2 >= mostSpacedValue) {
            return seats.length - 1 - start;
        }
        return mostSpacedValue;
    }

//    https://leetcode.com/problems/transpose-matrix/

    public static int[][] transpose(int[][] matrix) {
        int[][] transposedMatrix = new int[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                transposedMatrix[j][i] = matrix[i][j];
            }
        }
        return transposedMatrix;
    }

//    https://leetcode.com/problems/fair-candy-swap/

    public static int[] fairCandySwap(int[] a, int[] b) {
        int sumA = 0;
        Set<Integer> hsA = new HashSet<>();
        for (int number : a) {
            sumA += number;
            hsA.add(number);
        }
        int sumB = 0;
        for (int number : b) {
            sumB += number;
        }
        int[] result = new int[2];
        int diff = (sumA - sumB) / 2;
        for (int number : b) {
            if (hsA.contains(number + diff)) {
                result[0] = number + diff;
                result[1] = number;
                break;
            }
        }
        return result;
    }

//    https://leetcode.com/problems/monotonic-array/

    public static boolean isMonotonic(int[] a) {
        if (a[0] <= a[a.length - 1]) {
            for (int i = 1; i < a.length; i++) {
                if (a[i - 1] > a[i]) {
                    return false;
                }
            }
        } else {
            for (int i = 1; i < a.length; i++) {
                if (a[i - 1] < a[i]) {
                    return false;
                }
            }
        }
        return true;
    }

//    https://leetcode.com/problems/sort-array-by-parity/

    public static int[] sortArrayByParity(int[] a) {
        int evenPointer = 0;
        int oddPointer = a.length - 1;
        while (evenPointer < oddPointer) {
            if (a[evenPointer] % 2 != 0 && a[oddPointer] % 2 == 0) {
                int temp = a[evenPointer];
                a[evenPointer] = a[oddPointer];
                a[oddPointer] = temp;
                evenPointer++;
                oddPointer--;
            } else {
                if (a[evenPointer] % 2 == 0) {
                    evenPointer++;
                }
                if (a[oddPointer] % 2 != 0) {
                    oddPointer--;
                }
            }
        }
        return a;
    }

//    https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/

    public static boolean hasGroupsSizeX(int[] deck) {
        Map<Integer, Integer> hm = new HashMap<>();
        for (int number : deck) {
            hm.merge(number, 1, Integer::sum);
        }
        Integer frequency = null;
        boolean allEven = true;
        for (int key : hm.keySet()) {
            int value = hm.get(key);
            if (frequency == null) {
                frequency = value;
            }
            if (value % 2 != 0) {
                allEven = false;
            }
            if (!allEven) {
                frequency = findHCF(frequency, value);
            }
        }
        if (allEven) {
            return true;
        }
        return frequency >= 2;
    }

    static int findHCF(int a, int b) {
        int tempLarger;
        int tempSmaller;
        if (a > b) {
            tempLarger = a;
            tempSmaller = b;
        } else {
            tempLarger = b;
            tempSmaller = a;
        }
        while (tempSmaller != 0) {
            int temp = tempSmaller;
            tempSmaller = tempLarger % tempSmaller;
            tempLarger = temp;
        }
        return tempLarger;
    }

//    https://leetcode.com/problems/sort-array-by-parity-ii/

    public static int[] sortArrayByParityII(int[] a) {
        int i = 0;
        int j = 1;
        while (j < a.length && i < a.length) {
            if (a[j] % 2 == 0 && a[i] % 2 != 0) {
                int temp = a[i];
                a[i] = a[j];
                a[j] = temp;
                i = i + 2;
                j = j + 2;
            } else if (a[j] % 2 != 0 && a[i] % 2 == 0) {
                i = i + 2;
                j = j + 2;
            } else if (a[j] % 2 != 0) {
                j = j + 2;
            } else {
                i = i + 2;
            }
        }
        return a;
    }

//    https://leetcode.com/problems/valid-mountain-array/

    public static boolean validMountainArray(int[] a) {
        if (a.length < 3) {
            return false;
        }
        boolean goingUp = false;
        boolean once = false;
        for (int i = 1; i < a.length; i++) {
            if (a[i - 1] < a[i]) {
                goingUp = true;
            } else {
                if (goingUp && !once) {
                    goingUp = false;
                    once = true;
                } else if ((!goingUp && !once)){
                    return false;
                }
                if (!(a[i - 1] > a[i] && !goingUp)) {
                    return false;
                }
            }
        }
        return once && !goingUp;
    }

//    https://leetcode.com/problems/fibonacci-number/

    public static int fib(int n) {
        if (n == 0) {
            return 0;
        }
        int firstValue = 0;
        int secondValue = 1;
        for (int i = 1; i < n; i++) {
            int temp = secondValue + firstValue;
            firstValue = secondValue;
            secondValue = temp;
        }
        return secondValue;
    }

//    https://leetcode.com/problems/squares-of-a-sorted-array/

    public static int[] sortedSquares(int[] a) {
        int breakPoint = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] >= 0) {
                breakPoint = i;
                break;
            }
        }
        int [] result = new int[a.length];
        int i = breakPoint - 1;
        int j = breakPoint;
        int resultIndex = 0;
        while (i >= 0 && j < a.length) {
            if (a[i] * a[i] < a[j] * a[j]) {
                result[resultIndex] = a[i] * a[i];
                resultIndex++;
                i--;
            } else {
                result[resultIndex] = a[j] * a[j];
                resultIndex++;
                j++;
            }
        }
        while (i >= 0) {
            result[resultIndex] = a[i] * a[i];
            resultIndex++;
            i--;
        }
        while (j < a.length) {
            result[resultIndex] = a[j] * a[j];
            resultIndex++;
            j++;
        }
        return result;
    }

//    https://leetcode.com/problems/sum-of-even-numbers-after-queries/

    public static int[] sumEvenAfterQueries(int[] a, int[][] queries) {
        int sum = 0;
        for (int num : a) {
            if (num % 2 == 0) {
                sum += num;
            }
        }
        int [] answer = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            int value = queries[i][0];
            int index = queries[i][1];
            int oldValue = a[index];
            if (oldValue % 2 == 0 && value % 2 == 0) {
                sum = sum + value;
            } else if (oldValue % 2 != 0 && value % 2 != 0) {
                sum += oldValue + value;
            } else if (oldValue % 2 == 0) {
                sum -= oldValue;
            }
            a[index] = oldValue + value;
            answer[i] = sum;
        }
        return answer;
    }

}
