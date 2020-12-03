package dsalgo.leetcode.arrays;

import java.util.*;

public class ArrayMedium2 {

    static int preOrderIndex = 0;
    static int postOrderIndex = 0;
    static int minTotal = Integer.MAX_VALUE;
    static Set<String> mainSet = new HashSet<>();

    public static void main(String[] args) {
//        TreeNode root = buildTreeFromPreIn(new int[]{3, 9, 20, 15, 7}, new int[]{9, 3, 15, 20, 7});
//        System.out.println(root.val);
//        TreeNode postOrderRoot = buildTreeFromPostIn(new int[]{9,15,7,20,3}, new int[]{9,3,15,20,7});
//        System.out.println(postOrderRoot.val);
//        System.out.println(minimumTotal(
//                Stream.of(Stream.of(2).collect(Collectors.toList()),
//                Stream.of(3,4).collect(Collectors.toList()),
//                Stream.of(5,6,7).collect(Collectors.toList()),
//                Stream.of(4,1,8,3).collect(Collectors.toList())).collect(Collectors.toList())));
//        System.out.println(maxProduct(new int[]{-2, 3, -4}));
//        System.out.println(findPeakElement(new int[]{3,4,3,2,1}));
//        minDenominations();
//        System.out.println(minSubArrayLen(5, new int[]{4, 3, 2, 1}));
//        System.out.println(combinationSum3(9, 45));
//        System.out.println(summaryRanges(new int[]{0,1,2,4,5,7}));
//        System.out.println(majorityElement2(new int[]{1, 2}));
//        System.out.println(Arrays.toString(productExceptSelf(new int[]{1, 2, 3, 4})));
//        System.out.println(findDuplicate(new int[]{1, 1, 2}));
//        gameOfLife(new int[][]{{0,1,0},{0,0,1},{1,1,1},{0,0,0}});
//        RandomizedSet rs = new RandomizedSet();
//        rs.insert(1);
//        rs.remove(2);
//        rs.insert(2);
//        System.out.println(rs.getRandom());
//        rs.remove(1);
//        rs.insert(2);
//        System.out.println(rs.getRandom());
//        System.out.println(findDuplicates(new int[]{4,3,2,7,8,2,3,1}));
//        System.out.println(circularArrayLoop(new int[]{-8,-1,1,7,2}));
//        System.out.println(findPoisonedDuration(new int[]{0,1,2,3,4,5},1));
//        System.out.println(subarraySum(new int[]{1}, 0));
//        System.out.println(arrayNesting(new int[]{0,1,2,3,4}));
//        System.out.println(pivotOfOneZero(new int[]{0}, 0, 3));
//        System.out.println(coinChange(new int[]{186,419,83,408},6249));
//        combinationTest(new int[] {1,2,3}, 0, new ArrayList<>());
//        System.out.println(longestCommonPrefix(new String[]{"dog", "racecar", "car"}));
        System.out.println(reverseVowels("leetcode"));
    }

//    https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

    static int getIndex(int[] arr, int low, int high, int value) {
        if (low > high) {
            return -1;
        }
        for (int i = low; i <= high; i++) {
            if (arr[i] == value) {
                return i;
            }
        }
        return -1;
    }

    public static TreeNode buildTreeRecursivelyFromPreIn(int[] preorder, int[] inorder,
                                                         int inOrderStart, int inOrderEnd) {
        if (inOrderStart > inOrderEnd) {
            return null;
        }
        int currentNodeData = preorder[preOrderIndex];
        int currentNodeIndex = getIndex(inorder, inOrderStart, inOrderEnd, currentNodeData);
        preOrderIndex++;
        TreeNode left = buildTreeRecursivelyFromPreIn(preorder, inorder, inOrderStart, currentNodeIndex - 1);
        TreeNode right = buildTreeRecursivelyFromPreIn(preorder, inorder, currentNodeIndex + 1, inOrderEnd);
        TreeNode current = new TreeNode(currentNodeData);
        current.left = left;
        current.right = right;
        return current;
    }

//    https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/

    public static TreeNode buildTreeFromPreIn(int[] preorder, int[] inorder) {
        return buildTreeRecursivelyFromPreIn(preorder, inorder, 0, inorder.length - 1);
    }

    public static TreeNode buildTreeRecursivelyFromPostIn(int[] postOrder, int[] inorder,
                                                          int inOrderStart, int inOrderEnd) {
        if (inOrderStart > inOrderEnd) {
            return null;
        }
        int currentNodeData = postOrder[postOrderIndex];
        int currentNodeIndex = getIndex(inorder, inOrderStart, inOrderEnd, currentNodeData);
        postOrderIndex--;
        TreeNode right = buildTreeRecursivelyFromPostIn(postOrder, inorder, currentNodeIndex + 1, inOrderEnd);
        TreeNode left = buildTreeRecursivelyFromPostIn(postOrder, inorder, inOrderStart, currentNodeIndex - 1);
        TreeNode current = new TreeNode(currentNodeData);
        current.right = right;
        current.left = left;
        return current;
    }

    public static TreeNode buildTreeFromPostIn(int[] postorder, int[] inorder) {
        postOrderIndex = postorder.length - 1;
        return buildTreeRecursivelyFromPostIn(postorder, inorder, 0, inorder.length - 1);
    }

//    https://leetcode.com/problems/triangle/

    public static void minimumTotalRecursively(List<List<Integer>> triangle,
                                               int rowIndex, int moveValue, int sum, Set<String> s) {
        if (rowIndex == triangle.size()) {
            mainSet.addAll(s);
            minTotal = Math.min(minTotal, sum);
            return;
        }
        if (moveValue < 0 || moveValue >= triangle.get(rowIndex).size()) {
            return;
        }
        if (mainSet.contains(rowIndex + "-" + moveValue)) {
            return;
        }
        s.add(rowIndex + "-" + moveValue);
        sum += triangle.get(rowIndex).get(moveValue);
        minimumTotalRecursively(triangle, rowIndex + 1, moveValue, sum, s);
        minimumTotalRecursively(triangle, rowIndex + 1, moveValue + 1, sum, s);
    }

    public static int minimumTotal(List<List<Integer>> triangle) {
        int length = triangle.size();
        int[] dpArr = new int[length + 1];
        for (int i = length - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                dpArr[j] = Math.min(dpArr[j], dpArr[j + 1]) + triangle.get(i).get(j);
            }
        }
        return dpArr[0];
    }

//    https://leetcode.com/problems/maximum-product-subarray/

    public static int maxProduct(int[] nums) {
        int max = nums[0];
        int maxSoFar = nums[0];
        int minSoFar = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int temp = maxSoFar;
            maxSoFar = Math.max(maxSoFar * nums[i], Math.max(minSoFar * nums[i], nums[i]));
            minSoFar = Math.min(temp * nums[i], Math.min(minSoFar * nums[i], nums[i]));
            max = Math.max(max, maxSoFar);
        }
        return max;
    }

//    https://leetcode.com/problems/find-peak-element/

    public static int findPeakElement(int[] nums) {
        if (nums.length == 1) {
            return 0;
        }
        int low = 0;
        int high = nums.length - 1;
        if (nums[low] > nums[low + 1]) {
            return low;
        }
        if (nums[high] > nums[high - 1]) {
            return high;
        }
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (mid == 0) {
                low = low + 1;
            } else if (nums[mid] > nums[mid + 1] && nums[mid] > nums[mid - 1]) {
                return mid;
            } else if (low < mid && nums[mid] < nums[mid - 1]) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return -1;
    }

    static void minDenominations() {
        int a = 242;
        int count = 0;
        int[] d = new int[]{1, 2, 5, 10, 50, 100};
        int i = d.length - 1;
        while (a != 0) {
            if (a >= d[i]) {
                while (a / d[i] != 0) {
                    count = count + (a / d[i]);
                    a = a % d[i];
                    System.out.println(a + " " + d[i] + " " + i + " " + count);
                }
            } else {
                i--;
            }
        }
        System.out.println(count);
    }

//    https://leetcode.com/problems/minimum-size-subarray-sum/

    public static int minSubArrayLen(int s, int[] nums) {
        int min = nums.length + 1;
        int start = 0;
        int end = 0;
        int sum = 0;
        while (end != nums.length) {
            sum += nums[end];
            end++;
            while (sum >= s) {
                min = Math.min(min, end - start);
                sum -= nums[start];
                start++;
            }
        }
        return min == nums.length + 1 ? 0 : min;
    }

//    https://leetcode.com/problems/combination-sum-iii/

    static void createCombinationSum3(List<List<Integer>> result, int[] arr, int index,
                                      int k, int n, List<Integer> curr) {
        if (curr.size() == k && n == 0) {
            result.add(new ArrayList<>(curr));
            return;
        }
        for (int i = index; i < arr.length; i++) {
            curr.add(arr[i]);
            createCombinationSum3(result, arr, i + 1, k, n - arr[i], curr);
            curr.remove(curr.size() - 1);
        }
    }

    public static List<List<Integer>> combinationSum3(int k, int n) {
        int[] arr = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> curr = new ArrayList<>();
        createCombinationSum3(result, arr, 0, k, n, curr);
        return result;
    }

//    https://leetcode.com/problems/summary-ranges/

    public static String getRange(int min, int max) {
        StringBuilder result = new StringBuilder();
        result.append(min);
        if (min != max) {
            result.append("->").append(max);
        }
        return result.toString();
    }

    public static List<String> summaryRanges(int[] nums) {
        List<String> result = new ArrayList<>();
        if (nums.length == 0) {
            return result;
        }
        int start = nums[0];
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] + 1 != nums[i + 1]) {
                result.add(getRange(start, nums[i]));
                start = nums[i + 1];
            }
        }
        result.add(getRange(start, nums[nums.length - 1]));
        return result;
    }

//    https://leetcode.com/problems/majority-element-ii/

    public static List<Integer> majorityElement2(int[] nums) {
        int majorityElement1 = Integer.MIN_VALUE;
        int majorityElement2 = Integer.MIN_VALUE;
        int counter1 = 0;
        int counter2 = 0;
        for (int num : nums) {
            if (num == majorityElement1) {
                counter1++;
            } else if (num == majorityElement2) {
                counter2++;
            } else if (counter1 == 0) {
                majorityElement1 = num;
                counter1 = 1;
            } else if (counter2 == 0) {
                majorityElement2 = num;
                counter2 = 1;
            } else {
                counter1--;
                counter2--;
            }
        }
        List<Integer> result = new ArrayList<>();
        counter1 = 0;
        counter2 = 0;
        for (int num : nums) {
            if (num == majorityElement1) {
                counter1++;
            } else if (num == majorityElement2) {
                counter2++;
            }
        }
        if (counter1 > nums.length / 3) {
            result.add(majorityElement1);
        }
        if (counter2 > nums.length / 3) {
            result.add(majorityElement2);
        }
        return result;
    }

//    https://leetcode.com/problems/product-of-array-except-self/

    public static int[] productExceptSelf(int[] nums) {
        int[] result = new int[nums.length];
        result[nums.length - 1] = 1;
        int suffixProduct = 1;
        for (int i = nums.length - 2; i >= 0; i--) {
            suffixProduct *= nums[i + 1];
            result[i] = suffixProduct;
        }
        int prefixProduct = 1;
        for (int i = 0; i < nums.length; i++) {
            result[i] *= prefixProduct;
            prefixProduct *= nums[i];
        }
        return result;
    }

//    https://leetcode.com/problems/find-the-duplicate-number/

    static int binarySearchDuplicateInN(int[] arr, int low, int high) {
        if (low > high) {
            return -1;
        }
        int mid = low + (high - low) / 2;
        if (arr[mid] == arr[mid + 1]) {
            return arr[mid];
        }
        if (arr[mid] == mid + 1) {
            return binarySearchDuplicateInN(arr, mid + 1, high);
        } else {
            return binarySearchDuplicateInN(arr, low, mid);
        }
    }

    public static int findDuplicate(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            int value;
            if (nums[i] < 0) {
                value = nums[i] * -1;
            } else {
                value = nums[i];
            }
            if (nums[value] < 0) {
                return value;
            } else {
                nums[value] = -nums[value];
            }
        }
        return -1;
    }

//    https://leetcode.com/problems/game-of-life/

    public static int getValueFromIndex(int i, int j, int m, int n, int[][] arr) {
        if (i < 0 || i > m - 1 || j < 0 || j > n - 1) {
            return 0;
        }
        if (arr[i][j] > 0) {
            return 1;
        } else {
            return 0;
        }
    }

    public static void gameOfLife(int[][] board) {
        int m = board.length;
        int n = board[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int sum = getValueFromIndex(i - 1, j, m, n, board) + getValueFromIndex(i + 1, j, m, n, board) +
                        getValueFromIndex(i, j - 1, m, n, board) + getValueFromIndex(i, j + 1, m, n, board) +
                        getValueFromIndex(i - 1, j - 1, m, n, board) + getValueFromIndex(i - 1, j + 1, m, n, board) +
                        getValueFromIndex(i + 1, j - 1, m, n, board) + getValueFromIndex(i + 1, j + 1, m, n, board);
                if (board[i][j] == 0 && sum == 3) {
                    board[i][j] = -10;
                } else if (board[i][j] == 1 && sum < 2) {
                    board[i][j] = 10;
                } else if (board[i][j] == 1 && (sum == 2 || sum == 3)) {
                    board[i][j] = board[i][j];
                } else if (board[i][j] == 1 && sum > 3) {
                    board[i][j] = 10;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == -10) {
                    board[i][j] = 1;
                } else if (board[i][j] == 10) {
                    board[i][j] = 0;
                }
            }
        }
        System.out.println(Arrays.deepToString(board));
    }

//    https://leetcode.com/problems/find-all-duplicates-in-an-array/

    public static List<Integer> findDuplicates(int[] nums) {
        List<Integer> result = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < nums.length; i++) {
            int value;
            if (nums[i] < 0) {
                value = nums[i] * -1;
            } else {
                value = nums[i];
            }
            if (nums[value % n] < 0) {
                result.add(value);
            } else {
                nums[value % n] = -nums[value % n];
            }
        }
        return result;
    }

//    https://leetcode.com/problems/circular-array-loop/

    public static boolean circularArrayLoop(int[] nums) {
        if (nums.length <= 1) {
            return false;
        }
        int n = nums.length;
        for (int i = 0; i < nums.length; i++) {
            int slowPointer = i;
            int fastPointer = i;
            while (true) {
                int prevSlowPointer = slowPointer;
                slowPointer = (slowPointer + (((nums[(slowPointer)] % n) + n) % n)) % n;
                fastPointer = (fastPointer + (((nums[(fastPointer)] % n) + n) % n)) % n;
                fastPointer = (fastPointer + (((nums[(fastPointer)] % n) + n) % n)) % n;
                if (nums[prevSlowPointer] * nums[slowPointer] < 0) {
                    break;
                } else if (slowPointer == fastPointer && prevSlowPointer == slowPointer) {
                    break;
                } else if (slowPointer == fastPointer && slowPointer == i) {
                    return true;
                } else if (slowPointer == fastPointer) {
                    break;
                }
            }
        }
        return false;
    }


    static int increasedInstances(int instances, List<Integer> averageUtil) {
        System.out.println(averageUtil + " " + instances);
        final int upperLimit = 200000000;
        int size = averageUtil.size();
        for (int i = 0; i < size; i++) {
            int currentUtilization = averageUtil.get(i);
            if (currentUtilization < 25) {
                if (instances > 1) {
                    int divisionNo = instances / 2;
                    int cielIncease = instances % 2 == 0 ? 0 : 1;
                    instances = divisionNo + cielIncease;
                    i = i + 10;
                }
            } else if (currentUtilization > 60) {
                int increasedInstances = instances * 2;
                if (increasedInstances <= upperLimit) {
                    instances = increasedInstances;
                    i = i + 10;
                }
            }
        }
        return instances;
    }

    /* https://leetcode.com/problems/teemo-attacking/ */

    public static int findPoisonedDuration(int[] timeSeries, int duration) {
        int poisonedTime = 0;
        int tillTime = -1;
        for (int currTime : timeSeries) {
            if (currTime > tillTime) {
                poisonedTime += duration;
            } else {
                int newTillTime = currTime + duration - 1;
                poisonedTime = poisonedTime + newTillTime - tillTime;
            }
            tillTime = currTime + duration - 1;
        }
        return poisonedTime;
    }

    /* https://leetcode.com/problems/subarray-sum-equals-k/ */

    public static int subarraySum(int[] nums, int k) {
        int count = 0;
        Map<Integer, Integer> hm = new HashMap<>();
        hm.put(0, 1);
        int runningSum = 0;
        for (int num : nums) {
            runningSum += num;
            if (hm.containsKey(runningSum - k)) {
                count = count + hm.get(runningSum - k);
            }
            hm.merge(runningSum, 1, Integer::sum);
        }
        return count;
    }

//    https://leetcode.com/problems/array-nesting/

    public static int arrayNesting(int[] nums) {
        boolean[] visited = new boolean[nums.length];
        int max = Integer.MIN_VALUE;
        int count;
        for (int i = 0; i < nums.length; i++) {
            count = 0;
            int j = i;
            while (!visited[j]) {
                visited[j] = true;
                j = nums[j];
                count++;
            }
            max = Math.max(max, count);
        }
        return max;
    }

//    https://leetcode.com/problems/valid-triangle-number/

    public static int triangleNumber(int[] nums) {
        int n = nums.length - 1;
        Arrays.sort(nums);
        int count = 0;
        for (int i = n; i >= 2; i--) {
            int l = 0;
            int r = i - 1;
            while (l < r) {
                if (nums[l] + nums[r] > nums[i]) {
                    count += r - l;
                    r--;
                } else {
                    l++;
                }
            }
        }
        return count;
    }

//    find count of zeros in sorted array of 1 n 0

    static int pivotOfOneZero(int[] arr, int low, int high) {
        if (low > high) {
            return -1;
        } else if (arr[0] == 0) {
            return low;
        } else if (arr[arr.length - 1] == 1) {
            return arr.length;
        }
        int mid = low + (high - low) / 2;
        if (mid < high && arr[mid] == 1 && arr[mid + 1] == 0) {
            return mid + 1;
        } else if (low < mid && arr[mid] == 0 && arr[mid - 1] == 1) {
            return mid;
        } else if (arr[mid] == 1) {
            return pivotOfOneZero(arr, mid + 1, high);
        } else {
            return pivotOfOneZero(arr, low, mid - 1);
        }
    }

//    https://leetcode.com/problems/coin-change/submissions/

    public static int coinChange(int[] coins, int amount) {
        Arrays.sort(coins);
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 0; i <= amount; i++) {
            for (int coin : coins) {
                if (i >= coin) {
                    dp[i] = Math.min(dp[i], dp[i - coin]);
                } else {
                    break;
                }
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }

    static void combinationTest(int[] arr, int index, List<Integer> currentList) {
        System.out.println(currentList);
        if (index == arr.length) {
            return;
        }
        for (int i = index; i < arr.length; i++) {
            currentList.add(arr[i]);
            combinationTest(arr, i + 1, currentList);
            currentList.remove(currentList.size() - 1);
        }
    }

//   https://leetcode.com/problems/partition-labels/

    public static List<Integer> partitionLabels(String s) {
        List<Integer> partitionsLengths = new ArrayList<>();
        int[] lastValueOfChar = new int[26];
        for (int i = 0; i < s.length(); i++) {
            lastValueOfChar[s.charAt(i) - 'a'] = i;
        }
        int i = 0;
        while (i < s.length()) {
            int lastValueIndex = lastValueOfChar[s.charAt(i) - 'a'];
            int j = i;
            while (j <= lastValueIndex) {
                int currentLastIndex = lastValueOfChar[s.charAt(j) - 'a'];
                if (lastValueIndex < currentLastIndex) {
                    lastValueIndex = currentLastIndex;
                }
                j++;
            }
            partitionsLengths.add(j - i + 1);
            i = j;
        }
        return partitionsLengths;
    }

//    https://leetcode.com/problems/longest-common-prefix/

    public static String longestCommonPrefix(String[] strs) {
        StringBuilder commonPrefix = new StringBuilder();
        if (strs.length == 0) {
            return commonPrefix.toString();
        }
        int k = 0;
        for (char c : strs[0].toCharArray()) {
            for (int i = 1; i < strs.length; i++) {
                if (k >= strs[i].length() || c != strs[i].charAt(k)) {
                    return commonPrefix.toString();
                }
            }
            commonPrefix.append(c);
            k++;
        }
        return commonPrefix.toString();
    }

//    https://leetcode.com/problems/reverse-vowels-of-a-string/

    public static boolean isVowel(char element){
        if(element == 'a' ||element == 'e' ||element == 'i' ||element == 'o' ||element == 'u') {
            return true;
        }
        else return element == 'A' || element == 'E' || element == 'I' || element == 'O' || element == 'U';
    }

    public static String reverseVowels(String s) {
        int i = 0;
        int j = s.length() - 1;
        char [] sArr = s.toCharArray();
        while (i < j) {
            char left = sArr[i];
            char right = sArr[j];
            if (isVowel(left) && isVowel(right)) {
                sArr[i] = right;
                sArr[j] = left;
                i++;
                j--;
            } else if (isVowel(left)) {
                j--;
            } else if (isVowel(right)) {
                i++;
            } else {
                i++;
                j--;
            }
        }
        return String.valueOf(sArr);
    }

    /* https://leetcode.com/problems/insert-delete-getrandom-o1/ */

    static class RandomizedSet {

        Map<Integer, Integer> hm;
        List<Integer> arr;
        Random rand;

        public RandomizedSet() {
            hm = new HashMap<>();
            arr = new ArrayList<>();
            rand = new Random();
        }

        /**
         * Inserts a value to the set. Returns true if the set did not already contain the specified element.
         */
        public boolean insert(int val) {
            if (hm.containsKey(val)) {
                return false;
            } else {
                hm.put(val, 1);
                arr.add(val);
                return true;
            }
        }

        /**
         * Removes a value from the set. Returns true if the set contained the specified element.
         */
        public boolean remove(int val) {
            boolean flag = false;
            if (hm.containsKey(val)) {
                flag = true;
                hm.remove(val);
                int index = arr.indexOf(val);
                arr.set(index, arr.get(arr.size() - 1));
                arr.remove(arr.size() - 1);
            }
            return flag;
        }

        /**
         * Get a random element from the set.
         */
        public int getRandom() {
            int nextRand = rand.nextInt(arr.size());
            return arr.get(nextRand);
        }
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

}
