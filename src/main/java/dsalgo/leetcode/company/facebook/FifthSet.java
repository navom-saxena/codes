package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;
import javafx.util.Pair;

import java.util.*;

public class FifthSet {

    public static void main(String[] args) {
//        System.out.println(findTargetSumWays(new int[]{0,0,0,0,0,0,0,0,1},1));
//        System.out.println(Arrays.toString(maxSlidingWindow(new int[]{1,3,-1,-3,5,3,6,7},3)));
//        System.out.println(Arrays.deepToString(updateBoard(new char[][]{{'E','E','E'},{'E','E','E'},{'E','E','E'},{'E','E','E'}},new int[]{0,0})));
//        System.out.println(canPartition(new int[]{3,7,19,27,40}));
//        longestOnes(new int[]{0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1},3);
//        System.out.println(longestArithSeqLength(new int[]{9,4,7,2,10}));
//        System.out.println(isValidPalindrome("bacabaaa",2));
//        NumMatrix numMatrix = new NumMatrix(new int[][]{{2,4}, {-3,5}});
//        numMatrix.update(0,1,3);
//        numMatrix.update(1,1,-3);
//        numMatrix.update(0,1,1);
//        System.out.println(numMatrix.sumRegion(0, 0, 1, 1));
//        System.out.println(findItinerary(Arrays.asList(Arrays.asList("AXA", "EZE"), Arrays.asList("EZE", "AUA"), Arrays.asList("ADL", "JFK"), Arrays.asList("ADL", "TIA"), Arrays.asList("AUA", "AXA"), Arrays.asList("EZE", "TIA"), Arrays.asList("EZE", "TIA"), Arrays.asList("AXA", "EZE"), Arrays.asList("EZE", "ADL"), Arrays.asList("ANU", "EZE"), Arrays.asList("TIA", "EZE"), Arrays.asList("JFK", "ADL"), Arrays.asList("AUA", "JFK"), Arrays.asList("JFK", "EZE"), Arrays.asList("EZE", "ANU"), Arrays.asList("ADL", "AUA"), Arrays.asList("ANU", "AXA"), Arrays.asList("AXA", "ADL"), Arrays.asList("AUA", "JFK"), Arrays.asList("EZE", "ADL"), Arrays.asList("ANU", "TIA"), Arrays.asList("AUA", "JFK"), Arrays.asList("TIA", "JFK"), Arrays.asList("EZE", "AUA"), Arrays.asList("AXA", "EZE"), Arrays.asList("AUA", "ANU"), Arrays.asList("ADL", "AXA"), Arrays.asList("EZE", "ADL"), Arrays.asList("AUA", "ANU"), Arrays.asList("AXA", "EZE"), Arrays.asList("TIA", "AUA"), Arrays.asList("AXA", "EZE"), Arrays.asList("AUA", "SYD"), Arrays.asList("ADL", "JFK"), Arrays.asList("EZE", "AUA"), Arrays.asList("ADL", "ANU"), Arrays.asList("AUA", "TIA"), Arrays.asList("ADL", "EZE"), Arrays.asList("TIA", "JFK"), Arrays.asList("AXA", "ANU"), Arrays.asList("JFK", "AXA"), Arrays.asList("JFK", "ADL"), Arrays.asList("ADL", "EZE"), Arrays.asList("AXA", "TIA"), Arrays.asList("JFK", "AUA"), Arrays.asList("ADL", "EZE"), Arrays.asList("JFK", "ADL"), Arrays.asList("ADL", "AXA"), Arrays.asList("TIA", "AUA"), Arrays.asList("AXA", "JFK"), Arrays.asList("ADL", "AUA"), Arrays.asList("TIA", "JFK"), Arrays.asList("JFK", "ADL"), Arrays.asList("JFK", "ADL"), Arrays.asList("ANU", "AXA"), Arrays.asList("TIA", "AXA"), Arrays.asList("EZE", "JFK"), Arrays.asList("EZE", "AXA"), Arrays.asList("ADL", "TIA"), Arrays.asList("JFK", "AUA"), Arrays.asList("TIA", "EZE"), Arrays.asList("EZE", "ADL"), Arrays.asList("JFK", "ANU"), Arrays.asList("TIA", "AUA"), Arrays.asList("EZE", "ADL"), Arrays.asList("ADL", "JFK"), Arrays.asList("ANU", "AXA"), Arrays.asList("AUA", "AXA"), Arrays.asList("ANU", "EZE"), Arrays.asList("ADL", "AXA"), Arrays.asList("ANU", "AXA"), Arrays.asList("TIA", "ADL"), Arrays.asList("JFK", "ADL"), Arrays.asList("JFK", "TIA"), Arrays.asList("AUA", "ADL"), Arrays.asList("AUA", "TIA"), Arrays.asList("TIA", "JFK"), Arrays.asList("EZE", "JFK"), Arrays.asList("AUA", "ADL"), Arrays.asList("ADL", "AUA"), Arrays.asList("EZE", "ANU"), Arrays.asList("ADL", "ANU"), Arrays.asList("AUA", "AXA"), Arrays.asList("AXA", "TIA"), Arrays.asList("AXA", "TIA"), Arrays.asList("ADL", "AXA"), Arrays.asList("EZE", "AXA"), Arrays.asList("AXA", "JFK"), Arrays.asList("JFK", "AUA"), Arrays.asList("ANU", "ADL"), Arrays.asList("AXA", "TIA"), Arrays.asList("ANU", "AUA"), Arrays.asList("JFK", "EZE"), Arrays.asList("AXA", "ADL"), Arrays.asList("TIA", "EZE"), Arrays.asList("JFK", "AXA"), Arrays.asList("AXA", "ADL"), Arrays.asList("EZE", "AUA"), Arrays.asList("AXA", "ANU"), Arrays.asList("ADL", "EZE"), Arrays.asList("AUA", "EZE"))));
//        System.out.println(isMatch("zacabz","*a?b*"));
//        System.out.println(Arrays.toString(medianSlidingWindow(new int[]{-2147483648, -2147483648, 2147483647, -2147483648, -2147483648, -2147483648, 2147483647, 2147483647, 2147483647, 2147483647, -2147483648, 2147483647, -2147483648}, 3)));
//        System.out.println(generateParenthesis(3));
        System.out.println(knightDialer(2));
    }

//    https://leetcode.com/problems/target-sum/

    static int findTargetSumWaysUtil(int[] nums, int i, int sum, int target, int[][] dp) {
        if (i == nums.length) {
            if (sum == target) {
                return 1;
            }
            return 0;
        }
        if (dp[i][sum + 1000] != Integer.MIN_VALUE) return dp[i][sum + 1000];
        int add = findTargetSumWaysUtil(nums, i + 1, sum + nums[i], target, dp);
        int sub = findTargetSumWaysUtil(nums, i + 1, sum - nums[i], target, dp);
        dp[i][sum + 1000] = add + sub;
        return dp[i][sum + 1000];
    }

    public static int findTargetSumWays(int[] nums, int target) {
        int[][] dp = new int[nums.length][2001];
        for (int[] row : dp) Arrays.fill(row, Integer.MIN_VALUE);
        return findTargetSumWaysUtil(nums, 0, 0, target, dp);
    }

//    https://leetcode.com/problems/sliding-window-maximum/

    public static int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> queue = new ArrayDeque<>();
        int[] result = new int[nums.length - k + 1];
        int i = 0;
        while (i < k) {
            while (!queue.isEmpty() && nums[queue.peekLast()] < nums[i]) queue.removeLast();
            queue.addLast(i);
            i++;
        }
        i = 0;
        if (!queue.isEmpty()) result[i] = nums[queue.peekFirst()];
        i++;
        while (i + k <= nums.length) {
            while (!queue.isEmpty() && queue.peekFirst() < i) queue.removeFirst();
            while (!queue.isEmpty() && nums[queue.peekLast()] < nums[i + k - 1]) queue.removeLast();
            queue.addLast(i + k - 1);
            if (!queue.isEmpty()) result[i] = nums[queue.peekFirst()];
            i++;
        }
        return result;
    }

//    https://leetcode.com/problems/robot-room-cleaner/

    interface Robot {
        public boolean move();

        public void turnRight();

        public void clean();
    }

    static void processCleaning(int dir, int[] current, int[][] directions, Robot robot, Set<Pair<Integer, Integer>> cleaned) {
        robot.clean();
        cleaned.add(new Pair<>(current[0], current[1]));
        for (int i = 0; i < 4; i++) {
            int nextDir = (dir + i) % 4;
            int x = current[0] + directions[nextDir][0];
            int y = current[1] + directions[nextDir][1];
            if (!cleaned.contains(new Pair<>(x, y)) && robot.move()) {
                processCleaning(nextDir, new int[]{x, y}, directions, robot, cleaned);
                robot.turnRight();
                robot.turnRight();
                robot.move();
                robot.turnRight();
                robot.turnRight();
            }
            robot.turnRight();
        }
    }

    public static void cleanRoom(Robot robot) {
        int[] location = new int[]{0, 0};
        int[][] directions = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        Set<Pair<Integer, Integer>> cleaned = new HashSet<>();
        processCleaning(0, location, directions, robot, cleaned);
    }

//    https://leetcode.com/problems/longest-increasing-path-in-a-matrix/

    static int longestIncreasingPathUtil(int[][] matrix, int i, int j, int prev, int[][] dp) {
        if (i < 0 || i >= matrix.length || j < 0 || j >= matrix[i].length || prev >= matrix[i][j]) return 0;
        if (dp[i][j] != 0) return dp[i][j];
        else {
            dp[i][j] = Math.max(dp[i][j], longestIncreasingPathUtil(matrix, i + 1, j, matrix[i][j], dp));
            dp[i][j] = Math.max(dp[i][j], longestIncreasingPathUtil(matrix, i - 1, j, matrix[i][j], dp));
            dp[i][j] = Math.max(dp[i][j], longestIncreasingPathUtil(matrix, i, j + 1, matrix[i][j], dp));
            dp[i][j] = Math.max(dp[i][j], longestIncreasingPathUtil(matrix, i, j - 1, matrix[i][j], dp));
            dp[i][j]++;
        }
        return dp[i][j];
    }

    public static int longestIncreasingPath(int[][] matrix) {
        int[][] dp = new int[matrix.length][matrix[0].length];
        int[] count = new int[]{0};
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                longestIncreasingPathUtil(matrix, i, j, -1, dp);
                count[0] = Math.max(count[0], dp[i][j]);
            }
        }
        return count[0];
    }

//    https://leetcode.com/problems/minesweeper/

    static boolean inBoard(char[][] board, int x, int y) {
        return x >= 0 && x < board.length && y >= 0 && y < board[x].length;
    }


    public static char[][] updateBoard(char[][] board, int[] click) {
        if (board[click[0]][click[1]] == 'M') board[click[0]][click[1]] = 'X';
        else {
            int[][] directions = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, +1}, {1, -1}, {1, 1}};
            Set<String> visited = new HashSet<>();
            Queue<int[]> queue = new LinkedList<>();
            queue.add(new int[]{click[0], click[1]});
            while (!queue.isEmpty()) {
                int[] node = queue.remove();
                String str = node[0] + " " + node[1];
                if (visited.contains(str)) continue;
                visited.add(str);
                int x = node[0];
                int y = node[1];
                int minesNo = 0;
                for (int[] direction : directions) {
                    int nx = x + direction[0];
                    int ny = y + direction[1];
                    if (inBoard(board, nx, ny) && board[nx][ny] == 'M') minesNo++;
                }
                if (minesNo != 0) board[x][y] = (char) ('0' + minesNo);
                else {
                    board[x][y] = 'B';
                    for (int[] direction : directions) {
                        int nx = x + direction[0];
                        int ny = y + direction[1];
                        if (inBoard(board, nx, ny) && !visited.contains(nx + " " + ny) && board[nx][ny] == 'E')
                            queue.add(new int[]{nx, ny});
                    }
                }
            }
        }
        return board;
    }

//    https://leetcode.com/problems/permutation-in-string/

    static boolean matches(int[] a1, int[] a2) {
        for (int i = 0; i < 26; i++) {
            if (a1[i] != a2[i]) return false;
        }
        return true;
    }

    public static boolean checkInclusion(String s1, String s2) {
        if (s2.length() < s1.length()) return false;
        int[] s1Map = new int[26];
        for (char c : s1.toCharArray()) {
            s1Map[c - 'a']++;
        }
        int[] s2Map = new int[26];
        int k = s1.length();
        char[] s2Arr = s2.toCharArray();
        for (int i = 0; i < k; i++) {
            s2Map[s2Arr[i] - 'a']++;
        }
        if (matches(s1Map, s2Map)) return true;
        for (int i = k; i < s2Arr.length; i++) {
            s2Map[s2Arr[i] - 'a']++;
            s2Map[s2Arr[i - k] - 'a']--;
            if (matches(s1Map, s2Map)) return true;
        }
        return false;
    }

//    https://leetcode.com/problems/meeting-rooms/

    public static boolean canAttendMeetings(int[][] intervals) {
        if (intervals.length == 0) return true;
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        int[] prev = intervals[0];
        for (int i = 1; i < intervals.length; i++) {
            if (prev[1] >= intervals[i][0]) return false;
            prev = intervals[i];
        }
        return true;
    }

//    https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/

    static int updateHeight(TreeNode node) {
        if (node == null) return 0;
        if (node.left == null && node.right == null) return 1;
        return Math.max(updateHeight(node.left), updateHeight(node.right)) + 1;
    }

    static TreeNode getLCADeepest(TreeNode node, int[] height, int currH) {
        if (node == null) return null;
        if (node.left == null && node.right == null) {
            if (currH == height[0]) return node;
            return null;
        }
        TreeNode left = getLCADeepest(node.left, height, currH + 1);
        TreeNode right = getLCADeepest(node.right, height, currH + 1);
        if (left != null && right != null) return node;
        else if (left != null) return left;
        return right;
    }

    public static TreeNode subtreeWithAllDeepest(TreeNode root) {
        int[] height = new int[]{updateHeight(root)};
        return getLCADeepest(root, height, 1);
    }

//    https://leetcode.com/problems/design-circular-queue/

    static class MyCircularQueue {
        int[] arr;
        int k;
        int s;
        int front;
        int rear;

        public MyCircularQueue(int k) {
            arr = new int[k];
            this.k = k;
            this.s = 0;
            this.front = 0;
            this.rear = -1;
        }

        public boolean enQueue(int value) {
            if (isFull()) return false;
            rear = (rear + 1) % k;
            arr[rear] = value;
            s++;
            return true;
        }

        public boolean deQueue() {
            if (isEmpty()) return false;
            front = (front + 1) % k;
            s--;
            return true;
        }

        public int Front() {
            return isEmpty() ? -1 : arr[front];
        }

        public int Rear() {
            return isEmpty() ? -1 : arr[rear];
        }

        public boolean isEmpty() {
            return s == 0;
        }

        public boolean isFull() {
            return s == k;
        }
    }

//    https://leetcode.com/problems/valid-word-abbreviation/

    public static boolean validWordAbbreviation(String word, String abbr) {
        char[] wordArr = word.toCharArray();
        char[] abbArr = abbr.toCharArray();
        int i = 0;
        int j = 0;
        int n = wordArr.length;
        int m = abbArr.length;
        while (i < n && j < m) {
            int no = 0;
            if (Character.isDigit(abbArr[j]) && abbArr[j] == '0') return false;
            while (j < m && Character.isDigit(abbArr[j])) {
                no = no * 10 + (abbArr[j] - '0');
                j++;
            }
            i += no;
            if (i == n && j == m) return true;
            if (i >= n || j >= m || wordArr[i] != abbArr[j]) return false;
            i++;
            j++;
        }
        return i == n && j == m;
    }

//    https://leetcode.com/problems/total-hamming-distance/

    public static int totalHammingDistance(int[] nums) {
        int sum = 0;
        int n = nums.length;
        int[] bitsArr = new int[32];
        for (int no : nums) {
            int pos = 0;
            while (no > 0) {
                if ((no & 1) != 0) {
                    bitsArr[pos]++;
                }
                pos++;
                no = no >> 1;
            }
        }
        for (int setBit : bitsArr) {
            sum += setBit * (n - setBit);
        }
        return sum;
    }

//    https://leetcode.com/problems/partition-equal-subset-sum/

    static boolean canPartitionUtil(int[] nums, int sum, int index, int runningSum, Set<Integer> processed) {
        if (processed.contains(runningSum)) return false;
        if (index < nums.length && runningSum > sum / 2) return false;
        if (index == nums.length) {
            return runningSum == sum / 2;
        }
        int takeItSum = runningSum + nums[index];
        boolean takeIt = canPartitionUtil(nums, sum, index + 1, takeItSum, processed);
        if (takeIt) return true;
        else processed.add(takeItSum);
        boolean dontTake = canPartitionUtil(nums, sum, index + 1, runningSum, processed);
        if (dontTake) return true;
        else processed.add(runningSum);
        return false;
    }

    public static boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) sum += num;
        if (sum % 2 != 0) return false;
        Set<Integer> processed = new HashSet<>();
        return canPartitionUtil(nums, sum, 0, 0, processed);
    }

//    https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/

    static Pair<TreeNode, Integer> lcaDeepestLeavesUtil(TreeNode node) {
        if (node == null) return new Pair<>(null, 0);
        if (node.left == null && node.right == null) return new Pair<>(node, 1);
        Pair<TreeNode, Integer> left = lcaDeepestLeavesUtil(node.left);
        Pair<TreeNode, Integer> right = lcaDeepestLeavesUtil(node.right);
        if (left.getValue() > right.getValue()) return new Pair<>(left.getKey(), left.getValue() + 1);
        else if (left.getValue() < right.getValue()) return new Pair<>(right.getKey(), right.getValue() + 1);
        else return new Pair<>(node, left.getValue() + 1);
    }

    public static TreeNode lcaDeepestLeaves(TreeNode root) {
        return lcaDeepestLeavesUtil(root).getKey();
    }

//    https://leetcode.com/problems/bulb-switcher/

    static void sieveOfEratosthenes(boolean[] no, int n) {
        for (int i = 2; i * i <= n; i++) {
            if (!no[i]) {
                int j = i * i;
                while (j <= n) {
                    no[j] = true;
                    j += i;
                }
            }
        }
    }

    public static int bulbSwitch(int n) {
        return (int) (Math.sqrt(n));
    }

//    https://leetcode.com/problems/max-consecutive-ones-iii/

    static void longestOnesUtil(int[] nums, int k, int i, int[] maxLength) {
        if (i == nums.length) {
            int length = 0;
            int maxLengthNow = 0;
            for (int no : nums) {
                if (no == 1) {
                    length++;
                } else {
                    length = 0;
                }
                maxLengthNow = Math.max(maxLengthNow, length);
            }
            if (maxLengthNow > maxLength[0]) {
                maxLength[0] = maxLengthNow;
            }
            return;
        }
        if (nums[i] == 0 && k > 0) {
            nums[i] = 1;
            longestOnesUtil(nums, k - 1, i + 1, maxLength);
            nums[i] = 0;
        }
        longestOnesUtil(nums, k, i + 1, maxLength);
    }

    public static int longestOnes(int[] nums, int k) {
        int maxLength = 0;
        int i = 0;
        int j = 0;
        int n = nums.length;
        int zCount = 0;
        while (i < n) {
            while (i < n && zCount <= k) {
                if (nums[i] == 0) zCount++;
                i++;
                if (zCount <= k) maxLength = Math.max(maxLength, i - j + 1);
            }
            while (j < n && zCount > k) {
                if (nums[j] == 0) zCount--;
                j++;
            }
        }
        return maxLength;
    }

//    https://leetcode.com/problems/split-array-with-equal-sum/

    public static boolean splitArray(int[] nums) {
        int[] prefixSumArr = new int[nums.length];
        int prefixSum = 0;
        for (int i = 0; i < nums.length; i++) {
            prefixSum += nums[i];
            prefixSumArr[i] = prefixSum;
        }
        for (int j = 3; j <= nums.length - 4; j++) {
            Set<Integer> matchedSumFirstHalf = new HashSet<>();
            for (int i = 1; i < j - 1; i++) {
                if (prefixSumArr[i - 1] == prefixSumArr[j - 1] - prefixSumArr[i])
                    matchedSumFirstHalf.add(prefixSumArr[i - 1]);
            }
            for (int k = j + 2; k < nums.length - 1; k++) {
                final int o = prefixSumArr[k - 1] - prefixSumArr[j];
                if (o == prefixSumArr[nums.length - 1] - prefixSumArr[k])
                    if (matchedSumFirstHalf.contains(o)) return true;
            }
        }
        return false;
    }

//    https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/

    public static String removeDuplicates(String s) {
        Deque<Character> stack = new ArrayDeque<>();
        for (char c : s.toCharArray()) {
            if (!stack.isEmpty() && stack.peek() == c) stack.pop();
            else stack.push(c);
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) sb.append(stack.pop());
        return sb.reverse().toString();
    }

//    https://leetcode.com/problems/longest-arithmetic-subsequence/

    static void longestArithSeqLengthUtil(int[] nums, int i, int diff, int count, int prevIndex, int[] maxLength) {
        if (count > maxLength[0]) {
            maxLength[0] = count;
        }
        if (i == nums.length) return;
        if (count == 0) {
            longestArithSeqLengthUtil(nums, i + 1, diff, count + 1, i, maxLength);
            longestArithSeqLengthUtil(nums, i + 1, diff, count, prevIndex, maxLength);
        } else if (count == 1) {
            longestArithSeqLengthUtil(nums, i + 1, nums[i] - nums[prevIndex], count + 1, i, maxLength);
            longestArithSeqLengthUtil(nums, i + 1, diff, count, prevIndex, maxLength);
        } else if (count > 1) {
            if (nums[i] - nums[prevIndex] == diff) {
                longestArithSeqLengthUtil(nums, i + 1, diff, count + 1, i, maxLength);
            } else {
                longestArithSeqLengthUtil(nums, i + 1, diff, count, prevIndex, maxLength);
            }
        }
    }

    public static int longestArithSeqLength(int[] nums) {
//        int [] maxLength = new int[]{1};
//        longestArithSeqLengthUtil(nums, 0, Integer.MIN_VALUE, 0, -1, maxLength);
//        return maxLength[0];
        List<Map<Integer, Integer>> seqMaps = new ArrayList<>();
        int maxLength = 0;
        for (int i = 0; i < nums.length; i++) {
            Map<Integer, Integer> indexMap = new HashMap<>();
            for (int j = 0; j < i; j++) {
                int diff = nums[i] - nums[j];
                if (seqMaps.get(j).get(diff) != null) indexMap.put(diff, seqMaps.get(j).get(diff) + 1);
                else indexMap.put(diff, 2);
                maxLength = Math.max(maxLength, indexMap.get(diff));
            }
            seqMaps.add(indexMap);
        }
        return maxLength;
    }

//    https://leetcode.com/problems/permutations/

    static void swap(List<Integer> nums, int i, int j) {
        int temp = nums.get(i);
        nums.set(i, nums.get(j));
        nums.set(j, temp);
    }

    static void permuteUtil(int i, int n, List<Integer> running, List<List<Integer>> result) {
        if (i == n) result.add(new ArrayList<>(running));
        for (int j = i; j < n; j++) {
            swap(running, i, j);
            permuteUtil(i + 1, n, running, result);
            swap(running, j, i);
        }
    }

    public static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> running = new ArrayList<>();
        for (int no : nums) running.add(no);
        permuteUtil(0, nums.length, running, result);
        return result;
    }

//    https://leetcode.com/problems/find-k-closest-elements/

    static int closestBinarySearch(int[] arr, int low, int high, int x) {
        if (low > high) return -1;
        if (low == high) return low;
        int mid = low + (high - low) / 2;
        if (arr[mid] == x) return mid;
        if (arr[mid] > x) {
            if (low < mid && arr[mid - 1] < x) {
                return x - arr[mid - 1] <= arr[mid] - x ? mid - 1 : mid;
            } else return closestBinarySearch(arr, low, mid - 1, x);
        } else {
            if (mid < high && arr[mid + 1] > x) {
                return x - arr[mid] <= arr[mid + 1] - x ? mid : mid + 1;
            } else return closestBinarySearch(arr, mid + 1, high, x);
        }
    }

    public static List<Integer> findClosestElements(int[] arr, int k, int x) {
        int index = closestBinarySearch(arr, 0, arr.length - 1, x);
        int i = index;
        int j = index + 1;
        List<Integer> res = new ArrayList<>();
        while (k != 0) {
            if (i >= 0 && j < arr.length) {
                if (x - arr[i] <= arr[j] - x) {
                    i--;
                } else {
                    j++;
                }
            } else if (i >= 0) {
                i--;
            } else if (j < arr.length) {
                j++;
            } else break;
            k--;
        }
        for (int l = i + 1; l < j; l++) {
            if (l >= 0 && l < arr.length) res.add(arr[l]);
        }
        return res;
    }

//    https://leetcode.com/problems/valid-palindrome-iii/

    static int isValidPalindromeUtil(char[] arr, int i, int j, int k, int[][] memo) {
        if (i >= j) return 0;
        if (arr[i] == arr[j]) return memo[i][j] = isValidPalindromeUtil(arr, i + 1, j - 1, k, memo);
        if (memo[i][j] != Integer.MAX_VALUE) return memo[i][j];
        return memo[i][j] = 1 + Math.min(isValidPalindromeUtil(arr, i + 1, j, k, memo)
                , isValidPalindromeUtil(arr, i, j - 1, k, memo));
    }

    public static boolean isValidPalindrome(String s, int k) {
        char[] sArr = s.toCharArray();
        int[][] memo = new int[s.length()][s.length()];
        for (int[] a : memo) Arrays.fill(a, Integer.MAX_VALUE);
        return isValidPalindromeUtil(sArr, 0, sArr.length - 1, k, memo) <= k;
    }

//    https://leetcode.com/problems/range-sum-query-2d-mutable/

    static class NumMatrix {

        int rows;
        int cols;
        int[][] bit;

        public NumMatrix(int[][] matrix) {
            rows = matrix.length;
            cols = matrix[0].length;
            bit = new int[rows + 1][cols + 1];
            buildBIT(bit, matrix);
        }

        void updateBIT(int[][] bit, int r, int c, int val) {
            for (int i = r; i <= rows; i += lsb(i)) {
                for (int j = c; j <= cols; j += lsb(j)) {
                    bit[i][j] += val;
                }
            }
        }

        int queryBIT(int[][] bit, int r, int c) {
            int sum = 0;
            for (int i = r; i > 0; i -= lsb(i)) {
                for (int j = c; j > 0; j -= lsb(j)) {
                    sum += bit[i][j];
                }
            }
            return sum;
        }

        void buildBIT(int[][] bit, int[][] matrix) {
            for (int i = 1; i <= rows; i++) {
                for (int j = 1; j <= cols; j++) {
                    updateBIT(bit, i, j, matrix[i - 1][j - 1]);
                }
            }
        }

        int lsb(int n) {
            return n & -n;
        }

        public void update(int row, int col, int val) {
            int oldVal = sumRegion(row, col, row, col);
            row++;
            col++;
            updateBIT(bit, row, col, val - oldVal);
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            row1++;
            col1++;
            row2++;
            col2++;
            int a = queryBIT(bit, row2, col2);
            int b = queryBIT(bit, row1 - 1, col1 - 1);
            int c = queryBIT(bit, row2, col1 - 1);
            int d = queryBIT(bit, row1 - 1, col2);
            return a + b - (c + d);
        }
    }

//    https://leetcode.com/problems/reconstruct-itinerary/

    static boolean dfsLexical(Map<String, Set<String>> adj, String start, List<String> itinerary,
                              Map<String, Integer> ticketSet) {
        itinerary.add(start);
        if (ticketSet.isEmpty()) return true;
        for (String neighbour : adj.getOrDefault(start, new TreeSet<>())) {
            String key = start + " " + neighbour;
            int f = ticketSet.getOrDefault(key, 0);
            if (f > 1) ticketSet.put(key, f - 1);
            else ticketSet.remove(key);
            if (f >= 1) {
                boolean b = dfsLexical(adj, neighbour, itinerary, ticketSet);
                if (b) return true;
                ticketSet.put(key, f);
                itinerary.remove(itinerary.size() - 1);
            }
        }
        return false;
    }

    public static List<String> findItinerary(List<List<String>> tickets) {
        Map<String, Set<String>> adj = new HashMap<>();
        Map<String, Integer> ticketSets = new HashMap<>();
        for (List<String> ticket : tickets) {
            Set<String> neighbours = adj.getOrDefault(ticket.get(0), new TreeSet<>());
            neighbours.add(ticket.get(1));
            adj.put(ticket.get(0), neighbours);
            ticketSets.merge(ticket.get(0) + " " + ticket.get(1), 1, Integer::sum);
        }
        String start = "JFK";
        List<String> itinerary = new ArrayList<>();
        dfsLexical(adj, start, itinerary, ticketSets);
        return itinerary;
    }

//    https://leetcode.com/problems/median-of-two-sorted-arrays/

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) return findMedianSortedArrays(nums2, nums1);
        int l = nums1.length + nums2.length + 1;
        int low = 0;
        int high = nums1.length;
        while (low <= high) {
            int partitionX = (low + high) / 2;
            int partitionY = l / 2 - partitionX;
            int maxLeftX = partitionX <= 0 ? Integer.MIN_VALUE : nums1[partitionX - 1];
            int minRightX = partitionX >= nums1.length ? Integer.MAX_VALUE : nums1[partitionX];
            int maxLeftY = partitionY <= 0 ? Integer.MIN_VALUE : nums2[partitionY - 1];
            int minRightY = partitionY >= nums2.length ? Integer.MAX_VALUE : nums2[partitionY];
            if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
                if (l % 2 == 0) return Math.max(maxLeftX, maxLeftY);
                else return (Math.max(maxLeftX, maxLeftY) + Math.min(minRightX, minRightY)) / 2.0;
            } else if (maxLeftX > minRightY) high = partitionX - 1;
            else low = partitionX + 1;
        }
        return -1;
    }

//    https://leetcode.com/problems/decode-string/

    static StringBuilder decodeStringUtil(char[] arr, int[] index) {
        StringBuilder sb = new StringBuilder();
        while (index[0] < arr.length) {
            int k = 0;
            while (index[0] < arr.length && Character.isDigit(arr[index[0]])) {
                k = k * 10 + (arr[index[0]] - '0');
                index[0]++;
            }
            if (arr[index[0]] == '[') {
                index[0]++;
                StringBuilder temp = decodeStringUtil(arr, index);
                while (k > 0) {
                    sb.append(temp);
                    k--;
                }
            }
            else if (arr[index[0]] == ']') {
                return sb;
            } else {
                sb.append(arr[index[0]]);
            }
            index[0]++;
        }
        return sb;
    }

    public String decodeString(String s) {
        char [] sArr = s.toCharArray();
        return decodeStringUtil(sArr, new int []{0}).toString();
    }

//    https://leetcode.com/problems/find-median-from-data-stream/

    static class MedianFinder {

        PriorityQueue<Integer> maxHeap;
        PriorityQueue<Integer> minHeap;

        public MedianFinder() {
            maxHeap = new PriorityQueue<>((a,b) -> b - a);
            minHeap = new PriorityQueue<>();
        }

        public void addNum(int num) {
            if (minHeap.isEmpty() || num >= minHeap.peek()) {
                minHeap.add(num);
            } else {
                maxHeap.add(num);
            }
            while (Math.abs(minHeap.size() - maxHeap.size()) > 1) {
                if (minHeap.size() > maxHeap.size()) maxHeap.add(minHeap.remove());
                else minHeap.add(maxHeap.remove());
            }
        }

        public double findMedian() {
            if ((minHeap.size() + maxHeap.size()) % 2 == 0) return (minHeap.peek() + maxHeap.peek()) / 2.0;
            else return minHeap.size() > maxHeap.size() ? minHeap.peek() : maxHeap.peek();
        }
    }

//    https://leetcode.com/problems/kth-smallest-element-in-a-bst/

    static Integer kThSmallestV;

    static void kthSmallestUtil(TreeNode node, int [] kV, int k) {
        if (node == null) return;
        if (kThSmallestV != null) return;
        kthSmallestUtil(node.left, kV, k);
        if (kV[0] == k) {
            kThSmallestV = node.val;
            kV[0]++;
            return;
        }
        kV[0]++;
        kthSmallestUtil(node.right, kV, k);
    }

    public static int kthSmallest(TreeNode root, int k) {
        int [] kV = new int[]{1};
        kthSmallestUtil(root, kV, k);
        return kThSmallestV;
    }

//    https://leetcode.com/problems/backspace-string-compare/

    public boolean backspaceCompare(String s, String t) {
        char [] sArr = s.toCharArray();
        char [] tArr = t.toCharArray();
        int n = sArr.length;
        int m = tArr.length;
        int i = n - 1;
        int j = m - 1;
        int hashS = 0;
        int hashT = 0;
        while (i >= 0 || j >= 0) {
            while (i >= 0) {
                if (sArr[i] == '#') { i--; hashS++;}
                else if (hashS > 0) {i--; hashS--;}
                else break;
            }
            while (j >= 0) {
                if (tArr[j] == '#') { j--; hashT++;}
                else if (hashT > 0) {j--; hashT--;}
                else break;
            }
            if (i >= 0 && j >= 0 && sArr[i] != tArr[j]) return false;
            if ((i >= 0) != (j >= 0)) return false;
            i--;
            j--;
        }
        return true;
    }

//    https://leetcode.com/problems/wildcard-matching/

    public static boolean isMatch(String s, String p) {
        char [] sArr = s.toCharArray();
        char [] pArr = p.toCharArray();
        boolean [][] dp = new boolean[sArr.length + 1][pArr.length + 1];
        dp[0][0] = true;
        if (pArr.length > 0 && pArr[0] == '*') dp[0][1] = true;
        for (int j = 2; j < dp[0].length; j++) {
            dp[0][j] = dp[0][j - 1] && pArr[j - 1] == '*';
        }
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[i].length; j++) {
                if (pArr[j - 1] == '?' || pArr[j - 1] == sArr[i - 1]) dp[i][j] = dp[i - 1][j - 1];
                else if (pArr[j - 1] == '*') dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }

//    https://leetcode.com/problems/count-and-say/

    public static String countAndSay(int n) {
        StringBuilder sb = new StringBuilder();
        sb.append(1);
        for (int i = 2; i <= n; i++) {
            StringBuilder temp = new StringBuilder();
            int c = 1;
            int prev = sb.charAt(0) - '0';
            for (int j = 1; j < sb.length(); j++) {
                if (sb.charAt(j) - '0' == prev) c++;
                else {
                    temp.append(c);
                    temp.append(prev);
                    c = 1;
                    prev = sb.charAt(j) - '0';
                }
            }
            temp.append(c);
            temp.append(prev);
            sb = temp;
        }
        return sb.toString();
    }

//    https://leetcode.com/problems/top-k-frequent-words/


    public static List<String> topKFrequent(String[] words, int k) {
        Map<String,Integer> hm = new HashMap<>();
        for (String word : words) {
            hm.merge(word, 1, Integer::sum);
        }
        PriorityQueue<String> maxHeap = new PriorityQueue<>
                ((a,b)-> Objects.equals(hm.get(a), hm.get(b)) ? a.compareTo(b) : hm.get(b) - hm.get(a));
        maxHeap.addAll(hm.keySet());
        List<String> res = new ArrayList<>();
        while (k > 0) {
            res.add(maxHeap.remove());
            k--;
        }
        return res;
    }

//    https://leetcode.com/problems/sliding-window-median/

    static void balanceHeaps(PriorityQueue<Double> maxHeap, PriorityQueue<Double> minHeap) {
        while (Math.abs(maxHeap.size() - minHeap.size()) > 1 ||
                (!maxHeap.isEmpty() && !minHeap.isEmpty() && maxHeap.peek() > minHeap.peek())) {
            if (minHeap.size() > maxHeap.size()) maxHeap.add(minHeap.remove());
            else minHeap.add(maxHeap.remove());
        }
    }

    static void addInHeaps(PriorityQueue<Double> maxHeap, PriorityQueue<Double> minHeap, int [] nums, int i) {
        if (minHeap.isEmpty() || minHeap.peek() < nums[i]) {
            minHeap.add((double) nums[i]);
        } else maxHeap.add((double) nums[i]);
       balanceHeaps(maxHeap, minHeap);
    }

    static double computeMedian(PriorityQueue<Double> maxHeap, PriorityQueue<Double> minHeap) {
        if ((maxHeap.size() + minHeap.size()) % 2 == 0) {
            return (maxHeap.peek() + minHeap.peek()) / 2.0;
        } else {
            return minHeap.size() > maxHeap.size() ? minHeap.peek() : maxHeap.peek();
        }
    }

    public static double[] medianSlidingWindow(int[] nums, int k) {
        double [] res = new double[nums.length - k + 1];
        PriorityQueue<Double> maxHeap = new PriorityQueue<>((a,b) -> (int) (b - a));
        PriorityQueue<Double> minHeap = new PriorityQueue<>();
        int m = 0;
        int i = 0;
        while (i < k) {
            addInHeaps(maxHeap, minHeap, nums, i);
            i++;
        }
        res[m] = computeMedian(maxHeap, minHeap);
        m++;
        while (i < nums.length) {
            addInHeaps(maxHeap, minHeap, nums, i);
            double last = nums[i - k];
            if (!minHeap.isEmpty() && last >= minHeap.peek()) minHeap.remove(last);
            else maxHeap.remove(last);
            balanceHeaps(maxHeap, minHeap);
            res[m] = computeMedian(maxHeap, minHeap);
            i++;
            m++;
        }
        return res;
    }

//    https://leetcode.com/problems/generate-parentheses/

    static void generateParenthesisUtil(int [] opening, int [] closing, int n, StringBuilder sb, List<String> res) {
        if (opening[0] == n && closing[0] == n) {
            res.add(sb.toString());
            return;
        }
        if (opening[0] < n) {
            sb.append('(');
            opening[0]++;
            generateParenthesisUtil(opening, closing, n, sb, res);
            sb.deleteCharAt(sb.length() - 1);
            opening[0]--;
        }
        if (closing[0] < opening[0]) {
            sb.append(')');
            closing[0]++;
            generateParenthesisUtil(opening, closing, n, sb, res);
            sb.deleteCharAt(sb.length() - 1);
            closing[0]--;
        }
    }

    public static List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        int [] opening = new int[]{0};
        int [] closing = new int[]{0};
        generateParenthesisUtil(opening,closing, n, new StringBuilder(), res);
        return res;
    }

//    https://leetcode.com/problems/reverse-linked-list-ii/

    static Pair<ListNode,ListNode> reverseBetweenUtil(ListNode curr, int right, int counter) {
        if (curr == null) return new Pair<>(null,null);
        if (counter == right) {
            ListNode nextToReturned = curr.next;
            curr.next = null;
            return new Pair<>(curr, nextToReturned);
        }
        ListNode nextNode = curr.next;
        Pair<ListNode,ListNode> returned = reverseBetweenUtil(curr.next, right, counter + 1);
        curr.next = null;
        nextNode.next = curr;
        return returned;
    }

    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode prev = null;
        ListNode curr = head;
        int counter = 1;
        while (curr != null && counter != left) {
            prev = curr;
            curr = curr.next;
            counter++;
        }
        if (curr == null) return head;
        ListNode temp = curr;
        Pair<ListNode,ListNode> reversedData = reverseBetweenUtil(curr, right, counter);
        ListNode reversedHead = reversedData.getKey();
        ListNode nextToReversed = reversedData.getValue();
        if (prev == null) head = reversedHead;
        else prev.next = reversedHead;
        temp.next = nextToReversed;
        return head;
    }

//    https://leetcode.com/problems/knight-dialer/

    public static int knightDialer(int n) {
        Map<Integer,List<Integer>> adj = new HashMap<>();
        adj.put(0, Arrays.asList(4,6));
        adj.put(1, Arrays.asList(6,8));
        adj.put(2, Arrays.asList(7,9));
        adj.put(3, Arrays.asList(4,8));
        adj.put(4, Arrays.asList(3,9,0));
        adj.put(5, new ArrayList<>());
        adj.put(6, Arrays.asList(7,0,1));
        adj.put(7, Arrays.asList(2,6));
        adj.put(8, Arrays.asList(1,3));
        adj.put(9, Arrays.asList(2,4));

        int [][] dp = new int[10][2];
        for (int i = 0; i < dp.length; i++) {
            dp[i][0] = 1;
        }
        int numDigits = 2;
        while (numDigits <= n) {
            for (int i = 0; i < dp.length; i++) {
                dp[i][1] = 0;
                for (int fromNode : adj.getOrDefault(i, new ArrayList<>())) {
                    dp[i][1] = (dp[i][1] + (dp[fromNode][0] % 1000000007)) % 1000000007;
                }
            }
            for (int i = 0; i < dp.length; i++) {
                dp[i][0] = dp[i][1];
            }
            numDigits++;
        }
        int sum = 0;
        for (int[] ints : dp) sum = (sum + (ints[0] % 1000000007)) % 1000000007;
        return sum;
    }

//    https://leetcode.com/problems/number-of-distinct-islands/

    void dfsDistinctIslands(int [][] grid, int i, int j, int absI, int absJ,
                            Set<String> visited, boolean [][] seen) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length ||
                grid[i][j] == 0 || seen[i][j]) return;
        int iVal = i - absI;
        int jVal = j - absJ;
        String node = iVal + " " + jVal;
        if (visited.contains(node)) return;
        visited.add(node);
        seen[i][j] = true;
        dfsDistinctIslands(grid, i + 1, j, absI, absJ, visited, seen);
        dfsDistinctIslands(grid, i - 1, j, absI, absJ, visited, seen);
        dfsDistinctIslands(grid, i, j + 1, absI, absJ, visited, seen);
        dfsDistinctIslands(grid, i, j - 1, absI, absJ, visited, seen);
    }

    public int numDistinctIslands(int[][] grid) {
        Set<Set<String>> distinctIslands = new HashSet<>();
        boolean [][] seen = new boolean[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    Set<String> visited = new HashSet<>();
                    dfsDistinctIslands(grid, i, j, i, j, visited, seen);
                    if (!visited.isEmpty()) distinctIslands.add(visited);
                }
            }
        }
        return distinctIslands.size();
    }

//    https://leetcode.com/problems/spiral-matrix-iii/

    boolean withinBounds(int i, int j, int r, int c) {
        return i >= 0 && i < r && j >= 0 && j < c;
    }

    public int[][] spiralMatrixIII(int rows, int cols, int rStart, int cStart) {
        int n = rows * cols;
        int [][] res = new int[n][2];
        int no = 0;
        int leftC = cStart;
        int rightC = cStart + 1;
        int upR = rStart;
        int downR = rStart + 1;
        while (no <= n) {
            boolean processed = false;
            for (int j = leftC; j < rightC; j++) {
                if (!withinBounds(upR, j, rows, cols)) continue;
                res[no] = new int[]{upR,j};
                no++;
                processed = true;
            }
            leftC--;
            for (int i = upR; i < downR; i++) {
                if (!withinBounds(i, rightC, rows, cols)) continue;
                res[no] = new int[]{i,rightC};
                no++;
                processed = true;
            }
            upR--;
            for (int j = rightC; j > leftC; j--) {
                if (!withinBounds(downR, j, rows, cols)) continue;
                res[no] = new int[]{downR,j};
                no++;
                processed = true;
            }
            rightC++;
            for (int i = downR; i > upR; i--) {
                if (!withinBounds(i, leftC, rows, cols)) continue;
                res[no] = new int[]{i,leftC};
                no++;
                processed = true;
            }
            downR++;
            if (!processed) break;
        }
        return res;
    }

//    https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/

    public String removeDuplicates(String s, int k) {
        Deque<Pair<Character,Integer>> stack = new ArrayDeque<>();
        for (char c : s.toCharArray()) {
            if (stack.isEmpty() || stack.peek().getKey() != c) stack.push(new Pair<>(c,1));
            else {
                Pair<Character,Integer> t = stack.pop();
                stack.push(new Pair<>(t.getKey(),t.getValue() + 1));
            }
            if (!stack.isEmpty() && stack.peek().getValue() == k) stack.pop();
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            Pair<Character,Integer> t = stack.pop();
            for (int i = 1; i <= t.getValue(); i++) {
                sb.append(t.getKey());
            }
        }
        return sb.reverse().toString();
    }

//    https://leetcode.com/problems/find-largest-value-in-each-tree-row/

    void largestValuesUtil(TreeNode node, int i, List<Integer> res) {
        if (node == null) return;
        if (res.size() == i) res.add(node.val);
        else res.set(i, Math.max(res.get(i),node.val));
        largestValuesUtil(node.left, i + 1, res);
        largestValuesUtil(node.right, i + 1, res);
    }

    public List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        largestValuesUtil(root, 0, res);
        return res;
    }

}