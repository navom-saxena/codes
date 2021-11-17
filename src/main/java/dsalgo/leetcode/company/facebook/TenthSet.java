package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;
import javafx.util.Pair;
import org.apache.spark.sql.sources.In;

import java.math.BigInteger;
import java.util.*;

public class TenthSet {

    public static void main(String[] args) {
//        Skiplist skiplist = new Skiplist();
//        skiplist.add(1);
//        skiplist.add(2);
//        skiplist.add(3);
//        System.out.println(skiplist.search(0)); // return False
//        skiplist.add(4);
//        System.out.println(skiplist.search(1));  // return True
//        System.out.println(skiplist.erase(0));  // return False, 0 is not in skiplist.
//        System.out.println(skiplist.erase(1));  // return True
//        System.out.println(skiplist.search(1)); // return False, 1 has already been erased.

//        NumArray numArray = new NumArray(new int[]{1,3,5});

//        System.out.println(subarraysDivByK(new int[]{2,-2,2,-4},6));

    }

//     backtracking - choice, constraint, goal..choice - decision tree, for loop from 1 to 9 in sukodu,
//     pick or not pick all ahead numbers in arr, and for each choice, check if there is a constraint.
//    finally, base case should abe goal.

    //    https://leetcode.com/problems/asteroid-collision/

    public int[] asteroidCollision(int[] asteroids) {
        Deque<Integer> stack = new ArrayDeque<>();
        for (int no : asteroids) {
            if (no < 0) {
                while (!stack.isEmpty() && stack.peek() > 0 && stack.peek() < Math.abs(no)) stack.pop();
                if (!stack.isEmpty() && stack.peek() > 0 && stack.peek() == Math.abs(no)) stack.pop();
                else if (stack.isEmpty() || stack.peek() < 0) stack.push(no);
            } else stack.push(no);
        }
        int [] res = new int[stack.size()];
        for (int i = res.length - 1; i >= 0; i--) res[i] = stack.pop();
        return res;
    }

//    https://leetcode.com/problems/reverse-integer/

    public int reverse(int x) {
        int sign = x > 0 ? 1 : -1;
        x = Math.abs(x);
        long v = 0;
        while (x != 0) {
            int ones = x % 10;
            v = (v * 10) + ones;
            if (v >= Integer.MAX_VALUE || (sign == -1 && v * sign <= Integer.MIN_VALUE)) return 0;
            x = x / 10;
        }
        return (int) v * sign;
    }

//    https://leetcode.com/problems/non-overlapping-intervals/

    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        int l = 0;
        int r = 1;
        int count = 0;

        while (r < intervals.length) {
            int [] first = intervals[l];
            int [] second = intervals[r];
            if (first[1] <= second[0]) {
                l = r;
                r++;
            } else {
                if (first[1] >= second[1]) l = r;
                r++;
                count++;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/climbing-stairs/

    public int climbStairs(int n) {
        if (n == 1) return 1;
        int [] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i < dp.length; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[dp.length - 1];
    }

//    https://leetcode.com/problems/palindrome-number/

    public boolean isPalindrome(int x) {
        if (x < 0) return false;
        long v = 0;
        int o = x;
        while (x != 0) {
            int ones = x % 10;
            v = (v * 10) + ones;
            x /= 10;
        }
        while (o != 0 && v != 0) {
            if (o % 10 != v % 10) return false;
            o /= 10;
            v /= 10;
        }
        return true;
    }

//    https://leetcode.com/problems/maximum-depth-of-binary-tree/

    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return Math.max(left,right) + 1;
    }

//    https://leetcode.com/problems/invert-binary-tree/

    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        if (root.left == null && root.right == null) return root;
        TreeNode left = invertTree(root.left);
        root.left = invertTree(root.right);
        root.right = left;
        return root;
    }

//    https://leetcode.com/problems/balanced-binary-tree/

    boolean isBalancedValue = true;

    int isBalancedUtil(TreeNode node) {
        if (node == null) return 0;
        if (node.left == null && node.right == null) return 1;
        if (!isBalancedValue) return -1;
        int left = isBalancedUtil(node.left);
        int right = isBalancedUtil(node.right);
        if (Math.abs(left - right) > 1) isBalancedValue = false;
        return Math.max(left,right) + 1;
    }

    public boolean isBalanced(TreeNode root) {
        isBalancedUtil(root);
        return isBalancedValue;
    }

//    https://leetcode.com/problems/increasing-order-search-tree/

    TreeNode inorderHead = null;
    TreeNode inOrderCurr = null;

    void increasingBSTUtil(TreeNode node) {
        if (node == null) return;
        increasingBSTUtil(node.left);
        node.left = null;
        if (inorderHead == null) {
            inorderHead = node;
            inOrderCurr = node;
        } else {
            inOrderCurr.right = node;
            inOrderCurr = inOrderCurr.right;
        }
        increasingBSTUtil(node.right);
    }

    public TreeNode increasingBST(TreeNode root) {
        increasingBSTUtil(root);
        return inorderHead;
    }

//    https://leetcode.com/problems/reverse-bits/

    // last bit raised to the highest power so far because we are reversing. 1 at index = 0 will be 1 at
    // index 31 - index, then we divide n by 2.

    public int reverseBits(int n) {
        int v = 0;
        for (int i = 31; i >= 0; i--) {
            int lastBit = n & 1;
            v += lastBit << i;
            n = n >> 1;
        }
        return v;
    }

//    https://leetcode.com/problems/break-a-palindrome/

    public static String breakPalindrome(String palindrome) {
        if (palindrome.length() <= 1) return "";

        int n = palindrome.length();
        char [] p = palindrome.toCharArray();

        for (int i = 0; i < n / 2; i++) {
            if (p[i] > 'a') {
                p[i] = 'a';
                return String.valueOf(p);
            }
        }

        p[n - 1] = 'b';
        return String.valueOf(p);
    }

//    https://leetcode.com/problems/partition-to-k-equal-sum-subsets/

    static boolean canPartitionKSubsetsUtil(int [] nums, int index, int currSum,
                                            int c, int k, int targetSum, boolean [] visited) {
        if (c == k - 1) return true;
        if (currSum > targetSum) return false;
        if (currSum == targetSum) return canPartitionKSubsetsUtil(nums, 0, 0, c + 1, k, targetSum, visited);

        for (int i = index; i < nums.length; i++) {
            if (!visited[i]) {
                visited[i] = true;
                if (canPartitionKSubsetsUtil(nums, i + 1, currSum + nums[i], c, k, targetSum, visited)) return true;
                visited[i] = false;
            }
        }
        return false;
    }

    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = 0;
        for (int num : nums) sum += num;
        if (sum % k != 0) return false;
        Arrays.sort(nums);
        return canPartitionKSubsetsUtil(nums, 0, 0, 0, k,sum / k, new boolean[nums.length]);
    }

//    https://leetcode.com/problems/zigzag-conversion/

    public String convert(String s, int numRows) {
        if (numRows == 1) return s;

        Map<Integer,StringBuilder> rowsStr = new HashMap<>();
        char [] arr = s.toCharArray();
        int r = 0;
        boolean goingDown = true;

        for (char c : arr) {
            StringBuilder sb = rowsStr.getOrDefault(r, new StringBuilder());
            sb.append(c);
            rowsStr.put(r, sb);
            if (goingDown) {
                if (r + 1 == numRows) {
                    r = r - 1;
                    goingDown = false;
                } else r++;
            } else {
                if (r - 1 == -1) {
                    r = 1;
                    goingDown = true;
                } else r--;
            }
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < numRows; i++) {
            if (rowsStr.containsKey(i)) sb.append(rowsStr.get(i));
        }
        return sb.toString();
    }

//    https://leetcode.com/problems/valid-sudoku/

    public boolean isValidSudoku(char[][] board) {
        Map<Integer,Set<Integer>> rows = new HashMap<>();
        Map<Integer,Set<Integer>> cols = new HashMap<>();
        Map<Integer,Set<Integer>> boxes = new HashMap<>();

        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int hash = ((i / 3) * 3) + j / 3;
                    Set<Integer> r = rows.getOrDefault(i, new HashSet<>());
                    Set<Integer> c = cols.getOrDefault(j, new HashSet<>());
                    Set<Integer> b = boxes.getOrDefault(hash, new HashSet<>());

                    int no = board[i][j] - '0';
                    if (r.contains(no)) return false;
                    r.add(no);
                    rows.put(i, r);

                    if (c.contains(no)) return false;
                    c.add(no);
                    cols.put(j, c);

                    if (b.contains(no)) return false;
                    b.add(no);
                    boxes.put(hash, b);
                }
            }
        }
        return true;
    }

//    https://leetcode.com/problems/find-the-kth-largest-integer-in-the-array/

    String kthLargestNumberUtil(String [] nums, int k, int low, int high) {
        if (low > high) return "";
        if (low == high) return nums[low];
        String highS = nums[high];
        BigInteger pivotNo = new BigInteger(highS);
        int j = low;
        for (int i = low; i <= high; i++) {
            String noS = nums[i];
            BigInteger no = new BigInteger(noS);
            if (no.compareTo(pivotNo) < 0) {
                String temp = nums[j];
                nums[j] = noS;
                nums[i] = temp;
                j++;
            }
        }
        nums[high] = nums[j];
        nums[j] = highS;
        int n = nums.length;
        if (n - j == k) return nums[j];
        else if (n - j < k) return kthLargestNumberUtil(nums, k, low, j - 1);
        else return kthLargestNumberUtil(nums, k, j + 1, high);
    }

    public String kthLargestNumber(String[] nums, int k) {
//        return kthLargestNumberUtil(nums, k, 0, nums.length - 1);
        PriorityQueue<BigInteger> minHeap = new PriorityQueue<>();
        for (String num : nums) {
            minHeap.add(new BigInteger(num));
            if (minHeap.size() > k) minHeap.remove();
        }
        if (minHeap.isEmpty()) return "";
        return minHeap.peek().toString();
    }

//    https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/

    int gstValue = 0;

    public TreeNode bstToGst(TreeNode root) {
        if (root == null) return null;
        bstToGst(root.right);
        gstValue += root.val;
        root.val = gstValue;
        bstToGst(root.left);
        return root;
    }

//    https://leetcode.com/problems/coloring-a-border/

    boolean endOfGraph(int [][] grid, int row, int col, int oldColor, boolean [][] visited, int [][] directions) {
        if (row == 0 || row == grid.length - 1 || col == 0 || col == grid[row].length - 1) return true;
        for (int [] direction : directions) {
            int newR = row + direction[0];
            int newC = col + direction[1];
            if (grid[newR][newC] != oldColor && !visited[newR][newC]) return true;
        }
        return false;
    }

    void dfsColorBoundary(int [][] grid, int row, int col, int newColor, int oldColor,
                          boolean [][] visited, int [][] directions) {
        if (row < 0 || row >= grid.length || col < 0 || col >= grid[row].length
                || grid[row][col] != oldColor || visited[row][col]) return;
        visited[row][col] = true;
        if (endOfGraph(grid, row, col, oldColor, visited, directions)) {
            grid[row][col] = newColor;
        }
        for (int [] direction : directions) {
            dfsColorBoundary(grid, row + direction[0], col + direction[1], newColor, oldColor, visited, directions);
        }
    }

    public int[][] colorBorder(int[][] grid, int row, int col, int color) {
        int [][] directions = new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
        dfsColorBoundary(grid, row, col, color, grid[row][col], new boolean [grid.length][grid[0].length], directions);
        return grid;
    }

//    https://leetcode.com/problems/number-of-visible-people-in-a-queue/

    public int[] canSeePersonsCount(int[] heights) {
        int n = heights.length;
        Deque<Integer> stack = new ArrayDeque<>();
        int [] canSee = new int[n];

        for (int i = n - 1; i >= 0; i--) {
            int c = 0;
            while (!stack.isEmpty() && heights[stack.peek()] < heights[i]) {
                stack.pop();
                c++;
            }
            if (!stack.isEmpty()) c++;
            canSee[i] = c;
            stack.push(i);
        }

        return canSee;
    }

//    https://leetcode.com/problems/design-skiplist/

    static class SkipListNode {
        int v;
        SkipListNode next;
        SkipListNode prev;
        SkipListNode up;
        SkipListNode below;

        SkipListNode(int v) {
            this.v = v;
            next = null;
            prev = null;
            up = null;
            below = null;
        }

        @Override
        public String toString() {
            return "SkipListNode{" +
                    "v=" + v +
                    '}';
        }
    }

    static class Skiplist {

        int h;
        SkipListNode head;
        SkipListNode tail;
        Random random;
        Map<Integer,Integer> freq;

        public Skiplist() {
            h = -1;
            head = createEmptyLLAtLevel(null, null);
            tail = head.next;
            random = new Random();
            freq = new HashMap<>();
        }

        SkipListNode createEmptyLLAtLevel(SkipListNode head, SkipListNode tail) {
            h++;
            SkipListNode nHead = new SkipListNode(Integer.MIN_VALUE);
            SkipListNode nTail = new SkipListNode(Integer.MAX_VALUE);
            nHead.next = nTail;
            nTail.prev = nHead;
            if (head == null || tail == null) {
               return nHead;
            } else {
                head.up = nHead;
                nHead.below = head;
                tail.up= nTail;
                nTail.below = tail;
                head = head.up;
                tail = tail.up;
                this.head = head;
                this.tail = tail;
            }
            return head;
        }

        SkipListNode searchUtil(SkipListNode head, int target) {
            SkipListNode node = head;
            while (node.below != null) {
                node = node.below;
                while (node.next.v <= target) {
                    node = node.next;
                }
            }
            return node;
        }

        public boolean search(int target) {
            SkipListNode node = searchUtil(head, target);
            return node.v == target;
        }

        public void add(int num) {
            SkipListNode node = searchUtil(head, num);
            if (node.v == num) {
                freq.merge(num, 1, Integer::sum);
                return;
            }
            freq.put(num,1);
            SkipListNode newNode = new SkipListNode(num);
            SkipListNode next = node.next;
            newNode.next = next;
            next.prev = newNode;
            node.next = newNode;
            newNode.prev = node;

            SkipListNode q = node;
            SkipListNode levelDown = newNode;

            int level = 0;
            while (random.nextBoolean()) {

                if (level == h) createEmptyLLAtLevel(head, tail);

                while (q.up == null) {
                    q = q.prev;
                }
                q = q.up;
                SkipListNode levelUpNewNode = new SkipListNode(num);
                SkipListNode qn = q.next;
                levelUpNewNode.next = qn;
                qn.prev = levelUpNewNode;
                q.next = levelUpNewNode;
                levelUpNewNode.prev = q;
                levelUpNewNode.below = levelDown;
                levelDown.up = levelUpNewNode;
                levelDown = levelUpNewNode;
                level++;
            }
        }

        public boolean erase(int num) {
            SkipListNode node = searchUtil(head, num);
            if (node.v != num) return false;
            if (freq.containsKey(num)) {
                if (freq.get(num) == 1) {
                    freq.remove(num);
                    while (node.up != null) {
                        node.prev.next = node.next;
                        node.next.prev = node.prev;
                        node = node.up;
                    }
                }
                else freq.put(num, freq.get(num) - 1);
            }
            return true;
        }
    }

//    https://leetcode.com/problems/path-sum-iv/

    void addPathSumRec(int [][] tree, int depth, int nodePos, int runningSum, int [] pathSum) {
        if (tree[depth][nodePos] == -1) return;
        runningSum += tree[depth][nodePos];
        if (depth < 3 && tree[depth + 1][nodePos * 2] == -1 && tree[depth + 1][(nodePos * 2) + 1] == -1) {
            pathSum[0] +=runningSum;
            return;
        }
        if (depth == 3) {
            pathSum[0] +=runningSum;
            return;
        }
        addPathSumRec(tree, depth + 1, (nodePos * 2), runningSum, pathSum);
        addPathSumRec(tree, depth + 1, (nodePos * 2) + 1, runningSum, pathSum);
    }

    public int pathSum(int[] nums) {
        int [][] tree = new int[][]{{-1},{-1,-1},{-1,-1,-1,-1},{-1,-1,-1,-1,-1,-1,-1,-1}};
        for (int num : nums) {
            int value = num % 10;
            num /= 10;
            int levelPos = (num % 10) - 1;
            num /= 10;
            int depth = (num % 10) - 1;
            tree[depth][levelPos] = value;
        }
        int [] pathSum = new int[]{0};
        addPathSumRec(tree, 0, 0, 0, pathSum);
        return pathSum[0];
    }

//    https://leetcode.com/problems/string-compression-ii/

    int getLengthOfOptimalCompressionUtil(int start, int k, char [] arr, Map<String,Integer> dp) {
        if (dp.containsKey(start + " " + k)) return dp.get(start + " " + k);
        if (start == arr.length || arr.length - start <= k) return 0;

        int minLength = Integer.MAX_VALUE;
        int [] countArr = new int[26];
        int mostFreq = 0;

        for (int i = start; i < arr.length; i++) {
            countArr[arr[i] - 'a']++;
            mostFreq = Math.max(mostFreq, countArr[arr[i] - 'a']);
            int compressedLength = 1 + (mostFreq > 1 ? String.valueOf(mostFreq).length() : 0);

            if (k >= i - start + 1 - mostFreq) {
                minLength = Math.min(minLength,compressedLength +
                        getLengthOfOptimalCompressionUtil(i + 1, k - (i - start + 1 - mostFreq), arr, dp));
            }
        }

        dp.put(start + " " + k, minLength);
        return minLength;
    }

    public int getLengthOfOptimalCompression(String s, int k) {
        char [] arr = s.toCharArray();
        Map<String,Integer> dp = new HashMap<>();
        return getLengthOfOptimalCompressionUtil(0, k, arr, dp);
    }

//    https://leetcode.com/problems/number-of-music-playlists/

    long numMusicPlaylistsUtil(int currLen, int diffSongs, int goal, int n, int k, long [][] dp, int mod) {
        if (currLen > goal || diffSongs > n) return 0;
        if (currLen == goal) return diffSongs == n ? 1 : 0;
        if (dp[currLen][diffSongs] != -1) return dp[currLen][diffSongs];

        long ans = (numMusicPlaylistsUtil(currLen + 1, diffSongs, goal, n, k, dp, mod) * Math.max(0, diffSongs - k)) % mod;
        ans += (numMusicPlaylistsUtil(currLen + 1, diffSongs + 1, goal, n, k, dp, mod) * (n - diffSongs)) % mod;
        ans %= mod;

        dp[currLen][diffSongs] = ans;
        return ans;
    }

    public int numMusicPlaylists(int n, int goal, int k) {
        long [][] dp = new long[goal + 1][n + 1];
        for (long [] l : dp) Arrays.fill(l, -1);
        int mod = 1_000_000_007;
        return (int) numMusicPlaylistsUtil(0, 0, goal, n, k, dp, mod);
    }

//    https://leetcode.com/problems/maximum-average-subtree/

    double maxAv = 0;

    Pair<Double,Integer> maximumAverageSubtreeUtil(TreeNode node) {
        if (node == null) return new Pair<>(0.0,0);
        Pair<Double,Integer> left = maximumAverageSubtreeUtil(node.left);
        Pair<Double,Integer> right = maximumAverageSubtreeUtil(node.right);
        double subTreeSum = left.getKey() + right.getKey() + node.val;
        int subTreeCount = left.getValue() + right.getValue() + 1;
        maxAv = Math.max(maxAv, subTreeSum / subTreeCount);
        return new Pair<>(subTreeSum,subTreeCount);
    }

    public double maximumAverageSubtree(TreeNode root) {
        maximumAverageSubtreeUtil(root);
        return maxAv;
    }

//    https://leetcode.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/

    void calculateDepth(char[] s, int [] i, int[] substringG, int chosen) {
       while (i[0] < s.length) {
           if (s[i[0]] == '(') {
               substringG[i[0]] = chosen;
               i[0]++;
               calculateDepth(s, i, substringG, chosen == 0 ? 1 : 0);
           } else {
               substringG[i[0]] = chosen == 0 ? 1 : 0;
               i[0]++;
               return;
           }
       }
    }

    public int[] maxDepthAfterSplit(String seq) {
        int n = seq.length();
        char [] s = seq.toCharArray();
        int [] substringG = new int[n];

        calculateDepth(s,new int[]{0}, substringG,0);

        // above one is recursive, lower one is iterative

        int depth = 0;
        for (int i = 0; i < s.length; i++) {
            char c = s[i];
            if (c == '(') {
                depth++;
                substringG[i] = depth % 2;
            } else {
                depth--;
                substringG[i] = depth % 2 == 0 ? 1 : 0;
            }
        }
        return substringG;
    }

//    https://leetcode.com/problems/missing-number-in-arithmetic-progression/

    public int missingNumber(int[] arr) {
        int n = arr.length;
        int diff = (arr[n - 1] - arr[0]) / n;
//        for (int i = 0; i < n - 1; i++) {
//            int d = arr[i + 1] - arr[i];
//            if (d != diff) return arr[i] + diff;
//        }

        int low = 0;
        int high = n - 1;

        while (low < high) {

         int mid = low + (high - low) / 2;
         int supposed = (diff * mid) + arr[0];

         if (supposed == arr[mid]) low = mid + 1;
         else high = mid;
        }

        return arr[0] + (diff * low);
    }

//    https://leetcode.com/problems/sentence-similarity/

    public boolean areSentencesSimilar(String[] sentence1, String[] sentence2, List<List<String>> similarPairs) {
        if (sentence1.length != sentence2.length) return false;

        Map<String,Set<String>> mapping = new HashMap<>();

        for (List<String> similarPair : similarPairs) {

            String f = similarPair.get(0);
            String s = similarPair.get(1);
            Set<String> fN = mapping.getOrDefault(f, new HashSet<>());
            fN.add(s);
            mapping.put(f, fN);

            Set<String> sN = mapping.getOrDefault(s, new HashSet<>());
            sN.add(f);
            mapping.put(s, sN);
        }

        for (int i = 0; i < sentence1.length; i++) {
            String f = sentence1[i];
            String s = sentence2[i];

            if (f.equals(s) || mapping.getOrDefault(f, new HashSet<>()).contains(s)
                    || mapping.getOrDefault(s, new HashSet<>()).contains(f)) continue;
            else return false;
        }
        return true;
    }

//    https://leetcode.com/problems/cousins-in-binary-tree/

    int xDepth = -1;
    int yDepth = -1;
    TreeNode xParent = null;
    TreeNode yParent = null;

    boolean isCousinsUtil(TreeNode node, int x, int y, int depth, TreeNode parent) {
        if (node == null) return false;
        if (node.val == x) {
            if (yDepth != -1) {
                return depth == yDepth && yParent != parent;
            } else {
                xDepth = depth;
                xParent = parent;
                return false;
            }
        } else if (node.val == y) {
            if (xDepth != -1) {
                return depth == xDepth && xParent != parent;
            } else {
                yDepth = depth;
                yParent = parent;
                return false;
            }
        } else {
            return isCousinsUtil(node.left, x, y, depth + 1, node) || isCousinsUtil(node.right, x, y, depth + 1, node);
        }
    }

    public boolean isCousins(TreeNode root, int x, int y) {
        return isCousinsUtil(root, x, y, 0, null);
    }

//    https://leetcode.com/problems/range-sum-query-mutable/

    static class NumArray {
        int [] nums;
        int [] segmentTree;
        int n;

        public NumArray(int[] nums) {
            this.n = nums.length;
            this.nums = nums;
            int height = (int) (Math.ceil (Math.log (n) / Math.log (2)));
            int size = 2 * (int) Math.pow(2, height) - 1;
            segmentTree = new int[size];
            buildSegmentTree(nums, segmentTree, 0, 0, n - 1);
        }

        int buildSegmentTree(int [] nums, int [] segmentTree, int index, int low, int high) {
            if (low == high) {
                segmentTree[index] = nums[low];
            } else {
                int mid = low + (high - low) / 2;
                segmentTree[index] = buildSegmentTree(nums, segmentTree, 2 * index + 1, low, mid)
                        + buildSegmentTree(nums, segmentTree, 2 * index + 2, mid + 1, high);
            }
            return segmentTree[index];
        }

        void updateSegmentTree(int [] segmentTree, int index, int low, int high, int numsIndex, int diff) {
            if (low > numsIndex || high < numsIndex) return;

            segmentTree[index] += diff;

            if (low != high) {
                int mid = low + (high - low) / 2;
                updateSegmentTree(segmentTree, 2 * index + 1, low, mid, numsIndex, diff);
                updateSegmentTree(segmentTree, 2 * index + 2, mid + 1, high, numsIndex, diff);
            }
        }

        public void update(int index, int val) {
            int diff = val - nums[index];
            nums[index] = val;
            updateSegmentTree(segmentTree, 0, 0, n - 1, index, diff);
        }

        int calculateSumRange(int [] segmentTree, int index, int l, int r, int left, int right) {
            if (r < left || l > right) return 0;
            else if (left <= l && r <= right) return segmentTree[index];
            else {
                int mid = l + (r - l) / 2;
                return calculateSumRange(segmentTree, 2 * index + 1, l, mid, left, right)
                        + calculateSumRange(segmentTree, 2 * index + 2, mid + 1, r, left, right);
            }
        }

        public int sumRange(int left, int right) {
            return calculateSumRange(segmentTree, 0, 0, n - 1, left, right);
        }
    }

//    https://leetcode.com/problems/cherry-pickup/

    // wrong answer, dp solves the question, below template can be used for understanding dijkstra

    static class Cherry {
        int x;
        int y;
        int cost;
        List<int []> pathSF;

        Cherry(int x, int y, int cost) {
            this.x = x;
            this.y = y;
            this.cost = cost;
            pathSF = new ArrayList<>();
            pathSF.add(new int[]{x,y});
        }

        void add(List<int[]> tillNow) {
            this.pathSF.addAll(tillNow);
        }

    }

    Pair<Integer,List<int []>> dijkstraCherry(PriorityQueue<Cherry> maxHeap, int [][] grid, int destinationX,
                                              int destinationY, boolean [][] visited, int maxCherryPicked,
                                              List<int []> pathTaken, int [][] directions) {
        while (!maxHeap.isEmpty()) {
            Cherry cherry = maxHeap.remove();
            int i = cherry.x;
            int j = cherry.y;

            if (i == destinationX && j == destinationY) {
                maxCherryPicked += cherry.cost;
                pathTaken = cherry.pathSF;
                break;
            }

            if (visited[i][j]) continue;
            visited[i][j] = true;

            for (int [] direction : directions) {
                int newI = i + direction[0];
                int newJ = j + direction[1];

                if (newI < grid.length && newI >=0 && newJ < grid[0].length && newJ >= 0 &&
                        !visited[newI][newJ] && grid[newI][newJ] >= 0) {
                    Cherry neighbour = new Cherry(newI, newJ, grid[newI][newJ] + cherry.cost);
                    neighbour.add(cherry.pathSF);
                    maxHeap.add(neighbour);
                }
            }
        }
        maxHeap.clear();
        return new Pair<>(maxCherryPicked, pathTaken);
    }

    public int cherryPickup(int[][] grid) {

        PriorityQueue<Cherry> maxHeap = new PriorityQueue<>((a,b) -> b.cost - a.cost);
        maxHeap.add(new Cherry(0,0, grid[0][0]));

        int [][] downRight = new int[][]{{1,0},{0,1}};
        int [][] upLeft = new int[][]{{-1,0},{0,-1}};

        int destinationX = grid.length - 1;
        int destinationY = grid[0].length - 1;
        boolean [][] visited = new boolean[grid.length][grid[0].length];

        Pair<Integer,List<int []>> p = dijkstraCherry(maxHeap, grid, destinationX, destinationY,
                visited, 0, new ArrayList<>(), downRight);

        for (int [] node : p.getValue()) {
            grid[node[0]][node[1]] = 0;
        }
        visited = new boolean[grid.length][grid[0].length];
        destinationX = 0;
        destinationY = 0;
        int startX = grid.length - 1;
        int startY = grid[0].length - 1;
        maxHeap.add(new Cherry(startX, startY, grid[startX][startY]));

        p = dijkstraCherry(maxHeap, grid, destinationX, destinationY,
                visited, p.getKey(), new ArrayList<>(), upLeft);

        return p.getKey();
    }

//    https://leetcode.com/problems/maximum-number-of-occurrences-of-a-substring/

    public int maxFreq(String s, int maxLetters, int minSize, int maxSize) {
        Map<String,Integer> subStringsC = new HashMap<>();
        Map<Character,Integer> freq = new HashMap<>();
        int maxLen = 0;

        int j = 0;
        for (int i = 0; i < s.length(); i++) {
            freq.merge(s.charAt(i),1, Integer::sum);

            if (i - j + 1 > minSize || freq.size() > maxLetters) {
                freq.put(s.charAt(j), freq.get(s.charAt(j)) - 1);
                if (freq.get(s.charAt(j)) == 0) freq.remove(s.charAt(j));
                j++;
            }

            if (freq.size() <= maxLetters && i - j + 1 >= minSize) {
                String sub = s.substring(j, i + 1);
                subStringsC.merge(sub, 1, Integer::sum);
                maxLen = Math.max(maxLen, subStringsC.get(sub));
            }
        }
        return maxLen;
    }

//    https://leetcode.com/problems/the-maze-ii/

    static class Ball {
        int x;
        int y;
        int dist;
        int[] d; // u, d, r, l

        Ball(int x, int y, int dist, int[] d) {
            this.x = x;
            this.y = y;
            this.dist = dist;
            this.d = d;
        }


        @Override
        public String toString() {
            return "Ball{" +
                    "x=" + x +
                    ", y=" + y +
                    ", dist=" + dist +
                    ", d=" + Arrays.toString(d) +
                    '}';
        }

    }

    boolean withInBounds(int[][] maze, int x, int y) {
        return x >= 0 && x < maze.length && y >= 0 && y < maze[0].length
                && maze[x][y] != 1;
    }

    void preProcess(Deque<Ball> deque, int[] start, int[][] maze, Set<String> visited, int[][] directions) {
        for (int[] direction : directions) {

            int newX = start[0];
            int newY = start[1];

            int dist = 0;
            while (withInBounds(maze, newX + direction[0], newY + direction[1])) {
                dist++;
                newX += direction[0];
                newY += direction[1];
            }

            deque.add(new Ball(newX, newY, dist, direction));

        }
    }

    public int shortestDistance(int[][] maze, int[] start, int[] destination) {
        Deque<Ball> deque = new ArrayDeque<>();
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        int [][] minD = new int[maze.length][maze[0].length];
        for (int [] d : minD) Arrays.fill(d, Integer.MAX_VALUE);

        deque.add(new Ball(start[0], start[1], 0, null));

        while (!deque.isEmpty()) {
            Ball ball = deque.remove();

            if (ball.dist < minD[ball.x][ball.y]) {
                minD[ball.x][ball.y] = ball.dist;
            } else continue;

            for (int[] direction : directions) {
                if (direction == ball.d) continue;

                int newX = ball.x;
                int newY = ball.y;

                int dist = 0;
                while (withInBounds(maze, newX + direction[0], newY + direction[1])) {
                    dist++;
                    newX += direction[0];
                    newY += direction[1];
                }

                deque.add(new Ball(newX, newY, ball.dist + dist, direction));
            }
        }

        return minD[destination[0]][destination[1]] == Integer.MAX_VALUE ? - 1 : minD[destination[0]][destination[1]];
    }

//    https://leetcode.com/problems/subarray-sums-divisible-by-k/

    public static int subarraysDivByK(int[] nums, int k) {
        Map<Integer,Integer> prefixMap = new HashMap<>();
        prefixMap.put(0,1);
        int count = 0;

        int prefixSum = 0;
        for (int no : nums) {
            prefixSum += no;
            int mod = prefixSum % k;
            if (mod < 0) mod += k;

            count += prefixMap.getOrDefault(mod,0);

            prefixMap.put(mod, prefixMap.getOrDefault(mod,0) + 1);
        }

        return count;
    }

//    https://leetcode.com/problems/score-of-parentheses/

    public int scoreOfParentheses(String s) {
        Deque<Integer> stack = new ArrayDeque<>();
        char [] arr = s.toCharArray();

        for (char c : arr) {
            if (c == '(') stack.push(0);
            else {
                int val = 0;
                while (!stack.isEmpty() && stack.peek() != 0) val += stack.pop();
                val = val == 0 ? 1 : 2 * val;
                stack.pop();
                stack.push(val);
            }
        }
        int v = 0;
        while (!stack.isEmpty()) v += stack.pop();
        return v;
    }

//    https://leetcode.com/problems/search-in-a-sorted-array-of-unknown-size/

    interface ArrayReader {
      public int get(int index);
    }

    public int search(ArrayReader reader, int target) {
        int p = 0;

        while (true) {
            int atP = reader.get(p);
            if (atP >= target) break;
            else p = p == 0 ? 1 : p * 2;
        }

        int low = 0;
        int high = p;

        while (low <= high) {

            int mid = low + (high - low) / 2;
            int atP = reader.get(mid);
            if (atP == target) return mid;
            else if (atP > target) high = mid - 1;
            else low = mid + 1;
        }

        return -1;
    }

//    https://leetcode.com/problems/sliding-puzzle/

    String convertToStr(int [][] board) {
        StringBuilder sb = new StringBuilder();
        for (int[] ints : board) {
            for (int j = 0; j < board[0].length; j++) {
                sb.append(ints[j]);
            }
        }
        return sb.toString();
    }

    public int slidingPuzzle(int[][] board) {
        Set<String> visited = new HashSet<>();
        int [][] directions = new int[][]{{-1,0},{1,0},{0,-1},{0,1}};

        int [][] destination = new int[][]{{1,2,3},{4,5,0}};
        String desStr = convertToStr(destination);

        Deque<int [][]> deque = new ArrayDeque<>();
        int d = 0;

        deque.add(board);
        while (!deque.isEmpty()) {

            int n = deque.size();
            for (int i = 0; i < n; i++) {

                int[][] b = deque.remove();
                String bStr = convertToStr(b);

                if (bStr.equals(desStr)) return d;
                if (visited.contains(bStr)) continue;
                visited.add(bStr);

                int zeroX = -1;
                int zeroY = -1;

                outer : for (int r = 0; r < b.length; r++) {
                    for (int c = 0; c < b[0].length; c++) {
                        if (b[r][c] == 0) {
                            zeroX = r;
                            zeroY = c;
                            break outer;
                        }
                    }
                }

                for (int [] direction : directions) {
                    int newX = zeroX + direction[0];
                    int newY = zeroY + direction[1];

                    if (newX >= 0 && newX < board.length && newY >= 0 && newY < board[0].length) {

                        int [][] newB = new int[board.length][board[0].length];
                        for (int rows = 0; rows < newB.length; rows++) {
                            newB[rows] = Arrays.copyOf(b[rows], b[0].length);
                        }

                        newB[zeroX][zeroY] = newB[newX][newY];
                        newB[newX][newY] = 0;

                        if (!visited.contains(convertToStr(newB))) {
                            deque.add(newB);
                        }
                    }
                }
            }
            d++;
        }

        return -1;
    }

//    https://leetcode.com/problems/smallest-string-with-swaps/

    void dfsSmallestStr(int i, boolean [] visited, Map<Integer,Set<Integer>> adj,
                        List<Integer> indexes, List<Character> indexesC, char [] arr) {
        if (visited[i]) return;
        visited[i] = true;

        indexes.add(i);
        indexesC.add(arr[i]);

        for (int neighbour : adj.getOrDefault(i, new HashSet<>()))
            dfsSmallestStr(neighbour, visited, adj, indexes, indexesC, arr);
    }

    int find(int [] parent, int v) {
        if (parent[v] == -1) return v;
        else {
            parent[v] = find(parent, parent[v]);
            return parent[v];
        }
    }

    void union(int [] parent, int [] rank, int x, int y) {
        int pX = find(parent, x);
        int pY = find(parent, y);

        if (pX == pY) return;

        if (rank[pX] > rank[pY]) parent[pY] = pX;
        else if (rank[pX] < rank[pY]) parent[pX] = pY;
        else {
            parent[pX] = pY;
            rank[pY]++;
        }
    }

    public String smallestStringWithSwaps(String s, List<List<Integer>> pairs) {
//        char [] arr = s.toCharArray();
//        Map<Integer,Set<Integer>> adj = new HashMap<>();
//
//        for (List<Integer> pair : pairs) {
//            int from = pair.get(0);
//            int to = pair.get(1);
//
//            Set<Integer> toNeighbour = adj.getOrDefault(to, new HashSet<>());
//            toNeighbour.add(from);
//            adj.put(to, toNeighbour);
//
//            Set<Integer> fromNeighbour = adj.getOrDefault(from, new HashSet<>());
//            fromNeighbour.add(to);
//            adj.put(from, fromNeighbour);
//        }
//
//        Map<Integer,Character> pairing = new HashMap<>();
//        boolean [] visited = new boolean[s.length()];
//
//        for (int i = 0; i < visited.length; i++) {
//            if (adj.containsKey(i) && !visited[i]) {
//                List<Integer> indexes = new ArrayList<>();
//                List<Character> indexesC = new ArrayList<>();
//                dfsSmallestStr(i, visited, adj, indexes, indexesC, arr);
//                Collections.sort(indexes);
//                Collections.sort(indexesC);
//                for (int j = 0; j < indexes.size(); j++) pairing.put(indexes.get(j), indexesC.get(j));
//            }
//        }
//        StringBuilder sb = new StringBuilder();
//        for (int i = 0; i < arr.length; i++) {
//            if (pairing.containsKey(i)) sb.append(pairing.get(i));
//            else sb.append(arr[i]);
//        }
//        return sb.toString();

        // using union find
        char [] arr = s.toCharArray();
        int n = arr.length;

        int [] parent = new int[n];
        Arrays.fill(parent, -1);
        int [] rank = new int[n];

        Map<Integer,PriorityQueue<Character>> mapping = new HashMap<>();

        for (List<Integer> pair : pairs) {
            int from = pair.get(0);
            int to = pair.get(1);

            union(parent, rank, from, to);
        }

        for (int i = 0; i < n; i++) {
            int p = find(parent, i);
            PriorityQueue<Character> minHeap = mapping.getOrDefault(p, new PriorityQueue<>());
            minHeap.add(arr[i]);
            mapping.put(p, minHeap);
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            int p = find(parent, i);
            PriorityQueue<Character> minHeap = mapping.get(p);
            sb.append(minHeap.remove());
        }
        return sb.toString();
    }

//    https://leetcode.com/problems/construct-string-from-binary-tree/

    public String tree2str(TreeNode root) {
        if (root == null) return "";

        String left = tree2str(root.left);
        String right = tree2str(root.right);

        if (left.isEmpty() && right.isEmpty()) return String.valueOf(root.val);
        return root.val + "(" + left  + ")" + (right.isEmpty() ? "" : "(" + right + ")");
    }

//    https://leetcode.com/problems/longest-word-in-dictionary/

//         brute force, n2
//
//        for (String word : words) {
//            String w = word;
//            while (w.length() != 0) {
//                w = w.substring(0,w.length() - 1);
//                if (!wordsSet.contains(w)) break;
//            }
//            if (w.length() == 0) {
//                if (word.length() > result.length()) {
//                    result = word;
//                } else if (word.length() == result.length() && word.compareTo(result) < 0) {
//                    result = word;
//                }
//            }
//        }
//        return result;

    static class TrieLongest {
        char c;
        Map<Character,TrieLongest> children;

        TrieLongest(char c) {
            this.c = c;
            this.children = new HashMap<>();
        }

        @Override
        public String toString() {
            return "TrieLongest{" +
                    "c=" + c +
                    ", children=" + children +
                    '}';
        }
    }

    void insertInTrie(char [] word, int i, TrieLongest node) {
        if (i == word.length) return;

        char c = word[i];
        TrieLongest child = node.children.get(c);
        if (child == null) {
            child = new TrieLongest(c);
            node.children.put(c, child);
        }

        insertInTrie(word, i + 1, child);
    }

    void search(TrieLongest node, Set<String> wordsSet, String [] res, StringBuilder running) {
        if (node == null) return;

        running.append(node.c);
        if (wordsSet.contains(running.toString())) {

            if (running.length() > res[0].length()) {
                res[0] = running.toString();
            } else if (running.length() == res[0].length() && running.toString().compareTo(res[0]) < 0) {
                res[0] = running.toString();
            }

            for (TrieLongest child : node.children.values()) {
                search(child, wordsSet, res, running);
            }

        }
        running.deleteCharAt(running.length() - 1);
    }

    public String longestWord(String[] words) {
        Set<String> wordsSet = new HashSet<>(Arrays.asList(words));
        TrieLongest root = new TrieLongest(' ');
        String [] result = new String[]{""};

        for (String word : words) insertInTrie(word.toCharArray(), 0, root);

        for (TrieLongest ch : root.children.values()) search(ch, wordsSet, result, new StringBuilder());
        return result[0];
    }

//    https://leetcode.com/problems/partition-list/

    public ListNode partition(ListNode head, int x) {
        if (head == null) return null;

        ListNode sentinelBefore = new ListNode(Integer.MIN_VALUE);
        ListNode sentinelAfter = new ListNode(Integer.MIN_VALUE);

        ListNode smaller = sentinelBefore;
        ListNode greater = sentinelAfter;

        ListNode curr = head;

        while (curr != null) {
            ListNode next = curr.next;
            curr.next = null;
            if (curr.val < x) {
                smaller.next = curr;
                smaller = smaller.next;
            } else {
                greater.next = curr;
                greater = greater.next;
            }
            curr = next;
        }
        smaller.next = sentinelAfter.next;
        return sentinelBefore.next;
    }

//    https://leetcode.com/problems/h-index/

    public int hIndex(int[] citations) {
            int n = citations.length;
            Arrays.sort(citations);
            int low = 0;
            int high = citations.length - 1;

            int prev = -1;

            while (low <= high) {
                int mid = low + (high - low) / 2;

                if (n - mid <= citations[mid]) {
                    prev = n - mid;
                    high = mid - 1;
                } else low = mid + 1;

            }
            return prev;
    }

//    https://leetcode.com/problems/matrix-block-sum/

    public int[][] matrixBlockSum(int[][] mat, int k) {
        int m = mat.length;
        int n = mat[0].length;
        int [][] preSumArr = new int[m + 1][n + 1];

        for (int i = 1; i < preSumArr.length; i++) {
            for (int j = 1; j < preSumArr[0].length; j++) {
                preSumArr[i][j] = mat[i - 1][j - 1] + preSumArr[i - 1][j] + preSumArr[i][j - 1] - preSumArr[i - 1][j - 1];
            }
        }

        for (int i = 1; i < preSumArr.length; i++) {
            for (int j = 1; j < preSumArr[0].length; j++) {

                int endI = Math.min(m, i + k);
                int endJ = Math.min(n, j + k);

                int startI = Math.max(1, i - k);
                int startJ = Math.max(1, j - k);

                mat[i - 1][j - 1] = preSumArr[endI][endJ] - preSumArr[startI - 1][endJ] - preSumArr[endI][startJ - 1]
                        + preSumArr[startI - 1][startJ - 1];
            }
        }
        return mat;
    }

//    https://leetcode.com/problems/single-number-iii/

    public int[] singleNumber(int[] nums) {
        int xy = 0;
        for (int num : nums) xy ^= num;
        xy &= -xy ;

        int [] res = new int[]{0,0};
        for (int num : nums) {
            if ((num & xy) == 0) res[0] ^= num;
            else res[1] ^= num;
        }

        return res;
    }

//    https://leetcode.com/problems/combinations/

    void combineUtils(int i, int n, int k, List<Integer> arr, List<List<Integer>> res) {
        if (arr.size() == k) {
            res.add(new ArrayList<>(arr));
            return;
        }
        if (i <= n) {
            arr.add(i);
            combineUtils(i + 1, n, k, arr, res);
            arr.remove(arr.size() - 1);
            combineUtils(i + 1, n, k, arr, res);
        }
    }

    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        combineUtils(1, n, k, new ArrayList<>(), res);
        return res;
    }

//    https://leetcode.com/problems/permutation-sequence/

    void getPermutationUtils(int n, int k, int [] kTh, String [] res, StringBuilder sb, boolean [] visited) {
        if (sb.length() == n) {
            kTh[0]++;
            if (k == kTh[0]) {
                res[0] = sb.toString();
            }
            return;
        }
        for (int i = 1; i <= n; i++) {
            if (!visited[i]) {
                visited[i] = true;
                sb.append(i);
                getPermutationUtils(n, k, kTh, res, sb, visited);
                if (!res[0].equals("")) return;
                sb.deleteCharAt(sb.length() - 1);
                visited[i] = false;
            }
        }
    }

    public String getPermutation(int n, int k) {
        int [] kth = new int[]{0};
        String [] res = new String[]{""};
        boolean [] visited = new boolean[n + 1];
        getPermutationUtils(n, k, kth, res, new StringBuilder(), visited);
        return res[0];
    }

}
