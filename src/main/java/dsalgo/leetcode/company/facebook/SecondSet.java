package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;

import java.util.*;

public class SecondSet {

    public static void main(String[] args) {
//        System.out.println(Arrays.toString(maxSumOfThreeSubarrays(new int[]{7,13,20,19,19,2,10,1,1,19}, 3)));
//        System.out.println(checkSubarraySum(new int[] {0,1,0},1));
//        System.out.println(findKthLargest(new int[]{7,6,5,4,3,2,1},5));
//        System.out.println(Arrays.toString(exclusiveTime(8, Arrays.asList())));
//        System.out.println(lengthOfLongestSubstringKDistinct("aa",1));
//        System.out.println(maxSubArrayLen(new int[]{1,-1,5,-2,3},3));
//        System.out.println(wordBreak("catsandogcat",Arrays.asList("cats","dog","sand","and","cat","an")));
//        System.out.println(Arrays.deepToString(intervalIntersection(new int[][]{{0, 2}, {5, 10}, {13, 23}, {24, 25}},
//                new int[][]{{1, 5}, {8, 12}, {15, 24}, {25, 26}})));
//        System.out.println(palindromePairs(new String[]{"a",""}));
//        System.out.println(isBipartite(new int[][] {{1,3},{0,2},{1,3},{0,2}}));
//        System.out.println(binarySearchBucket(new int[]{3,17,18,25}, 18, 0, 3));
//        System.out.println(shortestDistance(new int[][]{{1,1},{0,1}}));
//        System.out.println(simplifyPath("/a/./b/../../c/"));
//        System.out.println(closest(new int[]{1,2,3,4,5,6},3.14));
//        TreeNode node = new TreeNode(1);
//        node.right = new TreeNode(8);
//        System.out.println(closestValue(node,3.14));
//        System.out.println(numFriendRequests(new int[]{20,30,100,110,120}));
//        System.out.println(isOneEditDistance("acbbda","abbdad"));
//        System.out.println(findCelebrity(2));
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(6);
        root.right = new TreeNode(3);
        root.right.left = new TreeNode(5);
        root.right.right = new TreeNode(7);
        System.out.println(verticalTraversal(root));
    }

//    https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/

    public static int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        int [] subArrSum = new int[nums.length];
        int sum = 0;
        int i = 0;
        while (i < k) {
            sum += nums[i];
            i++;
        }
        subArrSum[i - k] = sum;
        while (i < nums.length) {
            sum = sum + nums[i] - nums[i - k];
            subArrSum[i - k + 1] = sum;
            i++;
        }
        int [] leftMax = new int[nums.length];
        int max = Integer.MIN_VALUE;
        int maxIndex = -1;
        for (int j = 0; j < nums.length - k; j++) {
            if (max < subArrSum[j]) {
                max = subArrSum[j];
                maxIndex = j;
            }
            leftMax[j] = maxIndex;
        }

        int [] rightMax = new int[nums.length];
        max = Integer.MIN_VALUE;
        maxIndex = -1;
        for (int j = nums.length - k; j >= 0; j--) {
            if (max <= subArrSum[j]) {
                max = subArrSum[j];
                maxIndex = j;
            }
            rightMax[j] = maxIndex;
        }
        max = Integer.MIN_VALUE;
        int [] result = new int[3];
        for (int j = k; j <= nums.length - 2 * k; j++) {
            int sumSubArr = subArrSum[leftMax[j - k]] + subArrSum[j] + subArrSum[rightMax[j + k]];
            if (sumSubArr > max) {
                max = sumSubArr;
                result[0] = leftMax[j - k];
                result[1] = j;
                result[2] = rightMax[j + k];
            }
        }
        return result;
    }

//    https://leetcode.com/problems/clone-graph/

    static Map<Integer,Node> hm = new HashMap<>();

    static Node cloneGraphUtils(Node node) {
        Node visitCheck = hm.get(node.val);
        if (visitCheck != null) return visitCheck;

        Node clonedNode = new Node(node.val, new ArrayList<>());
        hm.put(node.val, clonedNode);
        for (Node neighbour : node.neighbors) {
            clonedNode.neighbors.add(cloneGraphUtils(neighbour));
        }
        return clonedNode;
    }

    public static Node cloneGraph(Node node) {
        if(node == null) return null;
        return cloneGraphUtils(node);
    }

//    https://leetcode.com/problems/continuous-subarray-sum/

    public static boolean checkSubarraySum(int[] nums, int k) {
        int sum = 0;
        Map<Integer,Integer> hm = new HashMap<>();
        hm.put(0, -1);
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            int rem = ((sum % k) + k) % k;
            Integer remHmIndex = hm.get(rem);
            if (remHmIndex != null)  {
                if (i - remHmIndex >= 2) return true;
            }
            else hm.put(rem, i);
        }
        return false;
    }

//    https://leetcode.com/problems/kth-largest-element-in-an-array/

    public static void swap(int [] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public static int findKthLargestUtil(int [] nums, int k, int low, int high) {
        if (low > high) return -1;
        int pivot = nums[low];
        int i = low + 1;
        int j = high;
        while (i < j) {
            while (i < high && nums[i] <= pivot) i++;
            while (j > low && nums[j] >= pivot) j--;
            if (i < j) swap(nums, i, j);
        }
        if (nums[low] > nums[j]) swap(nums,low,j);
        int check = j - (nums.length - k);
        if (check == 0) return nums[j];
        else if (check > 0) return findKthLargestUtil(nums, k, low, j - 1);
        else return findKthLargestUtil(nums, k, j + 1, high);
    }

    public static int findKthLargest(int[] nums, int k) {
       return findKthLargestUtil(nums,k, 0, nums.length - 1);
    }

//    https://leetcode.com/problems/exclusive-time-of-functions/

    public static int[] exclusiveTime(int n, List<String> logs) {
        int [] result = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        int prevTime = 0;
        for (String log : logs) {
            String [] logArr = log.split(":");
            int id = Integer.parseInt(logArr[0]);
            boolean start = logArr[1].equals("start");
            int time = Integer.parseInt(logArr[2]);
            if (start) {
                if (!stack.isEmpty()) result[stack.peek()] += time - prevTime;
                stack.push(id);
                prevTime = time;
            } else {
                result[stack.pop()] += time - prevTime + 1;
                prevTime = time + 1;
            }
        }
        return result;
    }

//    https://leetcode.com/problems/binary-tree-right-side-view/

    public static List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        queue.add(null);
        TreeNode prev = root;
        while (!queue.isEmpty()) {
            TreeNode node = queue.remove();
            if (node != null) {
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
                prev = node;
            } else {
                result.add(prev.val);
                if (!queue.isEmpty()) queue.add(null);
            }
        }
        return result;
    }

    public static void rightSideViewDepthUtil(TreeNode node, int level, List<Integer> result) {
        if (node == null) return;
        if (level == result.size()) {
            result.add(node.val);
        } else {
            result.set(level,node.val);
        }
        rightSideViewDepthUtil(node.left, level + 1, result);
        rightSideViewDepthUtil(node.right, level + 1, result);
    }

    public static List<Integer> rightSideViewDepth(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        rightSideViewDepthUtil(root, 0, result);
        return result;
    }

//    https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/

    public static int lengthOfLongestSubstringKDistinct(String s, int k) {
        Map<Character,Integer> hm = new HashMap<>();
        int j = 0;
        int maxLength = Integer.MIN_VALUE;
        for (int i = 0; i < s.length(); i++) {
            hm.merge(s.charAt(i), 1, Integer::sum);
            while (hm.size() > k) {
                int jValue = hm.get(s.charAt(j));
                if (jValue > 1) {
                    hm.put(s.charAt(j), jValue - 1);
                } else {
                    hm.remove(s.charAt(j));
                }
                j++;
            }
            maxLength = Math.max(maxLength,i - j + 1);
        }
        return maxLength;
    }

//    https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/

    public static int maxSubArrayLen(int[] nums, int k) {
       int sum = 0;
       int maxLength = 0;
       Map<Integer,Integer> hm = new HashMap<>();
       hm.put(0,-1);
       for (int i= 0 ; i < nums.length; i++) {
           sum += nums[i];
           hm.putIfAbsent(sum,i);
           Integer j = hm.get(sum - k);
           if (j != null) {
               maxLength = Math.max(maxLength, i - j);
           }
       }
       return maxLength;
    }

//    https://leetcode.com/problems/word-break/

    static Set<Integer> cache = new HashSet<>();

    public static boolean wordBreakUtil(String s, int i, Set<String> wordSet) {
        if (i >= s.length()) return true;
        for (int index = i; index <= s.length(); index++) {
            String leftSubStr = s.substring(i, index);
            String rightSubStr = s.substring(index);
            if (wordSet.contains(leftSubStr)) {
                if (wordSet.contains(rightSubStr)) {
                    return true;
                } else {
                    if (!cache.contains(index)) {
                        boolean rightSubStrCheck = wordBreakUtil(s, index, wordSet);
                        if (rightSubStrCheck) return true;
                        else cache.add(index);
                    }
                }
            }
        }
        return false;
    }

    public static boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        return wordBreakUtil(s, 0, wordSet);
    }

//    https://leetcode.com/problems/interval-list-intersections/

    public static int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        List<int []> result = new ArrayList<>();
        int m = firstList.length;
        int n = secondList.length;
        int i = 0;
        int j = 0;
        while (i < m && j < n) {
            int [] first = firstList[i];
            int [] second = secondList[j];
            if (first[1] < second[0]) {
                i++;
            } else if (second[1] < first[0]) {
                j++;
            } else if (first[0] <= second[0] && first[1] <= second[1]) {
                result.add(new int[]{second[0],first[1]});
                i++;
            } else if (second[0] <= first[0] && second[1] <= first[1]) {
                result.add(new int[]{first[0],second[1]});
                j++;
            } else if (first[0] <= second[0]) {
                result.add(new int[]{second[0],second[1]});
                j++;
            } else {
                result.add(new int[]{first[0],first[1]});
                i++;
            }
        }
        int [][] resArr = new int[result.size()][];
        int k = 0;
        for (int [] r : result) {
            resArr[k] = r;
            k++;
        }
        return resArr;
    }

//    https://leetcode.com/problems/accounts-merge/

    public static void dfsMerge(int i, Map<Integer,Set<Integer>> adjM, List<List<String>> accounts,
                                Set<Integer> processed, Set<String> ids) {
        if (processed.contains(i)) return;
        ids.addAll(accounts.get(i).subList(1,accounts.get(i).size()));
        processed.add(i);
        for (int neighbour : adjM.getOrDefault(i, new HashSet<>())) {
            dfsMerge(neighbour, adjM, accounts, processed, ids);
        }
    }

    public static List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<Integer,Set<Integer>> adjM = new HashMap<>();
        Map<String,Integer> hm = new HashMap<>();
        for (int i = 0; i < accounts.size(); i++) {
            List<String> account = accounts.get(i);
            for (int j = 1; j < account.size(); j++) {
                String id = account.get(j);
                if (hm.containsKey(id)) {
                    int foundAt = hm.get(id);
                    Set<Integer> foundAtNeighbour = adjM.getOrDefault(foundAt, new HashSet<>());
                    foundAtNeighbour.add(i);
                    adjM.put(foundAt,foundAtNeighbour);
                    Set<Integer> iNeighbour = adjM.getOrDefault(i, new HashSet<>());
                    iNeighbour.add(foundAt);
                    adjM.put(i,iNeighbour);
                } else {
                    hm.put(id,i);
                }
            }
        }
        List<List<String>> result = new ArrayList<>();
        Set<Integer> processed = new HashSet<>();
        for (int i = 0; i < accounts.size(); i++) {
            Set<String> ids = new HashSet<>();
            dfsMerge(i, adjM, accounts, processed, ids);
            if (ids.size() > 0) {
                PriorityQueue<String> minHeap = new PriorityQueue<>(String::compareTo);
                minHeap.addAll(ids);
                List<String> resAtI = new ArrayList<>();
                resAtI.add(accounts.get(i).get(0));
                while (!minHeap.isEmpty()) {
                 resAtI.add(minHeap.remove());
                }
                result.add(resAtI);
            }
        }
        return result;
    }

//    https://leetcode.com/problems/palindrome-pairs/

    static String reverse(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = s.length() - 1; i >= 0; i--) {
            sb.append(s.charAt(i));
        }
        return sb.toString();
    }

    static boolean isPalindrome(String s) {
        int i = 0;
        int j = s.length() - 1;
        while (i < j) {
            if (s.charAt(i++) != s.charAt(j--)) return false;
        }
        return true;
    }

    public static List<List<Integer>> palindromePairs(String[] words) {
        List<List<Integer>> result = new ArrayList<>();
        Map<String,Integer> hm = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            hm.put(reverse(words[i]), i);
        }
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            int n = word.length();
            if (hm.containsKey(word) && hm.get(word) != i) {
                result.add(Arrays.asList(i, hm.get(word)));
            }
            for (int j = 0; j < n; j++) {
                String left = word.substring(0, n - 1 - j);
                String right = word.substring(n - 1 - j);
                if (isPalindrome(right) && hm.containsKey(left)) {
                    result.add(Arrays.asList(i, hm.get(left)));
                }
            }
            for (int j = 0; j < n; j++) {
                String left = word.substring(0, j + 1);
                String right = word.substring(j + 1);
                if (isPalindrome(left) && hm.containsKey(right)) {
                    result.add(Arrays.asList(hm.get(right), i));
                }
            }
        }
        return result;
    }

//    https://leetcode.com/problems/binary-tree-paths/

    public static void binaryTreePathsUtil(TreeNode node, List<Integer> arr, List<String> result) {
        if (node == null) return;
        arr.add(node.val);
        if (node.left == null && node.right == null) {
            StringBuilder sb = new StringBuilder();
            int i;
            for (i = 0; i < arr.size() - 1; i++) {
                sb.append(arr.get(i));
                sb.append("->");
            }
            sb.append(arr.get(i));
            result.add(sb.toString());
        }
        binaryTreePathsUtil(node.left, arr, result);
        binaryTreePathsUtil(node.right,arr, result);
        arr.remove(arr.size() - 1);
    }

    public static List<String> binaryTreePaths(TreeNode root) {
        List<String> result = new ArrayList<>();
        List<Integer> arr = new ArrayList<>();
        binaryTreePathsUtil(root, arr, result);
        return result;
    }

//    https://leetcode.com/problems/is-graph-bipartite/

    static boolean isBipartiteUtil(int [][] graph, int [] check, int i, int in) {
        if (check[i] == 0) {
            check[i] = in;
            for (int n : graph[i]) {
                if (!isBipartiteUtil(graph, check, n, in == 1 ? 2 : 1)) return false;
            }
            return true;
        } else return check[i] == in;
    }

    public static boolean isBipartite(int[][] graph) {
        int [] check = new int[graph.length];
        for (int i = 0; i < graph.length; i++) {
           if (check[i] == 0 && !isBipartiteUtil(graph, check, i, 1)) return false;
        }
        return true;
    }

//    https://leetcode.com/problems/random-pick-index/

    static class Solution {

        Map<Integer,List<Integer>> hm;

        public Solution(int[] nums) {
            this.hm = new HashMap<>();
            for (int i = 0; i < nums.length; i++) {
                List<Integer> arr = hm.getOrDefault(nums[i],new ArrayList<>());
                arr.add(i);
                hm.put(nums[i],arr);
            }
        }

        public int pick(int target) {
            List<Integer> arr = this.hm.get(target);
            int randomNo = (int) ((Math.random() * (arr.size())));
            return arr.get(randomNo);
        }
    }

//    https://leetcode.com/problems/random-pick-with-weight/

    public static int binarySearchBucket(int [] prefixArr, int randNo, int low, int high) {
        if (low > high) return -1;
        int mid = low + (high - low) / 2;
        if (randNo <= prefixArr[mid] && randNo > (mid > 0 ? prefixArr[mid - 1] : -1)) {
            return mid;
        } else if (randNo > prefixArr[mid]) {
            return binarySearchBucket(prefixArr, randNo, mid + 1, high);
        } else {
            return binarySearchBucket(prefixArr, randNo, low, mid - 1);
        }
    }

    static class SolutionWeight {

        int [] prefixArr;
        int prefixSum;

        public SolutionWeight(int[] w) {
            this.prefixArr = new int[w.length];
            int prefixSum = 0;
            for (int i = 0; i < w.length; i++) {
                prefixSum += w[i];
                prefixArr[i] = prefixSum;
            }
            this.prefixSum = prefixSum;
        }

        public int pickIndex() {
            int randomNo = (int) ((Math.random() % (prefixSum)));
            return binarySearchBucket(this.prefixArr, randomNo, 0, prefixArr.length - 1);
        }
    }

//    https://leetcode.com/problems/shortest-distance-from-all-buildings/

    static int d = Integer.MAX_VALUE;

    static void shortestDistanceUtil(int [][] grid, int i, int j, int destinationI, int destinationJ, int dist) {
        if (i < 0 || i == grid.length || j < 0 || j == grid[i].length) return;
        else if (i == destinationI && j == destinationJ) { d = Math.min(d,dist);
            return;
        } else if (grid[i][j] > 0) return;
        grid[i][j] = 2;
        shortestDistanceUtil(grid, i + 1, j, destinationI, destinationJ, dist + 1);
        shortestDistanceUtil(grid, i - 1, j, destinationI, destinationJ, dist + 1);
        shortestDistanceUtil(grid, i, j + 1, destinationI, destinationJ, dist + 1);
        shortestDistanceUtil(grid, i, j - 1, destinationI, destinationJ, dist + 1);
        grid[i][j] = 0;
    }

    static void bfs(int [][] grid, int i, int j, int [][] dp, int [][] countGrid) {
        int n = grid.length;
        int m = grid[0].length;
        Queue<int []> queue = new LinkedList<>();
        queue.add(new int[]{i, j});
        queue.add(null);
        grid[i][j] = -1;
        int distance = 0;
        boolean [][] processed = new boolean[n][m];
        while (!queue.isEmpty()) {
            int [] node = queue.remove();
            if (node != null) {
                int indexI = node[0];
                int indexJ = node[1];
                if (processed[indexI][indexJ] || grid[indexI][indexJ] > 0) continue;
                processed[indexI][indexJ] = true;
                dp[indexI][indexJ] += distance;
                countGrid[indexI][indexJ]++;
                if (indexI > 0) queue.add(new int[]{indexI - 1, indexJ});
                if (indexI < n - 1) queue.add(new int[]{indexI + 1, indexJ});
                if (indexJ > 0) queue.add(new int[]{indexI, indexJ - 1});
                if (indexJ < m - 1) queue.add(new int[]{indexI, indexJ + 1});
            } else {
                if (!queue.isEmpty()) queue.add(null);
                distance++;
            }
        }
        grid[i][j] = 1;
    }

    public static int shortestDistance(int[][] grid) {
        int [][] dp = new int[grid.length][grid[0].length];
        int [][] countGrid = new int[grid.length][grid[0].length];
        int totalBuildings = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    bfs(grid, i,j, dp, countGrid);
                    totalBuildings++;
                }
            }
        }
        int minDistance = Integer.MAX_VALUE;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < dp[0].length; j++) {
                if (grid[i][j] == 0 && countGrid[i][j] == totalBuildings) {
                    minDistance = Math.min(minDistance, dp[i][j]);
                }
            }
        }
        return minDistance != Integer.MAX_VALUE ? minDistance : -1;
    }

//    https://leetcode.com/problems/simplify-path/

    public static String simplifyPath(String path) {
        String [] pathArr = path.split("/");
        List<String> arr = new ArrayList<>();
        for (String f : pathArr) {
            if (f.equals("..")) {
                if (arr.size() > 0) arr.remove(arr.size() - 1);
            } else if (!f.isEmpty() && !f.equals(".")) {
                arr.add(f);
            }
        }
        StringBuilder sb = new StringBuilder();
        for (String s : arr) {
            sb.append("/");
            sb.append(s);
        }
        return sb.length() > 0 ? sb.toString() : "/";
    }

//    https://leetcode.com/problems/closest-binary-search-tree-value/

    static int prev = Integer.MIN_VALUE;
    static double prevDiff = Integer.MAX_VALUE;

    public static void closestValueUtil(TreeNode node, double target) {
        if (node == null) return;
        closestValueUtil(node.left, target);
        if (prev != Integer.MIN_VALUE) {
            double currDiff = Math.abs(target - node.val);
            if (currDiff > prevDiff) return;
            prevDiff = currDiff;
        } else {
            prevDiff = Math.abs(target - node.val);
        }
        prev = node.val;
        closestValueUtil(node.right,target);
    }

    public static int closestValue(TreeNode root, double target) {
        closestValueUtil(root,target);
        return prev;
    }

    static int closestNo = Integer.MAX_VALUE;

    static int closestValueOptimised(TreeNode root, double target) {
        if (root == null) return closestNo;
        double diff = Math.abs(target - root.val);
        closestNo = diff < closestNo ? root.val : closestNo;
        if (target > root.val) return closestValueOptimised(root.right, target);
        else return closestValueOptimised(root.left, target);
    }

    static int closest(int [] nums, double target) {
        int low = 0;
        int high = nums.length - 1;
        while (high > low) {
            int mid = low + (high - low) / 2;
            int num = nums[mid];
            if (num > target) high = mid - 1;
            else low = mid + 1;
        }
        if (nums[low] > target) {
            return low > 0 && Math.abs(target - nums[low]) > Math.abs(target - nums[low - 1]) ? nums[low - 1] : nums[low];
        } else {
            return low < nums.length - 1 && Math.abs(target - nums[low]) > Math.abs(target - nums[low + 1]) ? nums[low + 1] : nums[low];
        }
    }

//    https://leetcode.com/problems/range-sum-query-2d-immutable/

    static class NumMatrix {

        int [][] dp;

        public NumMatrix(int[][] matrix) {
            dp = new int[matrix.length + 1][matrix[0].length + 1];
            for (int i = 1; i < dp.length; i++) {
                for (int j = 1; j < dp[0].length; j++) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1] - dp[i - 1][j - 1] + matrix[i - 1][j - 1];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            int r1 = row1 + 1;
            int r2 = row2 + 1;
            int c1 = col1 + 1;
            int c2 = col2 + 1;
            return dp[r2][c2] - dp[r2][c1 - 1] - dp[r1 - 1][c2] +  dp[r1 - 1][c1 - 1];
        }
    }

//    https://leetcode.com/problems/friends-of-appropriate-ages/

    static boolean canBefriend(int ageA, int ageB) {
        if (ageB <= 0.5 * ageA + 7) return false;
        return ageB <= ageA;
    }

    public static int numFriendRequests(int[] ages) {
       Map<Integer,Integer> agesCount = new HashMap<>();
       for (int age : ages) {
           agesCount.put(age,agesCount.getOrDefault(age, 0) + 1);
       }
       int requests = 0;
       for (int ageA : agesCount.keySet()) {
           for (int ageB : agesCount.keySet()) {
               int countA = agesCount.get(ageA);
               int countB = agesCount.get(ageB);
               if (canBefriend(ageA, ageB)) {
                   requests += countA * countB;
                   if (ageA == ageB) requests -= countA;
               }
           }
       }
       return requests;
    }

//    https://leetcode.com/problems/one-edit-distance/

    public static boolean isOneEditDistance(String s, String t) {
        int change = 0;
        int i = 0;
        int j = 0;
        char [] sArr = s.toCharArray();
        char [] tArr = t.toCharArray();
        int n = sArr.length;
        int m = tArr.length;
        while (i < n || j < m) {
            char s1 = ' ';
            char t1 = ' ';
            if (i < n) {
                s1 = sArr[i];
            }
            if (j < m) {
                t1 = tArr[j];
            }
            if (s1 == t1) {
                i++;
                j++;
            } else {
                if (change >= 1) return false;
                change++;
                if (n > m) i++;
                else if (n < m) j++;
                else { i++; j++;}
            }
        }
        return change == 1;
    }

//    https://leetcode.com/problems/flatten-binary-tree-to-linked-list/

    static TreeNode prevNodeLL = null;

    static void bTToLL(TreeNode node) {
       if (node == null) return;
       TreeNode temp = node.right;
       if (prevNodeLL == null) {
           prevNodeLL = node;
       } else {
           prevNodeLL.right = node;
           prevNodeLL = prevNodeLL.right;
       }
       bTToLL(node.left);
       node.left = null;
       bTToLL(temp);
    }

    public static void flatten(TreeNode root) {
        bTToLL(root);
    }

//    https://leetcode.com/problems/find-the-celebrity/

    static boolean knows(int a, int b) {
        return false;
    }

    public static int findCelebrity(int n) {
        int candidate = 0;
        for (int i = 1; i < n; i++) {
            if (knows(candidate,i)) {
                candidate = i;
            }
        }
        for (int i = 0; i < n; i++) {
            if (i != candidate && (knows(candidate, i) || (!knows(i, candidate)))) {
                return -1;
            }
        }
        return candidate;
    }

//    https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/

    public static List<List<Integer>> verticalTraversal(TreeNode root) {
        Map<Integer,List<Integer>> columnsMap = new HashMap<>();
        Map<TreeNode,Integer> shiftMap = new HashMap<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        queue.add(null);
        shiftMap.put(root,0);
        int minShift = Integer.MAX_VALUE;
        int maxShift = Integer.MIN_VALUE;
        while (!queue.isEmpty()) {
            int size = queue.size();
            Map<Integer,List<Integer>> rowMap = new HashMap<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.remove();
                if (node != null) {
                    int shift = shiftMap.get(node);
                    minShift = Math.min(minShift, shift);
                    maxShift = Math.max(maxShift, shift);
                    List<Integer> arr = rowMap.getOrDefault(shift, new ArrayList<>());
                    arr.add(node.val);
                    rowMap.put(shift, arr);
                    if (node.left != null) {
                        shiftMap.put(node.left, shift - 1);
                        queue.add(node.left);
                    }
                    if (node.right != null) {
                        shiftMap.put(node.right, shift + 1);
                        queue.add(node.right);
                    }
                } else {
                    if (!queue.isEmpty()) queue.add(null);
                    for (int key : rowMap.keySet()) {
                        List<Integer> rowNodes = rowMap.get(key);
                        Collections.sort(rowNodes);
                        List<Integer> colList = columnsMap.getOrDefault(key,new ArrayList<>());
                        colList.addAll(rowNodes);
                        columnsMap.put(key,colList);
                    }
                }
            }
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = minShift; i <= maxShift; i++) {
            res.add(columnsMap.get(i));
        }
        return res;
    }

    static class Node {
        public int val;
        public List<Node> neighbors;
        public Node() {
            val = 0;
            neighbors = new ArrayList<>();
        }
        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<>();
        }
        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    }

}