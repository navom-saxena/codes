package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;
import javafx.util.Pair;

import java.util.*;

public class SixthSet {

    public static void main(String[] args) {
//        System.out.println(canCross(new int[]{0,1,3,5,6,8,12,17}));
//        AllOne allOne = new AllOne();
//        allOne.inc("a");
//        allOne.inc("b");
//        allOne.inc("b");
//        allOne.inc("b");
//        allOne.inc("b");
//        allOne.dec("b");
//        allOne.dec("b");
//        System.out.println(allOne.getMaxKey());
//        System.out.println(allOne.getMinKey());
//        allOne.inc("ds");
//        allOne.dec("leet");
//        System.out.println(allOne.getMaxKey());
//        allOne.dec("a");
//        System.out.println(allOne.getMaxKey());
//        System.out.println(allOne.getMinKey());
//        System.out.println(numIslands2(3,3,new int[][]{{0,1},{1,2},{2,1},{1,0},{0,2},{0,0},{1,1}}));
//        System.out.println(shortestBridge(new int[][]{{0,1,0},{0,0,0},{0,0,1}}));
//        System.out.println(findNthDigit(11));
//        System.out.println(mincostTickets(new int[]{1,4,6,7,8,20}, new int[]{2,7,15}));
//        System.out.println(fractionToDecimal(1,6));
//        System.out.println(restoreIpAddresses("010010"));
//        System.out.println(longestConsecutive(new int[]{100,4,200,1,3,2}));
    }

//    https://leetcode.com/problems/stickers-to-spell-word/

    void minStickersUtil(String [] stickers, String target, int i, int count, int [] countArr, int [] minV) {
        if (i >= stickers.length) return;
        boolean allFound = true;
        for (char c : target.toCharArray()) {
            if (countArr[c - 'a'] > 0) {
                allFound = false;
                break;
            }
        }
        if (allFound) {
            minV[0] = Math.min(minV[0], count);
            return;
        }
        int [] copied = Arrays.copyOf(countArr,countArr.length);
        boolean found = false;
        for (char c : stickers[i].toCharArray()) {
            if (copied[c - 'a'] > 0) {
                found = true;
                copied[c - 'a']--;
            }
        }
        if (found) {
            minStickersUtil(stickers, target, i, count + 1, copied, minV);
            minStickersUtil(stickers, target,i + 1, count + 1, copied, minV);
        }
        minStickersUtil(stickers, target,i + 1, count, countArr, minV);
    }

    int minStickersUtilDp(String target, Map<Character,Integer> sticker, Map<String,Integer> dp,
                          List<Map<Character,Integer>> stickers) {
       if (target.length() == 0) return 0;
       if (dp.containsKey(target)) return dp.get(target);
       int res = 1;
       if (sticker.isEmpty()) res = 0;
       StringBuilder remaining = new StringBuilder();
       sticker = new HashMap<>(sticker);
       for (char c : target.toCharArray()) {
           if (sticker.containsKey(c) && sticker.get(c) > 0) sticker.put(c, sticker.get(c) - 1);
           else remaining.append(c);
       }
       if (remaining.length() == 0) return res;
       int used = Integer.MAX_VALUE;
       for (Map<Character,Integer> s : stickers) {
           if (!s.containsKey(remaining.charAt(0))) continue;
           used = Math.min(used, minStickersUtilDp(remaining.toString(), s, dp, stickers));
       }
       dp.put(remaining.toString(), used);
       return used != Integer.MAX_VALUE ? res + used : used;
    }

    public int minStickers(String[] stickers, String target) {
//        int [] minValue = new int[]{Integer.MAX_VALUE};
//        int [] countArr = new int[26];
//        for (char c : target.toCharArray()) {
//            countArr[c - 'a']++;
//        }
//        minStickersUtil(stickers, target,0, 0, countArr, minValue);
//        return minValue[0] == Integer.MAX_VALUE ? -1 : minValue[0];
        List<Map<Character,Integer>> stickersL = new ArrayList<>();
        for (String sticker : stickers) {
            Map<Character,Integer> m = new HashMap<>();
            for (char c : sticker.toCharArray()) m.merge(c, 1, Integer::sum);
            stickersL.add(m);
        }
        return minStickersUtilDp(target, new HashMap<>(), new HashMap<>(), stickersL);
    }

//    https://leetcode.com/problems/frog-jump/

    static int canCrossUtil(int [] stones, int start, int end, int k, int [][] dp) {
        if (dp[start][k] == 0) return dp[start][k];
        for (int i = start + 1; i < stones.length; i++) {
            int currGap = stones[i] - stones[start];
            if (currGap >= k - 1 && currGap <= k + 1) {
                if (canCrossUtil(stones, i, end, currGap, dp) == 1) {
                   return 1;
                }
            }
        }
        if (start == stones.length - 1) return 1;
        dp[start][k] = 0;
        return dp[start][k];
    }

    public static boolean canCross(int[] stones) {
        int [][] dp = new int[stones.length][stones.length];
        for (int[] row : dp) {
            Arrays.fill(row, -1);
        }
        int end = stones[stones.length - 1];
        return canCrossUtil(stones, 0, end, 0, dp) == 1;
    }

//    https://leetcode.com/problems/all-oone-data-structure/

    static class AllOneNode {
        int count;
        String value;
        AllOneNode next;
        AllOneNode prev;

        AllOneNode(int c, String k) {
            count = c;
            value = k;
        }
    }

//    static class AllOne {
//
//        Map<String,AllOneNode> map;
//        AllOneNode head;
//        AllOneNode tail;
//
//        /** Initialize your data structure here. */
//        public AllOne() {
//            map = new HashMap<>();
//        }
//
//        /** Inserts a new key <Key> with value 1. Or increments an existing key by 1. */
//        public void inc(String key) {
//            AllOneNode node;
//            if (map.containsKey(key)) {
//                node = map.get(key);
//                node.count++;
//                if (node == head) return;
//                if (node == tail) tail = node.prev;
//                AllOneNode p = node.prev;
//                if (p != null) p.next = node.next;
//                if (node.next != null) node.next.prev = p;
//                while (p != null && node.count > p.count) {
//                    p = p.prev;
//                }
//                if (p != null) {
//                    AllOneNode tmp = p.next;
//                    node.next = tmp;
//                    if (tmp != null) tmp.prev = node;
//                    else tail = node;
//                    node.prev = p;
//                    p.next = node;
//                } else {
//                    node.next = head;
//                    node.prev = null;
//                    head.prev = node;
//                    head = head.prev;
//                }
//            } else {
//                node = new AllOneNode(1, key);
//                map.put(key,node);
//                if (tail == null) {
//                    tail = node;
//                    if (head == null) head = node;
//                } else {
//                    tail.next = node;
//                    node.next = null;
//                    node.prev = tail;
//                    tail = tail.next;
//                }
//            }
//        }
//
//        /** Decrements an existing key by 1. If Key's value is 1, remove it from the data structure. */
//        public void dec(String key) {
//            AllOneNode node = map.get(key);
//            if (node.count > 1) {
//                node.count--;
//                AllOneNode n = node.next;
//                if (node == head) head = n;
//                if (node == tail) return;
//                if (n != null) n.prev = node.prev;
//                if (node.prev != null) node.prev.next = n;
//                else head = n;
//                while (n != null && node.count < n.count) {
//                    n = n.next;
//                }
//                if (n != null) {
//                    AllOneNode tmp = n.prev;
//                    node.prev = tmp;
//                    if (tmp != null) tmp.next = node;
//                    else head = node;
//                    node.next = n;
//                    n.prev = node;
//                } else {
//                    node.prev = tail;
//                    tail.next = node;
//                    node.next = null;
//                    tail = tail.next;
//                }
//            } else {
//                map.remove(key);
//                if (node == head || node == tail) {
//                    if (node == tail) {
//                        tail = tail.prev;
//                        tail.next = null;
//                    }
//                    if (node == head) {
//                        head = head.next;
//                        head.prev = null;
//                    }
//                }
//                else {
//                    node.prev.next = node.next;
//                    node.next.prev = node.prev;
//                }
//            }
//        }
//
//        public String getMaxKey() {
//            if (head == null) return "";
//            return head.value;
//        }
//
//        /** Returns one of the keys with Minimal value. */
//        public String getMinKey() {
//            if (tail == null) return "";
//            return tail.value;
//        }
//    }

    static class AllOne {

        Map<String,Integer> map;
        TreeSet<String> set;

        /** Initialize your data structure here. */
        public AllOne() {
            map = new HashMap<>();
            set = new TreeSet<>((x, y) -> {
                int fx = map.getOrDefault(x, 0);
                int fy = map.getOrDefault(y, 0);
                return fx == fy ? x.compareTo(y) : fx - fy;
            });
        }

        /** Inserts a new key <Key> with value 1. Or increments an existing key by 1. */
        public void inc(String key) {
            set.remove(key);
            map.put(key, map.getOrDefault(key, 0) + 1);
            set.add(key);
        }

        /** Decrements an existing key by 1. If Key's value is 1, remove it from the data structure. */
        public void dec(String key) {
            set.remove(key);
            map.put(key, map.getOrDefault(key, 0) - 1);
            if (map.get(key) > 0) {
                set.add(key);
            } else {
                map.remove(key);
            }
        }

        public String getMaxKey() {
            if (set.isEmpty()) return "";
            return set.last();
        }

        /** Returns one of the keys with Minimal value. */
        public String getMinKey() {
            if (set.isEmpty()) return "";
            return set.first();
        }
    }

//    https://leetcode.com/problems/maximum-vacation-days/

    static int dfsMaxVacationDays(int [][] flights, int [][] days, int currCity, int currWeek, int [][] dp) {
        int maxValue = 0;
        if (currWeek == days[0].length) return 0;
        if (dp[currCity][currWeek] != Integer.MIN_VALUE) return dp[currCity][currWeek];
        for (int c = 0; c < flights[currCity].length; c++) {
            if (flights[currCity][c] == 1 || currCity == c) {
                int v = days[c][currWeek] +
                        dfsMaxVacationDays(flights, days, c, currWeek + 1, dp);
                maxValue = Math.max(maxValue, v);
            }
        }
        dp[currCity][currWeek] = maxValue;
        return maxValue;
    }

    public static int maxVacationDays(int[][] flights, int[][] days) {
        int [][] dp = new int[days.length][days[0].length];
        for (int [] ints : dp) Arrays.fill(ints, Integer.MIN_VALUE);
       return dfsMaxVacationDays(flights, days, 0, 0, dp);
    }

//    https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/

    static int numWaysDFS(int steps, int arrLen, int curr, int mod, Map<Pair<Integer,Integer>,Integer> dp) {
        if (curr < 0 || curr >= arrLen) return 0;
        if (steps == 0) {
            if (curr == 0) return 1;
            return 0;
        }
        Pair<Integer,Integer> p = new Pair<>(steps,curr);
        if (dp.containsKey(p)) return dp.get(p);
        int c = 0;
        c = (c + numWaysDFS(steps - 1, arrLen, curr, mod, dp) % mod ) % mod;
        c = (c + numWaysDFS(steps - 1, arrLen, curr + 1, mod, dp) % mod ) % mod;
        c = (c + numWaysDFS(steps - 1, arrLen, curr - 1, mod, dp) % mod ) % mod;
        dp.put(p,c);
        return c;
    }

    public static int numWays(int steps, int arrLen) {
        int mod = 1000000007;
        Map<Pair<Integer,Integer>,Integer> dp = new HashMap<>();
        return numWaysDFS(steps, arrLen, 0, mod, dp);
    }

//    https://leetcode.com/problems/sum-root-to-leaf-numbers/

    void sumNumbersUtil(TreeNode node, int [] sum, int runningSum) {
        if (node == null) return;
        runningSum = (runningSum * 10) + node.val;
        if (node.left == null && node.right == null) {
            sum[0] += runningSum;
            return;
        }
        sumNumbersUtil(node.left, sum, runningSum);
        sumNumbersUtil(node.right, sum, runningSum);
    }

    public int sumNumbers(TreeNode root) {
        int [] sum = new int[]{0};
        sumNumbersUtil(root, sum, 0);
        return sum[0];
    }

//    https://leetcode.com/problems/number-of-islands-ii/

    public static List<Integer> numIslands2(int m, int n, int[][] positions) {
        List<Integer> res = new ArrayList<>();
        Map<Integer, Integer> islands = new HashMap<>();
        int[][] directions = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int id = 0;
        int islandsCount = 0;
        for (int[] position : positions) {
            int r = position[0];
            int c = position[1];
            if (islands.containsKey(r * n + c)) {
                res.add(islandsCount);
                continue;
            }
            Set<Integer> overlap = new HashSet<>();
            for (int[] direction : directions) {
                int newI = r + direction[0];
                int newJ = c + direction[1];
                if (newI < 0 || newI >= m || newJ < 0 || newJ >= n) continue;
                int hash = newI * n + newJ;
                if (islands.containsKey(hash)) overlap.add(islands.get(hash));
            }
            if (overlap.isEmpty()) {
                islands.put(r * n + c, id);
                id++;
                islandsCount++;
            } else if (overlap.size() == 1) {
                islands.put(r * n + c, overlap.iterator().next());
            } else {
                int rootId = overlap.iterator().next();
                for (Map.Entry<Integer,Integer> es: islands.entrySet()) {
                    int k = es.getKey();
                    int v = es.getValue();
                    if (overlap.contains(v)) {
                        islands.put(k, rootId);
                    }
                }
                islands.put(r * n + c, rootId);
                islandsCount -= overlap.size() - 1;
            }
            res.add(islandsCount);
        }
        return res;
    }

//    https://leetcode.com/problems/similar-string-groups/

    static boolean isSimilar(String w1, String w2) {
        int disSimilarCount = 0;
        for (int i = 0; i < w1.length(); i++) {
            if (w1.charAt(i) != w2.charAt(i)) disSimilarCount++;
            if (disSimilarCount > 2) return false;
        }
        return disSimilarCount == 0 || disSimilarCount == 2;
    }

    static void dfsSimilarGroups(String word, String [] words, Set<String> visited) {
        if (visited.contains(word)) return;
        visited.add(word);
        for (String w : words) {
            if (isSimilar(w, word)) dfsSimilarGroups(w, words, visited);
        }
    }

    public static int numSimilarGroups(String[] strs) {
       Set<String> visited = new HashSet<>();
       int disConnectedGraphs = 0;
       for (String str : strs) {
           if (visited.contains(str)) continue;
           dfsSimilarGroups(str, strs, visited);
           disConnectedGraphs++;
       }
       return disConnectedGraphs;
    }

//    https://leetcode.com/problems/balance-a-binary-search-tree/

    void inOrderPopulate(TreeNode node, List<Integer> arr) {
        if (node == null) return;
        inOrderPopulate(node.left, arr);
        arr.add(node.val);
        inOrderPopulate(node.right,arr);
    }

    TreeNode createBalancedBST(List<Integer> arr, int low, int high) {
        if (low > high) return null;
        if (low == high) return new TreeNode(arr.get(low));
        int mid = low + (high - low) / 2;
        TreeNode node = new TreeNode(arr.get(mid));
        node.left = createBalancedBST(arr, low, mid - 1);
        node.right = createBalancedBST(arr, mid + 1, high);
        return node;
    }

    public TreeNode balanceBST(TreeNode root) {
        List<Integer> arr = new ArrayList<>();
        inOrderPopulate(root, arr);
        return createBalancedBST(arr, 0, arr.size() - 1);
    }

//    https://leetcode.com/problems/shortest-bridge/

    static void shortestBridgeDFSMark(int [][] grid, int i, int j, Deque<int []> deque) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length
                || grid[i][j] == Integer.MAX_VALUE || grid[i][j] == 0) return;
        deque.add(new int[]{i,j});
        grid[i][j] = Integer.MAX_VALUE;
        shortestBridgeDFSMark(grid, i - 1, j, deque);
        shortestBridgeDFSMark(grid, i + 1, j, deque);
        shortestBridgeDFSMark(grid, i, j - 1, deque);
        shortestBridgeDFSMark(grid, i, j + 1, deque);
    }

    public static int shortestBridge(int[][] grid) {
        Deque<int []> deque = new ArrayDeque<>();
        a : for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    shortestBridgeDFSMark(grid, i, j, deque);
                    break a;
                }
            }
        }
        int [][] directions = new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
        int minDistance = 0;
        while (!deque.isEmpty()) {
            int n = deque.size();
            for (int k = 0; k < n; k++) {
                int[] node = deque.remove();
                int iTh = node[0];
                int jTh = node[1];
                if (iTh < 0 || iTh >= grid.length || jTh < 0 || jTh >= grid[iTh].length
                        || grid[iTh][jTh] == Integer.MIN_VALUE) continue;
                if (grid[iTh][jTh] == 1) return minDistance - 1;
                grid[iTh][jTh] = Integer.MIN_VALUE;
                for (int[] direction : directions) {
                    int newI = iTh + direction[0];
                    int newJ = jTh + direction[1];
                    deque.add(new int[]{newI, newJ});
                }
            }
            minDistance++;
        }
        return minDistance;
    }

//    https://leetcode.com/problems/minimum-area-rectangle/

    public static int minAreaRect(int[][] points) {
        Map<Integer,Set<Integer>> rows = new HashMap<>();
        for (int [] point : points) {
            int r = point[0];
            int c = point[1];
            Set<Integer> cols = rows.getOrDefault(r, new HashSet<>());
            cols.add(c);
            rows.put(r, cols);
        }
        int minArea = Integer.MAX_VALUE;
        for (int r1 : rows.keySet()) {
            Set<Integer> c1 = rows.get(r1);
            if (c1.size() == 1) continue;
            for (int r2 : rows.keySet()) {
                Set<Integer> c2 = rows.get(r2);
                if (c2.size() == 1 || r1 == r2) continue;
                List<Integer> c1Arr = new ArrayList<>(c1);
                Collections.sort(c1Arr);
                for (int i = 0; i < c1Arr.size() - 1; i++) {
                    int cF = c1Arr.get(i);
                    int cS = c1Arr.get(i + 1);
                    if (c2.contains(cF) && c2.contains(cS)) {
                        int area = Math.abs(r1 - r2) * (cS - cF);
                        minArea = Math.min(minArea, area);
                    }
                }
            }
        }
//        for (int [] p1 : points) {
//            for (int [] p2 : points) {
//                int r1 = p1[0];
//                int c1 = p1[1];
//                int r2 = p2[0];
//                int c2 = p2[1];
//                if (r1 == r2 || c1 == c2) continue;
//                if (rows.get(r1).contains(c2) && rows.get(r2).contains(c1)) {
//                    int area = Math.abs(r1 - r2) * Math.abs(c1 - c2);
//                    minArea = Math.min(minArea,area);
//                }
//            }
//        }
        return minArea != Integer.MAX_VALUE ? minArea : 0;
    }

//    https://leetcode.com/problems/search-a-2d-matrix-ii/

    boolean binarySearchSpecific(int [][] matrix, int rLow, int rHigh, int cLow, int cHigh, int target) {
        if (rLow > rHigh || cLow > cHigh) return false;
        int rMid = rLow + (rHigh - rLow) / 2;
        int cMid = cLow + (cHigh - cLow) / 2;
        if (matrix[rMid][cMid] == target) return true;
        if (matrix[rMid][cMid] < target) {
            if (rLow == rHigh) {
                return binarySearchSpecific(matrix, rLow, rHigh, cMid + 1, cHigh, target);
            } else if (cLow == cHigh) {
                return binarySearchSpecific(matrix, rMid + 1, rHigh, cLow, cHigh, target);
            }
        } else {
            if (rLow == rHigh) {
                return binarySearchSpecific(matrix, rLow, rHigh, cLow, cMid - 1, target);
            } else if (cLow == cHigh) {
                return binarySearchSpecific(matrix, rLow, rMid - 1, cLow, cHigh, target);
            }
        }
        return false;
    }

    boolean searchMatrixUtil(int [][] matrix, int rLow, int rHigh, int cLow, int cHigh, int target) {
        if (rLow > rHigh || cLow > cHigh) return false;
        int rMid = rLow + (rHigh - rLow) / 2;
        int cMid = cLow + (cHigh - cLow) / 2;
        if (matrix[rMid][cMid] == target) return true;
        if (binarySearchSpecific(matrix, rMid, rMid, cLow, cHigh, target) ||
                binarySearchSpecific(matrix, rLow, rHigh, cMid, cMid, target)) return true;
        if (matrix[rMid][cMid] < target) {
            return searchMatrixUtil(matrix, rMid + 1, rHigh, cMid + 1, cHigh, target) ||
                    searchMatrixUtil(matrix, rLow, rMid - 1, cMid, cHigh, target) ||
                    searchMatrixUtil(matrix, rMid, rHigh, cLow, cHigh - 1, target);
        } else {
            return searchMatrixUtil(matrix, rLow, rMid - 1, cLow, cMid - 1, target) ||
                    searchMatrixUtil(matrix, rMid + 1, rHigh, cLow, cMid, target) ||
                    searchMatrixUtil(matrix, rLow, rMid, cMid + 1, cHigh, target);
        }
    }

    public boolean searchMatrix(int[][] matrix, int target) {
//        return searchMatrixUtil(matrix, 0, matrix.length - 1, 0, matrix[0].length - 1, target);
        int r = 0;
        int c = matrix[0].length - 1;
        while (r < matrix.length && c >= 0) {
            if (matrix[r][c] == target) return true;
            else if (matrix[r][c] < target) r++;
            else c--;
        }
        return false;
    }

//    https://leetcode.com/problems/implement-trie-prefix-tree/

    static class TrieNode {
        char c;
        Map<Character,TrieNode> children;
        boolean flag;

        TrieNode(char c) {
            this.c = c;
            this.children = new HashMap<>();
        }
    }

    static class Trie {

        TrieNode root;

        /** Initialize your data structure here. */
        public Trie() {
            root = new TrieNode('.');
        }

        void insertUtils(char [] word, int index, TrieNode node) {
            if (index >= word.length) return;
            TrieNode c = node.children.get(word[index]);
            if (c == null) {
                c = new TrieNode(word[index]);
                node.children.put(word[index],c);
            }
            if (index == word.length - 1) {
                c.flag = true;
            } else {
                insertUtils(word, index + 1, c);
            }
        }

        /** Inserts a word into the trie. */
        public void insert(String word) {
            insertUtils(word.toCharArray(), 0, root);
        }

        boolean searchUtils(char [] word, int index, TrieNode node) {
            TrieNode c = node.children.get(word[index]);
            if (c == null) {
               return false;
            }
            if (index == word.length - 1) {
                return c.flag;
            } else {
                return searchUtils(word, index + 1, c);
            }
        }

        /** Returns if the word is in the trie. */
        public boolean search(String word) {
            return searchUtils(word.toCharArray(), 0, root);
        }

        boolean startsWithUtils(char [] word, int index, TrieNode node) {
            TrieNode c = node.children.get(word[index]);
            if (c == null) {
                return false;
            }
            if (index == word.length - 1) {
                return true;
            } else {
                return startsWithUtils(word, index + 1, c);
            }
        }

        /** Returns if there is any word in the trie that starts with the given prefix. */
        public boolean startsWith(String prefix) {
            return startsWithUtils(prefix.toCharArray(), 0, root);
        }

    }

//    https://leetcode.com/problems/evaluate-division/

    static double dfsConversionValue(String from, String to, Map<String,Set<String>> adj,
                                     Map<String,Double> weights, double initial, Set<String> visited) {
        if (from.equals(to)) return initial;
        if (!visited.contains(from)) {
            visited.add(from);
            for (String neighbour : adj.getOrDefault(from, new HashSet<>())) {
                double computed = dfsConversionValue(neighbour, to, adj, weights,
                        initial * weights.get(from + " " + neighbour), visited);
                if (computed != -1.0) return computed;
            }
        }
        return -1.0;
    }

    public static double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        Map<String,Set<String>> adj = new HashMap<>();
        Map<String,Double> weights = new HashMap<>();
        for (int i = 0; i < equations.size(); i++) {
            String from = equations.get(i).get(0);
            String to = equations.get(i).get(1);
            double weight = values[i];
            Set<String> fromNeighbours = adj.getOrDefault(from, new HashSet<>());
            fromNeighbours.add(to);
            adj.put(from, fromNeighbours);
            Set<String> toNeighbours = adj.getOrDefault(to, new HashSet<>());
            toNeighbours.add(from);
            adj.put(to, toNeighbours);
            weights.put(from + " " + to, weight);
            weights.put(to + " " + from, 1.0 / weight);
        }
        double [] res = new double[queries.size()];
        for (int i = 0; i < queries.size(); i++) {
            String from = queries.get(i).get(0);
            String to = queries.get(i).get(1);
            Set<String> visited = new HashSet<>();
            if (adj.containsKey(from) && adj.containsKey(to))
                res[i] = dfsConversionValue(from, to, adj, weights, 1.0, visited);
            else res[i] = -1.0;
        }
        return res;
    }

//    https://leetcode.com/problems/nth-digit/

    public static int findNthDigit(int n) {
        int len = 1;
        long count = 9;
        int start = 1;
        while (n > len * count) {
            n -= len * count;
            len++;
            count *= 10;
            start *= 10;
        }
        start += (n - 1) / len;
        String s = String.valueOf(start);
        return Character.getNumericValue(s.charAt((n - 1) % len));
    }

//    https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/

    static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    public Node connect(Node node) {
        if (node == null) return null;
        if (node.left != null && node.right != null) {
            node.left.next = node.right;
        }
        Node n = node.next;
        while (n != null) {
            Node l = node.right != null ? node.right : node.left;
            Node r = n.left != null ? n.left : n.right;
            if (l == null) break;
            if (r != null) {
                l.next = r;
                break;
            }
            n = n.next;
        }
        node.right = connect(node.right);
        node.left = connect(node.left);
        return node;
    }

//    https://leetcode.com/problems/minimum-cost-for-tickets/

    static int mincostTicketsUtil(int [] days, int [] cost, int [] duration, int i, int [] dp) {
        if (i >= days.length) return 0;
        if (dp[i] != Integer.MAX_VALUE) return dp[i];
        for (int k = 0; k < cost.length; k++) {
            int j = i;
            while (j < days.length && days[j] < days[i] + duration[k]) j++;
            dp[i] = Math.min(dp[i], mincostTicketsUtil(days, cost, duration, j, dp) + cost[k]);
        }
        return dp[i];
    }

    public static int mincostTickets(int[] days, int[] costs) {
        int [] dp = new int[days.length];
        Arrays.fill(dp, Integer.MAX_VALUE);
        int [] duration = new int[]{1,7,30};
        return mincostTicketsUtil(days, costs, duration,0, dp);
    }

//    https://leetcode.com/problems/increasing-triplet-subsequence/

    public boolean increasingTriplet(int[] nums) {
        int smallest = Integer.MAX_VALUE;
        int secondSmallest = Integer.MAX_VALUE;
        for (int num : nums) {
            if (num <= smallest) smallest = num;
            else if (num < secondSmallest) secondSmallest = num;
            else return true;
        }
        return false;
    }

//    https://leetcode.com/problems/remove-sub-folders-from-the-filesystem/

    static class TrieNodeFolder {
        String folder;
        Map<String,TrieNodeFolder> children;
        boolean end;

        TrieNodeFolder(String f) {
            this.folder = f;
            this.children = new HashMap<>();
        }
    }

    static class TrieFolder {
        TrieNodeFolder root;

        TrieFolder() {
            root = new TrieNodeFolder(".");
        }

        void insertUtils(String [] folder, int i, TrieNodeFolder node) {
            if (i >= folder.length) return;
            TrieNodeFolder c = node.children.get(folder[i]);
            if (c == null) {
                c = new TrieNodeFolder(folder[i]);
                node.children.put(folder[i],c);
            }
            if (i == folder.length - 1) c.end = true;
            else insertUtils(folder, i + 1, c);
        }

        void insert(String [] folder) {
            insertUtils(folder, 1, root);
        }

        boolean checkParentExistsUtil(String [] folder, int i, TrieNodeFolder node) {
            if (i >= folder.length) return false;
            TrieNodeFolder c = node.children.get(folder[i]);
            if (i != folder.length - 1 && c.end) return true;
            else return checkParentExistsUtil(folder, i + 1, c);
        }

        boolean checkParentExists(String [] folder) {
            return checkParentExistsUtil(folder, 1, root);
        }
    }

    public List<String> removeSubfolders(String[] folder) {
        Set<String> foldersSet = new HashSet<>(Arrays.asList(folder));
        TrieFolder trie = new TrieFolder();
        for (String f : folder) {
            trie.insert(f.split("/"));
        }
        for (String f : folder) {
            if (trie.checkParentExists(f.split("/"))) foldersSet.remove(f);
        }
        return new ArrayList<>(foldersSet);
    }

//    https://leetcode.com/problems/fraction-to-recurring-decimal/

    public static String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) return "0";
        StringBuilder ans = new StringBuilder();
        long n = numerator;
        long d = denominator;
        if ((n > 0 && d < 0) || (n < 0 && d > 0)) {
            ans.append("-");
            if (n < 0) n *= -1;
            else d *= -1;
        }
        long q = n / d;
        long r = n % d;
        ans.append(q);
        if (r == 0) return ans.toString();
        else {
            ans.append(".");
            Map<Long,Integer> remsBefore = new HashMap<>();
            while (r != 0) {
                if (remsBefore.containsKey(r)) {
                    int len = remsBefore.get(r);
                    ans.insert(len, '(');
                    ans.append(')');
                    break;
                } else {
                    remsBefore.put(r, ans.length());
                    r *= 10;
                    q = r / d;
                    r = r % d;
                    ans.append(q);
                }
            }
        }
        return ans.toString();
    }

//    https://leetcode.com/problems/restore-ip-addresses/

    static boolean validateIPAddress(String s) {
        String [] sArr = s.split("\\.");
        if (sArr.length != 4) return false;
        for (String v : sArr) {
            if (v.length() == 0 || v.length() > 3) return false;
            int value = Integer.parseInt(v);
            if (value < 0 || value > 255) return false;
            if (value != 0 && v.startsWith("0") || (value == 0 && v.length() > 1)) return false;
        }
        return true;
    }

    static void restoreIpAddressesUtil(char [] s, int i, String prev, int split, List<String> res, StringBuilder sb) {
       if (i == s.length) {
           if (split == 0) {
               System.out.println(sb.toString());
               if (sb.charAt(sb.length() - 1) != '.') res.add(sb.toString());
           }
           return;
       }
       String v = prev + s[i];
       int no = Integer.parseInt(v);
       if (no < 0 || no > 255) return;
       if ((no != 0 && v.startsWith("0")) || (no == 0 && v.length() > 1)) return;
       sb.append(s[i]);
       restoreIpAddressesUtil(s, i + 1, v, split, res, sb);
       if (split > 0 && i < s.length - 1) {
           sb.append('.');
           restoreIpAddressesUtil(s, i + 1, "", split - 1, res, sb);
           sb.deleteCharAt(sb.length() - 1);
       }
       sb.deleteCharAt(sb.length() - 1);
    }

    static void restoreIpAddressesUtil2(String s, int i, int splitIndex, String [] split, List<String> res) {
        if (i == s.length() && splitIndex == 4) {
            StringBuilder sb = new StringBuilder();
            for (String sp : split) {
                sb.append(sp);
                sb.append(".");
            }
            sb.deleteCharAt(sb.length() - 1);
            res.add(sb.toString());
            return;
        } else if (i == s.length() || splitIndex == 4) return;
        for (int len = 1; len <= 3 && i + len <= s.length(); len++) {
            String sub = s.substring(i, i + len);
            int no = Integer.parseInt(sub);
            if (no > 255 || (sub.length() > 1 && sub.startsWith("0"))) return;
            split[splitIndex] = sub;
            restoreIpAddressesUtil2(s, i + len, splitIndex + 1, split, res);
            split[splitIndex] = "";
        }
    }

    public static List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
//        restoreIpAddressesUtil(s.toCharArray(), 0, "", 3, res, new StringBuilder());
        restoreIpAddressesUtil2(s, 0, 0, new String[4], res);
        return res;
    }

//    https://leetcode.com/problems/intersection-of-three-sorted-arrays/

    public static List<Integer> arraysIntersection(int[] arr1, int[] arr2, int[] arr3) {
        List<Integer> res = new ArrayList<>();
        int i = 0;
        int j = 0;
        int k = 0;
        int m = arr1.length;
        int n = arr2.length;
        int o = arr3.length;
        while (i < m && j < n && k < o) {
            if (arr1[i] == arr2[j] && arr2[j] == arr3[k]) {
                res.add(arr1[i]);
                i++;
                j++;
                k++;
            } else {
                int max = Math.max(arr1[i], Math.max(arr2[j], arr3[k]));
                while (i < m && arr1[i] < max) i++;
                while (j < n && arr2[j] < max) j++;
                while (k < o && arr3[k] < max) k++;
            }
        }
        return res;
    }

//    https://leetcode.com/problems/minimum-depth-of-binary-tree/

    int minDepthUtils(TreeNode node) {
        if (node == null) return Integer.MAX_VALUE;
        if (node.left == null && node.right == null) return 1;
        int l = minDepthUtils(node.left);
        int r = minDepthUtils(node.right);
        return Math.min(l,r) + 1;
    }

    void minDepthUtils2(TreeNode node, int d, int[] minV) {
        if (node == null) return;
        if (node.left == null && node.right == null) {
            minV[0] = Math.min(minV[0], d);
            return;
        }
        minDepthUtils2(node.left, d + 1, minV);
        minDepthUtils2(node.right, d + 1, minV);
    }

    public int minDepth(TreeNode root) {
//        int min = minDepthUtils(root);
//        return min != Integer.MAX_VALUE ? min : 0;
        int [] minV = new int[]{Integer.MAX_VALUE};
        minDepthUtils2(root, 1, minV);
        return minV[0] == Integer.MAX_VALUE ? 0 : minV[0];
    }

//    https://leetcode.com/problems/average-of-levels-in-binary-tree/

    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> ans = new ArrayList<>();
        Deque<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int n = queue.size();
            double sum = 0;
            for (int i = 0; i < n; i++) {
                TreeNode node = queue.remove();
                sum += node.val;
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            ans.add(sum / n);
        }
        return ans;
    }

//    https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/

    static class MultiNode {
        public int val;
        public MultiNode prev;
        public MultiNode next;
        public MultiNode child;
    };

    MultiNode flattenUtil(MultiNode node) {
        if (node == null) return null;
        MultiNode prev = node;
        while (node != null) {
            if (node.child != null) {
                MultiNode lastOfUpper = flattenUtil(node.child);
                MultiNode temp = node.next;
                node.next = node.child;
                node.next.prev = node;
                node.child = null;
                lastOfUpper.next = temp;
                if (temp != null) temp.prev = lastOfUpper;
                prev = lastOfUpper;
                node = temp;
            } else {
                prev = node;
                node = node.next;
            }
        }
        return prev;
    }

    public MultiNode flatten(MultiNode head) {
        flattenUtil(head);
        return head;
    }

//    https://leetcode.com/problems/longest-consecutive-sequence/

    static int dfsConsecutive(int no, Map<Integer,Integer> numbers, int [] dp) {
        if (numbers.get(no) == null) return 0;
        if (dp[numbers.get(no)] != 0) return dp[numbers.get(no)];
        int v = 1 + dfsConsecutive(no + 1, numbers, dp);
        dp[numbers.get(no)] = v;
        return v;
    }

    public static int longestConsecutive(int[] nums) {
//        Map<Integer,Integer> numbers = new HashMap<>();
//        for (int i = 0; i < nums.length; i++) numbers.put(nums[i], i);
//        int max = Integer.MIN_VALUE;
//        int [] dp = new int[nums.length];
//        for (int i = 0; i < nums.length; i++) {
//            if (dp[i] == 0) dp[i] = dfsConsecutive(nums[i], numbers, dp);
//            max = Math.max(max, dp[i]);
//        }
//        return max;
        int max = 0;
        Set<Integer> numbers = new HashSet<>();
        for (int num : nums) numbers.add(num);
        for (int num : nums) {
            if (!numbers.contains(num - 1)) {
                int no = num;
                int count = 0;
                while (numbers.contains(no)) {
                    count++;
                    no += 1;
                }
                max = Math.max(max, count);
            }
        }
        return max;
    }

//    https://leetcode.com/problems/best-meeting-point/

    List<Integer> collectRows(int [][] grid) {
        List<Integer> rows = new ArrayList<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) rows.add(i);
            }
        }
        return rows;
    }

    List<Integer> collectCols(int [][] grid) {
        List<Integer> cols = new ArrayList<>();
        for (int j = 0; j < grid[0].length; j++) {
            for (int[] ints : grid) {
                if (ints[j] == 1) cols.add(j);
            }
        }
        return cols;
    }

    int getDistance(List<Integer> rc, int origin) {
        int d = 0;
        for (int p : rc) {
            d += Math.abs(p - origin);
        }
        return d;
    }

    public int minTotalDistance(int[][] grid) {
        List<Integer> rows = collectRows(grid);
        List<Integer> cols = collectCols(grid);
        int middleRow = rows.get(rows.size() / 2);
        int middleCol = cols.get(cols.size() / 2);

        return getDistance(rows, middleRow) + getDistance(cols, middleCol);
    }

//    https://leetcode.com/problems/previous-permutation-with-one-swap/

    public int[] prevPermOpt1(int[] arr) {
        for (int i = arr.length - 2; i >= 0; i--) {
            int maxNo = Integer.MIN_VALUE;
            int maxNoIndex = i;
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j] < arr[i] && arr[j] > maxNo) {
                    maxNo = arr[j];
                    maxNoIndex = j;
                }
            }
            if (maxNoIndex != i) {
                int temp = arr[i];
                arr[i] = arr[maxNoIndex];
                arr[maxNoIndex] = temp;
                return arr;
            }
        }
        return arr;
    }

//    https://leetcode.com/problems/the-maze/

    boolean boundsAndWallCheck(int [][] maze, int i, int j) {
        return i >= 0 && i < maze.length && j >= 0 && j < maze[0].length
                && maze[i][j] != 1;
    }

//    hasPathUnoptimized same method as hasPath
    public boolean hasPathUnoptimized(int[][] maze, int[] start, int[] destination) {
        Deque<int [][]> deque = new ArrayDeque<>();
        Set<String> visited = new HashSet<>();
        int [][] directions = new int[][]{{-1,0},{1,0},{0,-1},{0,1}};
        for (int [] direction : directions) {
            if (boundsAndWallCheck(maze, start[0] + direction[0], start[1] + direction[1]))
                deque.add(new int[][]{start, direction});
        }
        while (!deque.isEmpty()) {
            int [][] node = deque.remove();
            int [] curr = node[0];
            int [] direction = node[1];
            String s = curr[0] + " " + curr[1] + " " + direction[0] + " " + direction[1];
            if (visited.contains(s)) continue;
            visited.add(s);
            int newIth = curr[0] + direction[0];
            int newJth = curr[1] + direction[1];
            if (!boundsAndWallCheck(maze, newIth, newJth)) {
                if (curr[0] == destination[0] && curr[1] == destination[1]) return true;
                else {
                    for (int [] d : directions) {
                        int newIAfter = curr[0] + d[0];
                        int newJAfter = curr[1] + d[1];
                        if (boundsAndWallCheck(maze, newIAfter, newJAfter))
                            deque.add(new int[][]{new int[]{newIAfter,newJAfter}, d});
                    }
                }
            } else {
                deque.add(new int[][]{new int[]{newIth,newJth}, direction});
            }
        }
        return false;
    }

    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        Deque<int []> deque = new ArrayDeque<>();
        boolean [][] visited = new boolean[maze.length][maze[0].length];
        int [][] directions = new int[][]{{-1,0},{1,0},{0,-1},{0,1}};
        deque.add(start);
        while (!deque.isEmpty()) {
            int [] node = deque.remove();
            if (node[0] == destination[0] && node[1] == destination[1]) return true;
            for (int [] direction : directions) {
                int newI = node[0] + direction[0];
                int newJ = node[1] + direction[1];
                while (boundsAndWallCheck(maze, newI, newJ)) {
                    newI += direction[0];
                    newJ += direction[1];
                }
                newI -= direction[0];
                newJ -= direction[1];
                if (!visited[newI][newJ]) deque.add(new int[]{newI,newJ});
                visited[newI][newJ] = true;
            }
        }
        return false;
    }

//    https://leetcode.com/problems/angle-between-hands-of-a-clock/

    public double angleClock(int hour, int minutes) {
        double hourAngle = (minutes * 0.5) + (hour * 30);
        double minAngle = minutes * 6;
        double angle = Math.abs(minAngle - hourAngle);
        return angle > 180 ? 360 - angle : angle;
    }

//    https://leetcode.com/problems/closest-leaf-in-a-binary-tree/

    void getParent(TreeNode node, Map<TreeNode,TreeNode> parents) {
        if (node == null) return;
        if (node.left != null) parents.put(node.left, node);
        if (node.right != null) parents.put(node.right, node);
        getParent(node.left, parents);
        getParent(node.right, parents);
    }

    TreeNode getTarget(TreeNode node, int k) {
        if (node == null) return null;
        if (node.val == k) return node;
        TreeNode left = getTarget(node.left, k);
        if (left != null) return left;
        return getTarget(node.right, k);
    }

    int bfsTreeClosestLeaf(TreeNode start, Map<TreeNode,TreeNode> parents) {
        Deque<TreeNode> deque = new ArrayDeque<>();
        Set<TreeNode> visited = new HashSet<>();
        deque.add(start);
        while (!deque.isEmpty()) {
            int n = deque.size();
            for (int i = 0; i < n; i++) {
                TreeNode node = deque.remove();
                if (node.left == null && node.right == null) return node.val;
                if (visited.contains(node)) continue;
                visited.add(node);
                if (node.left != null) deque.add(node.left);
                if (node.right != null) deque.add(node.right);
                if (parents.get(node) != null) deque.add(parents.get(node));
            }
        }
        return Integer.MIN_VALUE;
    }

    public int findClosestLeaf(TreeNode root, int k) {
        Map<TreeNode,TreeNode> parents = new HashMap<>();
        parents.put(root, null);
        getParent(root, parents);
        TreeNode start = getTarget(root, k);
        return bfsTreeClosestLeaf(start, parents);
    }

//    https://leetcode.com/problems/remove-linked-list-elements/

    public ListNode removeElements(ListNode head, int val) {
        ListNode sentinel = new ListNode(Integer.MIN_VALUE);
        sentinel.next = head;
        ListNode prev = sentinel;
        ListNode curr = head;
        while (curr != null) {
            if (curr.val == val) prev.next = curr.next;
            else prev = curr;
            curr = curr.next;
        }
        return sentinel.next;
    }

//    https://leetcode.com/problems/remove-nth-node-from-end-of-list/

    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) return null;
        ListNode sentinel = new ListNode(Integer.MIN_VALUE);
        sentinel.next = head;
        ListNode fastP = sentinel;
        ListNode slowP = sentinel;
        while (fastP.next != null) {
            if (n <= 0) {
                slowP = slowP.next;
            }
            fastP = fastP.next;
            n--;
        }
        slowP.next = slowP.next.next;
        return sentinel.next;
    }

//    https://leetcode.com/problems/string-compression/

    public int compress(char[] chars) {
        int p = 0;
        int c = 1;
        for (int i = 0; i < chars.length; i++) {
            if (i < chars.length - 1 && chars[i] == chars[i + 1]) c++;
            else {
                chars[p] = chars[i];
                p++;
                if (c > 1) {
                    char [] charArr = String.valueOf(c).toCharArray();
                    for (char value : charArr) {
                        chars[p] = value;
                        p++;
                    }
                }
                c = 1;
            }
        }
        return p;
    }

//    https://leetcode.com/problems/pancake-sorting/

    int getMaxValueIndex(int [] arr, int limitLength) {
        int max = Integer.MIN_VALUE;
        int maxIndex = -1;
        for (int i = 0; i <= limitLength; i++) {
            if (arr[i] > max) {
                max = arr[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    void flip(int [] arr, int limitLength) {
        int i = 0;
        int j = limitLength;
        while (i < j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++;
            j--;
        }
    }

    public List<Integer> pancakeSort(int[] arr) {
        List<Integer> flipRes = new ArrayList<>();
        int n = arr.length - 1;
        while (n >= 0) {
            int maxVIndex = getMaxValueIndex(arr, n);
            if (maxVIndex != n) {
                if (maxVIndex != 0) {
                    flip(arr, maxVIndex);
                    flipRes.add(maxVIndex + 1);
                }
                flip(arr, n);
                flipRes.add(n + 1);
            }
            n--;
        }
        return flipRes;
    }

//    https://leetcode.com/problems/single-element-in-a-sorted-array/

    int singleNonDuplicateUtils(int [] arr, int low, int high) {
        if (low > high) return -1;
        if (low == high) return arr[low];
        int mid = low + (high - low) / 2;
        if (0 < mid && arr[mid] != arr[mid - 1] && arr[mid] != arr[mid + 1]) return arr[mid];
        if (mid == 0 && arr[mid] != arr[mid + 1]) return arr[mid];
        if (mid % 2 == 0) {
            if (mid < high && arr[mid] == arr[mid + 1]) return singleNonDuplicateUtils(arr, mid + 1, high);
            else return singleNonDuplicateUtils(arr, low, mid - 1);
        } else {
            if (mid < high && arr[mid] == arr[mid + 1]) return singleNonDuplicateUtils(arr, low, mid - 1);
            else return singleNonDuplicateUtils(arr, mid + 1, high);
        }
    }

    public int singleNonDuplicate(int[] nums) {
        return singleNonDuplicateUtils(nums, 0, nums.length - 1);
    }

//    https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/

    public int findMinArrowShots(int[][] points) {
        if (points.length == 0) return 0;
        Arrays.sort(points, Comparator.comparingInt(a -> a[1]));
        int arrows = 1;
        int pos = points[0][1];
        for (int i = 1; i < points.length; i++) {
            if (pos >= points[i][0]) continue;
            arrows++;
            pos = points[i][1];
        }
        return arrows;
    }

//    https://leetcode.com/problems/sort-list/


    public ListNode sortList(ListNode head) {
        if (head == null) return null;
        ListNode slowP = head;
        ListNode fastP = head;
        ListNode prev = head;
        while (fastP != null && fastP.next != null) {
            prev = slowP;
            fastP = fastP.next.next;
            slowP = slowP.next;
        }
        ListNode h2 = prev.next;
        prev.next = null;
        if (h2 == null) return head;
        ListNode sortedH1 =sortList(head);
        ListNode sortedH2 = sortList(h2);
        ListNode h1Curr = sortedH1;
        ListNode h2Curr = sortedH2;
        ListNode sentinel = new ListNode(Integer.MIN_VALUE);
        ListNode curr = sentinel;
        while (h1Curr != null && h2Curr != null) {
            if (h1Curr.val <= h2Curr.val) {
                ListNode h1CurrNext = h1Curr.next;
                h1Curr.next = null;
                curr.next = h1Curr;
                h1Curr = h1CurrNext;
            } else {
                ListNode h2CurrNext = h2Curr.next;
                h2Curr.next = null;
                curr.next = h2Curr;
                h2Curr = h2CurrNext;
            }
            curr = curr.next;
        }
        if (h1Curr != null) curr.next = h1Curr;
        else curr.next = h2Curr;

        return sentinel.next;
    }

//    https://leetcode.com/problems/tree-diameter/

    int treeDiameterDFS(Map<Integer,Set<Integer>> adj, Set<Integer> visited, int curr, int [] diameter) {
        if (visited.contains(curr)) return 0;
        visited.add(curr);
        int maxDepth = 0;
        int secondMaxDepth = 0;
        for (int neighbour : adj.getOrDefault(curr, new HashSet<>())) {
            int d = treeDiameterDFS(adj, visited, neighbour, diameter);
            if (d > maxDepth) {
                secondMaxDepth = maxDepth;
                maxDepth = d;
            }
            else if (d > secondMaxDepth) secondMaxDepth = d;;
        }
        diameter[0] = Math.max(diameter[0], maxDepth + secondMaxDepth + 1);
        return maxDepth + 1;
    }

    public int treeDiameter(int[][] edges) {
        Set<Integer> visited = new HashSet<>();
        Map<Integer,Set<Integer>> adj = new HashMap<>();
        for (int [] edge : edges) {
            int from = edge[0];
            int to = edge[1];
            Set<Integer> fromNeighbours = adj.getOrDefault(from, new HashSet<>());
            fromNeighbours.add(to);
            adj.put(from, fromNeighbours);
            Set<Integer> toNeighbours = adj.getOrDefault(to, new HashSet<>());
            toNeighbours.add(from);
            adj.put(to, toNeighbours);
        }
        int [] diameter = new int[]{Integer.MIN_VALUE};
        int d = treeDiameterDFS(adj, visited, edges[0][0], diameter);
        diameter[0] = Math.max(diameter[0], d);
        return diameter[0] == Integer.MIN_VALUE ? 0 : diameter[0] - 1;
    }

//    https://leetcode.com/problems/diagonal-traverse-ii/

    public int[] findDiagonalOrder(List<List<Integer>> nums) {
//        int r = nums.size();
//        int c = 0;
//        int total = 0;
//        for (List<Integer> rows : nums) {
//            c = Math.max(c, rows.size());
//            total += rows.size();
//        }
//        int [] result = new int[total];
//        int k = 0;
//        for (int z = 0; z < r; z++) {
//            for (int i = z; i >= 0; i--) {
//                int j = z - i;
//                if (j < nums.get(i).size()) {
//                    result[k] = nums.get(i).get(j);
//                    k++;
//                }
//            }
//        }
//        for (int y = 1; y < c; y++) {
//            int j = y;
//            for (int i = r - 1; i >= 0; i--) {
//                if (j < nums.get(i).size()) {
//                    result[k] = nums.get(i).get(j);
//                    k++;
//                }
//                j++;
//            }
//        }
//        return result;
        Map<Integer,List<Integer>> mapping = new HashMap<>();
        int total = 0;
        int minKey = Integer.MAX_VALUE;
        int maxKey = Integer.MIN_VALUE;
        for (int i = 0; i < nums.size(); i++) {
            for (int j = 0; j < nums.get(i).size(); j++) {
                int key = i + j;
                List<Integer> v = mapping.getOrDefault(key, new ArrayList<>());
                v.add(nums.get(i).get(j));
                mapping.put(i + j, v);
                total++;
                minKey = Math.min(minKey, key);
                maxKey = Math.max(maxKey, key);
            }
        }
        int [] res = new int[total];
        int k = 0;
        for (int i = minKey; i <= maxKey; i++) {
            List<Integer> v = mapping.getOrDefault(i, new ArrayList<>());
            for (int j = v.size() - 1; j >= 0; j--) {
                res[k] = v.get(j);
                k++;
            }
        }
        return res;
    }

}