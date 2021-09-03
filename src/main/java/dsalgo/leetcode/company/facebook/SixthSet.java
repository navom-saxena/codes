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
        System.out.println(numIslands2(3,3,new int[][]{{0,1},{1,2},{2,1},{1,0},{0,2},{0,0},{1,1}}));
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

}