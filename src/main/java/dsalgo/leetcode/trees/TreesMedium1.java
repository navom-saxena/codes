package dsalgo.leetcode.trees;

import java.util.*;
import dsalgo.leetcode.Models.*;
public class TreesMedium1 {

    public static void main(String[] args) {
        System.out.println(reorganizeString("aab"));
    }

//    https://leetcode.com/problems/path-sum-ii/

    static void pathSumUtil(TreeNode root, int sum, int runningSum,
                                           List<Integer> curr, List<List<Integer>> result) {
        if (root == null) {
            return;
        }
        curr.add(root.val);
        if (root.left == null && root.right == null) {
            if (runningSum == sum + root.val) {
                curr.add(root.val);
                result.add(new ArrayList<>(curr));
                curr.remove(curr.size() - 1);
            }
        }
        pathSumUtil(root.left, sum, runningSum + root.val, curr, result);
        pathSumUtil(root.right, sum, runningSum + root.val, curr, result);
        curr.remove(curr.size() - 1);
    }

    public static List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> curr = new ArrayList<>();
        int runningSum = 0;
        pathSumUtil(root, sum, runningSum, curr, result);
        return result;
    }

//    https://leetcode.com/problems/reorganize-string/

    public static String reorganizeString(String s) {
        Map<Character,Integer> counts = new HashMap<>();
        for (char c : s.toCharArray()) {
            counts.merge(c, 1, Integer::sum);
        }
        PriorityQueue<Character> maxHeap = new PriorityQueue<>((x,y) -> counts.get(y) - counts.get(x));
        maxHeap.addAll(counts.keySet());
        StringBuilder sb = new StringBuilder();
        while (maxHeap.size() > 1) {
            char current = maxHeap.remove();
            char next = maxHeap.remove();
            sb.append(current);
            sb.append(next);
            counts.put(current, counts.get(current) - 1);
            counts.put(next, counts.get(next) - 1);
            if (counts.get(current) > 0) {
                maxHeap.add(current);
            }
            if (counts.get(next) > 0) {
                maxHeap.add(next);
            }
        }
        if (!maxHeap.isEmpty()) {
            char last = maxHeap.remove();
            if (counts.get(last) > 1) {
                return "";
            }
            sb.append(last);
        }
        return sb.toString();
    }

    public static class Node {
        int freq;
        char data;

        Node(int freq, char data) {
            this.freq = freq;
            this.data = data;
        }
    }

}
