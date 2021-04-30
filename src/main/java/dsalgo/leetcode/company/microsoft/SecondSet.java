package dsalgo.leetcode.company.microsoft;

import java.util.*;

public class SecondSet {

    public static void main(String[] args) {
        System.out.println(maximalNetworkRank(5, new int[][]{{2,3},{0,3},{0,4},{4,1}}));
//        System.out.println(longestKUniqueSubStr("abcderrertss", 4));
    }

//    https://leetcode.com/problems/maximal-network-rank/

    public static int maximalNetworkRank(int n, int[][] roads) {
        Map<Integer, Set<Integer>> adj = new HashMap<>();
        for (int [] road : roads) {
            int to = road[0];
            int from = road[1];
            Set<Integer> fromList = adj.getOrDefault(from,new HashSet<>());
            Set<Integer> toList = adj.getOrDefault(to, new HashSet<>());
            fromList.add(to);
            toList.add(from);
            adj.put(from, fromList);
            adj.put(to, toList);
        }
        int max = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Set<Integer> firstNodeNeighbours = adj.getOrDefault(i,new HashSet<>());
                Set<Integer> secondNodeNeighbours = adj.getOrDefault(j,new HashSet<>());
                int currMax = firstNodeNeighbours.size() + secondNodeNeighbours.size();
                if (firstNodeNeighbours.contains(j)) currMax--;
                max = Math.max(max,currMax);
            }
        }
        return max;
    }

// abcderrertss
// errertss
    public static String longestKUniqueSubStr(String s, int k) {
        Map<Character,Integer> freq = new HashMap<>();
        int p1 = 0;
        int p2 = 0;
        int n = s.length();
        String curr = "";
        String max = "";
        int maxLength = 0;
        while (p2 < n) {
            if (freq.size() <= k && curr.length() >= maxLength) {
                max = curr;
                maxLength = curr.length();
            }
            if (freq.size() <= k) {
                char c = s.charAt(p2);
                freq.merge(c, 1, Integer::sum);
                curr = curr + c;
                p2++;
            } else {
                char c = s.charAt(p1);
                if (freq.get(c) == 1) {
                    freq.remove(c);
                } else {
                    freq.put(c, freq.get(c) - 1);
                }
                curr = curr.substring(1);
                p1++;
            }
        }
        System.out.println(maxLength);
        return max;
    }

}

