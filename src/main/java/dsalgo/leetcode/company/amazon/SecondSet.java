package dsalgo.leetcode.company.amazon;

import java.util.*;

public class SecondSet {

    public static void main(String[] args) {
        System.out.println(criticalConnections(4,
                Arrays.asList(Arrays.asList(0,1), Arrays.asList(1,2), Arrays.asList(2,0), Arrays.asList(1,3))));
    }

//    https://leetcode.com/problems/critical-connections-in-a-network/

    public static boolean bfsCriticalConnections(int n, Map<Integer,Set<Integer>> adj) {
        Queue<Integer> queue = new LinkedList<>();
        boolean [] visited = new boolean[n];
        queue.add(new ArrayList<>(adj.keySet()).get(0));
        while (!queue.isEmpty()) {
            Integer curr = queue.remove();
            if (!visited[curr]) {
                queue.addAll(adj.get(curr));
            }
            visited[curr] = true;
        }
        for (boolean isVisited : visited) {
            if (!isVisited) {
                return false;
            }
        }
        return true;
    }

    public static void targenAlgorithm(Map<Integer,Set<Integer>> adj, List<List<Integer>> bridges, int [] disc,
                                       int [] low, int [] parent, int node) {
        if (!adj.containsKey(node)) {
            return;
        }
        if (disc[node] != -1) {
            return;
        }
        disc[node] = time;
        low[node] = time;
        time++;
        for (Integer neighbour : adj.get(node)) {
            if (disc[neighbour] == -1) {
                parent[neighbour] = node;
                targenAlgorithm(adj,bridges, disc, low, parent, neighbour);
                low[node] = Math.min(low[node],low[neighbour]);
                if (disc[node] < low[neighbour]) {
                    List<Integer> arr = new ArrayList<>();
                    arr.add(node);
                    arr.add(neighbour);
                    bridges.add(arr);
                }
            } else if (parent[node] != neighbour) {
                low[node] = Math.min(low[node],disc[neighbour]);
            }
        }
    }

    static int time = 0;

    public static List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
        List<List<Integer>> bridges = new ArrayList<>();
        Map<Integer,Set<Integer>> adj = new HashMap<>();
        for (List<Integer> connection : connections) {
            int a = connection.get(0);
            int b = connection.get(1);
            Set<Integer> setA;
            Set<Integer> setB;
            if (adj.containsKey(a)) {
                setA = adj.get(a);
            } else {
                setA = new HashSet<>();
            }
            setA.add(b);
            if (adj.containsKey(b)) {
                setB = adj.get(b);
            } else {
                setB = new HashSet<>();
            }
            setB.add(a);
            adj.put(a, setA);
            adj.put(b, setB);
        }
        int [] disc = new int[n];
        int [] low = new int[n];
        int [] parent = new int[n];
        for (int i = 0; i < n; i++) {
            disc[i] = -1;
            low[i] = -1;
            parent[i] = -1;
        }
        targenAlgorithm(adj, bridges, disc, low, parent, 0);
        return bridges;
    }

}
