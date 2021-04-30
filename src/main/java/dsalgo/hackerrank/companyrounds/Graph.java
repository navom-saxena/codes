package dsalgo.hackerrank.companyrounds;


import java.util.*;

public class Graph {

    public static void main(String[] args) {
        System.out.println(unitConversion("10 feet mile"));
    }

//   * Unit Conversion
//   * We would like to convert between arbitrary sets of units (of a given type i.e length)
//   * Input will be provided in the form of:
//     * meter feet 3.28084
//     * feet inch 12
//     * mile meter 1609.34
//     * meter yard 1.09361
//     * mile furlong 7.99998
//     * etc
//   * For example, 1 mile in meters -> 1609.34
//     * 5 mile -> feet = ?
//     * 10 feet -> mile = ?
//     input = list of conversion factors and string as - 5 mile feet

    static class ConversionUnit {
        String unit;
        double value;

        ConversionUnit(String unit, double value) {
            this.unit = unit;
            this.value = value;
        }
    }

    public static double bfs(Map<String,Set<String>> adj, Map<String,Double> weights,
                             String from, String to, double multiplier) {
        Set<String> visited = new HashSet<>();
        Queue<ConversionUnit> queue = new LinkedList<>();
        queue.add(new ConversionUnit(from, multiplier));
        while (!queue.isEmpty()) {
            ConversionUnit node = queue.remove();
            if (!visited.contains(node.unit)) {
                if (node.unit.equals(to)) {
                    return node.value;
                }
                visited.add(node.unit);
                for (String neighbour : adj.getOrDefault(node.unit, new HashSet<>())) {
                    queue.add(new ConversionUnit(neighbour, node.value * weights.get(node.unit + "-" + neighbour)));
                }
            }
        }
        return -1;
    }

    public static double execute(List<String> inputConversions, String input) {
        Map<String,Set<String>> adj = new HashMap<>();
        Map<String,Double> weights = new HashMap<>();
        for (String inputConversion : inputConversions) {
            String [] inputArr = inputConversion.split(" ");
            String from = inputArr[0];
            String to = inputArr[1];
            double weight = Double.parseDouble(inputArr[2]);
            Set<String> toNeighbour = adj.getOrDefault(to, new HashSet<>());
            toNeighbour.add(from);
            adj.put(to, toNeighbour);
            Set<String> fromNeighbour = adj.getOrDefault(from, new HashSet<>());
            fromNeighbour.add(to);
            adj.put(from, fromNeighbour);
            weights.put(from + "-" + to, weight);
            weights.put(to + "-" + from, 1.0 / weight);
        }
        String [] inputAr = input.split(" ");
        return bfs(adj, weights, inputAr[1], inputAr[2], Double.parseDouble(inputAr[0]));
    }

    public static double unitConversion(String input) {
        List<String> inputConversions = Arrays.asList("meter feet 3.28084","feet inch 12", "mile meter 1609.34",
                "meter yard 1.09361", "mile furlong 7.99998");
        return execute(inputConversions, input);
    }

/*
In a dail pad..u can jump like a horse, for n hops, give no of diff combinations that can be created
  1 2 3
  4 5 6
  7 8 9
    0
From 6..u can jump to 1 and 7, 0..for n = 2 hops. Return Total digits = 6
  6 1 8
  6 1 6
  6 7 6
  6 7 2
  6 0 4
  6 0 6
*/

    static int count = 0;

    public static void calculateCombinationsDFS(Map<Integer,Set<Integer>> adj, int start, int hops) {
        if (hops == 0) {
            count++;
            return;
        }
        for (int landingNode : adj.getOrDefault(start, new HashSet<>())) {
            calculateCombinationsDFS(adj, landingNode, hops - 1);
        }
    }

    public static int calculateCombinations(Map<Integer,Set<Integer>> adj, int start, int hops) {
        int [][] hopsCount = new int[10][2];
        for (int [] values : hopsCount) {
            values[0] = 1;
        }
        int initialHops = 1;
        while (initialHops <= hops) {
            for (int i = 0; i <= 9; i++) {
                hopsCount[i][1] = 0;
                for (int landingNode : adj.getOrDefault(i, new HashSet<>())) {
                    hopsCount[i][1] += hopsCount[landingNode][0];
                }
            }
            for (int i = 0; i <= 9; i++) {
                hopsCount[i][1] = hopsCount[i][0];
            }
            initialHops++;
        }
        return hopsCount[start][0];
    }

    public static int noCombinations(int start, int hops) {
        Map<Integer,Set<Integer>> adj = new HashMap<>();
        Set<Integer> one = new HashSet<>();
        one.add(6);
        one.add(8);
        adj.put(1, one);
        Set<Integer> two = new HashSet<>();
        two.add(7);
        two.add(9);
        adj.put(2, two);
        Set<Integer> three = new HashSet<>();
        three.add(4);
        three.add(8);
        adj.put(3, three);
        Set<Integer> four = new HashSet<>();
        four.add(3);
        four.add(9);
        four.add(0);
        adj.put(4, four);
        Set<Integer> five = new HashSet<>();
        adj.put(5, five);
        Set<Integer> six = new HashSet<>();
        one.add(1);
        one.add(7);
        one.add(0);
        adj.put(6, six);
        Set<Integer> seven = new HashSet<>();
        one.add(2);
        one.add(6);
        adj.put(7, seven);
        Set<Integer> eight = new HashSet<>();
        one.add(6);
        one.add(8);
        adj.put(8, eight);
        Set<Integer> nine = new HashSet<>();
        one.add(2);
        one.add(7);
        adj.put(9, nine);
        Set<Integer> zero = new HashSet<>();
        one.add(4);
        one.add(6);
        adj.put(0, zero);

//        calculateCombinationsDFS(adj, start, hops);
//        return count;
        return calculateCombinations(adj, start, hops);
    }

}
