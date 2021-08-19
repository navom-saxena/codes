package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;

import java.util.*;

public class FBRec {

    public static void main(String[] args) {
//        System.out.println(matchingPairs("abcd", "cabd"));
//        System.out.println(getBillionUsersDay(new float[]{1.01f,1.02f}));
        System.out.println(canEdit("cat","cats"));
    }

    //    fb_rec

//    rotational cipher

    String rotationalCipher(String input, int rotationFactor) {
        int n = input.length();
        char [] arr = input.toCharArray();

        for (int i = 0; i < n; i++) {
            int c = arr[i];
            if (Character.isAlphabetic(c)) {
                if (Character.isLowerCase(c)) arr[i] = (char) (((c - 'a' + rotationFactor) % 26) + 'a');
                else  arr[i] = (char) (((c - 'A' + rotationFactor) % 26) + 'A');
            } else if (Character.isDigit(c)) {
                arr[i] = (char) (((c - '0' + rotationFactor) % 10) + '0');
            }
        }
        return String.valueOf(arr);
    }

//    reverse to make equal

    boolean areTheyEqual(int[] array_a, int[] array_b) {
        Map<Integer, Integer> a = new HashMap<>();
        Map<Integer, Integer> b = new HashMap<>();
        for (int x1 : array_a) a.merge(x1, 1, Integer::sum);
        for (int x2 : array_b) b.merge(x2, 1, Integer::sum);

        return a.equals(b);
    }

//    Passing Yearbooks

    int processGraph(int [] arr, int x, int destination, Map<Integer,Integer> cMap, int count) {
        if (x == destination) {
            cMap.put(x, count);
            return count;
        }
        int c = processGraph(arr, arr[x - 1], destination, cMap, count + 1);
        cMap.put(x,c);
        return c;
    }

    int[] findSignatureCounts(int[] arr) {
        int [] output = new int[arr.length];
        Map<Integer,Integer> cMap = new HashMap<>();
        for (int i = 0; i <  arr.length; i++) {
            int x = arr[i];
            if (cMap.get(x) == null) {
                int c = processGraph(arr, arr[x - 1], x, cMap, 1);
                output[i] = c;
            } else {
                output[i] = cMap.get(x);
            }
        }
        return output;
    }

//    median stream

    int[] findMedian(int[] arr) {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        int [] output = new int [arr.length];
        for (int i = 0; i <arr.length; i++) {
            if (minHeap.isEmpty() || maxHeap.isEmpty() || arr[i] > minHeap.peek()) {
                minHeap.add(arr[i]);
            } else {
                maxHeap.add(arr[i]);
            }
            int d = Math.abs(minHeap.size() - maxHeap.size());
            while(d > 1) {
                if (minHeap.size() > maxHeap.size()) maxHeap.add(minHeap.remove());
                else minHeap.add(maxHeap.remove());
                d = Math.abs(maxHeap.size() - minHeap.size());
            }
            if (((maxHeap.size() + minHeap.size()) % 2 == 0) && !maxHeap.isEmpty() && !minHeap.isEmpty()) {
                output[i] = (maxHeap.peek() + minHeap.peek()) / 2;
            } else if (maxHeap.size() > minHeap.size()) {
                output[i] = maxHeap.peek();
            } else if (maxHeap.size() < minHeap.size()) {
                output[i] = minHeap.peek();
            }
        }
        return output;
    }

//    matching pairs

    static int matchingPairs(String s, String t) {
        if (s == null || t == null || s.length() != t.length()) return 0;
        Set<Character> unmatchedS = new HashSet<>();
        Set<Character> unmatchedT = new HashSet<>();
        Set<String> unmatchedPair = new HashSet<>();

        char [] sArr = s.toCharArray();
        char [] tArr = t.toCharArray();
        int n = s.length();
        int count = 0;
        boolean pairFound = false;

        for (int i = 0; i < n; i++) {
            if (sArr[i] == tArr[i]) count++;
            else if (!pairFound) {
                if (unmatchedPair.contains(tArr[i] + " " + sArr[i])) {
                    pairFound = true;
                    continue;
                }
                unmatchedPair.add(sArr[i] + " " + tArr[i]);
                unmatchedS.add(sArr[i]);
                unmatchedT.add(tArr[i]);
            }
        }
        if (count == n) return count - 2;
        if (pairFound) return count + 2;
        else {
            for (char unMS : unmatchedS) {
                if (unmatchedT.contains(unMS)) return count + 1;
            }
        }
        return count;
    }

    // slow sums

    int getTotalTime(int[] arr) {
        // Write your code here
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a,b) -> b - a);
        for (int a : arr) {
            maxHeap.add(a);
        }
        int penality = 0;
        int max = maxHeap.remove();
        if (maxHeap.isEmpty()) return max;
        while(!maxHeap.isEmpty()) {
            int no = maxHeap.remove();
            max += no;
            penality += max;
        }
        return penality;
    }

//    Element Swapping

    static class Data {
        int no;
        int index;

        Data(int no, int index) {
            this.no = no;
            this.index = index;
        }
    }

    int[] findMinArray(int[] arr, int k) {
        // Write your code here
        PriorityQueue<Data> minHeap = new PriorityQueue<>((a,b) -> a.no - b.no);
        int n = arr.length;
        for (int i = 0; i < n; i++) {
            minHeap.add(new Data(arr[i],i));
        }
        int i = 0;
        int [] output = new int[n];
        while(k > 0 && !minHeap.isEmpty()) {
            Data smallest = minHeap.remove();
            if (k >= smallest.index && i != smallest.index) {
                arr[smallest.index] = -1;
                output[i] = smallest.no;
                k = k - smallest.index;
                i++;
            }
        }
        for (int value : arr) {
            if (value != -1) {
                output[i] = value;
                i++;
            }
        }
        return output;
    }

//    Seating Arrangements

    int minOverallAwkwardness(int[] arr) {
        int maxDiff = Integer.MIN_VALUE;
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a,b) -> b - a);
        for (int a : arr) {
            maxHeap.add(a);
        }
        int max = maxHeap.remove();
        int fMax = maxHeap.remove();
        int sMax = maxHeap.remove();
        maxDiff = Math.max(maxDiff,Math.abs(max - fMax));
        maxDiff = Math.max(maxDiff,Math.abs(max - sMax));
        while(maxHeap.size() >= 2) {
            int a = maxHeap.remove();
            int b = maxHeap.remove();
            maxDiff = Math.max(maxDiff,Math.abs(fMax - a));
            maxDiff = Math.max(maxDiff,Math.abs(sMax - b));
            fMax = a;
            sMax = b;
        }
        if (maxHeap.size() == 1) {
            int a = maxHeap.remove();
            maxDiff = Math.max(maxDiff,Math.abs(fMax - a));
            maxDiff = Math.max(maxDiff,Math.abs(sMax - a));
        }
        return maxDiff;
    }

//    Number of Visible Nodes

    int visibleNodes(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        int left = visibleNodes(root.left);
        int right = visibleNodes(root.right);
        return Math.max(left,right) + 1;
    }

//    Nodes in a Subtree

    static class NArrNode {
        public int val;
        public List<NArrNode> children;

        public NArrNode() {
            val = 0;
            children = new ArrayList<NArrNode>();
        }

        public NArrNode(int _val) {
            val = _val;
            children = new ArrayList<NArrNode>();
        }

        public NArrNode(int _val, ArrayList<NArrNode> _children) {
            val = _val;
            children = _children;
        }
    }

    static class Query {
        int u;
        char c;
        Query(int u, char c) {
            this.u = u;
            this.c = c;
        }
    }

    // Add any helper functions you may need here

    Map<Character,Integer> processSubTree(NArrNode node, Map<Integer,Map<Character,Integer>> mainMap, char [] sArr) {
        Map<Character,Integer> fMap = new HashMap<>();
        if (node == null) return fMap;
        if (node.children.size() == 0) {
            fMap.put(sArr[node.val - 1], 1);
            return fMap;
        }
        fMap.put(sArr[node.val - 1], 1);
        for (NArrNode child : node.children) {
            Map<Character,Integer> cFMap = processSubTree(child,mainMap,sArr);
            cFMap.forEach((k,v) -> fMap.merge(k,v, Integer::sum));
        }
        mainMap.put(node.val,fMap);
        return fMap;
    }

    int[] countOfNodes(NArrNode root, ArrayList<Query> queries, String s) {
        Map<Integer,Map<Character,Integer>> mainMap = new HashMap<>();
        processSubTree(root, mainMap, s.toCharArray());
        int [] output = new int[queries.size()];
        int i = 0;
        for (Query query : queries) {
            output[i] = mainMap.get(query.u).get(query.c);
            i++;
        }
        return output;
    }

//    Revenue Milestones

    int binarySearchCiel(int [] p, int low, int high, int x) {
        if (low > high || x > p[high]) return  -1;
        if (x <= p[low]) return low;
        int mid = low + (high - low) / 2;
        if (p[mid] == x) return mid;
        else if (p[mid] < x) {
            if (mid < high && p[mid + 1] >= x) return mid + 1;
            else return binarySearchCiel(p, mid + 1, high, x);
        } else {
            if (low < mid && p[mid - 1] < x) return mid;
            else return binarySearchCiel(p, low, mid - 1, x);
        }
    }


    int[] getMilestoneDays(int[] revenues, int[] milestones) {
        int n = revenues.length;
        int m = milestones.length;
        int [] output = new int[m];
        int [] prefixArr = new int [n];
        int sum = 0;

        for (int i = 0; i < n; i++) {
            sum += revenues[i];
            prefixArr[i] = sum;
        }

        for (int i = 0; i < m; i++) {
            int index = binarySearchCiel(prefixArr, 0, n - 1, milestones[i]);
            output[i] = index + 1;
        }

        return output;
    }

//    billion users

    int getBillionUsersDay(float[] growthRates) {
        double logSum = 0;
        int t = 1;
        double bill = Math.pow(10, 9);
        while (logSum < bill) {
            t *= 2;
            logSum = 0;
            for (double g : growthRates) {
                logSum += Math.pow(g, t);
            }
        }
        while (logSum >= bill) {
            t--;
            logSum = 0;
            for (double g : growthRates) {
                logSum += Math.pow(g, t);
            }
        }
        return t + 1;
    }

//    Encrypted Words

    void process(char [] arr, int low, int high, StringBuilder sb) {
        if (low > high) return;
        int mid = low + (high - low) / 2;
        sb.append(arr[mid]);
        process(arr, low, mid - 1, sb);
        process(arr,mid + 1, high, sb);
    }


    String findEncryptedWord(String s) {
        StringBuilder sb = new StringBuilder();
        process(s.toCharArray(),0, s.length() - 1, sb);
        return sb.toString();
    }

//    Change in a Foreign Currency

    boolean canGetExactChangeUtil(int targetMoney, int[] denominations, Set<Integer> processed) {
        if (targetMoney < 0) return false;
        if (targetMoney == 0) return true;
        if (processed.contains(targetMoney)) return false;
        for (int denomination : denominations) {
            if (canGetExactChangeUtil(targetMoney - denomination, denominations, processed)) {
                return true;
            }
        }
        processed.add(targetMoney);
        return false;
    }

    boolean canGetExactChange(int targetMoney, int[] denominations) {
        Set<Integer> processed = new HashSet<>();
        return canGetExactChangeUtil(targetMoney,denominations, processed);
    }

//    Balanced Split

    boolean balancedSplitExists(int[] arr) {
        // Write your code here
        Arrays.sort(arr);
        int n = arr.length;
        int [] prefixArr = new int[n];
        int [] suffixArr = new int [n];
        int prefixSum = 0;
        int suffixSum = 0;
        for (int i = 0; i < n; i++) {
            prefixSum += arr[i];
            prefixArr[i] = prefixSum;
        }
        for (int i = n - 1; i >= 0; i--) {
            suffixSum += arr[i];
            suffixArr[i] = suffixSum;
        }

        for (int i = 0; i < n - 1; i++) {
            if (arr[i] == arr[i + 1]) continue;
            if (prefixArr[i] == suffixArr[i + 1]) return true;
        }
        return false;
    }

//    counting triangles

    static class Sides {
        int a;
        int b;
        int c;
        Sides(int a,int b,int c){
            this.a = a;
            this.b = b;
            this.c = c;
        }
    }

    int countDistinctTriangles(ArrayList<Sides> arr) {
        // Write your code here
        Map<String,Integer> freqMap = new HashMap<>();
        for (Sides s : arr) {
            int [] side = new int [3];
            side[0] = s.a;
            side[1] = s.b;
            side[2] = s.c;
            Arrays.sort(side);
            StringBuilder sb = new StringBuilder();
            for (int s1 : side) sb.append(s1);
            freqMap.merge(sb.toString(), 1, Integer::sum);
        }
        return freqMap.size();
    }

//    Queue Removals

    int[] findPositions(int[] arr, int x) {
        Deque<Integer> queue = new ArrayDeque<>();
        int [] output = new int[x];
        int o = 0;
        for (int i = 0; i < arr.length; i++) {
            queue.add(i);
        }
        for (int i = 0; i < x; i++) {
            int maxIndex = -1;
            int max = Integer.MIN_VALUE;
            Deque<Integer> temp = new ArrayDeque<>();
            int j = 0;
            while(j < x && !queue.isEmpty()) {
                int e = queue.remove();
                if (arr[e] > max) {
                    max = arr[e];
                    maxIndex = e;
                }
                temp.add(e);
                j++;
            }
            output[o] = maxIndex + 1;
            o++;
            while (!temp.isEmpty()) {
                int e = temp.remove();
                if (e != maxIndex) {
                    arr[e] = arr[e] > 0 ? arr[e] - 1 : arr[e];
                    queue.add(e);
                }
            }
        }
        return output;
    }

//    reverse operations

//    int[] arr_1 = {1, 2, 8, 9, 12, 16};
//    int[] expected1 = {1, 8, 2, 9, 16, 12};
//    int[] arr_2 = {2, 18, 24, 3, 5, 7, 9, 6, 12};
//    int[] expected2 = {24, 18, 2, 3, 5, 7, 9, 12, 6};

    static class Tup {
        Node r;
        Node rem;
    }

    Tup reverseUtil(Node node) {
        if (node == null || node.next == null || node.next.val % 2 != 0) {
            Tup t = new Tup();
            if (node == null) return t;
            t.rem = node.next;
            t.r = node;
            node.next = null;
            return t;
        }
        Tup t = reverseUtil(node.next);
        Node r1 = node.next;
        node.next = null;
        r1.next = node;
        return t;
    }

    Node reverse(Node head) {
        if (head == null) return null;
        Node curr = head;
        if (curr.val % 2 == 0) {
            Node nextN = head;
            Tup t = reverseUtil(curr);
            head = t.r;
            nextN.next = t.rem;
            curr = t.rem;
        }
        while(curr != null) {
            if (curr.next != null && curr.next.val % 2 == 0) {
                Node nextN = curr.next;
                Tup t = reverseUtil(curr.next);
                curr.next = t.r;
                nextN.next = t.rem;
                curr = t.rem;
            } else {
                curr = curr.next;
            }
        }
        return head;
    }

//    Minimizing Permutations

    void reverse(int [] arr, Map<Integer,Integer> map, int low, int high) {
        int i = low;
        int j = high;
        while(i < j) {
            swap(arr, i,j);
            map.put(arr[i],i);
            map.put(arr[j],j);
            i++;
            j--;

        }
    }

    void swap(int [] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    int minOperations(int[] arr) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            map.put(arr[i],i);
            minHeap.add(arr[i]);
        }
        int index = 0;
        int swap = 0;
        for (int i = 0; i < arr.length; i++) {
            int n = minHeap.remove();
            if (n < arr[index]) {
                reverse(arr, map, Math.min(index, map.get(n)), Math.max(index, map.get(n)));
                swap++;
            }
            index++;
        }
        return swap;
    }

    int minOperationsGraph(int [] arr) {
        int len = arr.length;
        StringBuilder sbS = new StringBuilder();
        for (int a : arr) {
            sbS.append(a);
        }
        String initial = sbS.toString();
        StringBuilder sbD = new StringBuilder();
        Arrays.sort(arr);
        for (int a : arr) {
            sbD.append(a);
        }
        String result = sbD.toString();
        Deque<String> queue = new ArrayDeque<>();
        int d = 0;
        Set<String> visited = new HashSet<>();
        queue.add(initial);
        while (!queue.isEmpty()) {
            int n = queue.size();
            for (int i = 0; i < n; i++) {
                String node = queue.remove();
                if (node.equals(result)) return d;
                if (visited.contains(node)) continue;
                visited.add(node);
                StringBuilder rev = new StringBuilder();
                for (int j = 0; j < len; j++) {
                    for (int k = len - 1; k >= j; k--) {
                        rev.append(node.charAt(k));
                    }
                    String reversed = node.substring(0,j) + rev;
                    queue.add(reversed);
                }
            }
            d++;
        }
        return d;
    }

//    Above-Average Subarrays

    List<int []> aboveAverageSubArr(int [] arr) {
        List<int []> res = new ArrayList<>();
        int n = arr.length;
        int totalSum = 0;
        for (int no : arr) totalSum += no;
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = i; j >= 0; j--) {
                sum += arr[j];
                int len = i - j + 1;
                if ((sum / len) > (totalSum - sum)/ n - len) {
                    res.add(new int[]{j + 1,i + 1});
                }
            }
        }
        return res;
    }

//    https://www.geeksforgeeks.org/look-and-say-sequence/

    public List<String> lookAndSay(int n) {
        List<String> res = new ArrayList<>();
        res.add("1");
        for (int i = 1; i <= n; i++) {
            Map<Integer,Integer> freq = new HashMap<>();
            char [] arr = res.get(i - 1).toCharArray();
            for (char c : arr) {
                freq.merge(c - '0',1,Integer::sum);
            }
            StringBuilder sb = new StringBuilder();
            for (char c : arr) {
                sb.append(freq.get(c - '0'));
                sb.append(c - '0');
            }
            res.add(sb.toString());
        }
        return res;
    }

//    Edit Distance

    public static boolean canEdit(String s1, String s2) {
        if (Math.abs(s1.length() - s2.length()) > 1) return false;
        char [] arr1 = s1.toCharArray();
        char [] arr2 = s2.toCharArray();
        int lengthFlag = 0;
        if (arr1.length > arr2.length) lengthFlag = 1;
        else if (arr1.length < arr2.length) lengthFlag = 2;
        int i = 0;
        int j = 0;
        int n = arr1.length;
        int m = arr2.length;
        boolean foundOnce = false;
        while (i < n && j < m) {
            if (arr1[i] == arr2[j]) {
                i++;
                j++;
            } else {
                if (foundOnce) return false;
                foundOnce = true;
                switch (lengthFlag) {
                    case 0: {
                        i++;
                        j++;
                    }
                    break;
                    case 1: {
                        i++;
                    }
                    break;
                    case 2: {
                        j++;
                    }
                    break;
                }
            }
        }
        return foundOnce || n != m;
    }

//    https://leetcode.com/problems/moving-average-from-data-stream/

    static class MovingAverage {
        int size;
        Node head;
        Node tail;
        double sum;
        int n;

        public MovingAverage(int size) {
            this.size = size;
            this.head = null;
            this.tail = null;
            this.sum = 0;
            this.n = 0;
        }

        public double next(int val) {
            sum += val;
            Node node = new Node(val);
            n++;
            if (head == null && tail == null) {
                head = node;
                tail = node;
            } else {
                tail.next = node;
                tail = tail.next;
            }
            if (n > size && head != null) {
               sum -= head.val;
               head = head.next;
           }
            return sum / (Math.min(n, size));
        }
    }

}