package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;

import java.util.*;

public class EleventhSet {

    public static void main(String[] args) {
        List<List<String>> accounts = new ArrayList<>();
        List<String> s = new ArrayList<>();
        Collections.sort(s);
    }

//    https://leetcode.com/problems/jewels-and-stones/

    public int numJewelsInStones(String jewels, String stones) {
        Set<Character> jewelsSet = new HashSet<>();
        for (char j : jewels.toCharArray()) jewelsSet.add(j);

        int count = 0;
        for (char s : stones.toCharArray()) {
            if (jewelsSet.contains(s)) count++;
        }

        return count;
    }

//    https://leetcode.com/problems/binary-tree-inorder-traversal/

    void inorderTraversalUtil(TreeNode node, List<Integer> inorderList) {
        if (node == null) return;
        inorderTraversalUtil(node.left, inorderList);
        inorderList.add(node.val);
        inorderTraversalUtil(node.right, inorderList);
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> inorderList = new ArrayList<>();
        inorderTraversalUtil(root, inorderList);
        return inorderList;
    }

//    https://leetcode.com/problems/binary-tree-postorder-traversal/

    void postorderTraversalUtil(TreeNode node, List<Integer> postorderList) {
        if (node == null) return;
        postorderTraversalUtil(node.left, postorderList);
        postorderTraversalUtil(node.right, postorderList);
        postorderList.add(node.val);
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> postorderList = new ArrayList<>();
        postorderTraversalUtil(root, postorderList);
        return postorderList;
    }

//    https://leetcode.com/problems/implement-queue-using-stacks/

    static class MyQueue {

        Deque<Integer> s1;
        Deque<Integer> s2;

        public MyQueue() {
            s1 = new ArrayDeque<>();
            s2 = new ArrayDeque<>();
        }

        public void push(int x) {
            s1.push(x);
        }

        public int pop() {
            if (!s2.isEmpty()) return s2.pop();
            while (!s1.isEmpty()) s2.push(s1.pop());
            return s2.pop();
        }

        public int peek() {
            if (!s2.isEmpty()) return s2.peek();
            while (!s1.isEmpty()) s2.push(s1.pop());
            return !s2.isEmpty() ? s2.peek() : -1;
        }

        public boolean empty() {
            return s1.isEmpty() && s2.isEmpty();
        }
    }

//    https://leetcode.com/problems/binary-search/

    public int search(int[] nums, int target) {
        int low = 0;
        int high = nums.length - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }

//    https://leetcode.com/problems/same-tree/

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null || q == null) {
            return p == null && q == null;
        }
        if (p.val != q.val) return false;
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

//    https://leetcode.com/problems/remove-duplicates-from-sorted-list/

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return null;

        ListNode prev = head;
        ListNode curr = prev.next;

        while (curr != null) {
            if (curr.val != prev.val) {
                prev.next = curr;
                prev = prev.next;
            }
            curr = curr.next;
        }
        prev.next = null;

        return head;
    }

//    https://leetcode.com/problems/linked-list-cycle/

    public boolean hasCycle(ListNode head) {
        ListNode fastPointer = head;
        ListNode slowPointer = head;

        while (fastPointer != null && fastPointer.next != null && slowPointer != null) {
            slowPointer = slowPointer.next;
            fastPointer = fastPointer.next.next;

            if (slowPointer == fastPointer) return true;
        }
        return false;
    }

//    https://leetcode.com/problems/set-mismatch/

    public int[] findErrorNums(int[] nums) {
//        boolean [] visited = new boolean[nums.length + 1];
//        int repeat = 0;
//        int missing = 0;
//
//        for (int num : nums) {
//            if (visited[num]) repeat = num;
//            visited[num] = true;
//        }
//
//        for (int i = 1; i < visited.length; i++) {
//            if (!visited[i]) {
//                missing = i;
//            }
//        }
//
//        return new int[]{repeat, missing};

//        int n = nums.length;
//        int repeat = 0;
//        for (int i = 0; i < n; i++) {
//            int atIndex = Math.abs(nums[i]) % n;
//            if (nums[atIndex] < 0) {
//                repeat = atIndex;
//            }
//            nums[atIndex] = nums[atIndex] * -1;
//        }
//        int missing = 0;
//        for (int i = 0; i < n; i++) {
//            if (nums[i] > 0 && i != repeat) missing = i;
//        }
//        repeat = repeat == 0 ? n : repeat;
//        missing = missing == 0 ? n : missing;
//        return new int[]{repeat, missing};

        int n = nums.length;
        int xy = 0;
        for (int i = 1; i <= n; i++) {
            xy ^= i;
        }
        for (int num : nums) xy ^= num;
        xy &= -xy;
        int a = 0;
        int b = 0;
        for (int num : nums) {
            if ((num & xy) == 0) a ^= num;
            else b ^= num;
        }
        for (int i = 1; i <= n; i++) {
            if ((i & xy) == 0) a ^= i;
            else b ^= i;
        }
        for (int num : nums) {
            if (num == a) {
                return new int[]{a,b};
            }
        }
        return new int[]{b, a};
    }

//    https://leetcode.com/problems/excel-sheet-column-title/

    public String convertToTitle(int columnNumber) {
        StringBuilder sb = new StringBuilder();
        while (columnNumber != 0) {
            int mod = (columnNumber % 26);
            if (mod == 0) mod = 26;
            sb.append((char) ('A' + mod - 1));
            columnNumber -= mod;
            columnNumber /= 26;
        }
        return sb.reverse().toString();
    }

//    https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/

    public int findLeastNumOfUniqueInts(int[] arr, int k) {
        Map<Integer,Integer> freqMap = new HashMap<>();
        for (int num : arr) freqMap.merge(num, 1, Integer::sum);

        PriorityQueue<Integer> minHeap = new PriorityQueue<>(Comparator.comparingInt(freqMap::get));
        minHeap.addAll(freqMap.keySet());

        while (!minHeap.isEmpty() && k > 0) {
            int leastFNo = minHeap.remove();
            int f = freqMap.getOrDefault(leastFNo, 0);
            if (f == 1) freqMap.remove(leastFNo);
            else {
                freqMap.put(leastFNo, f - 1);
                minHeap.add(leastFNo);
            }
            k--;
        }

        return freqMap.size();
    }

//    https://leetcode.com/problems/time-based-key-value-store/

    static class TimeMap {

        Map<String,TreeMap<Integer,String>> timeMap;

        public TimeMap() {
            timeMap = new HashMap<>();
        }

        public void set(String key, String value, int timestamp) {
            TreeMap<Integer,String> treeMap = timeMap.getOrDefault(key, new TreeMap<>());
            treeMap.put(timestamp, value);
            timeMap.put(key, treeMap);
        }

        public String get(String key, int timestamp) {
            TreeMap<Integer,String> treeMap = timeMap.getOrDefault(key, new TreeMap<>());
            Integer ts = treeMap.floorKey(timestamp);
            if (ts == null) return "";
            return treeMap.get(ts);
        }
    }

//    https://leetcode.com/problems/delete-and-earn/

    public int deleteAndEarn(int[] nums) {
        int [] sums = new int[10005];
        for (int num : nums) {
            sums[num] += num;
        }
        int [] dp = new int[10005];
        dp[1] = sums[1];
        dp[2] = Math.max(sums[1],sums[2]);

        for (int i = 3; i < dp.length; i++) {
            dp[i] = Math.max(sums[i] + dp[i - 2], dp[i - 1]);
        }

        return Math.max(dp[dp.length - 1], dp[dp.length - 2]);
    }

//    https://leetcode.com/problems/house-robber-iii/

    Map<TreeNode,Integer> dpRobber = new HashMap<>();

    public int rob(TreeNode root) {
       if (root == null) return 0;
       if (dpRobber.containsKey(root)) return dpRobber.get(root);

       int gChildSum = 0;
       if (root.left != null) gChildSum += rob(root.left.left) + rob(root.left.right);
       if (root.right != null) gChildSum += rob(root.right.left) + rob(root.right.right);

       int childSum = 0;
       childSum += rob(root.left) + rob(root.right);

       int maxSum = Math.max(root.val + gChildSum, childSum);

       dpRobber.put(root, maxSum);
       return dpRobber.get(root);
    }

//    https://leetcode.com/problems/find-and-replace-in-string/

    public String findReplaceString(String s, int[] indices, String[] sources, String[] targets) {
        Map<Integer,Integer> reverseIndices = new HashMap<>();
        for (int i = 0; i < indices.length; i++) {
            reverseIndices.put(indices[i],i);
        }
        StringBuilder sb = new StringBuilder();
        int i = 0;
        while (i < s.length()) {
            if (reverseIndices.containsKey(i)) {
                int j = reverseIndices.get(i);
                if (sources[j].equals(s.substring(i, i + sources[j].length()))) {
                    sb.append(targets[j]);
                    i = i + sources[j].length();
                } else {
                    sb.append(s.charAt(i));
                    i++;
                }
            } else {
                sb.append(s.charAt(i));
                i++;
            }

        }
        return sb.toString();
    }

//    https://leetcode.com/problems/permutations-ii/

    void permuteUniqueUtils(int n, Map<Integer,Integer> freq, List<Integer> running,
                            List<List<Integer>> res) {
        if (running.size() == n) {
            res.add(new ArrayList<>(running));
            return;
        }
        for (int no : freq.keySet()) {

                int f = freq.getOrDefault(no, 0);
                if (f == 0) continue;
                freq.put(no, f - 1);

                running.add(no);
                permuteUniqueUtils(n, freq, running, res);

                freq.put(no, freq.getOrDefault(no, 0) + 1);
                running.remove(running.size() - 1);
        }
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Map<Integer,Integer> freq = new HashMap<>();
        for (int num : nums) freq.put(num, freq.getOrDefault(num, 0) + 1);
        permuteUniqueUtils(nums.length, freq, new ArrayList<>(), res);
        return res;
    }

//    https://leetcode.com/problems/combination-sum-iv/

    int combinationSum4Util(int [] nums, int target, int [] dp) {
        if (target == 0) return 1;

        if (dp[target] != -1) return dp[target];
        int res = 0;

        for (int num : nums) {
            if (num <= target) {
                res += combinationSum4Util(nums, target - num, dp);
            }
        }
        dp[target] = res;
        return res;
    }

    public int combinationSum4(int[] nums, int target) {
        int [] dp = new int[target + 1];
        Arrays.fill(dp, -1);
        dp[0] = 0;
       return combinationSum4Util(nums, target, dp);
    }

//    https://leetcode.com/problems/01-matrix/

    public int[][] updateMatrix(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        Deque<int []> deque = new ArrayDeque<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mat[i][j] == 0) deque.add(new int[]{i,j});
            }
        }

        int d = 0;
        int [][] directions = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
        boolean [][] visited = new boolean[m][n];

        while (!deque.isEmpty()) {
            int size = deque.size();
            for (int i = 0; i < size; i++) {
                int [] node = deque.remove();
                int x = node[0];
                int y = node[1];

                if (visited[x][y]) continue;
                visited[x][y] = true;

                if (mat[x][y] == 1) {
                    mat[x][y] = d;
                }

                for (int [] direction : directions) {
                    int newX = x + direction[0];
                    int newY = y + direction[1];

                    if (newX >= 0 && newX < m && newY >= 0 && newY < n && !visited[newX][newY]) {
                        deque.add(new int[]{newX, newY});
                    }
                }
            }
            d++;
        }

        return mat;
    }

//    https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/

    public ListNode deleteDuplicates2(ListNode head) {
        if (head == null) return null;

        ListNode sentinel = new ListNode(Integer.MIN_VALUE);
        ListNode prev = sentinel;
        ListNode curr = head;

        while (curr != null) {
            ListNode next = curr.next;
            if (next != null && curr.val == next.val) {
                while (next != null && curr.val == next.val) next = next.next;
                prev.next = next;
            } else {
                prev.next = curr;
                prev = prev.next;
            }
            curr = next;
        }

        return sentinel.next;
    }

//    https://leetcode.com/problems/evaluate-reverse-polish-notation/

    public int evalRPN(String[] tokens) {
        Deque<Integer> stack = new ArrayDeque<>();
        Set<String> operators = new HashSet<>();
        operators.add("+");
        operators.add("-");
        operators.add("*");
        operators.add("/");

        for (String token : tokens) {
            if (operators.contains(token)) {
                int second = stack.pop();
                int first = stack.pop();
                int res;
                switch (token) {
                    case "+":
                        res = first + second;
                        break;
                    case "-":
                        res = first - second;
                        break;
                    case "*":
                        res = first * second;
                        break;
                    default:
                        res = first / second;
                        break;
                }
                stack.push(res);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }

        if (stack.isEmpty()) return Integer.MIN_VALUE;
        return stack.pop();
    }

//    https://leetcode.com/problems/remove-k-digits/

    public String removeKdigits(String num, int k) {
        Deque<Character> stack = new ArrayDeque<>();
        for (char c : num.toCharArray()) {
            while (k > 0 && !stack.isEmpty() && stack.peek() > c) {
                stack.pop();
                k--;
            }
            if (stack.isEmpty() && c == '0') continue;
            stack.push(c);
        }

        while (k > 0 && !stack.isEmpty()) {
            stack.pop();
            k--;
        }

        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        String s = sb.reverse().toString();
        return s.length() == 0 ? "0" : s;
    }

//    https://leetcode.com/problems/contains-duplicate-iii/

    // use t >= 0 to solve nums[i] - nums[j] <= t and nums[i] - nums[j] >= 0 -->  nums[i] >= nums[j] -->
    // nums[i] is ceil of nums[j];

    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeSet<Long> ts = new TreeSet<>();

        for (int i = 0; i < nums.length; i++) {
            Long ceilingKey = ts.ceiling((long) nums[i]);
            Long floorKey = ts.floor((long) nums[i]);

            if ((ceilingKey != null && ceilingKey - nums[i] <= t) || (floorKey != null && nums[i] - floorKey <= t)) return true;

            ts.add((long) nums[i]);
            if (ts.size() > k) ts.remove((long) nums[i - k]);
        }

        return false;
    }

//    https://leetcode.com/problems/unique-word-abbreviation/

    static class ValidWordAbbr {

        Map<String,Set<String>> revMapping;

        public ValidWordAbbr(String[] dictionary) {
            revMapping = new HashMap<>();

            for (String word : dictionary) {
                String abb = convert(word);

                Set<String> abbWords = revMapping.getOrDefault(abb, new HashSet<>());
                abbWords.add(word);
                revMapping.put(abb, abbWords);
            }
        }

        String convert(String word) {
            String abb;

            if (word.length() <= 2) abb = word;
            else abb = "" + word.charAt(0) + (word.length() - 2) + word.charAt(word.length() - 1);

            return abb;
        }

        public boolean isUnique(String word) {
            String abb = convert(word);

            return (revMapping.get(abb) == null) || (revMapping.containsKey(abb)
                    && revMapping.get(abb).size() == 1 && revMapping.get(abb).contains(word));
        }
    }

//    https://leetcode.com/problems/exam-room/

    static class ExamRoom {
        TreeSet<Integer> ts;
        int n;

        public ExamRoom(int n) {
            this.n = n;
            ts = new TreeSet<>();
        }

        public int seat() {
           int seatNo = 0;

           if (ts.size() > 0) {
               Integer prev = null;
               int maxD = ts.first();

               for (int num : ts) {
                   if (prev != null) {
                       int d = (num - prev) / 2;
                       if (d > maxD) {
                           maxD = d;
                           seatNo = prev + d;
                       }
                   }
                   prev = num;
               }

               if ((n - 1 - ts.last()) > maxD) {
                   seatNo = n - 1;
               }
           }

           ts.add(seatNo);
           return seatNo;
        }

        public void leave(int p) {
            ts.remove(p);
        }
    }

//    https://leetcode.com/problems/maximum-width-ramp/

    public int maxWidthRamp(int[] nums) {
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < nums.length; i++) {
            if (stack.isEmpty() || nums[stack.peek()] > nums[i]) stack.push(i);
        }

        int maxWidth = 0;
        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && nums[stack.peek()] <= nums[i]) {
                maxWidth = Math.max(maxWidth, i - stack.pop());
            }
        }
        return maxWidth;
    }

    //    https://leetcode.com/problems/number-of-operations-to-make-network-connected/

    int find(int [] parent, int a) {
        if (parent[a] == a) return a;
        else return parent[a] = find(parent, parent[a]);
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

    public int makeConnected(int n, int[][] connections) {
        int [] parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }

        int [] rank = new int[n];

        for (int [] connection : connections) union(parent, rank, connection[0], connection[1]);

        Set<Integer> components = new HashSet<>();
        for (int i = 0; i < n; i++) {
            parent[i] = find(parent, i);
            components.add(parent[i]);
        }

        int c = components.size();
        int totalEdges = connections.length;
        int minimumEdgesForC = (n - 1) - (c - 1);
        int redundantEdges = totalEdges - minimumEdgesForC;

        if (totalEdges < n - 1) return -1;
        if (redundantEdges < c - 1) return -1;
        return c - 1;

        // better sol - check if totalEdges < n - 1; return -1, else it means redundant edges are there, return c - 1
    }

//    https://leetcode.com/problems/repeated-string-match/

    public int repeatedStringMatch(String a, String b) {
       int m = a.length();
       int n = b.length();

       int maxRepeat = (n / m) + 2;
       StringBuilder sb = new StringBuilder(a);

       for (int i = 1; i <= maxRepeat; i++) {
           if (sb.toString().contains(b)) return i;
           sb.append(a);
       }

       return -1;
    }

//    https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/submissions/

    public int shortestSubarray(int[] nums, int k) {

        int n = nums.length;
        ArrayDeque<Integer> queue = new ArrayDeque<>();

        long [] prefixArr = new long [n + 1];
        prefixArr[0] = 0;

        for (int i = 1; i <= n; i++) {
            prefixArr[i] = prefixArr[i - 1] + nums[i - 1];
        }

        int minL = Integer.MAX_VALUE;
        for (int r = 0; r <= n; r++) {
            long pr = prefixArr[r];

            while(!queue.isEmpty() && pr - prefixArr[queue.peekFirst()] >= k) {
                minL = Math.min(minL, r - queue.removeFirst());
            }

            while (!queue.isEmpty() && pr <= prefixArr[queue.peekLast()]) queue.removeLast();
            queue.addLast(r);
        }

        return minL == Integer.MAX_VALUE ? -1 : minL;
    }

//    https://leetcode.com/problems/print-immutable-linked-list-in-reverse/submissions/

    interface ImmutableListNode {
        void printValue(); // print the value of this node.
        ImmutableListNode getNext(); // return the next node.
    };

    public void printLinkedListInReverse(ImmutableListNode head) {
        if (head == null) return;

        printLinkedListInReverse(head.getNext());
        head.printValue();
    }

//    https://leetcode.com/problems/defanging-an-ip-address/

    public String defangIPaddr(String address) {

        char [] a = address.toCharArray();
        StringBuilder sb = new StringBuilder();

        for (char c : a) {
            if (c == '.') {
                sb.append("[");
                sb.append(".");
                sb.append("]");
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

//    https://leetcode.com/problems/distribute-coins-in-binary-tree/submissions/

    int ans;

    public int distributeCoinsUtil(TreeNode root) {
        if (root == null) return 0;

        int l = distributeCoinsUtil(root.left);
        int r = distributeCoinsUtil(root.right);

        ans += Math.abs(l) + Math.abs(r);

        return l + r + root.val - 1;
    }

    public int distributeCoins(TreeNode root) {
        ans = 0;
        distributeCoinsUtil(root);
        return ans;
    }

}
