package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;
import java.util.*;

public class NinthSet {

    public static void main(String[] args) {

    }

//    https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/

    public int minTaps(int n, int[] ranges) {
        int taps = 0;
        int min = 0;
        int max = 0;
        while (max < n) {
            for (int i = 0; i < ranges.length; i++) {
                int left = i - ranges[i];
                int right = i + ranges[i];
                if (left <= min && right > max) {
                    max = right;
                }
            }
            if (min == max) return -1;
            min = max;
            taps++;
        }
        return taps;
    }

//    https://leetcode.com/problems/design-circular-deque/

    static class DDLNode {
        int v;
        DDLNode prev;
        DDLNode next;

        DDLNode(int v) {
            this.v = v;
        }

    }

    static class MyCircularDequeLL {

        DDLNode first;
        DDLNode last;
        int k;
        int n;

        /** Initialize your data structure here. Set the size of the deque to be k. */
        public MyCircularDequeLL(int k) {
            first = new DDLNode(Integer.MIN_VALUE);
            last = new DDLNode(Integer.MIN_VALUE);
            first.next = last;
            last.prev = first;
            first.prev = last;
            last.next = first;
            this.k = k;
            this.n = 0;
        }

        /** Adds an item at the front of Deque. Return true if the operation is successful. */
        public boolean insertFront(int value) {
            if (n == k) return false;
            DDLNode node = new DDLNode(value);
            DDLNode temp = first.next;
            first.next = node;
            node.prev = first;
            node.next = temp;
            temp.prev = node;
            n++;
            return true;
        }

        /** Adds an item at the rear of Deque. Return true if the operation is successful. */
        public boolean insertLast(int value) {
            if (n == k) return false;
            DDLNode node = new DDLNode(value);
            DDLNode temp = last.prev;
            last.prev = node;
            node.next = last;
            node.prev = temp;
            temp.next = node;
            n++;
            return true;
        }

        /** Deletes an item from the front of Deque. Return true if the operation is successful. */
        public boolean deleteFront() {
            if (n == 0) return false;
            DDLNode toDelete = first.next;
            toDelete.next.prev = toDelete.prev;
            toDelete.prev.next = toDelete.next;
            n--;
            return true;
        }

        /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
        public boolean deleteLast() {
            if (n == 0) return false;
            DDLNode toDelete = last.prev;
            toDelete.next.prev = toDelete.prev;
            toDelete.prev.next = toDelete.next;
            n--;
            return true;
        }

        /** Get the front item from the deque. */
        public int getFront() {
            if (n == 0) return -1;
            return first.next.v;
        }

        /** Get the last item from the deque. */
        public int getRear() {
            if (n == 0) return -1;
            return last.prev.v;
        }

        /** Checks whether the circular deque is empty or not. */
        public boolean isEmpty() {
            return n == 0;
        }

        /** Checks whether the circular deque is full or not. */
        public boolean isFull() {
            return n == k;
        }
    }

    static class MyCircularDeque {

        int [] deque;
        int start;
        int end;
        int k;
        int n;

        /** Initialize your data structure here. Set the size of the deque to be k. */
        public MyCircularDeque(int k) {
            deque = new int[k];
            start = 0;
            end = k - 1;
            this.k = k;
            this.n = 0;
        }

        /** Adds an item at the front of Deque. Return true if the operation is successful. */
        public boolean insertFront(int value) {
            if (n == k) return false;
            deque[start] = value;
            start = (start + 1) % k;
            n++;
            return true;
        }

        /** Adds an item at the rear of Deque. Return true if the operation is successful. */
        public boolean insertLast(int value) {
            if (n == k) return false;
            deque[end] = value;
            end = (end - 1 + k) % k;
            n++;
            return true;
        }

        /** Deletes an item from the front of Deque. Return true if the operation is successful. */
        public boolean deleteFront() {
            if (n == 0) return false;
            start = (start - 1 + k) % k;
            n--;
            return true;
        }

        /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
        public boolean deleteLast() {
            if (n == 0) return false;
            end = (end + 1) % k;
            n--;
            return true;
        }

        /** Get the front item from the deque. */
        public int getFront() {
            if (n == 0) return -1;
            return deque[(start - 1 + k) % k];
        }

        /** Get the last item from the deque. */
        public int getRear() {
            if (n == 0) return -1;
            return deque[(end + 1) % k];
        }

        /** Checks whether the circular deque is empty or not. */
        public boolean isEmpty() {
            return n == 0;
        }

        /** Checks whether the circular deque is full or not. */
        public boolean isFull() {
            return n == k;
        }
    }

//    https://leetcode.com/problems/optimize-water-distribution-in-a-village/

    public int minCostToSupplyWater(int n, int[] wells, int[][] pipes) {
        Map<Integer,Set<int []>> adj = new HashMap<>();
        for (int [] pipe : pipes) {
            int from = pipe[0];
            int to = pipe[1];
            int cost = pipe[2];

            Set<int []> fromNeighbours = adj.getOrDefault(from, new HashSet<>());
            fromNeighbours.add(new int[]{to,cost});
            adj.put(from, fromNeighbours);

            Set<int []> toNeighbours = adj.getOrDefault(to, new HashSet<>());
            toNeighbours.add(new int[]{from,cost});
            adj.put(to, toNeighbours);
        }

        for (int i = 0; i < wells.length; i++) {
            int from = 0;
            int to = i + 1;
            int cost = wells[i];

            Set<int []> fromNeighbours = adj.getOrDefault(from, new HashSet<>());
            fromNeighbours.add(new int[]{to,cost});
            adj.put(from, fromNeighbours);

            Set<int []> toNeighbours = adj.getOrDefault(to, new HashSet<>());
            toNeighbours.add(new int[]{from,cost});
            adj.put(to, toNeighbours);
        }

        PriorityQueue<int []> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        minHeap.add(new int[]{0,0});

        int totalCost = 0;
        boolean [] visited = new boolean[n + 1];

        while (!minHeap.isEmpty()) {
            int [] node = minHeap.remove();

            int currNode = node[0];
            int cost = node[1];

            if (visited[currNode]) continue;
            visited[currNode] = true;

            totalCost += cost;

            for (int [] neighbour : adj.getOrDefault(currNode, new HashSet<>())) {
                if (!visited[neighbour[0]]) minHeap.add(neighbour);
            }
        }
        return totalCost;
    }

//    https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/

    int [] maxSumBSTUtil(TreeNode node, int [] maxSum) {
        if (node == null) return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE, 0};
        if (node.left == null && node.right == null) {
            maxSum[0] = Math.max(maxSum[0],node.val);
            return new int[] {node.val, node.val, node.val};
        }
        int [] left = maxSumBSTUtil(node.left, maxSum);
        int [] right = maxSumBSTUtil(node.right, maxSum);
        if (left != null && right != null && left[1] < node.val && right[0] > node.val) {
            int sum = left[2] + right[2] + node.val;
            maxSum[0] = Math.max(maxSum[0], sum);
            return new int[]{Math.min(left[0], node.val), Math.max(node.val, right[1]), sum};
        }
        return null;
    }

    public int maxSumBST(TreeNode root) {
        int [] maxSum = new int[]{0};
        maxSumBSTUtil(root, maxSum);
        return maxSum[0];
    }

//    https://leetcode.com/problems/minimum-insertions-to-balance-a-parentheses-string/

    public int minInsertions(String s) {
        char [] sArr = s.toCharArray();
        Deque<Character> stack = new ArrayDeque<>();
        int ans = 0;
        int i = 0;
        while (i < sArr.length) {
            if (sArr[i] == '(') stack.push('(');
            else {
                if (i < sArr.length - 1 && sArr[i + 1] == ')') {
                    if (stack.size() > 0) stack.pop();
                    else ans += 1;
                    i++;
                } else {
                    if (stack.size() > 0) {
                        stack.pop();
                        ans++;
                    } else ans += 2;
                }
            }
            i++;
        }
        return ans + (2 * stack.size());
    }

//    https://leetcode.com/problems/rotated-digits/

    boolean inCategory(int x, int [] arr) {
        for (int no : arr) if (no == x) return true;
        return false;
    }

    public int rotatedDigits(int n) {
        int [] nonMirror = new int[]{3,4,7};
        int [] diffMirror = new int[]{2,5,6,9};
        int count = 0;
        for (int no = 1; no <= n; no++) {
            boolean noInDiff = false;
            int j = no;
            while (j != 0) {
                int x = j % 10;
                if (inCategory(x, nonMirror)) break;
                if (inCategory(x, diffMirror)) noInDiff = true;
                j /= 10;
            }
            if (j == 0 && noInDiff) count++;
        }
        return count;
    }

//    https://leetcode.com/problems/longest-repeating-character-replacement/

    public int characterReplacement(String s, int k) {
        char [] arr = s.toCharArray();
        int [] alpha = new int[26];
        int j = 0;
        int maxFreq = 0;
        int maxLength = 0;

        for (int i = 0; i < arr.length; i++) {
            char c = arr[i];
            alpha[c - 'A']++;
            maxFreq = Math.max(maxFreq, alpha[c - 'A']);

            while (i - j + 1 - maxFreq > k) {
                alpha[arr[j] - 'A']--;
                j++;
            }

            maxLength = Math.max(maxLength, i - j + 1);
        }
        return maxLength;
    }

//    https://leetcode.com/problems/encode-and-decode-strings/

    public static class Codec {

        public String encode(List<String> strs) {

            String codec = Character.toString((char)257);
            StringBuilder sb = new StringBuilder();

            for (String s : strs) {
                sb.append(s);
                sb.append(codec);
            }
            sb.deleteCharAt(sb.length() - 1);
            return sb.toString();
        }

        // Decodes a single string to a list of strings.
        public List<String> decode(String s) {
            String codec = Character.toString((char)257);
            return Arrays.asList(s.split(codec, -1));
        }
    }

//    https://leetcode.com/problems/complete-binary-tree-inserter/

    static class CBTInserter {

        TreeNode root;
        Deque<TreeNode> deque;

        public CBTInserter(TreeNode root) {
            this.root = root;
            if (root != null) {
                deque = new ArrayDeque<>();
                deque.add(root);
                while (!deque.isEmpty() && deque.peekFirst().left != null && deque.peekFirst().right != null) {
                    TreeNode node = deque.remove();
                    deque.add(node.left);
                    deque.add(node.right);
                }
            }
        }

        public int insert(int val) {
            TreeNode node = new TreeNode(val);
            if (root == null || deque.isEmpty()) {
                root = node;
                deque.add(root);
                return -1;
            }
            TreeNode firstToAdd = deque.peekFirst();
            if (firstToAdd.left == null) firstToAdd.left = node;
            else {
                firstToAdd.right = node;
                deque.add(firstToAdd.left);
                deque.add(firstToAdd.right);
                deque.remove();
            }
            return firstToAdd.val;
        }

        public TreeNode get_root() {
            return root;
        }
    }

//    https://leetcode.com/problems/minimum-height-trees/

    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) return Collections.singletonList(0);
        Map<Integer,Set<Integer>> adj = new HashMap<>();
        for (int[] edge : edges) {
            int from = edge[0];
            int to = edge[1];
            Set<Integer> fromNeighbours = adj.getOrDefault(from, new HashSet<>());
            fromNeighbours.add(to);
            adj.put(from, fromNeighbours);
            Set<Integer> toNeighbours = adj.getOrDefault(to, new HashSet<>());
            toNeighbours.add(from);
            adj.put(to, toNeighbours);
        }
        Deque<Integer> deque = new ArrayDeque<>();
        for (int node : adj.keySet()) {
            if (adj.get(node).size() == 1) deque.add(node);
        }
        while (n > 2) {
            int size = deque.size();
            for (int i = 0; i < size; i++) {
                int leaf = deque.remove();
                Set<Integer> neighbours = adj.get(leaf);
                for (int neighbour : neighbours) {
                    adj.get(neighbour).remove(leaf);
                    if (adj.get(neighbour).size() == 1) deque.add(neighbour);
                }
            }
            n -= size;
        }
        return new ArrayList<>(deque);
    }

//    https://leetcode.com/problems/remove-duplicate-letters/

    public String removeDuplicateLetters(String s) {
        char [] arr = s.toCharArray();
        int n = arr.length;
        Deque<Integer> stack = new ArrayDeque<>();
        boolean [] visited = new boolean[26];
        int [] lastIndex = new int[26];
        for (int i = 0; i < n; i++) {
            char c = arr[i];
            lastIndex[c - 'a'] = i;
        }
        for (int i = 0; i < n; i++) {
            int c = arr[i] - 'a';
            if (visited[c]) continue;
            while (!stack.isEmpty() && stack.peek() > c && i < lastIndex[stack.peek()]) {
                visited[stack.pop()] = false;
            }
            stack.push(c);
            visited[c] = true;
        }
        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append((char) (stack.pop() + 'a'));
        }
        return sb.reverse().toString();
    }

//    https://leetcode.com/problems/maximum-frequency-stack/

    static class FreqStack {

        Map<Integer,Integer> freqMap;
        Map<Integer,Deque<Integer>> sameFreqMap;
        int maxFreq;

        public FreqStack() {
            freqMap = new HashMap<>();
            sameFreqMap = new HashMap<>();
            maxFreq = 0;
        }

        public void push(int val) {
           int f = freqMap.getOrDefault(val, 0) + 1;
           freqMap.put(val, f);

           maxFreq = Math.max(maxFreq, f);

           Deque<Integer> sameFreqStack = sameFreqMap.getOrDefault(f, new ArrayDeque<>());
           sameFreqStack.push(val);
           sameFreqMap.put(f, sameFreqStack);
        }

        public int pop() {
            int v = sameFreqMap.get(maxFreq).pop();

            int f = freqMap.getOrDefault(v, 0);
            if (f <= 1) freqMap.remove(v);
            else freqMap.put(v, f - 1);

            if (sameFreqMap.get(maxFreq).size() == 0) maxFreq--;

            return v;
        }
    }

//    https://leetcode.com/problems/pacific-atlantic-water-flow/

    void bfsOcean(int [][] heights, Deque<int []> deque, Set<String> visited, boolean [][] bfsMark,
                  int [][] directions, int m, int n) {
        while (!deque.isEmpty()) {
            int [] node = deque.remove();
            int i = node[0];
            int j = node[1];
            String hash = i + " " + j;
            if (visited.contains(hash)) continue;
            visited.add(hash);
            bfsMark[i][j] = true;
            for (int [] direction : directions) {
                int newI = i + direction[0];
                int newJ = j + direction[1];
                if (newI >= 0 && newI < m && newJ >= 0 && newJ < n
                        && !visited.contains(newI + " " + newJ) && heights[newI][newJ] >= heights[i][j])
                    deque.add(new int[]{newI,newJ});
            }
        }
    }

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        int m = heights.length;
        int n = heights[0].length;

        Deque<int []> deque = new ArrayDeque<>();
        int [][] directions = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
        Set<String> visited = new HashSet<>();

        boolean [][] bfs1 = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            deque.add(new int[]{i,0});
        }
        for (int j = 1; j < n; j++) {
            deque.add(new int[]{0,j});
        }
        bfsOcean(heights, deque, visited, bfs1, directions, m, n);

        visited.clear();
        boolean [][] bfs2 = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            deque.add(new int[]{i,n - 1});
        }
        for (int j = n - 2; j >= 0; j--) {
            deque.add(new int[]{m - 1,j});
        }
        bfsOcean(heights, deque, visited, bfs2, directions, m, n);

        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (bfs1[i][j] && bfs2[i][j]) res.add(Arrays.asList(i,j));
            }
        }
        return res;
    }

//    https://leetcode.com/problems/sort-the-matrix-diagonally/

    void diagonalSortRecursive(int [][] mat, int i, int j, int m, int n, List<Integer> arr) {
        if (i == m || j == n) {
            Collections.sort(arr);
            return;
        }
        arr.add(mat[i][j]);
        diagonalSortRecursive(mat, i + 1, j + 1, m, n, arr);
        mat[i][j] = arr.remove(arr.size() - 1);
    }

    public int[][] diagonalSort(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;

        for (int i = 0; i < m; i++) {
            List<Integer> arr = new ArrayList<>();
            diagonalSortRecursive(mat, i, 0, m, n, arr);
        }
        for (int j = 1; j < n; j++) {
            List<Integer> arr = new ArrayList<>();
            diagonalSortRecursive(mat, 0, j, m, n, arr);
        }
        return mat;
    }

//    https://leetcode.com/problems/path-sum/

    boolean hasPathSumUtil(TreeNode node, int targetSum, int pathSum) {
        if (node ==null) return false;
        pathSum += node.val;
        if (node.left == null && node.right == null) {
            return pathSum == targetSum;
        }
        return hasPathSumUtil(node.left, targetSum, pathSum) || hasPathSumUtil(node.right, targetSum, pathSum);
    }

    public boolean hasPathSum(TreeNode root, int targetSum) {
        return hasPathSumUtil(root, targetSum, 0);
    }

//    https://leetcode.com/problems/convert-bst-to-greater-tree/

    int greaterSum = 0;

    public TreeNode convertBST(TreeNode root) {
        if (root == null) return null;
        convertBST(root.right);
        greaterSum += root.val;
        root.val = greaterSum;
        convertBST(root.left);
        return root;
    }

//    https://leetcode.com/problems/odd-even-linked-list/

    public ListNode oddEvenList(ListNode head) {
        if (head == null) return null;

        ListNode oddSentinel = new ListNode(Integer.MIN_VALUE);
        ListNode evenSentinel = new ListNode(Integer.MIN_VALUE);
        ListNode oddPointer = oddSentinel;
        ListNode evenPointer = evenSentinel;

        int c = 1;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = null;
            if (c % 2 != 0) {
                oddPointer.next = curr;
                oddPointer = oddPointer.next;
            } else {
                evenPointer.next = curr;
                evenPointer = evenPointer.next;
            }
            c++;
            curr = next;
        }
        // if we don't disconnect curr and its next, we have to make evenPointer.next = null as last evenPointer
        // can point to last odd.
        oddPointer.next = evenSentinel.next;
        return oddSentinel.next;
    }

//    https://leetcode.com/problems/sum-of-two-integers/

    public int getSum(int a, int b) {
        int x = Math.abs(a);
        int y = Math.abs(b);
        if (x < y) return getSum(b,a);

        int sign = a > 0 ? 1 : -1;
        if (a * b >= 1) {
            while (y != 0) {
                int sumWoC = x ^ y;
                y = (x & y) << 1;
                x = sumWoC;
            }
        } else {
            while (y != 0) {
                int diffWoB = x ^ y;
                y = ((~x) & y) << 1;
                x = diffWoB;
            }
        }

        return x * sign;
    }

//    https://leetcode.com/problems/next-greater-element-ii/

    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int [] nGE = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = n - 1; i >= 0; i--) stack.push(nums[i]);
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() < nums[i]) stack.pop();
            nGE[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(nums[i]);
        }
        return nGE;
    }

//    https://leetcode.com/problems/valid-perfect-square/

    public boolean isPerfectSquare(int num) {
        if (num < 2) return true;
        long low = 2;
        long high = num / 2;
        while (low <= high) {
            long randomNo = low + ((high - low) / 2);
            long itsSq = randomNo * randomNo;
            if (itsSq == num) return true;
            else if (itsSq > num) high = randomNo - 1;
            else low = randomNo + 1;
        }
        return false;
    }

//    https://leetcode.com/problems/binary-tree-pruning/

    boolean pruneTreeUtil(TreeNode node) {
        if (node == null) return false;
        if (node.left == null && node.right == null) return node.val == 1;
        boolean left = pruneTreeUtil(node.left);
        boolean right = pruneTreeUtil(node.right);
        if (!right) node.right = null;
        if (!left) node.left = null;
        return left || right || node.val == 1;
    }

    public TreeNode pruneTree(TreeNode root) {
        return pruneTreeUtil(root) ? root : null;
    }

//    https://leetcode.com/problems/next-greater-element-i/

    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer,Integer> n1 = new HashMap<>();
        for (int i = 0; i < nums1.length; i++) n1.put(nums1[i],i);

        Deque<Integer> stack = new ArrayDeque<>();
        int [] res = new int[nums1.length];

        for (int i = nums2.length - 1; i >= 0; i--) {
            if (n1.containsKey(nums2[i])) {
                while (!stack.isEmpty() && stack.peek() <= nums2[i]) stack.pop();
                res[n1.get(nums2[i])] = stack.isEmpty() ? -1 : stack.peek();
            }
            stack.push(nums2[i]);
        }

        return res;
    }

//    https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/

    public int maxEvents(int[][] events) {
        Arrays.sort(events, Comparator.comparingInt(a -> a[0]));
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        int days = 0;
        int currDay = events[0][0];
        int i = 0;
        while (i < events.length || !minHeap.isEmpty()) {

            while (i < events.length && events[i][0] <= currDay) {
                minHeap.add(events[i][1]);
                i++;
            }

            while (!minHeap.isEmpty() && minHeap.peek() < currDay) minHeap.remove();

            if (!minHeap.isEmpty()) {
                minHeap.remove();
                days++;
            }
            currDay++;
        }
        return days;
    }

//    https://leetcode.com/problems/peak-index-in-a-mountain-array/

    public int peakIndexInMountainArray(int[] arr) {
        int low = 0;
        int high = arr.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] > arr[mid + 1] && arr[mid - 1] < arr[mid]) return mid;
            else if (arr[mid] < arr[mid + 1]) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }

//    https://leetcode.com/problems/sort-characters-by-frequency/

    public String frequencySort(String s) {
        Map<Character,Integer> freqMap = new HashMap<>();
        for (char c : s.toCharArray()) {
            freqMap.put(c, freqMap.getOrDefault(c,0) + 1);
        }
//        PriorityQueue<Character> maxHeap = new PriorityQueue<>((a,b) -> freqMap.get(b) - freqMap.get(a));
//        maxHeap.addAll(freqMap.keySet());
//        StringBuilder sb = new StringBuilder();
//        while (!maxHeap.isEmpty()) {
//            char mostFreqC = maxHeap.remove();
//            int freq = freqMap.get(mostFreqC);
//            for (int i = 0; i < freq; i++) {
//                sb.append(mostFreqC);
//            }
//        }
//        return sb.toString();
        List<StringBuilder> bucket = new ArrayList<>();
        for (int i = 0; i <= s.length(); i++) bucket.add(new StringBuilder());
        for (char k : freqMap.keySet()) {
            int f = freqMap.get(k);
            StringBuilder sb = bucket.get(f);
            for (int i = 0; i < f; i++) sb.append(k);
        }
        StringBuilder sb = new StringBuilder();
        for (int i = bucket.size() - 1; i >= 0; i--) {
            StringBuilder sbI = bucket.get(i);
            if (sbI.length() != 0) sb.append(sbI);
        }
        return sb.toString();
    }

//    https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/

    // to remember - special case - when nums[mid] == nums[high] (0,1,3,3,3), we always have to go left, but in scenarios
    // like (3,3,1,3), we should go right ? no because in such scenarios, nums[low] == nums[mid] == nums[high], and we
    // decrement both counters to decrease arr size.

    public int findMin(int[] nums) {
        int low = 0;
        int high = nums.length - 1;
        while(low < high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] < nums[high]) high = mid;
            else if (nums[mid] > nums[high]) low = mid + 1;
            else if (nums[mid] == nums[high]) high--;
        }
        return nums[low];
    }

//    https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/

    TreeNode sortedArrayToBSTUtil(int [] nums, int low, int high) {
        if (low > high) return null;
        else if (low == high) return new TreeNode(nums[low]);
        int mid = low + (high - low) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = sortedArrayToBSTUtil(nums, low, mid - 1);
        node.right = sortedArrayToBSTUtil(nums, mid + 1, high);
        return node;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBSTUtil(nums, 0, nums.length - 1);
    }

//    https://leetcode.com/problems/merge-two-binary-trees/

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) return null;
        else if (root1 == null || root2 == null) {
            return root1 == null ? root2 : root1;
        }
        TreeNode node = new TreeNode(root1.val + root2.val);
        node.left = mergeTrees(root1.left, root2.left);
        node.right = mergeTrees(root1.right, root2.right);
        return node;
    }

//    https://leetcode.com/problems/palindrome-partitioning/

    boolean isPalindrome(String s) {
        char [] arr = s.toCharArray();
        int i = 0;
        int j = arr.length - 1;
        while (i < j) {
            if (arr[i] != arr[j]) return false;
            i++;
            j--;
        }
        return true;
    }

    void partitionUtils(String s, int i, List<List<String>> palindromes, List<String> running) {
        if (i == s.length()) {
            palindromes.add(new ArrayList<>(running));
            return;
        }
        for (int j = i + 1; j <= s.length(); j++) {
            String s1 = s.substring(i, j);
            if (!isPalindrome(s1)) continue;
            running.add(s1);
            partitionUtils(s, j, palindromes, running);
            running.remove(running.size() - 1);
        }
    }

    public List<List<String>> partition(String s) {
        List<List<String>> palindromes = new ArrayList<>();
        partitionUtils(s, 0, palindromes, new ArrayList<>());
        return palindromes;
    }

//    https://leetcode.com/problems/cheapest-flights-within-k-stops/

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        Map<Integer,Set<int []>> adj = new HashMap<>();
        for (int [] flight : flights) {
            int from = flight[0];
            int to = flight[1];
            int cost = flight[2];

            Set<int []> fromNeighbours = adj.getOrDefault(from, new HashSet<>());
            fromNeighbours.add(new int[]{to, cost, 0});
            adj.put(from, fromNeighbours);
        }

        PriorityQueue<int []> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        minHeap.addAll(adj.getOrDefault(src, new HashSet<>()));
        Map<Integer,Integer> verticesStop = new HashMap<>();

        while (!minHeap.isEmpty()) {
            int [] values = minHeap.remove();
            int node = values[0];
            int cost = values[1];
            int stop = values[2];

            if (node == dst) return cost;
            if (stop >= k || (verticesStop.containsKey(node) && verticesStop.get(node) < stop)) continue;
            verticesStop.put(node,stop);

            for (int [] neighbour : adj.getOrDefault(node, new HashSet<>())) {
                minHeap.add(new int[] {neighbour[0], neighbour[1] + cost, stop + 1});
            }
        }
        return -1;
    }

//    https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/

    void dfsCountComponents(Map<Integer,Set<Integer>> adj, boolean [] visited, int i) {
        if (visited[i]) return;
        visited[i] = true;
        for (int neighbour : adj.getOrDefault(i, new HashSet<>())) dfsCountComponents(adj, visited, neighbour);
    }

    public int countComponents(int n, int[][] edges) {
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

        boolean [] visited = new boolean[n];
        int count = 0;

        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                dfsCountComponents(adj, visited, i);
                count++;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/min-stack/

    static class MinStack {

        Deque<Integer> mainStack;
        Deque<Integer> minStack;

        public MinStack() {
            mainStack = new ArrayDeque<>();
            minStack = new ArrayDeque<>();
        }

        public void push(int val) {
            mainStack.push(val);
            if (minStack.isEmpty()) minStack.push(val);
            else minStack.push(minStack.peek() < val ? minStack.peek() : val);
        }

        public void pop() {
            mainStack.pop();
            minStack.pop();
        }

        public int top() {
            if (mainStack.isEmpty()) return -1;
            return mainStack.peek();
        }

        public int getMin() {
            if (minStack.isEmpty()) return -1;
            return minStack.peek();
        }
    }

}