package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class FourthSet {

    public static void main(String[] args) {
//        TreeNode root = str2tree("4(2(3)(1))(6(5))");
//        System.out.println();
//        System.out.println(solveNQueens(7));
//        System.out.println(addBoldTag("aaabbcc", new String[]{"aaa","aab","bc"}));
//        System.out.println(lengthOfLIS(new int[]{0,1,0,3,2,3}));
//        System.out.println(validUtf8(new int[]{115,100,102,231,154,132,13,10}));
//        System.out.println(countSubstrings("abc"));
//        System.out.println(getColorFul(3245));
        lengthOfLISUtil(new int[]{1,3,2},0, new ArrayList<>(), new int []{0});
    }

//    https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/

    static TreeNode convertToBST(List<Integer> arr, int low, int high) {
        if (low > high) return null;
        if (low == high) return new TreeNode(arr.get(low));
        int mid = low + (high - low) / 2;
        TreeNode node = new TreeNode(arr.get(mid));
        node.left = convertToBST(arr, low, mid - 1);
        node.right = convertToBST(arr, mid + 1, high);
        return node;
    }

    public static TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        ListNode fastPointer = head;
        ListNode slowPointer = head;
        ListNode prevToSlow = head;
        while (fastPointer != null && fastPointer.next != null) {
            prevToSlow = slowPointer;
            fastPointer = fastPointer.next.next;
            slowPointer = slowPointer.next;
        }
        prevToSlow.next = null;
        ListNode r = slowPointer.next;
        slowPointer.next = null;
        TreeNode node = new TreeNode(slowPointer.val);
        if (slowPointer == head) return node;
        node.left = sortedListToBST(head);
        node.right = sortedListToBST(r);
        return node;
    }

//    https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/

    static int minAncestorDiff = Integer.MIN_VALUE;

    static void processMaxAncestorDiff(TreeNode node, int maxValue, int minValue) {
        if (node == null) {
            minAncestorDiff = Math.max(minAncestorDiff, maxValue - minValue);
            return;
        }
        processMaxAncestorDiff(node.left, Math.max(maxValue, node.val), Math.min(minValue, node.val));
        processMaxAncestorDiff(node.right, Math.max(maxValue, node.val), Math.min(minValue, node.val));
    }

    public static int maxAncestorDiff(TreeNode root) {
        if (root == null) return 0;
        processMaxAncestorDiff(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
        return minAncestorDiff;
    }

//    https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/

    public static int minAddToMakeValid(String s) {
        char [] sArr = s.toCharArray();
        int ans = 0;
        int extraClosing = 0;
        for (char c : sArr) {
            ans += c == '(' ? 1 : -1;
            if (ans == -1) {
                ans = 0;
                extraClosing++;
            }
        }
        return ans + extraClosing;
    }

//    https://leetcode.com/problems/making-a-large-island/

    static int dfsLargestIsland(int [][] grid, int i, int j, int n, boolean [][] visited) {
        if (i < 0 || i >= n || j < 0 || j >= n || visited[i][j] || grid[i][j] == 0) return 0;
        visited[i][j] = true;
        return 1 + dfsLargestIsland(grid, i + 1, j, n, visited) +
                dfsLargestIsland(grid, i - 1, j, n, visited) +
                dfsLargestIsland(grid, i, j + 1, n, visited) +
                dfsLargestIsland(grid, i, j - 1, n, visited);
    }

    static void dfsUpdateLargestIsland(int [][] grid, int i, int j,
                                       int n, int x) {
        if (i < 0 || i >= n || j < 0 || j >= n || grid[i][j] == 0 || grid[i][j] == x) return;
        grid[i][j] = x;
        dfsUpdateLargestIsland(grid,i + 1, j, n, x);
        dfsUpdateLargestIsland(grid,i - 1, j, n, x);
        dfsUpdateLargestIsland(grid,i, j + 1, n, x);
        dfsUpdateLargestIsland(grid,i, j - 1, n, x);
    }

    static int checkLargestSize(int [][] grid, int i, int j, int n, int [] area) {
        Set<Integer> uniqueDirections = new HashSet<>();
        int up = i > 0 ? grid[i - 1][j] : 0;
        int down = i < n - 1 ? grid[i + 1][j] : 0;
        int left = j > 0 ? grid[i][j - 1] : 0;
        int right = j < n - 1 ? grid[i][j + 1] : 0;
        if (up != 0) uniqueDirections.add(up);
        if (down != 0) uniqueDirections.add(down);
        if (left != 0) uniqueDirections.add(left);
        if (right != 0) uniqueDirections.add(right);
        int islandSum = 1;
        for (int key : uniqueDirections) islandSum += area[key];
        return islandSum;
    }

    public static int largestIsland(int[][] grid) {
        int n = grid.length;
        int index = 2;
        int largestLength = Integer.MIN_VALUE;
        boolean [][] visited = new boolean[n][n];
        int [] area = new int[n * n + 2];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1 && !visited[i][j]) {
                    int islandSize = dfsLargestIsland(grid, i, j, n, visited);
                    area[index] = islandSize;
                    dfsUpdateLargestIsland(grid, i, j, n, index);
                    index++;
                }
            }
        }
        boolean noZeros = true;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    noZeros = false;
                    int largestSizeAdded = checkLargestSize(grid, i, j, n, area);
                    largestLength = Math.max(largestLength,largestSizeAdded);
                }
            }
        }
        return noZeros ? n * n : largestLength;
    }

//    https://leetcode.com/problems/construct-binary-tree-from-string/

    static TreeNode str2treeUtil(char [] sArr, int [] index) {
        index[0]++;
        int n = sArr.length;
        if (n <= index[0]) return null;
        int no = 0;
        int sign = 1;
        if (sArr[index[0]] == '-') {
            sign = -1;
            index[0]++;
        }
        while (index[0] < n && Character.isDigit(sArr[index[0]])) {
            no = no * 10 + (sArr[index[0]++] - '0');
        }
        no *= sign;
        TreeNode node = new TreeNode(no);
        if (index[0] < n && sArr[index[0]] == '(') {
            node.left = str2treeUtil(sArr, index);
        }
        if (index[0] < n && sArr[index[0]] == '(' && node.left != null) {
            node.right = str2treeUtil(sArr, index);
        }
        if (index[0] < n && sArr[index[0]] == ')') {
            index[0]++;
            return node;
        }
        return node;
    }

    public static TreeNode str2tree(String s) {
        int [] index = new int[]{-1};
        char [] sArr = s.toCharArray();
        return str2treeUtil(sArr, index);
    }

//    https://leetcode.com/problems/n-queens/

    static void solveNQueensUtil(int row, int n, boolean [] cols, boolean [] d1, boolean [] d2, int queensCount,
                                 char [][] board, Set<List<String>> boards) {
        if (row == n) {
            List<String> boardStr = new ArrayList<>();
            for (char[] r : board) {
                boardStr.add(String.valueOf(r));
            }
            boards.add(boardStr);
        }
        for (int j = 0; j < n; j++) {
            int gap = row - j >= 0 ? row - j : row - j + n + n;
            if (!cols[j] && !d1[gap] && !d2[row + j]) {
                board[row][j] = 'Q';
                queensCount++;
                cols[j] = true;
                d1[gap] = true;
                d2[row + j] = true;
                solveNQueensUtil(row + 1, n, cols, d1, d2, queensCount, board, boards);
                board[row][j] = '.';
                queensCount--;
                cols[j] = false;
                d1[gap] = false;
                d2[row + j] = false;
            }
        }
    }

    public static List<List<String>> solveNQueens(int n) {
        boolean [] columns = new boolean[n];
        boolean [] gapD1 = new boolean[n + n];
        boolean [] diffD2 = new boolean[n + n];
        Set<List<String>> boards = new HashSet<>();
        char [][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = '.';
            }
        }
        int queensCount = 0;
        solveNQueensUtil(0, n, columns, gapD1, diffD2, queensCount, board, boards);
        return new ArrayList<>(boards);
    }

//    https://leetcode.com/problems/basic-calculator/

    static int calculateUtil(char [] sArr, int [] index) {
        index[0]++;
        int result = 0;
        int sign = 1;
        while (index[0] < sArr.length) {
            char c = sArr[index[0]];
            if (c == ' ' || c == '+') {
                index[0]++;
            }
            else if (c == '-') {
                sign = -1;
                index[0]++;
            }
            else if (c == '(') {
                result += calculateUtil(sArr, index) * sign;
                sign = 1;
            } else if (c == ')') {
                index[0]++;
                return sign * result;
            } else {
             int no = 0;
             while (index[0] < sArr.length && Character.isDigit(sArr[index[0]])) {
                 no = no * 10 + (sArr[index[0]] - '0');
                 index[0]++;
             }
             result += no * sign;
             sign = 1;
            }
        }
        return result;
    }

    public static int calculate(String s) {
       int [] index = new int[]{-1};
       return calculateUtil(s.toCharArray(), index);
    }

//    https://leetcode.com/problems/add-bold-tag-in-string/

    public static String addBoldTag(String s, String[] words) {
        List<int []> markersIndex = new ArrayList<>();
        for (String word : words) {
            int l = word.length();
            int i = 0;
            while (i + l <= s.length()) {
                if (s.substring(i, i + l).equals(word)) markersIndex.add(new int[]{i,i + l});
                i++;
            }
        }
        markersIndex.sort(Comparator.comparingInt(a -> a[0]));
        List<int []> arr = new ArrayList<>();
        if (markersIndex.size() == 0 || markersIndex.get(0).length < 2) return s;
        arr.add(markersIndex.get(0));
        for (int [] x : markersIndex) {
            int [] prev = arr.get(arr.size() - 1);
            if (prev[1] < x[0]) {
                arr.add(x);
            } else {
                prev[1] = Math.max(x[1], prev[1]);
            }
        }
        int index = 0;
        StringBuilder sb = new StringBuilder();
        char [] sArr = s.toCharArray();
        int i = 0;
        while (i < sArr.length) {
            if (index < arr.size()) {
                int[] bold = arr.get(index);
                if (i == bold[0]) {
                    sb.append("<b>");
                } else if (i == bold[1]) {
                    sb.append("</b>");
                    index++;
                }
            }
            sb.append(sArr[i]);
            i++;
        }
        if (index < arr.size()) {
            sb.append("</b>");
        }
        return sb.toString();
    }

//    https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/

    static class HeapNode {
        int i;
        int j;
        int v;

        HeapNode(int i, int j, int v) {
            this.i = i;
            this.j = j;
            this.v = v;
        }
    }

    public static int kthSmallest(int[][] matrix, int k) {
       PriorityQueue<HeapNode> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> a.v));
       int n = Math.min(k, matrix.length);
       int m = matrix[0].length;
       for (int i = 0; i < n; i++) {
           minHeap.add(new HeapNode(i, 0, matrix[i][0]));
       }
       while (k-- > 0) {
           if (!minHeap.isEmpty()) {
               HeapNode node = minHeap.remove();
               if (node.j < m - 1) {
                   minHeap.add(new HeapNode(node.i, node.j + 1,matrix[node.i][node.j + 1]));
               }
           }
       }
       return minHeap.remove().v;
    }

//    https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/

    static class RandomizedCollection {

        Map<Integer,Set<Integer>> indexMapping;
        List<Integer> arr;
        Random rand = new java.util.Random();

        public RandomizedCollection() {
            this.indexMapping = new HashMap<>();
            this.arr = new ArrayList<>();
        }

        public boolean insert(int val) {
            arr.add(val);
            Set<Integer> s = indexMapping.getOrDefault(val, new HashSet<>());
            s.add(arr.size() - 1);
            indexMapping.put(val,s);
            return s.size() == 1;
        }

        public boolean remove(int val) {
            Set<Integer> indexes = indexMapping.getOrDefault(val,new HashSet<>());
            if (indexes.size() == 0) return false;
            int swapIndex = indexes.iterator().next();
            indexes.remove(swapIndex);
            int lastNo = arr.get(arr.size() - 1);
            Set<Integer> lastValueIndexes = indexMapping.getOrDefault(lastNo, new HashSet<>());
            arr.set(swapIndex,lastNo);
            lastValueIndexes.add(swapIndex);
            lastValueIndexes.remove(arr.size() - 1);
            indexMapping.put(lastNo,lastValueIndexes);
            arr.remove(arr.size() - 1);
            return true;
        }

        int binarySearchRandomNoCiel(List<Integer> prefixArr, int low, int high, int randomNo) {
            if (low >= high) return low;
            int mid = low + (high - low) / 2;
            if (randomNo <= prefixArr.get(mid) && randomNo > (mid > 0 ? prefixArr.get(mid - 1) : -1)) {
                return mid;
            }
            return (randomNo > prefixArr.get(mid))
                    ? binarySearchRandomNoCiel(prefixArr, mid + 1, high, randomNo)
                    : binarySearchRandomNoCiel(prefixArr, low, mid - 1, randomNo);
        }

        public int getRandom() {
            int index = rand.nextInt(arr.size());
            return arr.get(index);
        }
    }

    static void lengthOfLISUtil(int [] nums, int index, List<Integer> arr, int [] maxLength) {
        if (index == nums.length) {
            System.out.println(arr);
            boolean flag = false;
            for (int i = 1; i < arr.size(); i++) {
                if (arr.get(i - 1) >= arr.get(i)) {
                    flag = true;
                    break;
                }
            }
            if (!flag) maxLength[0] = Math.max(maxLength[0],arr.size());
            return;
        }
        for (int i = index; i < nums.length; i++) {
            lengthOfLISUtil(nums, i + 1, arr, maxLength);
            arr.add(nums[i]);
            lengthOfLISUtil(nums, i + 1, arr, maxLength);
            arr.remove(arr.size() - 1);
        }
    }

//    https://leetcode.com/problems/longest-increasing-subsequence/

    public static int lengthOfLIS(int[] nums) {
//        int [] maxLength = new int[]{0};
//        lengthOfLISUtil(nums,0, new ArrayList<>(), maxLength);
//        return maxLength[0];
        int [] dp = new int[nums.length];
        Arrays.fill(dp,1);
        int maxLength = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i],dp[j] + 1);
                }
            }
            maxLength = Math.max(maxLength,dp[i]);
        }
        return maxLength;
    }

//    https://leetcode.com/problems/utf-8-validation/

    public static boolean validUtf8(int[] data) {
        int noOfBytes = 0;
        int mask1 = 1 << 7;
        int mask2 = 1 << 6;
        for (int num : data) {
            if (noOfBytes == 0) {
                int mask = 1 << 7;
                while ((mask & num) != 0) {
                    mask = mask >> 1;
                    noOfBytes++;
                }
                if (noOfBytes == 0) continue;
                if (noOfBytes == 1 || noOfBytes > 4) return false;
            } else {
                if (!((mask1 & num) != 0 && (mask2 & num) == 0)) return false;
            }
            noOfBytes--;
        }
        return noOfBytes == 0;
    }

//    https://leetcode.com/problems/valid-parenthesis-string/

    static boolean checkValidStringUtil(char [] arr) {
        int result = 0;
        for (char c : arr) {
            if (c == ')') {
                result--;
            } else if (c == '(') {
                result++;
            }
            if (result < 0) break;
        }
        return result == 0;
    }

    static boolean checkValidStringRec(char [] arr, int index, int result) {
        if (index == arr.length) {
            return result == 0;
        }
        for (int i = index; i < arr.length; i++) {
            if (arr[i] != '*') {
                if (arr[i] == ')') {
                    result--;
                } else if (arr[i] == '(') {
                    result++;
                }
                if (result < 0) return false;
                if (checkValidStringRec(arr, i + 1, result)) return true;
            } else {
                arr[i] = '(';
                if (checkValidStringRec(arr, i + 1, result + 1)) return true;
                arr[i] = ')';
                int r = result - 1;
                if (r < 0) return false;
                if (checkValidStringRec(arr, i + 1, r)) return true;
                arr[i] = '*';
            }
        }
        return false;
    }

    public static boolean checkValidString(String s) {
//        char [] sArr = s.toCharArray();
//        boolean check = checkValidStringUtil(sArr);
//        if (check) return true;
//        return checkValidStringRec(sArr, 0, 0);
        Deque<Integer> parenthesisStack = new ArrayDeque<>();
        Deque<Integer> starStack = new ArrayDeque<>();
        char [] sArr = s.toCharArray();
        for (int i = 0; i < sArr.length; i++) {
            if (sArr[i] == '(') parenthesisStack.push(i);
            else if (sArr[i] == ')') {
                if (!parenthesisStack.isEmpty()) parenthesisStack.pop();
                else if (!starStack.isEmpty()) starStack.pop();
                else return false;
            } else {
                starStack.push(i);
            }
        }
        while (!parenthesisStack.isEmpty()) {
            int poppedStartingBracket = parenthesisStack.pop();
            if (starStack.isEmpty()) return false;
            else if (starStack.peek() < poppedStartingBracket) return false;
            starStack.pop();
        }
        return true;
    }

//    https://leetcode.com/problems/basic-calculator-iii/

    static int calculateUtilTwo(char [] arr, int [] index) {
        index[0]++;
        if (index[0] >= arr.length) return 0;
        Deque<Integer> stack = new ArrayDeque<>();
        char prevSign = '+';
        while (index[0] < arr.length) {
            int no = 0;
            if (arr[index[0]] == '(') {
                no = calculateUtilTwo(arr,index);
            } else if (arr[index[0]] == ')') {
                index[0]++;
                break;
            } else if (Character.isDigit(arr[index[0]])) {
                while (index[0] < arr.length && Character.isDigit(arr[index[0]])) {
                    no = no * 10 + (arr[index[0]] - '0');
                    index[0]++;
                }
            } else {
                prevSign = arr[index[0]];
                index[0]++;
                continue;
            }
            if (prevSign == '+') stack.push(no);
            else if (prevSign == '-') stack.push(no * -1);
            else if (prevSign == '*') stack.push(stack.pop() * no);
            else if (prevSign == '/') stack.push(stack.pop() / no);
        }
        int s = 0;
        while (!stack.isEmpty()) s += stack.pop();
        return s;
    }

    public static int calculateTwo(String s) {
        char [] sArr = s.toCharArray();
        int [] index = new int[]{-1};
        return calculateUtilTwo(sArr,index);
    }

//    https://leetcode.com/problems/palindromic-substrings/

    public static int countSubstrings(String s) {
        char [] sArr = s.toCharArray();
        int n = sArr.length;
        int count = 0;
        for (int i = 0; i < n; i++) {
            int a = i;
            int b = i;
            while (a >= 0 && b < n && sArr[a] == sArr[b]) {
                count += 1;
                a--;
                b++;
            }
            a = i - 1;
            b = i;
            while (a >= 0 && b < n && sArr[a] == sArr[b]) {
                count += 1;
                a--;
                b++;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/

    static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node parent;
    }

    public Node lowestCommonAncestor(Node p, Node q) {
        Node pointerP = p;
        Node pointerQ = q;
        while (pointerP != pointerQ) {
            pointerP = pointerP.parent;
            pointerQ = pointerQ.parent;

            if (pointerP == null) pointerP = q;
            if (pointerQ == null) pointerQ = p;
        }
        return pointerP;
    }

//    https://leetcode.com/problems/largest-bst-subtree/

    static class BSTRangeCount {
        Integer minValue;
        Integer maxValue;
        Integer count;

        BSTRangeCount(Integer minValue, Integer maxValue, Integer count) {
            this.minValue = minValue;
            this.maxValue = maxValue;
            this.count = count;
        }
    }

    static int largestBSTSubtreeSize = 0;

    static BSTRangeCount largestBSTSubtreeUtil(TreeNode node) {
        if (node == null) return null;
        if (node.left == null && node.right == null) {
            largestBSTSubtreeSize = Math.max(largestBSTSubtreeSize,1);
            return new BSTRangeCount(node.val,node.val,1);
        }
        BSTRangeCount left = largestBSTSubtreeUtil(node.left);
        BSTRangeCount right = largestBSTSubtreeUtil(node.right);
        if (left != null && right != null) {
            if (left.minValue != null && left.maxValue != null && right.minValue != null && right.maxValue != null) {
                if (left.maxValue < node.val && right.minValue > node.val) {
                    int subTreeCount = 1 + left.count + right.count;
                    largestBSTSubtreeSize = Math.max(largestBSTSubtreeSize, subTreeCount);
                    return new BSTRangeCount(left.minValue, right.maxValue, subTreeCount);
                }
            }
        } else if (left != null && left.minValue != null && left.maxValue != null) {
            if (left.maxValue < node.val) {
                int subTreeCount = 1 + left.count;
                largestBSTSubtreeSize = Math.max(largestBSTSubtreeSize, subTreeCount);
                return new BSTRangeCount(left.minValue, node.val, subTreeCount);
            }
        } else if (right != null && right.minValue != null && right.maxValue != null) {
            if (right.minValue > node.val) {
                int subTreeCount = 1 + right.count;
                largestBSTSubtreeSize = Math.max(largestBSTSubtreeSize, subTreeCount);
                return new BSTRangeCount(node.val, right.maxValue, subTreeCount);
            }
        }
        return new BSTRangeCount(null,null,-1);
    }

    public static int largestBSTSubtree(TreeNode root) {
        largestBSTSubtreeUtil(root);
        return largestBSTSubtreeSize;
    }

//    https://leetcode.com/problems/minimum-moves-to-move-a-box-to-their-target-location/

    public static int minPushBox(char[][] grid) {
        int [] s = new int[2];
        int [] b = new int[2];
        int [] t = new int[2];
        int n = grid.length;
        int m = grid[0].length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 'S') {
                    s[0] = i;
                    s[1] = j;
                } else if (grid[i][j] == 'B') {
                    b[0] = i;
                    b[1] = j;
                } else if (grid[i][j] == 'T') {
                    t[0] = i;
                    t[1] = j;
                }
            }
        }
        int [][] directions = new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
        Set<String> visited = new HashSet<>();
        PriorityQueue<int []> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        minHeap.add(new int[]{distance(b[0], b[1], t[0], t[1]),0,s[0],s[1],b[0],b[1]});
        while (!minHeap.isEmpty()) {
            int [] node = minHeap.remove();
            int dist = node[0];
            int moves = node[1];
            int sI = node[2];
            int sJ = node[3];
            int bI = node[4];
            int bJ = node[5];
            String str = sI + " " + sJ + " " + bI + " " + bJ;
            if (visited.contains(str)) continue;
            visited.add(str);
            if (bI == t[0] && bJ == t[1]) return moves;
            for (int [] direction : directions) {
                int newI = sI + direction[0];
                int newJ = sJ + direction[1];
                if (!isValid(newI,newJ, grid)) continue;
                if (newI == bI && newJ == bJ) {
                    int newBI = bI + direction[0];
                    int newBJ = bJ + direction[1];
                    if (isValid(newBI, newBJ, grid)) {
                    minHeap.add(new int[]
                            {distance(newBI, newBJ, t[0], t[1]) + moves + 1, moves + 1, newI, newJ, newBI, newBJ});
                    }
                } else {
                    minHeap.add(new int[]{dist, moves, newI, newJ, bI, bJ});
                }
            }
        }
        return -1;
    }

    static boolean isValid(int x, int y, char [][] grid) {
        return x >= 0 && x < grid.length && y >= 0 && y < grid[x].length && grid[x][y] != '#';
    }

    static int distance(int x1, int y1, int x2, int y2) {
        return Math.abs(x1 - x2) + Math.abs(y1 - y2);
    }

//    https://algorithms.tutorialhorizon.com/colorful-numbers/

    public static boolean getColorFul(int Number) {

        Set<Integer> product = new HashSet<>();

        char [] numArr = String.valueOf(Number).toCharArray();
        int n = numArr.length;

        for (int i = 1; i < n; i++) {
            int k = 0;
            int productMul = 1;

            while(k < i) {
                productMul *= numArr[k] - '0';
                k++;
            }

            if (product.contains(productMul)) return false;
            product.add(productMul);

            while(k < n) {
                productMul *= numArr[k] - '0';
                productMul /= numArr[k - i] - '0';
                if (product.contains(productMul)) return false;
                product.add(productMul);
                System.out.println(product + " " + productMul);
                k++;
            }
        }
        return true;
    }

//    https://www.hackerrank.com/challenges/qheap1/problem

    public static void minHeapDelAnywhere(String[] args) throws IOException {
        /* Enter your code here. Read input from STDIN. Print output to STDOUT. Your class should be named Solution. */


        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int q = Integer.parseInt(br.readLine());

        List<Integer> arr = new ArrayList<>();
        Map<Integer,Integer> mapping = new HashMap<>();

        for (int q1 = 0; q1 < q; q1++) {

            String [] inputs = br.readLine().split(" ");
            String input = inputs[0];
            switch(input) {
                case "1" : {
                    int value = Integer.parseInt(inputs[1]);
                    arr.add(value);
                    mapping.put(value, arr.size() - 1);
                    heapifyUp(arr, mapping);
                }
                break;
                case "2" : {
                    int value = Integer.parseInt(inputs[1]);
                    int dIndex = mapping.get(value);
                    arr.set(dIndex, arr.get(arr.size() - 1));
                    mapping.put(arr.get(arr.size() - 1), dIndex);
                    arr.remove(arr.size() - 1);
                    heapifyDown(arr, mapping, dIndex);
                }
                break;
                case "3" : {
                    System.out.println(arr.get(0));
                }
                break;
            }

        }

    }

    static void heapifyUp(List<Integer> arr, Map<Integer,Integer> map) {
        int childIndex = arr.size() - 1;
        int parentIndex = childIndex / 2;
        while(childIndex != parentIndex) {
            if (arr.get(childIndex) < arr.get(parentIndex)) {
                int temp = arr.get(childIndex);
                arr.set(childIndex, arr.get(parentIndex));
                map.put(arr.get(childIndex), childIndex);
                arr.set(parentIndex, temp);
                map.put(arr.get(parentIndex), parentIndex);
                childIndex = parentIndex;
                parentIndex = childIndex / 2;
            } else break;
        }
    }

    static void heapifyDown(List<Integer> arr, Map<Integer,Integer> map, int dIndex) {
        int parentIndex = dIndex;
        int n = arr.size();
        while(parentIndex < n) {
            int leftChildIndex = parentIndex * 2 + 1;
            int rightChildIndex = parentIndex * 2 + 2;
            if ((leftChildIndex < n)) {
                int smallerChildIndex;
                if (rightChildIndex >= n) {
                    smallerChildIndex = leftChildIndex;
                } else {
                    smallerChildIndex = arr.get(leftChildIndex) < arr.get(rightChildIndex) ? leftChildIndex : rightChildIndex;
                }
                if (arr.get(parentIndex) > arr.get(smallerChildIndex)) {
                    int temp = arr.get(parentIndex);
                    arr.set(parentIndex, arr.get(smallerChildIndex));
                    map.put(arr.get(parentIndex), parentIndex);
                    arr.set(smallerChildIndex, temp);
                    map.put(temp, smallerChildIndex);
                    parentIndex = smallerChildIndex;
                } else break;
            } else break;
        }
    }

//    https://www.hackerrank.com/challenges/the-quickest-way-up/problem

    public static int quickestWayUp(List<List<Integer>> ladders, List<List<Integer>> snakes) {
        Map<Integer,Integer> lS = new HashMap<>();
        for (List<Integer> ladder : ladders) {
            lS.put(ladder.get(0),ladder.get(1));
        }
        for (List<Integer> snake : snakes) {
            lS.put(snake.get(0),snake.get(1));
        }
        Set<Integer> visited = new HashSet<>();
        Deque<Integer> queue = new ArrayDeque<>();
        queue.add(1);
        int hops = 0;
        while(!queue.isEmpty()) {
            int n = queue.size();
            System.out.println(queue + " " + hops);
            for (int i = 0; i < n; i++) {
                int node = queue.remove();
                if (node == 100) return hops;
                if (visited.contains(node)) continue;
                visited.add(node);
                for (int j = 1; j <= 6; j++) {
                    int newNode = node + j;
                    if (newNode > 100) continue;
                    queue.add(lS.getOrDefault(newNode, newNode));
                }
            }
            hops++;
        }
        return -1;
    }

}