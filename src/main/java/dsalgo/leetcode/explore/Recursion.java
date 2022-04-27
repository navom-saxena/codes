package dsalgo.leetcode.explore;

import dsalgo.leetcode.Models.*;
import javafx.util.Pair;

import java.util.*;
import java.util.Arrays;
import java.util.stream.Collectors;

public class Recursion {

    public static void main(String[] args) {
        int [] a = new int[]{0,1,2,3,4,5,6};
        int [] leftArr = Arrays.copyOfRange(a,0,a.length / 2);
        int [] rightArr = Arrays.copyOfRange(a, (a.length / 2), a.length);
        System.out.println(Arrays.toString(leftArr));
        System.out.println(Arrays.toString(rightArr));
    }

//    https://leetcode.com/explore/learn/card/recursion-i/250/principle-of-recursion/1440/

    void reverseStringUtil(char [] s, int i, int j) {
        if (i >= j) return;
        char a = s[i];
        char b = s[j];
        s[i] = b;
        s[j] = a;
        reverseStringUtil(s, i + 1, j - 1);
    }

    public void reverseString(char[] s) {
        reverseStringUtil(s, 0, s.length - 1);
    }

//    https://leetcode.com/explore/learn/card/recursion-i/250/principle-of-recursion/1681/

    public ListNode swapPairs(ListNode head) {
        if (head == null) return null;
        if (head.next == null) return head;
        ListNode nextN = head.next;
        ListNode nextToP = nextN.next;
        nextN.next = head;
        head.next = swapPairs(nextToP);
        return nextN;
    }

//    https://leetcode.com/explore/learn/card/recursion-i/251/scenario-i-recurrence-relation/3233/

    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null) return null;
        if (root.val == val) return root;
        else if (root.val < val) return searchBST(root.right, val);
        else return searchBST(root.left, val);
    }

//    https://leetcode.com/explore/learn/card/recursion-i/251/scenario-i-recurrence-relation/3234/

    int getRowUtil(int i, int j, int [][] dp) {
        if (i == 0 || j == 0 || i == j) return dp[i][j] = 1;
        if (dp[i][j] != 0) return dp[i][j];
        return dp[i][j] = getRowUtil(i - 1, j - 1, dp) + getRowUtil(i - 1, j, dp);
    }

    public List<Integer> getRow(int rowIndex) {
        List<Integer> res = new ArrayList<>();
        int [][] dp = new int[rowIndex + 1][rowIndex + 1];
        for (int j = 0; j <= rowIndex; j++) {
            res.add(getRowUtil(rowIndex, j, dp));
        }
        return res;
    }

//    https://leetcode.com/explore/learn/card/recursion-i/255/recursion-memoization/1661/

    int [] fibonacciDp = new int[31];

    public int fib(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        if (fibonacciDp[n] != 0) return fibonacciDp[n];
        return fibonacciDp[n] = fib(n - 1) + fib(n - 2);
    }

//    https://leetcode.com/explore/learn/card/recursion-i/255/recursion-memoization/1662/

    int [] climbStairsDp = new int[46];

    public int climbStairs(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        if (n == 2) return 2;
        if (climbStairsDp[n] != 0) return climbStairsDp[n];
        return climbStairsDp[n] = climbStairs(n - 1) + climbStairs(n - 2);
    }

//    https://leetcode.com/explore/learn/card/recursion-i/256/complexity-analysis/2380/

    public double myPow(double x, int n) {
        if (n == 0) return 1;
        else if (n == 1) return x;
        else if (n == -1) return 1/x;

        double y = myPow(x, n / 2);
        if (n >= 0) return n % 2 == 0 ? y * y : y * y * x;
        else return n % 2 == 0 ? y * y : y * y * (1/x);
    }

//    https://leetcode.com/explore/learn/card/recursion-i/253/conclusion/2382/

    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode sentinel = new ListNode(Integer.MIN_VALUE);
        ListNode curr = sentinel;
        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                curr.next = list1;
                list1 = list1.next;
            } else {
                curr.next = list2;
                list2 = list2.next;
            }
            curr = curr.next;
        }
        if (list1 != null) curr.next = list1;
        else curr.next = list2;
        return sentinel.next;
    }

    public ListNode mergeTwoListsRec(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        else if (l2 == null) return l1;
        else if (l1.val < l2.val) {
            l1.next = mergeTwoListsRec(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoListsRec(l1, l2.next);
            return l2;
        }
    }

//    https://leetcode.com/explore/learn/card/recursion-i/253/conclusion/1675/

    public int kthGrammar(int n, int k) {
        if (n == 1) return 0;
        int parent = kthGrammar(n - 1, (k + 1) / 2);
        if (parent == 0) return k % 2 == 0 ? 1 : 0;
        else return k % 2 == 0 ? 0 : 1;
    }

//    https://leetcode.com/explore/learn/card/recursion-i/253/conclusion/2384/

    List<TreeNode> generateTreesUtil(int low, int high) {
        List<TreeNode> res = new ArrayList<>();
        if (low > high) res.add(null);
        if (low == high) res.add(new TreeNode(low));
        else {
            for (int i = low; i <= high; i++) {
                List<TreeNode> leftN = generateTreesUtil(low, i - 1);
                List<TreeNode> rightN = generateTreesUtil(i + 1, high);
                for (TreeNode l : leftN) {
                    for (TreeNode r : rightN) {
                        TreeNode root = new TreeNode(i);
                        root.left = l;
                        root.right = r;
                        res.add(root);
                    }
                }
            }
        }
        return res;
    }

    public List<TreeNode> generateTrees(int n) {
        return generateTreesUtil(1, n);
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/470/divide-and-conquer/2944/

    public int[] sortArray(int[] nums) {
        if (nums.length >= 2) {
            int [] leftArr = Arrays.copyOfRange(nums,0,nums.length / 2);
            int [] rightArr = Arrays.copyOfRange(nums, nums.length / 2, nums.length);

            int [] leftArrS = sortArray(leftArr);
            int [] rightArrS = sortArray(rightArr);

            return mergeSortedArr(leftArrS, rightArrS);
        }
        return nums;
    }

    int [] mergeSortedArr(int [] left, int [] right) {
        int [] nums = new int[left.length + right.length];
        int p = 0;
        int l = 0;
        int r = 0;

        while (l < left.length && r < right.length) {
            if (left[l] <= right[r]) {
                nums[p] = left[l];
                l++;
            } else {
                nums[p] = right[r];
                r++;
            }
            p++;
        }

        while (l < left.length) {
            nums[p] = left[l];
            l++;
            p++;
        }

        while (r < right.length) {
            nums[p] = right[r];
            r++;
            p++;
        }

        return nums;
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/470/divide-and-conquer/2874/

    boolean isValidBSTUtils(TreeNode node, long low, long high) {
        if (node == null) return true;
        return node.val > low && node.val < high
                && isValidBSTUtils(node.left, low, node.val) && isValidBSTUtils(node.right, node.val, high);
    }

    public boolean isValidBST(TreeNode root) {
        return isValidBSTUtils(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/470/divide-and-conquer/2872/

    public boolean searchMatrix(int[][] matrix, int target) {
        int c = matrix[0].length - 1;
        int r = 0;
        while (c >= 0 && r < matrix.length) {
            if (matrix[r][c] == target) return true;
            else if (matrix[r][c] > target) c--;
            else r++;
        }
        return false;
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/472/backtracking/2804/

    void totalNQueensUtil(int r, boolean [] rows, boolean [] cols, boolean [] d1, boolean [] d2, int n, int [] res) {
        if (r == n) {
            res[0]++;
            return;
        }
        for (int c = 0; c < n; c++) {
            if (rows[r] || cols[c] || d1[r - c + n] || d2[r + c]) continue;
            rows[r] = true;
            cols[c] = true;
            d1[r - c + n] = true;
            d2[r + c] = true;

            totalNQueensUtil(r + 1, rows, cols, d1, d2, n, res);

            rows[r] = false;
            cols[c] = false;
            d1[r - c + n] = false;
            d2[r + c] = false;
        }
    }

    public int totalNQueens(int n) {
        boolean [] rows = new boolean[n];
        boolean [] cols = new boolean[n];
        boolean [] d1 = new boolean[2 * n + 1];
        boolean [] d2 = new boolean[2 * n + 1];
        int [] res = new int[]{0};
        totalNQueensUtil(0, rows, cols, d1, d2, n, res);
        return res[0];
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/472/backtracking/2794/

    interface Robot {
        boolean move();
        void turnLeft();
        void turnRight();
        void clean();
    }

    void goBack(Robot robot) {
        robot.turnRight();
        robot.turnRight();
        robot.move();
        robot.turnRight();
        robot.turnRight();
    }

    void processCleaning(int [] current, int dir, Robot robot) {
        robot.clean();
        cleaned.add(new Pair<>(current[0],current[1]));

        for (int i = 0; i < 4; i++) {
            int newDir = (dir + i) % 4;

            int x = current[0] + directions[newDir][0];
            int y = current[1] + directions[newDir][1];

            if (!cleaned.contains(new Pair<>(x,y)) && robot.move()) {
                processCleaning(new int[]{x,y}, newDir, robot);
                goBack(robot);
            }

            robot.turnRight();
        }
    }

    int[][] directions = new int[][]{{-1,0},{0,1},{1,0},{0,-1}};
    Set<Pair<Integer, Integer>> cleaned = new HashSet<>();

    public  void cleanRoom(Robot robot) {
        int[] location = new int[]{0,0};
        processCleaning(location, 0, robot);
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/472/backtracking/2796/

    boolean sudokuSolved = false;

    void solveSudokuUtil(char [][] board, int r, int c, Map<Integer,Set<Character>> rows,
                            Map<Integer,Set<Character>> cols, Map<Integer,Set<Character>> box) {
        if (r == 9) {
            sudokuSolved = true;
            return;
        }
        int newC = c + 1 == 9 ? 0 : c + 1;
        int newR = c + 1 == 9 ? r + 1 : r;
        int hash = (r / 3) * 3 + (c / 3);

        if (board[r][c] == '.') {
            for (char no = '1'; no <= '9'; no++) {
                if (rows.get(r).contains(no) || cols.get(c).contains(no) || box.get(hash).contains(no)) continue;

                board[r][c] = no;
                rows.get(r).add(no);
                cols.get(c).add(no);
                box.get(hash).add(no);

                solveSudokuUtil(board, newR, newC, rows, cols, box);
                if (sudokuSolved) return;

                board[r][c] = '.';
                rows.get(r).remove(no);
                cols.get(c).remove(no);
                box.get(hash).remove(no);
            }
        } else {
            solveSudokuUtil(board, newR, newC, rows, cols, box);
        }
    }

    public void solveSudoku(char[][] board) {
        Map<Integer,Set<Character>> rows = new HashMap<>();
        Map<Integer,Set<Character>> cols = new HashMap<>();
        Map<Integer,Set<Character>> box = new HashMap<>();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                int hash = (i / 3) * 3 + (j / 3);
                Set<Character> r = rows.getOrDefault(i, new HashSet<>());
                Set<Character> c = cols.getOrDefault(j, new HashSet<>());
                Set<Character> b = box.getOrDefault(hash, new HashSet<>());

                if (board[i][j] != '.') {
                    char v = board[i][j];
                    r.add(v);
                    c.add(v);
                    b.add(v);
                }
                rows.put(i,r);
                cols.put(j,c);
                box.put(hash,b);
            }
        }
        solveSudokuUtil(board, 0, 0, rows, cols, box);
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/472/backtracking/2798/

    void combineUtils(int curr, int k, int n, List<Integer> arr, List<List<Integer>> res) {
        if (arr.size() == k) {
            res.add(new ArrayList<>(arr));
            return;
        }
        for (int c = curr; c <= n; c++) {
            arr.add(c);
            combineUtils(c + 1, k, n, arr, res);
            arr.remove(arr.size() - 1);
        }
    }

    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        combineUtils(1, k, n, new ArrayList<>(), res);
        return res;
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/503/recursion-to-iteration/2772/

    void generateParenthesisUtil(int openClose, int n, StringBuilder sb, List<String> res) {
        if (sb.length() == 2 * n) {
            if (openClose == 0) res.add(sb.toString());
            return;
        }
       if (openClose < n) {
           sb.append("(");
           generateParenthesisUtil(openClose + 1, n, sb, res);
           sb.deleteCharAt(sb.length() - 1);
       }
       if (openClose > 0) {
           sb.append(")");
           generateParenthesisUtil(openClose - 1, n, sb, res);
           sb.deleteCharAt(sb.length() - 1);
       }
    }

    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        generateParenthesisUtil(0, n, new StringBuilder(), res);
        return res;
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/507/beyond-recursion/2899/

    static class Node {
        public int val;
        public Node left;
        public Node right;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }
    }

    Node prev;
    Node head;

    void treeToDoublyListUtil(Node node) {
        if (node == null) return;
        treeToDoublyListUtil(node.left);
        if (head == null) {
            head = node;
        } else {
            prev.right = node;
            node.left = prev;
        }
        prev = node;
        treeToDoublyListUtil(node.right);
    }

    public Node treeToDoublyList(Node root) {
        if (root == null) return null;
        treeToDoublyListUtil(root);
        head.left = prev;
        prev.right = head;
        return head;
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/507/beyond-recursion/2901/

    public int largestRectangleArea(int[] heights) {
        int n = heights.length;

        int [] leftMax = new int[n];
        int [] rightMax = new int[n];

        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i]) stack.pop();
            leftMax[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }

        stack.clear();

        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i]) stack.pop();
            rightMax[i] = stack.isEmpty() ? n: stack.peek();
            stack.push(i);
        }

        int maxR = 0;
        for (int i = 0; i < n; i++) {
            int rec = heights[i] * (rightMax[i] - leftMax[i] - 1);
            maxR = Math.max(maxR, rec);
        }
        return maxR;
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/507/beyond-recursion/2903/

    void swap(int i, int j, int [] nums) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    void permuteUtils(int[] nums, int index, List<List<Integer>> res) {
        if (index == nums.length) {
            res.add(Arrays.stream(nums).boxed().collect(Collectors.toList()));
            return;
        }
        for (int i = index; i < nums.length; i++) {
            swap(i, index, nums);
            permuteUtils(nums, index + 1, res);
            swap(index, i, nums);
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        permuteUtils(nums, 0, res);
        return res;
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/507/beyond-recursion/2905/

    List<String> combinations;
    String d;

    void backtrackLetterCombinations(int i, StringBuilder sb, Map<Character,List<String>> map) {
        if (sb.length() == d.length()) {
            combinations.add(sb.toString());
            return;
        }
        for (String c : map.get(d.charAt(i))) {
            sb.append(c);
            backtrackLetterCombinations(i + 1, sb, map);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    public List<String> letterCombinations(String digits) {
        Map<Character,List<String>> mapping = new HashMap<>();
        mapping.put('2', Arrays.asList("a","b","c"));
        mapping.put('3', Arrays.asList("d","e","f"));
        mapping.put('4', Arrays.asList("g","h","i"));
        mapping.put('5', Arrays.asList("j","k","l"));
        mapping.put('6', Arrays.asList("m","n","o"));
        mapping.put('7', Arrays.asList("p","q","r","s"));
        mapping.put('8', Arrays.asList("t","u","v"));
        mapping.put('9', Arrays.asList("w","x","y","z"));

        combinations = new ArrayList<>();
        d = digits;
        if (d.length() == 0) return combinations;
        backtrackLetterCombinations(0, new StringBuilder(), mapping);
        return combinations;
    }

//    https://leetcode.com/explore/learn/card/recursion-ii/507/beyond-recursion/3006/

    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<Point> points = new ArrayList<>();
        List<List<Integer>> res = new ArrayList<>();
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a,b) -> b - a);

        for (int [] b : buildings) {
            int s = b[0];
            int e = b[1];
            int h = b[2];
            points.add(new Point(s, -h));
            points.add(new Point(e, h));
        }

        Collections.sort(points);
        int currH = 0;
        maxHeap.add(0);
        for (Point p : points) {
            int x = p.x;
            int h = p.h;

            if (h < 0) {
                maxHeap.add(-h);
            } else {
                maxHeap.remove(h);
            }
            if (!maxHeap.isEmpty() && currH != maxHeap.peek()) {
                List<Integer> value = new ArrayList<>();
                value.add(x);
                value.add(maxHeap.peek());
                res.add(value);
                currH = maxHeap.peek();
            }
        }

        return res;
    }

    static class Point implements Comparable<Point> {
        int x;
        int h;

        Point(int x, int h) {
            this.x = x;
            this.h = h;
        }

        @Override
        public int compareTo(Point o) {
            if (this.x != o.x) return this.x - o.x;
            else return this.h - o.h;
        }
    }
}