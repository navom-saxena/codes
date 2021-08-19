package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;

import java.util.*;

public class ThirdSet {

    public static void main(String[] args) {
//        wallsAndGates(new int[][]
//                {{0,2147483647,2147483647,2147483647,2147483647},
//        {2147483647,0,2147483647,2147483647,2147483647},
//        {2147483647,2147483647,0,2147483647,2147483647},
//        {2147483647,2147483647,2147483647,0,2147483647},
//        {2147483647,2147483647,2147483647,2147483647,0}});
//        System.out.println(isNumber(".e1"));
//        System.out.println(myPow(34.00515,-3));
//        List<NestedInteger> l = new ArrayList<>();
//        List<NestedInteger> l1 = new ArrayList<>();
//        l1.add(new NestedIntegerImplemented(1));
//        l1.add(new NestedIntegerImplemented(1));
//        l.add(new NestedIntegerImplemented(l1));
//        l.add(new NestedIntegerImplemented(2));
//        List<NestedInteger> l3 = new ArrayList<>();
//        l3.add(new NestedIntegerImplemented(1));
//        l3.add(new NestedIntegerImplemented(1));
//        l.add(new NestedIntegerImplemented(l3));
//        NestedIterator iterator = new NestedIterator(l);
//        while (iterator.hasNext()) {
//            System.out.println(iterator.next());
//        }
//        System.out.println(calculate("0-2147483647"));
//        System.out.println(findStrobogrammatic(1));
//        ListNode head = new ListNode(1);
//        ListNode curr = head;
//        for (int i = 2; i <= 5; i++) {
//            curr.next = new ListNode(i);
//            curr = curr.next;
//        }
//        reorderList(head);
//        System.out.println(missingElement(new int[]{1,2,4},3));
//        System.out.println(rearrangeString("aabbcc", 3));
//        System.out.println(customSortString("cba","abcd"));
        System.out.println(minKnightMoves(5,5));
    }

//    https://leetcode.com/problems/walls-and-gates/

    public static void wallsAndGates(int[][] rooms) {
        int n = rooms.length;
        int m = rooms[0].length;
        Queue<int []> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (rooms[i][j] == 0) {
                    rooms[i][j] = 1;
                    queue.add(new int[]{i, j});
                    rooms[i][j] = 0;
                }
            }
        }
        int [][] directions = new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
        while (!queue.isEmpty()) {
            int [] node = queue.remove();
            int i = node[0];
            int j = node[1];
            for (int [] direction : directions) {
                int newI = i + direction[0];
                int newJ = j + direction[1];
                if (newI < 0 || newI >= n || newJ < 0 || newJ >= m || rooms[newI][newJ] != Integer.MAX_VALUE) continue;
                rooms[newI][newJ] = rooms[i][j] + 1;
                queue.add(new int[]{newI,newJ});
            }
        }
    }

//    https://leetcode.com/problems/valid-number/

    public static boolean isNumber(String s) {
        int eCount = 0;
        int decimalCount = 0;
        int digitCount = 0;
        char [] sArr = s.toCharArray();
        for (int i = 0; i < sArr.length; i++) {
            if (Character.isDigit(sArr[i])) {
                digitCount++;
            }
            else if (sArr[i] == '+' || sArr[i] == '-') {
                if (i > 0 && sArr[i - 1] != 'e' && sArr[i - 1] != 'E') return false;
                else if (i == sArr.length - 1 || sArr[i + 1] == 'e' || sArr[i + 1] == 'E') return false;
            } else if (Character.isAlphabetic(sArr[i])) {
                if (i == 0 || i == sArr.length - 1 || !(sArr[i] == 'e' || sArr[i] == 'E')) return false;
                eCount++;
                if (eCount > 1) return false;
            } else if (sArr[i] == '.') {
                if (decimalCount >= 1 || eCount >= 1) return false;
                else {
                    if (i == 0 && sArr.length > 1 && !Character.isDigit(sArr[i + 1])) return false;
                    else if (i > 0 && i == sArr.length - 1 && !Character.isDigit(sArr[i - 1])) return false;
                    else if (i > 0 && i < sArr.length - 1 && !(Character.isDigit(sArr[i - 1]) || Character.isDigit(sArr[i + 1]))) return false;
                }
                decimalCount++;
            }
        }
        return digitCount > 0;
    }

//    https://leetcode.com/problems/intersection-of-two-arrays/

    public static int[] intersection(int[] nums1, int[] nums2) {
        int [] arr1 = new int[1001];
        for (int num : nums1) {
            arr1[num] = 1;
        }
        int [] res = new int[nums2.length];
        int k = 0;
        for (int num : nums2) {
            if (arr1[num] > 0) {
                res[k] = num;
                k++;
                arr1[num] = 0;
            }
        }
        int [] finalRes = new int[k];
        System.arraycopy(res, 0, finalRes, 0, k);
        return finalRes;
    }

//    https://leetcode.com/problems/powx-n/

    public static double myPow(double x, int n) {
        if (n == 0) return 1;
        else if (n == 1) return x;
        else if (n == -1) return 1.0 / x;
        Double half = myPow(x,n / 2);
        if (n % 2 == 0) {
            return half * half;
        } else {
            return half * half * (n > 0 ? x : 1.0 / x);
        }
    }

//    https://leetcode.com/problems/flatten-nested-list-iterator/

    public interface NestedInteger {

        public boolean isInteger();

        public Integer getInteger();

        public List<NestedInteger> getList();

    }

    public static class NestedIntegerImplemented implements NestedInteger {

        List<NestedInteger> l;
        Integer value;

        NestedIntegerImplemented(int i) {
            this.value = i;
        }

        NestedIntegerImplemented(List<NestedInteger> l) {
            this.l = l;
        }

        @Override
        public boolean isInteger() {
            return value != null;
        }

        @Override
        public Integer getInteger() {
            return value;
        }

        @Override
        public List<NestedInteger> getList() {
            return l;
        }
    }

    public static class NestedIterator implements Iterator<Integer> {

        static class StackClass {
            List<NestedInteger> nestedList;
            int i;

            StackClass(List<NestedInteger> nestedList, int i) {
                this.nestedList = nestedList;
                this.i = i;
            }
        }

        Deque<StackClass> stack;

        public NestedIterator(List<NestedInteger> nestedList) {
            stack = new ArrayDeque<>();
            stack.push(new StackClass(nestedList,-1));
        }

        static void flattenNestedList(List<Integer> nestedListFlattened, List<NestedInteger> nestedList) {
            for (NestedInteger nestedInteger : nestedList) {
                if (nestedInteger.isInteger()) {
                    nestedListFlattened.add(nestedInteger.getInteger());
                } else {
                    flattenNestedList(nestedListFlattened, nestedInteger.getList());
                }
            }
        }

        @Override
        public Integer next() {
            StackClass s = stack.peek();
            if (s == null) return null;
            return s.nestedList.get(s.i).getInteger();
        }

        @Override
        public boolean hasNext() {
            if (stack.isEmpty()) return false;
            else {
                while (!stack.isEmpty()) {
                    StackClass s = stack.peek();
                    List<NestedInteger> l = s.nestedList;
                    s.i++;
                    if (s.i == l.size()) {
                        stack.pop();
                    }
                    else if (l.get(s.i).isInteger()) return true;
                    else {
                        stack.push(new StackClass(l.get(s.i).getList(),-1));
                    }
                }
            }
            return false;
        }
    }

//    https://leetcode.com/problems/basic-calculator-ii/

    public static int calculate(String s) {
        Deque<Integer> stack = new ArrayDeque<>();
        char [] sArr = s.toCharArray();
        int n = sArr.length;
        char prevSign = ' ';
        for (int i = 0; i < n; i++) {
            char c = sArr[i];
            if (c == ' ') {
                continue;
            } else if (Character.isDigit(c)) {
                int digit = 0;
                while (i < n && Character.isDigit(sArr[i])) {
                    digit = (digit * 10) + (sArr[i] - '0');
                    i++;
                }
                i--;
                if (prevSign == '*') stack.push(stack.pop() * digit);
                else if (prevSign == '/') stack.push(stack.pop() / digit);
                else if (prevSign == '-') {
                    stack.push(digit * -1);
                    prevSign = ' ';
                }
                else stack.push(digit);
            } else {
                prevSign = c;
            }
        }
        int result = 0;
        while (!stack.isEmpty()) {
            result += stack.pop();
        }
        return result;
    }

//    https://leetcode.com/problems/range-sum-of-bst/

    static int rangeSum = 0;

    static void rangeSumBSTUtil(TreeNode node, int low, int high) {
        if (node == null) return;
        if (node.val >= low && node.val <= high) {
            rangeSum += node.val;
        }
        if (node.val <= low) {
            rangeSumBSTUtil(node.right, low, high);
        } else if (node.val >=   high) {
            rangeSumBSTUtil(node.left, low, high);
        } else {
            rangeSumBSTUtil(node.left, low, high);
            rangeSumBSTUtil(node.right, low, high);
        }
    }

    public int rangeSumBST(TreeNode root, int low, int high) {
        rangeSumBSTUtil(root, low, high);
        return rangeSum;
    }

//    https://leetcode.com/problems/nested-list-weight-sum/

    static int nestedListSum = 0;

    static void depthSumUtil(List<NestedInteger> nestedList, int depth) {
        for (NestedInteger nestedInteger : nestedList) {
            if (nestedInteger.isInteger()) {
                nestedListSum += nestedInteger.getInteger() * depth;
            } else {
                depthSumUtil(nestedInteger.getList(), depth + 1);
            }
        }
    }

    public static int depthSum(List<NestedInteger> nestedList) {
        depthSumUtil(nestedList, 1);
        return nestedListSum;
    }

//    https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/

    public static Node insert(Node head, int insertVal) {
        if (head == null) {
            Node n = new Node(insertVal);
            n.next = n;
            return n;
        }
        Node prev = null;
        Node curr = null;
        Node greatest = head;
        while (prev != head) {
            if (prev == null) {
                prev = head;
                curr = head.next;
            }
            greatest = prev.val > head.val ? prev : greatest;
            if ((insertVal >= prev.val && insertVal <= curr.val)) {
                prev.next = new Node(insertVal);
                prev.next.next = curr;
                return head;
            }
            prev = curr;
            curr = curr.next;
        }
        Node n = new Node(insertVal);
        Node temp = greatest.next;
        greatest.next = n;
        n.next = temp;
        return head;
    }

//    https://leetcode.com/problems/strobogrammatic-number-ii/

    static void findStrobogrammaticUtil(char [][] mapping, char [] runningStr, int low, int high, List<String> res) {
        if (low > high) {
            if ((runningStr[0] != '0' || runningStr.length == 1))
            res.add(String.valueOf(runningStr));
            return;
        }
        for (char [] map : mapping) {
            if (low == high && (map[0] == '6' || map[0] == '9')) continue;
            runningStr[low] = map[0];
            runningStr[high] = map[1];
            findStrobogrammaticUtil(mapping, runningStr, low + 1, high - 1, res);
        }
    }

    public static List<String> findStrobogrammatic(int n) {
        char [][] mapping = new char[][]{{'0','0'},{'1','1'},{'6','9'},{'8','8'},{'9','6'}};
        List<String> res = new ArrayList<>();
        char [] runningStr = new char[n];
        findStrobogrammaticUtil(mapping, runningStr, 0, n - 1, res);
        return res;
    }

//    https://leetcode.com/problems/reorder-list/

    static ListNode reverseList(ListNode node) {
        if (node == null) return null;
        if (node.next == null) return node;
        ListNode reversed = reverseList(node.next);
        node.next.next = node;
        node.next = null;
        return reversed;
    }

    public static void reorderList(ListNode head) {
        if (head == null) return;
        ListNode slowPointer = head;
        ListNode fastPointer = head;
        while (fastPointer != null && fastPointer.next != null) {
            fastPointer = fastPointer.next.next;
            slowPointer = slowPointer.next;
        }
        if (fastPointer != null) slowPointer = slowPointer.next;
        ListNode reversedNode = reverseList(slowPointer);
        ListNode curr = head;
        while (curr != null && reversedNode != null) {
            ListNode temp = curr.next;
            ListNode reversedTemp = reversedNode.next;
            curr.next = reversedNode;
            reversedNode.next = temp;
            curr = temp;
            reversedNode = reversedTemp;
        }
        if (curr != null) curr.next = null;
    }

//    https://leetcode.com/problems/design-tic-tac-toe/

    static class TicTacToe {

        int [] rowCount;
        int [] columnCount;
        int d1;
        int d2;
        int n;

        public TicTacToe(int n) {
            rowCount = new int[n];
            columnCount = new int[n];
            d1 = 0;
            d2 = 0;
            this.n = n;
        }

        public int move(int row, int col, int player) {
            boolean isPlayer1 = player == 1;
            if (isPlayer1) {
                rowCount[row]++;
                columnCount[col]++;
            } else {
                rowCount[row]--;
                columnCount[col]--;
            }
            if (row == col) {
                if (isPlayer1) d1++;
                else d1--;
            }
            if (row + col == n - 1) {
                if (isPlayer1) d2++;
                else d2--;
            }
            if (Math.abs(rowCount[row]) == n || Math.abs(columnCount[col]) == n
                    || Math.abs(d1) == n || Math.abs(d2) == n) {
                return player;
            }
            return 0;
        }
    }

//    https://leetcode.com/problems/group-shifted-strings/

    public List<List<String>> groupStrings(String[] strings) {
        Map<String,List<String>> mapping = new HashMap<>();
        for (String s : strings) {
            StringBuilder sb = new StringBuilder();
            char checker = s.charAt(0);
            for (char c : s.toCharArray()) {
                sb.append((c - checker + 26) % 26);
                sb.append("-");
            }
            List<String> v = mapping.getOrDefault(sb.toString(),new ArrayList<>());
            v.add(s);
            mapping.put(sb.toString(),v);
        }
        return new ArrayList<>(mapping.values());
    }

//    https://leetcode.com/problems/missing-element-in-sorted-array/

    public static int missingElementUtil(int [] nums, int k, int offset, int low, int high) {
        if (low >= high) {
            int suggestedNo = offset + low;
            int missingNos = nums[low] - suggestedNo;
            return nums[low] + k - missingNos;
        }
        int mid = low + (high - low) / 2;
        int midNo = nums[mid];
        int suggestedNo = offset + mid;
        int missingNos = midNo - suggestedNo;
        if (mid < high && missingNos < k && nums[mid + 1] - (suggestedNo + 1) >= k) {
            return midNo + k - missingNos;
        }
        if (missingNos >= k) {
            return missingElementUtil(nums, k, offset, low, mid - 1);
        } else {
            return missingElementUtil(nums, k, offset, mid + 1, high);
        }
    }

    public static int missingElement(int[] nums, int k) {
        int offSet = nums[0];
        return missingElementUtil(nums, k, offSet, 0, nums.length - 1);
    }

//    https://leetcode.com/problems/island-perimeter/

    static int islandParam;

    static void islandPerimeterUtil(int [][] grid, int i, int j, boolean[][] visited) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == 0) {
            islandParam++;
            return;
        }
        if (visited[i][j]) return;
        visited[i][j] = true;
        islandPerimeterUtil(grid, i - 1, j, visited);
        islandPerimeterUtil(grid, i + 1, j, visited);
        islandPerimeterUtil(grid, i, j - 1, visited);
        islandPerimeterUtil(grid, i, j + 1, visited);
    }

    public int islandPerimeter(int[][] grid) {
        islandParam = 0;
        boolean [][] visited = new boolean[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    islandPerimeterUtil(grid, i, j, visited);
                    break;
                }
            }
        }
        return islandParam;
    }

//    https://leetcode.com/problems/goat-latin/

    public static String toGoatLatin(String sentence) {
        Set<Character> vowels = new HashSet<>();
        vowels.add('a');
        vowels.add('e');
        vowels.add('i');
        vowels.add('o');
        vowels.add('u');
        vowels.add('A');
        vowels.add('E');
        vowels.add('I');
        vowels.add('O');
        vowels.add('U');
        StringBuilder sb = new StringBuilder();
        String [] sArr = sentence.split(" ");
        for (int i = 0; i < sArr.length; i++) {
            String word = sArr[i];
            StringBuilder sb1 = new StringBuilder();
            boolean isVowel = true;
            if (!vowels.contains(word.charAt(0))) {
                sb1.append(word.charAt(0));
                isVowel = false;
            }
            sb1.append("ma");
            for (int j = 1; j <= i + 1; j++) {
                sb1.append("a");
            }
            if (!isVowel) {
                sb.append(word.substring(1)).append(sb1);
            } else {
                sb.append(word).append(sb1);
            }
            if (i != sArr.length - 1) sb.append(" ");
        }
        return sb.toString();
    }

//    https://leetcode.com/problems/check-completeness-of-a-binary-tree/

    public static boolean isCompleteTree(TreeNode root) {
        if (root == null) return true;
        boolean allLevelsFilled = true;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.remove();
            if (!allLevelsFilled && node != null) return false;
            if (node == null) {
                allLevelsFilled = false;
                continue;
            }
            queue.add(node.left);
            queue.add(node.right);
        }
        return true;
    }

//    https://leetcode.com/problems/rearrange-string-k-distance-apart/

    public static void insertionAdd(Deque<Character> queue, char [] words, char c) {
        Deque<Character> tempStack = new ArrayDeque<>();
        int charFreq = words[c - 'a'];
        while (!queue.isEmpty() && words[queue.peekLast() - 'a'] < charFreq) {
            tempStack.push(queue.removeLast());
        }
        queue.addLast(c);
        while (!tempStack.isEmpty()) {
            queue.addLast(tempStack.pop());
        }
    }

    public static String rearrangeString(String s, int k) {
        if (k == 0) return s;
        char [] words = new char[26];
        int size = 0;
        for (char c : s.toCharArray()) {
            if (words[c - 'a'] == 0) size++;
            words[c - 'a']++;
        }
        if (size < k) return "";
        Deque<Character> queue = new ArrayDeque<>();
        for (int i = 0; i < words.length; i++) {
            if (words[i] > 0) {
                char c = (char) (i + 'a');
                insertionAdd(queue, words, c);
            }
        }
        StringBuilder sb = new StringBuilder();
        while (!queue.isEmpty()) {
            List<Character> temp = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                if (!queue.isEmpty()) {
                    char c = queue.removeFirst();
                    sb.append(c);
                    temp.add(c);
                }
            }
            for (char c : temp) {
                int f = words[c - 'a']--;
                if (f > 1) {
                    insertionAdd(queue, words, c);
                }
            }
            size = 0;
            for (char word : words) {
                if (word > 0) size++;
            }
            if (temp.size() < k && size > 0) return "";
        }
        return sb.toString();
    }

//    https://leetcode.com/problems/strobogrammatic-number/

    public static boolean isStrobogrammatic(String num) {
        Map<Character,Character> mapping = new HashMap<>();
        mapping.put('0','0');
        mapping.put('1','1');
        mapping.put('6','9');
        mapping.put('8','8');
        mapping.put('9','6');
        int i = 0;
        int j = num.length() - 1;
        char [] numArr = num.toCharArray();
        while (i <= j) {
            Character stro = mapping.get(numArr[i]);
            if (stro == null) return false;
            else if (i == j && (numArr[i] == '6' || numArr[i] == '9')) return false;
            else if (stro != numArr[j]) return false;
            i++;
            j--;
        }
        return true;
    }

//    https://leetcode.com/problems/custom-sort-string/

    public static String customSortString(String order, String str) {
        int [] orderFreq = new int[26];
        int [] strFreq = new int[26];
        char [] orderArr = order.toCharArray();
        for (char c : orderArr) {
            orderFreq[c - 'a']++;
        }
        for (char c : str.toCharArray()) {
            strFreq[c - 'a']++;
        }
        StringBuilder sb = new StringBuilder();
        for (char c : orderArr) {
            int strLetterF = strFreq[c - 'a'];
            for (int i = 0; i < strLetterF; i++) {
                sb.append(c);
            }
        }
        for (int i = 0; i < strFreq.length; i++) {
            if (strFreq[i] > 0 && orderFreq[i] == 0) {
                int strLetterF = strFreq[i];
                for (int j = 0; j < strLetterF; j++) {
                    sb.append((char) (i + 'a'));
                }
            }
        }
        return sb.toString();
    }

//    https://leetcode.com/problems/intersection-of-two-arrays-ii/

    public static int[] intersect(int[] nums1, int[] nums2) {
        int [] num1Freq = new int[1001];
        int [] num2Freq = new int[1001];
        for (int num : nums1) {
            num1Freq[num]++;
        }
        for (int num : nums2) {
            num2Freq[num]++;
        }
        int [] intersectionArr = new int[1001];
        int j = 0;
        for (int i = 0; i < 1001; i++) {
            int minFreq = Math.min(num1Freq[i],num2Freq[i]);
            for (int k = 0; k < minFreq; k++) {
                intersectionArr[j] = i;
                j++;
            }
        }
        int [] res = new int[j];
        System.arraycopy(intersectionArr, 0, res, 0, j);
        return res;
    }

    static boolean reachTillTarget(TreeNode node, TreeNode target, Deque<TreeNode> parents) {
        if (node == null) return false;
        if (node == target) return true;
        parents.push(node);
        boolean left = reachTillTarget(node.left, target, parents);
        if (left) return true;
        else {
            boolean right = reachTillTarget(node.right, target, parents);
            if (right) return true;
            parents.pop();
        }
        return false;
    }

    public static void processAtDistanceK(TreeNode node, int i, int k, List<Integer> res) {
        if (i > k || node == null) return;
        if (i == k) {
            res.add(node.val);
            return;
        }
        processAtDistanceK(node.left,i + 1, k, res);
        processAtDistanceK(node.right, i + 1, k, res);
    }

//    https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/

    public static List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        List<Integer> result = new ArrayList<>();
        processAtDistanceK(target,0, k, result);
        Deque<TreeNode> parents = new ArrayDeque<>();
        reachTillTarget(root, target, parents);
        int i = 1;
        TreeNode child = target;
        while (!parents.isEmpty()) {
            TreeNode parent = parents.pop();
            if (i == k) {
                result.add(parent.val);
                break;
            }
            if (child == parent.left) {
                processAtDistanceK(parent.right, i + 1, k, result);
            } else {
                processAtDistanceK(parent.left, i + 1, k, result);
            }
            child = parent;
            i++;
        }
        return result;
    }

//    https://leetcode.com/problems/battleships-in-a-board/

    public static void dfsBattleShip(char [][] board, int i, int j, boolean [][] visited) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || board[i][j] == '.' || visited[i][j]) {
            return;
        }
        visited[i][j] = true;
        dfsBattleShip(board, i + 1, j, visited);
        dfsBattleShip(board, i - 1, j, visited);
        dfsBattleShip(board, i, j + 1, visited);
        dfsBattleShip(board, i, j - 1, visited);
    }

    public static void markForward(char [][] board, int i, int j, boolean [][] visited) {
        while (i < board.length && board[i][j] == 'X' && !visited[i][j]) {
            visited[i][j] = true;
            i++;
        }
    }

    public static void markDownward(char [][] board, int i, int j, boolean [][] visited) {
        while (j < board[i].length && board[i][j] == 'X' && !visited[i][j]) {
            visited[i][j] = true;
            j++;
        }
    }

    public static int countBattleships(char[][] board) {
        int count = 0;
        boolean [][] visited = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == 'X' && !visited[i][j]) {
                    visited[i][j] = true;
                    if (i + 1 < board.length && board[i + 1][j] == 'X') {
                        markForward(board, i + 1, j, visited);
                    } else if (j + 1 < board[i].length && board[i][j + 1] == 'X') {
                        markDownward(board, i, j + 1, visited);
                    }
                    count++;
                }
            }
        }
        return count;
    }

//    https://leetcode.com/problems/serialize-and-deserialize-bst/

    public static class Codec {

        static void serializeTree(TreeNode node, StringBuilder sb) {
            if (node == null) {
                sb.append("-");
                sb.append(".");
                return;
            }
            sb.append(node.val);
            sb.append(".");
            serializeTree(node.left, sb);
            serializeTree(node.right, sb);
        }

        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            serializeTree(root, sb);
            return sb.toString();
        }

        static TreeNode deserializeTree(String [] arr, int [] i) {
            i[0]++;
            String s = arr[i[0]];
            if (s.equals("-")) return null;
            TreeNode node = new TreeNode(Integer.parseInt(s));
            node.left = deserializeTree(arr, i);
            node.right = deserializeTree(arr, i);
            return node;
        }

        public TreeNode deserialize(String data) {
            String [] dataArr = data.split("\\.");
            int [] i = new int[0];
            return deserializeTree(dataArr, i);
        }

    }

//    https://leetcode.com/problems/buildings-with-an-ocean-view/

    public static int[] findBuildings(int[] heights) {
        int tallestSoFar = -1;
        int count = 0;
        for (int i = heights.length - 1; i >= 0; i--) {
            int h = heights[i];
            if (h > tallestSoFar) {
                tallestSoFar = h;
                heights[i] = -h;
                count++;
            }
        }
        int [] res = new int[count];
        int j = 0;
        for (int i = 0; i < heights.length; i++) {
            if (heights[i] < 0) {
                res[j] = i;
                j++;
            }
        }
        return res;
    }

//    https://leetcode.com/problems/diagonal-traverse/

    public int[] findDiagonalOrder(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int r = 0;
        int c = 0;
        int [] res = new int[m * n];
        for (int i = 0; i < res.length; i++) {
            res[i] = mat[r][c];
            if ((r + c) % 2 == 0) {
                if (c == n - 1) r++;
                else if (r == 0) c++;
                else {
                    r--;
                    c++;
                }
            } else {
                if (r == m - 1) c++;
                else if (c == 0) r++;
                else {
                    r++;
                    c--;
                }
            }
        }
        return res;
     }

//    https://leetcode.com/problems/inorder-successor-in-bst/

    public static TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
         TreeNode successor = null;
         while (root != null) {
             if (root.val <= p.val) {
                 root = root.right;
             } else {
                 successor = root;
                 root = root.left;
             }
         }
         return successor;
    }

//    https://leetcode.com/problems/longest-valid-parentheses/

    public static int longestValidParentheses(String s) {
        char [] sArr = s.toCharArray();
        Deque<Integer> stack = new ArrayDeque<>();
        int maxValid = 0;
        stack.push(-1);
        for (int i = 0; i < sArr.length; i++) {
            char c = sArr[i];
            if (c == '(') stack.push(i);
            else {
                stack.pop();
                if (stack.isEmpty()) stack.push(i);
                else maxValid = Math.max(maxValid,i - stack.peek());
            }
        }
        return maxValid;
    }

//    https://leetcode.com/problems/minimum-knight-moves/

    public static int minKnightMoves(int x, int y) {
        if (x == 0 && y == 0) return 0;
        int [][] moves = new int[][]{{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
        int [][] board = new int[602][602];
        int [][] visited = new int[602][602];
        int x0 = 300;
        int y0 = 300;
        x = x + 300;
        y = y + 300;
        visited[x0][y0] = 1;
        visited[x][y] = 2;
        Deque<int []> queue = new ArrayDeque<>();
        queue.addLast(new int[]{x0,y0});
        queue.addLast(new int[]{x,y});
        while (!queue.isEmpty()) {
            int n = queue.size();
            for (int i = 0; i < n; i++) {
                int[] node = queue.removeFirst();
                int x1 = node[0];
                int y1 = node[1];
                int visitedValue = visited[x1][y1];
                for (int [] move : moves) {
                    int x2 = x1 + move[0];
                    int y2 = y1 + move[1];
                    if (((x2 == x && y2 == y) || (x2 == x0 && y2 == y0)) && visited[x2][y2] != visitedValue) return board[x1][y1] + 1;
                    if (x2 < 0 || x2 >= board.length || y2 < 0 || y2 >= board[x2].length ||
                            visited[x2][y2] == visitedValue) continue;
                    queue.addLast(new int[]{x2,y2});
                    if (board[x2][y2] > 0) return board[x2][y2] + board[x1][y1] + 1;
                    board[x2][y2] = board[x1][y1] + 1;
                    visited[x2][y2] = visitedValue == 1 ? 1 : 2;
                }
            }
        }
        return -1;
    }

}
