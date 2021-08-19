package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.TreeNode;

import java.util.*;

public class FirstSet {

    public static void main(String[] args) {
//        System.out.println(removeInvalidParentheses("((()((s((((()"));
//        System.out.println(addBinary("11","1"));
//        System.out.println(minRemoveToMakeValid("(a(b(c)d)"));
//        System.out.println(isAlienSorted(new String[]{"hello","leetcode"},"hlabcdefgijkmnopqrstuvwxyz"));
//        System.out.println(read(new char[10],3));
//        System.out.println(validPalindrome("aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga"));
//        System.out.println(leastInterval(new char[]{'A','A','A','A','A','A','B','C','D','E','F','G'}, 2));
//        System.out.println(isMatch("aa","a"));
//        System.out.println(numDecodings("12"));
//        System.out.println(addOperators("123456789",45));
//        System.out.println(divide(-2147483648,-2147483648));
//        WordDictionary ws = new WordDictionary();
//        ws.addWord("bad");
//        System.out.println(isPalindrome("A man, a plan, a canal: Panama"));
//        System.out.println(alienOrder(new String[]{"qb","qts","qs","qa","s"}));
//        System.out.println(addStrings("0","0"));
//        System.out.println(minMeetingRooms(new int[][]{{0,30},{5,10},{15,20}}));
        System.out.println(findAnagrams("abab","ab"));
    }

//    https://leetcode.com/problems/remove-invalid-parentheses/

    public static int getMinInvalid(String s) {
        Deque<Character> stack = new ArrayDeque<>();
        for (char c : s.toCharArray()) {
            if (c == '(') {
                stack.push(c);
            } else if (c == ')') {
                if (!stack.isEmpty() && stack.peek() == '(') {
                    stack.pop();
                } else {
                    stack.push(c);
                }
            }
        }
        return stack.size();
    }

    public static void removeInvalidParenthesesUtil(List<String> result, Set<String> hs,
                                                    int minInvalid, String s) {
        if (minInvalid == 0) {
            if (getMinInvalid(s) == 0) {
                result.add(s);
            }
            return;
        }
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(' || s.charAt(i) == ')') {
                String left = s.substring(0, i);
                String right = s.substring(i + 1);
                String sAfterRemoval = left + right;
                if (!hs.contains(sAfterRemoval)) {
                    removeInvalidParenthesesUtil(result, hs, minInvalid - 1, left + right);
                    hs.add(sAfterRemoval);
                }
            }
        }
    }


    public static List<String> removeInvalidParentheses(String s) {
        List<String> result = new ArrayList<>();
        Set<String> hs = new HashSet<>();
        int minInvalid = getMinInvalid(s);
        removeInvalidParenthesesUtil(result, hs, minInvalid, s);
        return result;
    }

//    https://leetcode.com/problems/add-binary/

    public static String addBinary(String a, String b) {
        StringBuilder res = new StringBuilder();
        int i = a.length() - 1;
        int j = b.length() - 1;
        int carryOver = 0;
        while (i >= 0 || j >= 0) {
            int aa = (i >= 0 && a.charAt(i) == '1') ? 1 : 0;
            int bb = (j >= 0 && b.charAt(j) == '1') ? 1 : 0;
            int sum = aa + bb + carryOver;
            res.append(sum % 2);
            carryOver = sum / 2;
            i--;
            j--;
        }
        if (carryOver > 0) res.append(carryOver);
        return res.reverse().toString();
    }

//    https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses

    public static String minRemoveToMakeValid(String s) {
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else if (s.charAt(i) == ')') {
                if (!stack.isEmpty() && s.charAt(stack.peek()) == '(') {
                    stack.pop();
                } else {
                    stack.push(i);
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            if (!stack.isEmpty() && i == stack.peekLast()) {
                stack.removeLast();
                continue;
            }
            sb.append(s.charAt(i));
        }
        return sb.toString();
    }

//    https://leetcode.com/problems/verifying-an-alien-dictionary/

    public static boolean compare(String w1, String w2, Map<Character,Integer> ordering) {
        int i = 0;
        int j = 0;
        while (i < w1.length() && j < w2.length()) {
            int diff = ordering.get(w1.charAt(i)) - ordering.get(w2.charAt(j));
            if (diff < 0) return true;
            else if (diff > 0) return false;
            i++;
            j++;
        }
        return w1.length() - w2.length() <= 0;
    }

    public static boolean isAlienSorted(String[] words, String order) {
        Map<Character,Integer> ordering = new HashMap<>();
        for (int i = 0; i < order.length(); i++) {
            ordering.put(order.charAt(i), i);
        }
        for (int i = 1; i < words.length; i++) {
            boolean c = compare(words[i - 1], words[i], ordering);
            if (!c) return false;
        }
        return true;
    }

//    https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/

    public static int read4(char[] buf4) {
        buf4[0] = 'a';
        buf4[1] = 'b';
        buf4[2] = 'c';
        return 3;
    }

    static Deque<Character> queue = new ArrayDeque<>();
    public static int read(char[] buf, int n) {
        int i = 0;

        while (!queue.isEmpty() && n > 0) {
            buf[i] = queue.removeFirst();
            i++;
            n--;
        }
        while (n > 0) {
            char [] buf4 = new char[4];
            int j = read4(buf4);
            if (j == 0) return i;
            int k = 0;
            while (n > 0 && k < j) {
                buf[i] = buf4[k];
                k++;
                i++;
                n--;
            }
            while (k < j) {
                queue.addLast(buf4[k]);
                k++;
            }
        }
        return i;
    }

//    https://leetcode.com/problems/valid-palindrome-ii/

    public static boolean validPalindromeUtil(String s, int i, int j, boolean flag) {
        if (i >= j) return true;
        else if (s.charAt(i) == s.charAt(j)) return validPalindromeUtil(s, i + 1, j - 1, flag);
        else if (!flag) {
            return validPalindromeUtil(s, i + 1, j, true) || validPalindromeUtil(s, i, j - 1, true);
        } else {
            return false;
        }
    }

    public static boolean validPalindrome(String s) {
        return validPalindromeUtil(s, 0, s.length() - 1, false);
    }

//    https://leetcode.com/problems/serialize-and-deserialize-binary-tree/

    public static class Codec {

        void serializeUtil(TreeNode node, StringBuilder sb) {
            if (node == null) {
                sb.append(".,");
                return;
            }
            sb.append(node.val);
            sb.append(",");
            serializeUtil(node.left, sb);
            serializeUtil(node.right, sb);
        }

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            serializeUtil(root, sb);
            return sb.toString();
        }

        TreeNode deserializeUtil(Deque<String> queue) {
            if (queue.isEmpty()) return null;
            String s = queue.removeFirst();
            if (s.equals(".")) return null;
            TreeNode node = new TreeNode(Integer.parseInt(s));
            node.left = deserializeUtil(queue);
            node.right = deserializeUtil(queue);
            return node;
        }

        static Deque<String> queue = new ArrayDeque<>();

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
        for (String s : data.split(",")) {
            queue.addLast(s);
        }
        return deserializeUtil(queue);
        }

    }

//    https://leetcode.com/problems/task-scheduler/

    public static int leastInterval(char[] tasks, int n) {
        Map<Character,Integer> mapping = new HashMap<>();
        for (char c : tasks) {
            mapping.merge(c, 1, Integer::sum);
        }
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a,b) -> b - a);
        maxHeap.addAll(mapping.values());
        int time = 0;
        while (!maxHeap.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            for (int i = 0; i < n + 1; i++) {
                if (!maxHeap.isEmpty()) {
                    temp.add(maxHeap.remove());
                }
            }
            for (int freq : temp) {
                if (--freq > 0) maxHeap.add(freq);
            }
            time += maxHeap.isEmpty() ? temp.size() : n + 1;
        }
        return time;
    }

//    https://leetcode.com/problems/regular-expression-matching/

    public static boolean isMatchUtilsDP(String s, String p) {
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        dp[0][0] = true;
        for (int i = 1; i < dp[0].length; i++) {
           if (p.charAt(i - 1) == '*') dp[0][i] = dp[0][i - 2];
        }
        for (int i = 1; i < dp.length; i++) {
           for (int j = 1; j < dp[0].length; j++) {
               if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                   dp[i][j] = dp[i - 1][j - 1];
               } else if (p.charAt(j - 1) == '*') {
                   dp[i][j] = dp[i][j - 2];
                   if (s.charAt(i - 1) == p.charAt(j - 2) || p.charAt(j - 2) == '.') {
                       dp[i][j] = dp[i][j] | dp[i - 1][j];
                   }
               } else dp[i][j] = false;
           }
       }
       return dp[dp.length - 1][dp[0].length - 1];
    }


    public static boolean isMatchUtils(String s, String p, int i, int j) {
        if (i < 0 && j < 0) return true;
        else if (i >= 0 && j >= 0 && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.')) return isMatchUtils(s, p, i - 1, j - 1);
        else if (j >= 0 && p.charAt(j) == '*') {
            if (isMatchUtils(s, p, i, j - 2)) return true;
            else if (i >= 0 && j > 0 && (s.charAt(i) == p.charAt(j - 1) || p.charAt(j - 1) == '.'))
                return isMatchUtils(s, p, i - 1, j);
        }
        return false;
    }

    public static boolean isMatch(String s, String p) {
//        return isMatchUtils(s, p, s.length() - 1, p.length() - 1);
        return isMatchUtilsDP(s,p);
    }

//    https://leetcode.com/problems/first-bad-version/

    static boolean isBadVersion(int version) {
        return true;
    }

    static int firstBad = -1;

    public static int binarySearchBV(int low, int high) {
        if (low > high) return firstBad;
        int mid = low + (high - low) / 2;
        boolean isBad = isBadVersion(mid);
        if (isBad) {
            firstBad = mid;
            return binarySearchBV(low, mid - 1);
        } else {
            return binarySearchBV(mid + 1, high);
        }
    }

    public int firstBadVersion(int n) {
        return binarySearchBV(0, n);
    }

//    https://leetcode.com/problems/decode-ways/

    static int numDecodingsCount = 0;

    static void numDecodingsUtil(Set<String> hs, String s, int index) {
        if (index == s.length()) {
            numDecodingsCount++;
            return;
        }
        StringBuilder no = new StringBuilder();
        for (int i = index; i < index + 2 && i < s.length(); i++) {
            no.append(s.charAt(i) - '0');
            if (hs.contains(no.toString())) {
                numDecodingsUtil(hs, s, i + 1);
            } else return;
        }
    }

    public static int numDecodings(String s) {
        int [] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        for (int i = 2; i < dp.length; i++) {
            int oneDigit = Integer.parseInt(s.substring(i - 1, i));
            int twoDigit = Integer.parseInt(s.substring(i - 2, i));
            if (oneDigit != 0) {
                dp[i] += dp[i - 1];
            }
            if (twoDigit >= 10 && twoDigit <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[dp.length - 1];
    }

//    https://leetcode.com/problems/binary-search-tree-iterator/

    static class BSTIterator {

        List<Integer> inorderTraversalList;
        int i;

        public BSTIterator(TreeNode root) {
            this.inorderTraversalList = new ArrayList<>();
            inorderTraversal(root, inorderTraversalList);
            this.i = -1;
        }

        void inorderTraversal(TreeNode root, List<Integer> inOrderList) {
            if (root == null) return;
            inorderTraversal(root.left, inOrderList);
            inOrderList.add(root.val);
            inorderTraversal(root.right, inOrderList);
        }

        public int next() {
            this.i++;
            return this.inorderTraversalList.get(i);
        }

        public boolean hasNext() {
            return this.i < this.inorderTraversalList.size() - 1;
        }
    }

//  https://leetcode.com/problems/expression-add-operators/

    static void addOperatorsUtil(String num, int index, long prev, long curr, long value,
                                 int target, String sb, List<String> result) {
      if (index == num.length()) {
           if (target == value && curr == 0) {
               result.add(sb);
           }
          return;
      }
      curr = curr * 10 + num.charAt(index) - '0';

      if (curr > 0) {
          addOperatorsUtil(num, index + 1, prev, curr, value, target, sb, result);
      }

      if (sb.isEmpty()) {
          addOperatorsUtil(num, index + 1, curr, 0, value + curr, target, sb + curr, result);
          return;
      }
      addOperatorsUtil(num, index + 1, curr, 0, value + curr, target, sb + "+" + curr, result);
      addOperatorsUtil(num, index + 1, -curr, 0, value - curr, target, sb + "-" + curr, result);
      addOperatorsUtil(num, index + 1,prev * curr, 0, value - prev + prev * curr, target, sb + "*" + curr, result);
    }

    public static List<String> addOperators(String num, int target) {
        List<String> result = new LinkedList<>();
        addOperatorsUtil(num, 0, 0, 0, 0, target, "", result);
        return result;
    }

//    https://leetcode.com/problems/divide-two-integers/

    public static int divide(int dividend, int divisor) {
        long absDividend = Math.abs((long) dividend);
        long absDivisor = Math.abs((long) divisor);
        long sign = 1;
        if (((dividend < 0 && divisor > 0) || (dividend > 0 && divisor < 0))) {
            sign = -1;
        }
        if (absDividend < absDivisor) return 0;
        long quotient = 1;
        while (absDivisor < absDividend) {
            absDivisor += absDivisor;
            quotient *= 2;
        }
        int smallestDivisor = Math.abs(divisor);
        while (absDividend < absDivisor) {
            absDivisor -= smallestDivisor;
            quotient--;
        }
        long finalQ = quotient * sign;
        if (finalQ >= Integer.MAX_VALUE || finalQ <= Integer.MIN_VALUE) {
            return finalQ >= Integer.MAX_VALUE ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        }
        return (int) finalQ;
    }

//    https://leetcode.com/problems/design-add-and-search-words-data-structure/

    static class TrieNode {
        char c;
        boolean isWordEnd;
        Map<Character,TrieNode> children;

        TrieNode(char c) {
            this.c = c;
            this.children = new HashMap<>();
        }
    }

    static class WordDictionary {

        TrieNode root;

        public WordDictionary() {
            this.root = new TrieNode('*');
        }

        static void addWordUtil(TrieNode node, String word, int index) {
            if (index < word.length()) {
                TrieNode child = node.children.get(word.charAt(index));
                if (child == null) {
                    char c = word.charAt(index);
                    child = new TrieNode(c);
                    node.children.put(c, child);
                }
                if (index != word.length() - 1) addWordUtil(child, word,index + 1);
                else child.isWordEnd = true;
            }
        }

        public void addWord(String word) {
           addWordUtil(this.root, word, 0);
            System.out.println();
        }

        boolean wordSearchUtil(TrieNode node, String word, int index) {
            if (index < word.length()) {
                TrieNode child = node.children.get(word.charAt(index));
                if (child != null) {
                    if (index < word.length() - 1) {
                        return wordSearchUtil(child,word, index + 1);
                    } else {
                        return child.isWordEnd;
                    }
                } else if (word.charAt(index) == '.' && index < word.length() - 1) {
                    for (TrieNode ch : node.children.values()) {
                        if (wordSearchUtil(ch, word, index + 1)) {
                            return true;
                        }
                    }
                    return false;
                } else if (word.charAt(index) == '.' && index == word.length() - 1) {
                    for (TrieNode ch : node.children.values()) {
                        if (ch.isWordEnd) {
                            return true;
                        }
                    }
                    return false;
                }
                return false;
            }
            return false;
        }

        public boolean search(String word) {
            return wordSearchUtil(this.root, word, 0);
        }

    }

//    https://leetcode.com/problems/valid-palindrome/

    public static boolean isPalindrome(String s) {
        int i = 0;
        int j = s.length() - 1;
        while (i < j) {
            char ci = Character.toLowerCase(s.charAt(i));
            char cj = Character.toLowerCase(s.charAt(j));
            boolean ciB = (Character.isAlphabetic(ci) || Character.isDigit(ci));
            boolean cjB = (Character.isAlphabetic(cj) || Character.isDigit(cj));
            if (ciB && cjB) {
                if (ci == cj) {
                    i++;
                    j--;
                } else {
                    return false;
                }
            } else {
                if (!ciB) {
                    i++;
                } else {
                    j--;
                }
            }
        }
        return true;
    }

//    https://leetcode.com/problems/alien-dictionary/

    public static String alienOrder(String[] words) {
        Map<Character,Set<Character>> adj = new HashMap<>();
        Set<Character> all = new HashSet<>();
        Map<Character,Integer> degree = new HashMap<>();
        for (String word : words) {
            for (int i = 0; i < word.length(); i++) {
                all.add(word.charAt(i));
            }
        }
        for (int i = 1; i < words.length; i++) {
            String prev = words[i - 1];
            String curr = words[i];
            int j = 0;
            int k = 0;
            boolean diffFound = false;
            while (j < prev.length() && k < curr.length()) {
                char jth = prev.charAt(j);
                char kth = curr.charAt(k);
                if (prev.charAt(j) == curr.charAt(k)) {
                    j++;
                    k++;
                    continue;
                }
                diffFound = true;
                if (!adj.getOrDefault(jth, new HashSet<>()).contains(kth)) {
                    Set<Character> neighbours = adj.getOrDefault(jth, new HashSet<>());
                    neighbours.add(kth);
                    adj.put(jth, neighbours);
                    degree.merge(kth, 1, Integer::sum);
                }
                break;
            }
            if (!diffFound && prev.length() > curr.length()) return "";
        }
        Deque<Character> queue = new ArrayDeque<>();
        for (char c : all) {
            if (degree.get(c) == null) {
                queue.addLast(c);
            }
        }
        StringBuilder sb = new StringBuilder();
        while (!queue.isEmpty()) {
            char c = queue.removeFirst();
            sb.append(c);
            for (char neighbour : adj.getOrDefault(c, new HashSet<>())) {
                degree.put(neighbour, degree.get(neighbour) - 1);
                if (degree.get(neighbour) == 0) {
                    queue.addLast(neighbour);
                }
            }
        }
        return all.size() == sb.length() ? sb.toString() : "";
    }

//    https://leetcode.com/problems/add-strings/

    public static String addStrings(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int i = num1.length() - 1;
        int j = num2.length() - 1;
        int carryOver = 0;
        while (i >= 0 && j >= 0) {
            int first = num1.charAt(i) - '0';
            int second = num2.charAt(j) - '0';
            int sum = first + second + carryOver;
            sb.insert(0, sum % 10);
            carryOver = sum / 10;
            i--;
            j--;
        }
        while (i >= 0) {
            int first = num1.charAt(i) - '0';
            int sum = first + carryOver;
            sb.insert(0, sum % 10);
            carryOver = sum / 10;
            i--;
        }
        while (j >= 0) {
            int second = num2.charAt(j) - '0';
            int sum = second + carryOver;
            sb.insert(0, sum % 10);
            carryOver = sum / 10;
            j--;
        }
        if (carryOver > 0) sb.insert(0, carryOver);
        return sb.toString();
    }

//    https://leetcode.com/problems/meeting-rooms-ii/

    public static int minMeetingRooms(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        PriorityQueue<int []> minHeap = new PriorityQueue<>((a,b) -> (a[1] != b[1] ? a[1] - b[1] : a[0] - b[0]));
        int maxCount = 0;
        for (int [] interval : intervals) {
            while (!minHeap.isEmpty() && interval[0] >= minHeap.peek()[1]) {
                minHeap.remove();
            }
            minHeap.add(interval);
            maxCount = Math.max(maxCount, minHeap.size());
        }
        return maxCount;
    }

//    https://leetcode.com/problems/binary-tree-vertical-order-traversal/

    public static List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Map<Integer,List<Integer>> hm = new HashMap<>();
        Deque<TreeNode> queue = new ArrayDeque<>();
        Map<TreeNode,Integer> mapping = new HashMap<>();
        int minKey = Integer.MAX_VALUE;
        int maxKey = Integer.MIN_VALUE;
        mapping.put(root, 0);
        queue.addLast(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.removeFirst();
            int shift = mapping.get(node);
            minKey = Math.min(minKey, shift);
            maxKey = Math.max(maxKey, shift);
            List<Integer> level = hm.getOrDefault(shift, new ArrayList<>());
            level.add(node.val);
            hm.put(shift, level);
            if (node.left != null) {
                queue.addLast(node.left);
                mapping.put(node.left, shift - 1);
            }
            if (node.right != null) {
                queue.addLast(node.right);
                mapping.put(node.right, shift + 1);
            }
        }
        for (int i = minKey; i <= maxKey; i++) {
            result.add(hm.get(i));
        }
        return result;
    }

//    https://leetcode.com/problems/leftmost-column-with-at-least-a-one/

    interface BinaryMatrix {
        int get(int row, int col);
        List<Integer> dimensions();
    }

    public int binarySearchFloorBin(int row, int leftCol, int rightCol, BinaryMatrix binaryMatrix) {
        int leftMost = Integer.MAX_VALUE;
        while (leftCol <= rightCol) {
            int mid = leftCol + (rightCol - leftCol) / 2;
            int check = binaryMatrix.get(row, mid);
            if (check == 0) {
                leftCol = mid + 1;
            } else {
                leftMost = mid;
                rightCol = mid - 1;
            }
        }
        return leftMost;
    }

    public int leftMostColumnWithOne(BinaryMatrix binaryMatrix) {
        List<Integer> dimension = binaryMatrix.dimensions();
        int row = dimension.get(0);
        int column = dimension.get(1);
        int result = Integer.MAX_VALUE;
        int rightLength = column - 1;
        for (int r = 0; r < row; r++) {
            int firstLeft = binarySearchFloorBin(r, 0, rightLength, binaryMatrix);
            if (firstLeft == 0) return firstLeft;
            result = Math.min(result, firstLeft);
            rightLength = Math.min(rightLength, firstLeft);
        }
        return result == Integer.MAX_VALUE ? -1 : result;
    }

//    https://leetcode.com/problems/sparse-matrix-multiplication/

    public static int[][] multiply(int[][] mat1, int[][] mat2) {
        int [][] result = new int[mat1.length][mat2[0].length];
        for (int i = 0; i < mat1.length; i++) {
            for (int j = 0; j < mat1[0].length; j++) {
                int m1Value = mat1[i][j];
                if (m1Value == 0) continue;
                for (int k = 0; k < result[i].length; k++) {
                    result[i][k] += m1Value * mat2[j][k];
                }
            }
        }
        return result;
    }

    public int[][] multiply2(int[][] mat1, int[][] mat2) {
        Map<Integer,Map<Integer,Integer>> m1 = new HashMap<>();
        for (int i = 0 ; i < mat1.length; i++) {
            Map<Integer,Integer> cMap = new HashMap<>();
            for (int j = 0 ; j < mat1[i].length; j++) {
                if (mat1[i][j] != 0) cMap.put(j, mat1[i][j]);
            }
            if (cMap.size() > 0) m1.put(i, cMap);
        }

        Map<Integer,Map<Integer,Integer>> m2 = new HashMap<>();
        for (int j = 0 ; j < mat2[0].length; j++) {
            Map<Integer,Integer> rMap = new HashMap<>();
            for (int i = 0 ; i < mat2.length; i++) {
                if (mat2[i][j] != 0) rMap.put(i, mat2[i][j]);
            }
            if (rMap.size() > 0) m2.put(j, rMap);
        }
        System.out.println(m1);
        System.out.println(m2);
        int [][] result = new int[mat1.length][mat2[0].length];
        for (int m1Row : m1.keySet()) {
            Map<Integer,Integer> m1RowCol = m1.get(m1Row);
           for (int m2Col : m2.keySet()) {
               Map<Integer,Integer> m2ColRow = m2.get(m2Col);
               for (int c : m1RowCol.keySet()) {
                   result[m1Row][m2Col] += m1RowCol.get(c) * m2ColRow.getOrDefault(c, 0);
               }
           }
        }
        return result;
    }

//    https://leetcode.com/problems/read-n-characters-given-read4/

    public static int read1(char[] buf, int n) {
        int i = 0;
        while (n > 0) {
            char [] buff4 = new char[4];
            int r4 = read4(buff4);
            if (r4 == 0) return i;
            for (int j = 0; j < r4; j++) {
                if (n == 0) return i;
                buf[i] = buff4[j];
                i++;
                n--;
            }
        }
        return i;
    }

//    https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/

    static Node head = null;
    static Node prev = null;

    public static void treeToDoublyListUtil(Node node) {
        if (node == null) return;
        treeToDoublyListUtil(node.left);
        if (head == null) {
            head = node;
            prev = node;
        } else {
            prev.right = node;
            node.left = prev;
            prev = node;
        }
        treeToDoublyListUtil(node.right);
    }

    public static Node treeToDoublyList(Node root) {
        if (root != null) {
            treeToDoublyListUtil(root);
            prev.right = head;
            head.left = prev;
            return head;
        } else {
            return root;
        }
    }

//    https://leetcode.com/problems/find-all-anagrams-in-a-string/

    public static List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new ArrayList<>();
        int pLength = p.length();
        int [] pArr = new int[26];
        for (int i = 0 ; i < p.length(); i++) {
            pArr[p.charAt(i) - 'a']++;
        }
        int [] sArr = new int[26];
        for (int i = 0; i < s.length(); i++) {
            sArr[s.charAt(i) - 'a']++;
            if (i >= pLength) {
                sArr[i - pLength]--;
            }
            if (Arrays.equals(sArr,pArr)) result.add(i - pLength + 1);
        }
        return result;
    }

    static class Node {
        public int val;
        public Node left;
        public Node right;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val,Node _left,Node _right) {
            val = _val;
            left = _left;
            right = _right;
        }
    }

}
