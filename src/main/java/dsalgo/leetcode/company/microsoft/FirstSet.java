package dsalgo.leetcode.company.microsoft;

import dsalgo.leetcode.Models.*;

import java.util.*;

public class FirstSet {

    public static void main(String[] args) {
//        System.out.println(reverseWords("a good   example"));
//        System.out.println(maxLength(Arrays.asList("un", "iq", "ue")));
//        TreeNode root = new TreeNode(1);
//        root.left = new TreeNode(2);
//        root.left.left = new TreeNode(4);
//        root.left.right = new TreeNode(5);
//        root.left.right.left = new TreeNode(7);
//        root.left.right.right = new TreeNode(8);
//        root.right = new TreeNode(3);
//        root.right.left = new TreeNode(6);
//        root.right.left.left = new TreeNode(9);
//        root.right.left.right = new TreeNode(10);
//        System.out.println(boundaryOfBinaryTree(root));
//        System.out.println(removeComments(new String[]{"/*Test program */"}));
//        System.out.println(firstMissingPositive(new int[]{3,4,-1,1}));
//        System.out.println(minCost("cddcdcae", new int[]{4, 8, 8, 4, 4, 5, 4, 2}));
//        System.out.println(minReorderUnoptimised(5, new int[][]{{1, 0}, {1, 2}, {3, 2}, {3, 4}}));
//        System.out.println(minReorder(5, new int[][]{{1, 0}, {1, 2}, {3, 2}, {3, 4}}));
//        System.out.println(modifyString("?????"));
//        Codec codec = new Codec();
//        Node root = new Node(1);
//        root.children = Arrays.asList(new Node(2), new Node(3), new Node(4), new Node(5));
//        System.out.println(codec.serialize(root));
//        System.out.println(minDeletions("ceabaacb"));
//        System.out.println(longestPalindrome("a"));
//        System.out.println(letterCombinationsUnoptimised("23"));
//        System.out.println(numWays("aa"));
//        System.out.println(getSkyline(new int[][]{{2,9,10},{3,7,15},{5,12,12},{15,20,10},{19,24,8}}));
//        Node root = new Node(1);
//        root.left = new Node(2);
//        root.right = new Node(3);
//        root.left.left = new Node(4);
//        root.left.right = new Node(5);
//        root.right.left = new Node(6);
//        root.right.right = new Node(7);
//        System.out.println(connect(root));
//        ListNode l1 = new ListNode(7);
//        l1.next = new ListNode(2);
//        l1.next.next = new ListNode(4);
//        l1.next.next.next = new ListNode(3);
//        ListNode l2 = new ListNode(5);
//        l2.next = new ListNode(6);
//        l2.next.next = new ListNode(4);
//        ListNode l3 = addTwoNumbers(l1,l2);
//        System.out.println(l3);
//        System.out.println(multiply("6666666666666666666666666","5555555555555555555555555"));
//        spiralMatrix(new int[][]{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}});
//        System.out.println(maxRepOpt1("abcdef"));
        System.out.println(validIPAddress("2001:0db8:85a3:0:0:8A2E:0370:7334"));
    }

//    https://leetcode.com/problems/reverse-words-in-a-string/

    public static String reverseWords(String s) {
        StringBuilder sentence = new StringBuilder();
        String[] sArr = s.split(" ");
        for (int i = sArr.length - 1; i >= 0; i--) {
            if (!sArr[i].isEmpty()) {
                sentence.append(sArr[i]);
                sentence.append(" ");
            }
        }
        return sentence.toString().trim();
    }

//    https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/

    public static int maxLengthUnique(String str) {
        int[] countArr = new int[26];
        for (char c : str.toCharArray()) {
            if (countArr[c - 'a'] > 0) {
                return -1;
            } else {
                countArr[c - 'a']++;
            }
        }
        return str.length();
    }

    public static void maxLengthUtil(List<String> arr, String curr, int index, int[] maxLength) {
        if (index == arr.size() && maxLengthUnique(curr) > maxLength[0]) {
            System.out.println(curr);
            maxLength[0] = curr.length();
            return;
        }
        if (index == arr.size()) {
            System.out.println(curr);
            return;
        }
        maxLengthUtil(arr, curr, index + 1, maxLength);
        maxLengthUtil(arr, curr + arr.get(index), index + 1, maxLength);
    }

    public static int maxLength(List<String> arr) {
        int[] maxLength = new int[]{Integer.MIN_VALUE};
        maxLengthUtil(arr, "", 0, maxLength);
        return maxLength[0];
    }

//    https://leetcode.com/problems/boundary-of-binary-tree/

    public static void leftBoundary(TreeNode node, List<Integer> result) {
        if (node == null) {
            return;
        }
        if (node.left != null || node.right != null) {
            result.add(node.val);
        }
        if (node.left != null) {
            leftBoundary(node.left, result);
        } else if (node.right != null) {
            leftBoundary(node.right, result);
        }
    }

    public static void bottomView(TreeNode node, List<Integer> result) {
        if (node == null) {
            return;
        }
        if (node.left == null && node.right == null) {
            result.add(node.val);
        }
        bottomView(node.left, result);
        bottomView(node.right, result);
    }

    public static void rightBoundary(TreeNode node, List<Integer> result) {
        if (node == null) {
            return;
        }
        boolean isRightMost = false;
        if (node.left != null || node.right != null) {
            isRightMost = true;
        }

        if (node.right != null) {
            rightBoundary(node.right, result);
        } else if (node.left != null) {
            rightBoundary(node.left, result);
        }
        if (isRightMost) {
            result.add(node.val);
        }
    }

    public static List<Integer> boundaryOfBinaryTree(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        result.add(root.val);
        leftBoundary(root.left, result);
        if (root.left != null || root.right != null) bottomView(root, result);
        rightBoundary(root.right, result);
        return result;
    }

//    https://leetcode.com/problems/remove-comments/

    public static List<String> removeComments(String[] source) {
        boolean inBlock = false;
        StringBuilder newline = new StringBuilder();
        List<String> ans = new ArrayList<>();
        for (String line : source) {
            int i = 0;
            char[] chars = line.toCharArray();
            if (!inBlock) newline = new StringBuilder();
            while (i < line.length()) {
                if (!inBlock && i + 1 < line.length() && chars[i] == '/' && chars[i + 1] == '*') {
                    inBlock = true;
                    i++;
                } else if (inBlock && i + 1 < line.length() && chars[i] == '*' && chars[i + 1] == '/') {
                    inBlock = false;
                    i++;
                } else if (!inBlock && i + 1 < line.length() && chars[i] == '/' && chars[i + 1] == '/') {
                    break;
                } else if (!inBlock) {
                    newline.append(chars[i]);
                }
                i++;
            }
            if (!inBlock && newline.length() > 0) {
                ans.add(new String(newline));
            }
        }
        return ans;
    }

//    https://leetcode.com/problems/first-missing-positive/

    public static int firstMissingPositive(int[] nums) {
//        int [] countArr = new int[301];
//        for (int num : nums) {
//            if (num > 0 && num < 301) countArr[num]++;
//        }
//        for (int i = 1; i < countArr.length; i++) {
//            if (countArr[i] == 0) {
//                return i;
//            }
//        }
//        return -1;
        int n = nums.length;
        if (n == 0) return 1;
        if (n == 1 && nums[0] != 1) return 1;
        else if (n == 1) return 2;

        boolean notOne = true;
        for (int num : nums) {
            if (num == 1) {
                notOne = false;
                break;
            }
        }
        if (notOne) return 1;

        for (int i = 0; i < n; i++) {
            if (nums[i] <= 0 || nums[i] > n) nums[i] = 1;
        }
        for (int i = 0; i < n; i++) {
            int a = Math.abs(nums[i]);
            if (a == n) nums[0] = -n;
            else nums[a] = -Math.abs(nums[a]);
        }
        for (int i = 1; i < n; i++) {
            if (nums[i] > 0) {
                return i;
            }
        }
        if (nums[0] > 0) return n;
        return n + 1;
    }

//    https://leetcode.com/problems/minimum-deletion-cost-to-avoid-repeating-letters/

    public static int minCost(String s, int[] cost) {
        int minCost = 0;
        int i = 0;
        while (i < s.length()) {
            int j = i;
            int sum = 0;
            int maxInRepetition = 0;
            while (j < s.length() && s.charAt(i) == s.charAt(j)) {
                sum += cost[j];
                maxInRepetition = Math.max(maxInRepetition, cost[j]);
                j++;
            }
            minCost += sum - maxInRepetition;
            i = j;
        }
        return minCost;
    }

//    https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/

    public static int[] bfsDistance(Map<Integer, List<Integer>> adj, int n) {
        int[] distance = new int[n];
        boolean[] visited = new boolean[n];
        int depth = 0;
        Queue<Integer> queue = new LinkedList<>();
        queue.add(0);
        queue.add(null);
        while (!queue.isEmpty()) {
            Integer currNode = queue.remove();
            if (currNode != null) {
                if (!visited[currNode]) {
                    distance[currNode] = depth;
                    queue.addAll(adj.getOrDefault(currNode, new ArrayList<>()));
                    visited[currNode] = true;
                }
            } else {
                depth++;
                if (!queue.isEmpty()) {
                    queue.add(null);
                }
            }
        }
        return distance;
    }

    public static int minReorderUnoptimised(int n, int[][] connections) {
        Map<Integer, List<Integer>> adjOneSided = new HashMap<>();
        Map<Integer, List<Integer>> adjBi = new HashMap<>();
        for (int[] connection : connections) {
            int from = connection[0];
            int to = connection[1];

            List<Integer> oneSidedList = adjOneSided.getOrDefault(from, new ArrayList<>());
            oneSidedList.add(to);
            adjOneSided.put(from, oneSidedList);

            List<Integer> biListFrom = adjBi.getOrDefault(from, new ArrayList<>());
            biListFrom.add(to);
            adjBi.put(from, biListFrom);
            List<Integer> biListTo = adjBi.getOrDefault(to, new ArrayList<>());
            biListTo.add(from);
            adjBi.put(to, biListTo);
        }
        int count = 0;
        int[] distanceFromZero = bfsDistance(adjBi, n);
        for (int key : adjBi.keySet()) {
            if (key != 0) {
                List<Integer> biList = adjBi.get(key);
                List<Integer> oneSidedList = adjOneSided.getOrDefault(key, new ArrayList<>());
                if (oneSidedList.isEmpty()) {
                    count++;
                } else {
                    int minDistanceBi = Integer.MAX_VALUE;
                    for (int node : biList) {
                        minDistanceBi = Math.min(minDistanceBi, distanceFromZero[node]);
                    }
                    int minDistanceOneSided = Integer.MAX_VALUE;
                    for (int node : oneSidedList) {
                        minDistanceOneSided = Math.min(minDistanceOneSided, distanceFromZero[node]);
                    }
                    if (minDistanceBi != minDistanceOneSided) {
                        count++;
                    }
                }
            }
        }
        return count;
    }

    public static int minReorder(int n, int[][] connections) {
        int count = 0;
        Map<Integer, List<Integer>> adj = new HashMap<>();
        Set<String> actualPath = new HashSet<>();
        for (int[] connection : connections) {
            int from = connection[0];
            int to = connection[1];
            List<Integer> fromList = adj.getOrDefault(from, new ArrayList<>());
            fromList.add(to);
            adj.put(from, fromList);
            List<Integer> toList = adj.getOrDefault(to, new ArrayList<>());
            toList.add(from);
            adj.put(to, toList);
            actualPath.add(from + "-" + to);
        }
        Queue<Integer> q = new LinkedList<>();
        boolean[] visited = new boolean[n];
        q.add(0);
        while (!q.isEmpty()) {
            int currNode = q.remove();
            if (!visited[currNode]) {
                for (int neighbour : adj.getOrDefault(currNode, new ArrayList<>())) {
                    if (!actualPath.contains(neighbour + "-" + currNode) && !visited[neighbour]) {
                        count++;
                    }
                    q.add(neighbour);
                }
                visited[currNode] = true;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/count-good-nodes-in-binary-tree/

    static int count = 0;

    public static void goodNodeUtil(TreeNode node, int maxSoFar) {
        if (node == null) {
            return;
        }
        if (node.val >= maxSoFar) {
            count++;
        }
        maxSoFar = Math.max(maxSoFar, node.val);
        goodNodeUtil(node.left, maxSoFar);
        goodNodeUtil(node.right, maxSoFar);
    }

    public static int goodNodes(TreeNode root) {
        count = 0;
        goodNodeUtil(root, Integer.MIN_VALUE);
        return count;
    }

//    https://leetcode.com/problems/replace-all-s-to-avoid-consecutive-repeating-characters/

    public static char getChar(char a, char b) {
      if ((((a - 'a' + 1) % 26) + 'a')!= b) {
          return (char) (((a - 'a' + 1) % 26) + 'a');
      } else {
          return (char) (((b - 'a' + 1) % 26) + 'a');
      }
    }

    public static char getChar(char a) {
        return (char) (((a - 'a' + 1) % 26) + 'a');
    }

    public static String modifyString(String s) {
        int i = 0;
        char [] sArr = s.toCharArray();
        while (i < s.length()) {
            if (i + 1 < sArr.length && sArr[i] == '?' && sArr[i + 1] != '?') {
                if (i == 0) sArr[i] = getChar(sArr[i + 1]);
                else sArr[i] = getChar(sArr[i - 1], sArr[i + 1]);
            } else if (sArr[i] == '?') {
                if (i == 0) sArr[i] = 'a';
                else sArr[i] = getChar(sArr[i - 1]);
            }
            i++;
        }
        return String.valueOf(sArr);
    }

//    https://leetcode.com/problems/minimum-deletions-to-make-character-frequencies-unique/

    public static int minDeletions(String s) {
        int [] countArr = new int[26];
        Set<Integer> frequencies = new HashSet<>();
        for (char c : s.toCharArray()) {
            countArr[c - 'a']++;
        }
        int count = 0;
        for (int f : countArr) {
            while (f > 0 && frequencies.contains(f)) {
                count++;
                f--;
            }
            if (f > 0) frequencies.add(f);
        }
        return count;
    }

//    https://leetcode.com/problems/longest-palindromic-substring/

    public static String longestPalindrome(String s) {
        String maxString = "";
        for (int i = 0; i < s.length(); i++) {
            int start = i;
            int end = i;
            while (start >= 0 && end < s.length() && s.charAt(start) == s.charAt(end)) {
                if (end - start + 1 > maxString.length()) {
                    maxString = s.substring(start,end + 1);
                }
                start--;
                end++;
            }
            start = i - 1;
            end = i;
            while (start >= 0 && end < s.length() && s.charAt(start) == s.charAt(end)) {
                if (end - start + 1 > maxString.length()) {
                    maxString = s.substring(start,end + 1);
                }
                start--;
                end++;
            }
        }
        return maxString;
    }

//    https://leetcode.com/problems/letter-combinations-of-a-phone-number/

    public static List<String> letterCombinationsUnoptimised(String digits) {
        if (digits.isEmpty()) {
            return new ArrayList<>();
        }
        Map<Character,List<String>> mapping = new HashMap<>();
        int i = 0;
        for (char c = '2'; c <= '9'; c++) {
            List<String> alphabets = new ArrayList<>();
            for (int j = 0; j < 3; j++) {
                alphabets.add(String.valueOf((char) (i++ + 'a')));
            }
            if (c == '7' || c == '9') {
                alphabets.add(String.valueOf((char) (i++ + 'a')));
            }
            mapping.put(c, alphabets);
        }
        Queue<String> queue = new LinkedList<>(mapping.get(digits.charAt(0)));
        int size = queue.size();
        for (int j = 1; j < digits.length(); j++) {
            char c = digits.charAt(j);
            List<String> alphabetsForChar = mapping.getOrDefault(c, new ArrayList<>());
            for (int k = 0; k < size; k++) {
                String currProcessed = queue.remove();
                for (String alphabet : alphabetsForChar) {
                    queue.add(currProcessed + alphabet);
                }
            }
            size = queue.size();
        }
        return new ArrayList<>(queue);
    }

//    using backtracking

    public static void backTrack(String digits, Map<Character,String> mapping, int index,
                                 StringBuilder sb, List<String> res) {
        if (index == digits.length()) {
         res.add(sb.toString());
         return;
        }
         String alphabets = mapping.get(digits.charAt(index));
         for (char c : alphabets.toCharArray()) {
             sb.append(c);
             backTrack(digits, mapping, index + 1, sb, res);
             sb.deleteCharAt(sb.length() - 1);
        }
    }

    public static List<String> letterCombinations(String digits) {
        if (digits.isEmpty()) {
            return new ArrayList<>();
        }
        Map<Character,String> mapping = new HashMap<>();
        mapping.put('2',"abc");
        mapping.put('3',"def");
        mapping.put('4',"ghi");
        mapping.put('5',"jkl");
        mapping.put('6',"mno");
        mapping.put('7',"pqrs");
        mapping.put('8',"tuv");
        mapping.put('9',"wxyz");

        List<String> result = new ArrayList<>();
        backTrack(digits, mapping, 0, new StringBuilder(), result);
        return result;
    }

//    https://leetcode.com/problems/palindrome-permutation/

    public static boolean canPermutePalindrome(String s) {
        int [] countArr = new int[123];
        int oddCount = 0;
        for (char c : s.toCharArray()) {
            countArr[c]++;
            if (countArr[c] % 2 != 0) oddCount++;
            else oddCount--;
        }
        return oddCount <= 1;
    }

//    https://leetcode.com/problems/number-of-ways-to-split-a-string/

    public static int numWays(String s) {
        long onesCount = 0;
        long n = s.length();
        int mod = 1_000_000_007;
        for (char c : s.toCharArray()) {
            if (c == '1') {
                onesCount++;
            }
        }
        if (onesCount == 0) {
            return (int) ((((n - 1) * (n - 2)) / 2) % mod);
        }
        if (onesCount % 3 != 0) {
            return 0;
        }
        long runningOnesCount = 0;
        long [] indexes = new long[6];
        int index = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '1') {
                runningOnesCount++;
                if (runningOnesCount == 1) {
                    indexes[index] = i;
                    index++;
                }
                if (runningOnesCount == onesCount / 3) {
                    indexes[index] = i;
                    runningOnesCount = 0;
                    index++;
                }
            }
        }
        long w1 = (indexes[2] - indexes[1]);
        long w2 = (indexes[4] - indexes[3]);
        return (int) ((w1 * w2) % mod);
    }

//    https://leetcode.com/problems/the-skyline-problem/

    static class BuildingPoint implements Comparable<BuildingPoint> {
        int x;
        int height;
        boolean isStart;

        BuildingPoint(int x, int height, boolean isStart) {
            this.x = x;
            this.height = height;
            this.isStart = isStart;
        }

        public int compareTo(BuildingPoint o) {
            if (this.x != o.x) {
                return this.x - o.x;
            } else {
                return (this.isStart ? -this.height : this.height) - (o.isStart ? -o.height : o.height);
            }
        }

    }

    public static List<List<Integer>> getSkyline(int[][] buildings) {
        BuildingPoint [] buildingPoints = new BuildingPoint[buildings.length * 2];
        int j = 0;
        for (int[] building : buildings) {
            buildingPoints[j] = new BuildingPoint(building[0], building[2], true);
            buildingPoints[j + 1] = new BuildingPoint(building[1], building[2], false);
            j = j + 2;
        }
        Arrays.sort(buildingPoints);

        TreeMap<Integer,Integer> maxHeap = new TreeMap<>();
        maxHeap.put(0,1);
        List<List<Integer>> result = new ArrayList<>();
        int maxPrevHeight = 0;
        for (BuildingPoint buildingPoint : buildingPoints) {
            if (buildingPoint.isStart) {
                maxHeap.merge(buildingPoint.height, 1, Integer::sum);
            } else {
                if (maxHeap.get(buildingPoint.height) > 1) {
                    maxHeap.put(buildingPoint.height, maxHeap.get(buildingPoint.height) - 1);
                } else {
                    maxHeap.remove(buildingPoint.height);
                }
            }
            int currMaxHeight = maxHeap.lastKey();
            if (currMaxHeight != maxPrevHeight) {
                result.add(Arrays.asList(buildingPoint.x, buildingPoint.isStart ? buildingPoint.height : currMaxHeight));
                maxPrevHeight = currMaxHeight;
            }
        }
        return result;
    }

//    https://leetcode.com/problems/populating-next-right-pointers-in-each-node/

    static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    public static void connectHelper(Node root) {
        if (root == null) {
            return;
        }
        if (root.left != null) {
            root.left.next = root.right;
        }
        if (root.right != null) {
            if (root.next!= null) {
                root.right.next = root.next.left;
            } else {
                root.right.next = null;
            }
        }
        connectHelper(root.left);
        connectHelper(root.right);
    }

    public static Node connect(Node root) {
        if (root == null) {
            return null;
        }
        connectHelper(root);
        return root;
    }

//    https://leetcode.com/problems/add-two-numbers-ii/

    public static int findLength(ListNode listNode) {
        int length = 0;
        while (listNode != null) {
            length++;
            listNode = listNode.next;
        }
        return length;
    }

    public static int calculate(ListNode l1, ListNode l2, int d) {
       if (l1 == null && l2 == null) {
           return 0;
       }
       if (l1 == null) {
           return l2.val;
       }
       if (l2 == null) {
           return l1.val;
       }
       int carryOver = (d > 0) ? calculate(l1.next, l2,d - 1) : calculate(l1.next, l2.next, d - 1);
        int ones;
        if (d > 0) {
            ones = (l1.val + carryOver) % 10;
           carryOver = (l1.val + carryOver) / 10;
        } else {
            ones = (l1.val + l2.val + carryOver) % 10;
           carryOver = (l1.val + l2.val + carryOver) / 10;
        }
        l1.val = ones;
        return carryOver;
    }

    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int n1 = findLength(l1);
        int n2 = findLength(l2);
        if  (n1 > n2) {
            int carryOver = calculate(l1, l2,n1 - n2);
            if (carryOver != 0) {
                ListNode head = new ListNode(carryOver);
                head.next = l1;
                return head;
            } else {
                return l1;
            }
        } else {
            int carryOver = calculate(l2, l1,n2 - n1);
            if (carryOver != 0) {
                ListNode head = new ListNode(carryOver);
                head.next = l2;
                return head;
            } else {
                return l2;
            }
        }
    }

//    https://leetcode.com/problems/inorder-successor-in-bst-ii/

    static class NodeP {
        public int val;
        public NodeP left;
        public NodeP right;
        public NodeP parent;
    }

    public static NodeP findLeftMost(NodeP node) {
        while (node.left != null) {
            node = node.left;
        }
        return node;
    }

    public static NodeP inorderSuccessor(NodeP node) {
        if (node == null) {
            return null;
        }
        if (node.right != null) {
            return findLeftMost(node.right);
        } else if (node.parent != null && node.parent.left == node) {
            return node.parent;
        } else {
            while (node.parent != null && node.parent.right == node) {
                node = node.parent;
            }
            return node.parent;
        }
    }

//    recursive

    public static NodeP findLeftMostRec(NodeP node) {
        if (node.left == null) {
            return node;
        }
        return findLeftMostRec(node.left);
    }

    public static NodeP goUp(NodeP node, int val) {
        if (node == null) return null;
        if (node.val > val) return node;
        return goUp(node.parent, val);
    }

    public static NodeP inorderSuccessorRec(NodeP node) {
        if (node.parent == null || node.right != null) return findLeftMostRec(node.right);
        else return goUp(node.parent, node.val);
    }

//    https://leetcode.com/problems/multiply-strings/

    public static String multiply(String num1, String num2) {
        int n1 = num1.length();
        int n2 = num2.length();
        if (num1.equals("0") || num2.equals("0")) return "0";
        List<String> singleMultiplies = new ArrayList<>();
        for (int i = n1 - 1; i >= 0; i--) {
            int no = Integer.parseInt(String.valueOf(num1.charAt(i)));
            StringBuilder singleMultiply = new StringBuilder();
            int k = n1 - 1 - i;
            while (k > 0) {
                singleMultiply.append(0);
                k--;
            }
            int carryOver = 0;
            for (int j = n2 - 1; j >= 0; j--) {
                int anotherNo = Integer.parseInt(String.valueOf(num2.charAt(j)));
                singleMultiply.append(((anotherNo * no) + carryOver) % 10);
                carryOver = ((anotherNo * no) + carryOver) / 10;
            }
            if (carryOver > 0) singleMultiply.append(carryOver);
            singleMultiplies.add(singleMultiply.toString());
        }
        StringBuilder finalMultiplication = new StringBuilder();
        int carryOver = 0;
        int i = 0;
        while (true) {
            boolean flag = false;
            int sum = 0;
            for (String singleM : singleMultiplies) {
                if (i < singleM.length()) {
                    sum += Integer.parseInt(String.valueOf(singleM.charAt(i)));
                    flag = true;
                }
            }
            i++;
            if (!flag) break;
            finalMultiplication.append((sum + carryOver) % 10);
            carryOver = (sum + carryOver) / 10;
        }
        if (carryOver > 0) finalMultiplication.append(carryOver);
        return (finalMultiplication.reverse().toString());
    }

//    spiral matrix

    public static void spiralMatrix(int [][] matrix) {
        List<Integer> result = new ArrayList<>();
        int top = 0;
        int bottom = matrix.length - 1;
        int left = 0;
        int right = matrix[0].length - 1;
        int direction = 0;
        while (top <= bottom && left <= right) {
            if (direction == 0) {
                for (int j = left; j <= right; j++) {
                    result.add(matrix[top][j]);
                }
                top++;
            } else if (direction == 1) {
                for (int i = top; i <= bottom; i++) {
                    result.add(matrix[i][right]);
                }
                right--;
            } else if (direction == 2) {
                for (int j = right; j >= left; j--) {
                    result.add(matrix[bottom][j]);
                }
                bottom--;
            } else if (direction == 3) {
                for (int i = bottom; i >= top; i--) {
                    result.add(matrix[i][left]);
                }
                left++;
            }
            direction = (direction + 1) % 4;
        }
        System.out.println(result);
    }

//    https://leetcode.com/problems/swap-for-longest-repeated-character-substring/

    public static int maxRepOpt1(String text) {
       Map<Character,List<Integer>> positions = new HashMap<>();
       for (int i = 0; i < text.length(); i++) {
           List<Integer> arr = positions.getOrDefault(text.charAt(i), new ArrayList<>());
           arr.add(i);
           positions.put(text.charAt(i),arr);
       }
       int maxLength = 0;
       for (List<Integer> arr : positions.values()) {
           int maxSum = 0;
           int pre = 0;
           int curr = 1;
           for (int i = 0; i < arr.size() - 1; i++) {
               if (arr.get(i) + 1 == arr.get(i + 1)) {
                   curr++;
               } else if (arr.get(i) + 2 == arr.get(i + 1)) {
                   int sum = pre + curr + 1;
                   pre = curr;
                   curr = 1;
                   if (sum > arr.size()) sum--;
                   maxSum = Math.max(maxSum, sum);
               } else {
                   int sum = pre + curr + 1;
                   pre = 0;
                   curr = 1;
                   if (sum > arr.size()) sum--;
                   maxSum = Math.max(maxSum, sum);
               }
           }
           int sum = pre + curr + 1;
           if (sum > arr.size()) sum--;
           maxSum = Math.max(maxSum, sum);
           maxLength = Math.max(maxSum, maxLength);
       }
       return maxLength;
    }

//    https://leetcode.com/problems/bulb-switcher-iii/

    public static int numTimesAllBlue(int[] light) {
        int count = 0;
        int max = -1;
        for (int i = 0; i < light.length; i++) {
            max = Math.max(max, light[i]);
            if (max == i + 1) count++;
        }
        return count;
    }

//    https://leetcode.com/problems/validate-ip-address/

    public static boolean isIP4Address(String ip) {
        String [] arr = ip.split("\\.");
        int dots = 0;
        for (char c : ip.toCharArray()) {
            if (c == '.') dots++;
        }
        if (arr.length != 4 || dots != 3) return false;
        for (String subNet : arr) {
            if (subNet.length() < 1 || subNet.length() > 3) return false;
            int no = 0;
            for (int i = 0; i < subNet.length(); i++) {
                if (Character.isDigit(subNet.charAt(i))) {
                    no += (subNet.charAt(i) - '0') * Math.pow(10, subNet.length() - 1 - i);
                }
                else return false;
            }
            if (no < 0 || no > 255 || (no != 0 && subNet.charAt(0) == '0') || (no == 0 && subNet.length() != 1)) return false;
        }
        return true;
    }

    public static boolean isIP6Address(String ip) {
        Set<Character> hs = new HashSet<>();
        for(char c = 'A'; c <= 'F'; c++) {
            hs.add(c);
            hs.add(Character.toLowerCase(c));
        }
        String [] arr = ip.split(":");
        int colons = 0;
        for (char c : ip.toCharArray()) {
            if (c == ':') colons++;
        }
        if (arr.length != 8 || colons != 7) return false;
        for (String subNet : arr) {
            if (subNet.length() < 1 || subNet.length() > 4) return false;
            for (char c : subNet.toCharArray()) {
                if (Character.isLetter(c) && !hs.contains(c))  {
                    return false;
                }
            }
        }
        return true;
    }

    public static String validIPAddress(String ip) {
        if (isIP4Address(ip)) return "IPv4";
        else if (isIP6Address(ip)) return "IPv6";
        else return "Neither";
    }

//    https://leetcode.com/problems/serialize-and-deserialize-n-ary-tree/

    static class Codec {

        private static void serializeHelper(NAryNode node, StringBuilder sb) {
            if (node == null) {
                sb.append("null").append(",");
                return;
            }
            sb.append(node.val).append(",");
            int childrenSize = node.children != null ? node.children.size() : 0;
            sb.append(childrenSize).append(",");
            for (int i = 0; i < childrenSize; i++) {
                serializeHelper(node.children.get(i), sb);
            }
        }

        // Encodes a tree to a single string.
        public String serialize(NAryNode root) {
            StringBuilder sb = new StringBuilder();
            serializeHelper(root, sb);
            return sb.toString();
        }

        private static NAryNode deserializeHelper(Queue<String> queue) {
            if (queue.isEmpty()) {
                return null;
            }
            String nodeData = queue.remove();
            if (nodeData.equals("null")) {
                return null;
            }
            NAryNode root = new NAryNode(Integer.parseInt(nodeData));
            int childrenSize = Integer.parseInt(queue.remove());
            root.children = new ArrayList<>();
            for (int i = 0; i < childrenSize; i++) {
                root.children.add(deserializeHelper(queue));
            }
            return root;
        }

        // Decodes your encoded data to tree.
        public NAryNode deserialize(String data) {
            Queue<String> queue = new LinkedList<>(Arrays.asList(data.split(",")));
            return deserializeHelper(queue);
        }
    }

    static class NAryNode {
        public int val;
        public List<NAryNode> children;

        public NAryNode() {}

        public NAryNode(int _val) {
            val = _val;
        }

        public NAryNode(int _val, List<NAryNode> _children) {
            val = _val;
            children = _children;
        }
    }

}



