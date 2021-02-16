package dsalgo.leetcode.company.amazon;

import javafx.util.Pair;

import java.util.*;

public class Explore {

    public static void main(String[] args) {
//        System.out.println(lengthOfLongestSubstring("bbbbb"));
//        System.out.println(myAtoi("20000000000000000000"));
//        System.out.println(maxArea(new int[]{1,8,6,2,5,4,8,3,7}));
//        System.out.println(intToRoman(4));
//        System.out.println(romanToInt("MCMXCIV"));
//        System.out.println(threeSum(new int[]{-2,0,1,1,2}));
//        System.out.println(threeSumClosest(new int[]{-1,2,1,-4},1));
//        System.out.println(strStr("mississippi","issip"));
//        System.out.println(minWindow("a","b"));
//        System.out.println(compareVersion("7.5.2.4","7.5.3"));
//        System.out.println(Arrays.toString(productExceptSelf(new int[]{1, 2, 3, 4})));
//        System.out.println(isValid("]"));
//        System.out.println(isValidBST(new TreeNode(2147483647)));
//        TreeNode root = new TreeNode(3);
//        root.left = new TreeNode(9);
//        root.right = new TreeNode(20);
//        root.right.left = new TreeNode(15);
//        root.right.right = new TreeNode(7);
//        System.out.println(zigzagLevelOrder(root));
//        TreeNode root = new TreeNode(5);
//        root.right = new TreeNode(8);
//        root.right.left = new TreeNode(13);
//        root.right.right = new TreeNode(4);
//        root.right.right.right = new TreeNode(1);
//        System.out.println(maxPathSum(root));
//        System.out.println(ladderLength("hot","dog", Arrays.asList("hot","dog")));
//        System.out.println(findLadders("a","c", Arrays.asList("a","b","c")));
//        System.out.println(canFinish(2,new int[][]{{1,0},{0,1}}));
//        System.out.println(kMaxSumCombinations(new int[]{4,2,5,1}, new int[]{8,0,3,5}));
        System.out.println(cutOffTree(Arrays.asList(Arrays.asList(54581641,64080174,24346381,69107959),Arrays.asList(86374198,61363882,68783324,79706116),Arrays.asList(668150,92178815,89819108,94701471),Arrays.asList(83920491,22724204,46281641,47531096),Arrays.asList(89078499,18904913,25462145,60813308))));

    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2961/

    public static int lengthOfLongestSubstring(String s) {
        int maxLength = 0;
        int length = 0;
        Set<Character> hs = new HashSet<>();
        int i = 0;
        int j = 0;
        while (j < s.length()) {
            if (!hs.contains(s.charAt(j))) {
                hs.add(s.charAt(j));
                length++;
                j++;
                maxLength = Math.max(maxLength,length);
            } else {
                hs.remove(s.charAt(i));
                i++;
                length--;
            }
        }
        return maxLength;
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2962/

    public static int myAtoi(String s) {
        int result = 0;
        int lastResult;
        int sign = 1;
        int i = 0;
        while (i < s.length() && s.charAt(i) == ' ') {
            i++;
        }
        if (i < s.length() && (s.charAt(i) == '-' || s.charAt(i) == '+')) {
            if (s.charAt(i) == '-') {
                sign = -1;
            }
            i++;
        }
        while (i < s.length() && Character.isDigit(s.charAt(i))) {
            lastResult = result;
            result = result * 10 + s.charAt(i) - '0';

            if (result / 10 != lastResult) {
                return (sign == 1) ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            i++;
        }
        return result;
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2963/

    public static int maxArea(int[] height) {
        int max = 0;
        int i = 0;
        int j = height.length - 1;
        while (i < j) {
            max = Math.max(max, Math.min(height[i],height[j]) * (j - i));
            if (height[i] < height[j]) {
                i++;
            } else {
                j--;
            }
        }
        return max;
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2964/

    public static String intToRoman(int num) {
        int [] numArr = new int[]{1000,900,500,400,100,90,50,40,10,9,5,4,1};
        String [] romanNo = new String[]{"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
        StringBuilder sb = new StringBuilder();
        int i = 0;
        while (num != 0 && i < numArr.length) {
            if (num >= numArr[i]) {
                num -= numArr[i];
                sb.append(romanNo[i]);
            } else {
                i++;
            }
        }
        return sb.toString();
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2965/

    public static int romanToInt(String s) {
        Map<String,Integer> hm = new HashMap<>();
        hm.put("M",1000);
        hm.put("CM",900);
        hm.put("D",500);
        hm.put("CD",400);
        hm.put("C",100);
        hm.put("XC",90);
        hm.put("L",50);
        hm.put("XL",40);
        hm.put("X",10);
        hm.put("IX",9);
        hm.put("V",5);
        hm.put("IV",4);
        hm.put("I",1);

        int i = 0;
        int result = 0;
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c == 'I') {
                if (i < s.length() - 1 && s.charAt(i + 1) == 'V') {
                    result += hm.get("IV");
                    i = i + 2;
                } else if (i < s.length() - 1 && s.charAt(i + 1) == 'X') {
                    result += hm.get("IX");
                    i = i + 2;
                } else {
                    result += hm.get("I");
                    i++;
                }
            } else if (c == 'X') {
                if (i < s.length() - 1 && s.charAt(i + 1) == 'L') {
                    result += hm.get("XL");
                    i = i + 2;
                } else if (i < s.length() - 1 && s.charAt(i + 1) == 'C') {
                    result += hm.get("XC");
                    i = i + 2;
                } else {
                    result += hm.get("X");
                    i++;
                }
            } else if (c == 'C') {
                if (i < s.length() - 1 && s.charAt(i + 1) == 'D') {
                    result += hm.get("CD");
                    i = i + 2;
                } else if (i < s.length() - 1 && s.charAt(i + 1) == 'M') {
                    result += hm.get("CM");
                    i = i + 2;
                } else {
                    result += hm.get("C");
                    i++;
                }
            } else {
                result += hm.get(Character.toString(c));
                i++;
            }
        }
        return result;
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2966/

    public static List<List<Integer>> threeSum(int[] nums) {
      List<List<Integer>> result = new ArrayList<>();
      Arrays.sort(nums);
      for (int i = 0; i < nums.length; i++) {
          if (i > 0 && nums[i] == nums[i - 1]) {
              continue;
          }
          int j = i + 1;
          int k = nums.length - 1;
          int a = nums[i];
          while (j < k) {
              while (j - 1 != i && j < nums.length && nums[j] == nums[j - 1]) {
                  j++;
              }
              while (k >= 0 && k < nums.length - 1 && nums[k] == nums[k + 1]) {
                  k--;
              }
              if (j >= k) {
                  break;
              }
              int b = nums[j];
              int c = nums[k];
              if (a + b + c == 0) {
                  List<Integer> arr = new ArrayList<>();
                  arr.add(a);
                  arr.add(b);
                  arr.add(c);
                  result.add(arr);
                  j++;
                  k--;
              } else if (a + b + c < 0) {
                  j++;
              } else {
                  k--;
              }
          }
      }
      return result;
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2967/

    public static int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int absResult = Integer.MAX_VALUE;
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            int a = nums[i];
            int j = i + 1;
            int k = nums.length - 1;
            while (j < k) {
                int b = nums[j];
                int c = nums[k];
                int sum = a + b + c;
                int currAbsResult = Math.abs(target - sum);
                if (currAbsResult < absResult) {
                    absResult = currAbsResult;
                    result = sum;
                }
                if (sum == target) {
                    j++;
                    k--;
                } else if (sum < target) {
                    j++;
                } else {
                    k--;
                }
            }
        }
        return result;
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2968/

    public static int strStr(String haystack, String needle) {
        if (needle.length() == 0) {
            return 0;
        }
        for (int i = 0; i <= haystack.length() - needle.length(); i++) {
            int j;
            for (j = 0; j < needle.length(); j++) {
                if (haystack.charAt(j + i) != needle.charAt(j)) {
                    break;
                }
            }
            if (j == needle.length()) {
                return i;
            }
        }
        return -1;
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2969/

    public static void rotate(int[][] matrix) {
        int n = matrix.length - 1;
        for (int i = 0; i < matrix.length / 2; i++) {
            for (int j = 0; j < Math.ceil(matrix.length / 2.0); j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - j][i];
                matrix[n - j][i] = matrix[n - i][n - j];
                matrix[n - i][n - j] = matrix[j][n - i];
                matrix[j][n - i] = temp;
            }
        }
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/902/

    public static String minWindow(String s, String t) {
        String result = "";
        Map<Character,Integer> sMap = new HashMap<>();
        Map<Character,Integer> tMap = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            tMap.merge(t.charAt(i), 1, Integer::sum);
        }
        int desiredCount = t.length();
        int count = 0;
        int i = -1;
        int j = -1;
        while (true) {
            boolean f1 = false;
            boolean f2 = false;
            while (i < s.length() - 1 && desiredCount > count) {
                i++;
                char c = s.charAt(i);
                sMap.merge(c,1,Integer::sum);
                if (tMap.getOrDefault(c,0) >= sMap.getOrDefault(c,0)) {
                    count++;
                }
                f1 = true;
            }
            while (j < i && desiredCount == count) {
                String tempResult = s.substring(j + 1,i + 1);
                if (result.length() == 0 || result.length() > tempResult.length()) {
                    result = tempResult;
                }
                j++;
                char c = s.charAt(j);
                if (sMap.get(c) == 1) {
                    sMap.remove(c);
                } else {
                    sMap.put(c,sMap.get(c) - 1);
                }
                if (sMap.getOrDefault(c,0) < tMap.getOrDefault(c,0)) {
                    count--;
                }
                f2 = true;
            }
            if (!f1 && !f2) {
                break;
            }
        }
        return result;
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/502/

    public static int compareVersion(String version1, String version2) {
        int result = 0;
        String [] v1Arr = version1.split("\\.");
        String [] v2Arr = version2.split("\\.");
        int i = 0;
        int j = 0;
        while (i < v1Arr.length && j < v2Arr.length) {
            int v1 = Integer.parseInt(v1Arr[i]);
            int v2 = Integer.parseInt(v2Arr[j]);
            if (v1 > v2) {
                return  1;
            } else if (v1 < v2) {
                return -1;
            }
            i++;
            j++;
        }
       while (i < v1Arr.length) {
           int v1 = Integer.parseInt(v1Arr[i]);
           if (v1 > 0) {
               return 1;
           }
           i++;
       }
        while (j < v2Arr.length) {
            int v2 = Integer.parseInt(v2Arr[j]);
            if (v2 > 0) {
                return -1;
            }
            j++;
        }
        return result;
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/499/

    public static int[] productExceptSelf(int[] nums) {
        int [] output = new int[nums.length];
        output[output.length - 1] = 1;
        for (int i = output.length - 2; i >= 0; i--) {
            output[i] = output[i + 1] * nums[i + 1];
        }
        int runningProduct = 1;
        for (int i = 0; i < nums.length; i++) {
            output[i] *= runningProduct;
            runningProduct *= nums[i];
        }
        return output;
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/481/

    public static final String [] thousands = new String[]{"","Thousand","Million","Billion"};
    public static final String [] lessThanTwenty = new String[]{"","One","Two","Three","Four","Five","Six","Seven","" +
            "Eight","Nine","Ten","Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen","Seventeen","Eighteen","Nineteen"};
    public static final String [] tens = new String[]{"","","Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"};

    public static void helper(StringBuilder sb, int num) {
        if (num == 0) {
            return;
        }
        if (num < 20) {
            sb.append(lessThanTwenty[num]).append(" ");
        } else if (num < 100) {
            sb.append(tens[num / 10]).append(" ");
            helper(sb, num % 10);
        } else {
            sb.append(lessThanTwenty[num / 100]).append(" Hundred ");
            helper(sb, num % 100);
        }
    }

    public static String numberToWords(int num) {
        if (num == 0) {
            return "Zero";
        }
        StringBuilder sb = new StringBuilder();
        int index = 0;
        while (num > 0) {
            if (num % 1000 != 0) {
                StringBuilder temp = new StringBuilder();
                helper(temp, num % 1000);
                sb.insert(0, temp.append(thousands[index]).append(" "));
            }
            index++;
            num /= 1000;
        }
        return sb.toString().trim();
    }

//    https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2972/

    public static boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '[' || c == '{' || c == '(') {
                stack.push(c);
            } else if (!stack.isEmpty() && c == ']' && stack.peek() == '[') {
                stack.pop();
            } else if (!stack.isEmpty() && c == '}' && stack.peek() == '{') {
                stack.pop();
            } else if (!stack.isEmpty() && c == ')' && stack.peek() == '(') {
                stack.pop();
            } else {
                return false;
            }
        }
        return stack.isEmpty();
    }

//    https://leetcode.com/explore/interview/card/amazon/77/linked-list/513/

    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carryOver = 0;
        ListNode curr1 = l1;
        ListNode curr2 = l2;
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;
        while (curr1 != null && curr2 != null) {
            curr.next = new ListNode((curr1.val + curr2.val + carryOver) % 10);
            carryOver = (curr1.val + curr2.val + carryOver) / 10;
            curr1 = curr1.next;
            curr2 = curr2.next;
            curr = curr.next;
        }
        while (curr1 != null) {
            curr.next = new ListNode((curr1.val + carryOver) % 10);
            carryOver = (curr1.val+ carryOver) / 10;
            curr1 = curr1.next;
            curr = curr.next;
        }
        while (curr2 != null) {
            curr.next = new ListNode((curr2.val + carryOver) % 10);
            carryOver = (curr2.val+ carryOver) / 10;
            curr2 = curr2.next;
            curr = curr.next;
        }
        while (carryOver != 0) {
            curr.next = new ListNode(carryOver % 10);
            carryOver = carryOver / 10;
            curr = curr.next;
        }
        return dummy.next;
    }

//    https://leetcode.com/explore/interview/card/amazon/77/linked-list/2977/

    public static ListNode reverse(ListNode node, int k) {
        ListNode prev = null;
        ListNode curr = node;
        ListNode next;
        while (curr != null && k != 0) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
            k--;
        }
        return prev;
    }

    public static ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(Integer.MIN_VALUE);
        ListNode afterKNode;
        ListNode curr = head;
        ListNode prev = dummy;
        while (curr != null) {
            int i = 0;
            afterKNode = curr;
            while (afterKNode != null && i < k) {
                afterKNode = afterKNode.next;
                i++;
            }
            ListNode reversed = curr;
            if (i == k) {
                reversed = reverse(curr, k);
            }
            prev.next = reversed;
            prev = curr;
            curr = afterKNode;
        }
        return dummy.next;
    }

//    https://leetcode.com/explore/interview/card/amazon/77/linked-list/2979/

    public static ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode reversedList = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return reversedList;
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/514/

    public static boolean checkBST(TreeNode curr, long min, long max) {
        if (curr == null) {
            return true;
        }
        if (curr.val >= max || curr.val <= min) {
            return false;
        }
        return checkBST(curr.left, min, curr.val) && checkBST(curr.right, curr.val, max);
    }

    public static boolean isValidBST(TreeNode root) {
        return checkBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/507/

    public static boolean checkSymmetry(TreeNode left, TreeNode right) {
        if (left == null || right == null) {
            return left == null && right == null;
        }
        if (left.val != right.val) {
            return false;
        }
        return checkSymmetry(left.left, right.right) && checkSymmetry(left.right, right.left);
    }

    public static boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return checkSymmetry(root.left, root.right);
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/506/

    public static List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        List<Integer> level = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        queue.add(null);
        while (!queue.isEmpty()) {
            TreeNode node = queue.remove();
            if (node != null) {
                level.add(node.val);
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            } else {
                result.add(new ArrayList<>(level));
                level.clear();
                if (!queue.isEmpty()) {
                    queue.add(null);
                }
            }
        }
        return result;
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/2980/

    public static List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        int levelNo = 0;
        List<Integer> level = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        queue.add(null);
        while (!queue.isEmpty()) {
            TreeNode node = queue.remove();
            if (node != null) {
                level.add(node.val);
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            } else {
                if (levelNo % 2 != 0) {
                    Collections.reverse(level);
                }
                result.add(new ArrayList<>(level));
                level.clear();
                if (!queue.isEmpty()) {
                    queue.add(null);
                }
                levelNo++;
            }
        }
        return result;
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/2981/

    static int maxSum = Integer.MIN_VALUE;

    public static int maxSumPathUtil(TreeNode curr) {
        if (curr == null) {
            return 0;
        }
        int leftSubTreeMaxSum = maxSumPathUtil(curr.left);
        int rightSubTreeMaxSum = maxSumPathUtil(curr.right);
        int maxSumSubTree = leftSubTreeMaxSum + rightSubTreeMaxSum + curr.val;
        int maxSumOfEitherBranchOrCurrNode = Math.max(Math.max(leftSubTreeMaxSum,rightSubTreeMaxSum) + curr.val, curr.val);
        int maxSumAtNode = Math.max(maxSumSubTree, maxSumOfEitherBranchOrCurrNode);
        maxSum = Math.max(maxSum, maxSumAtNode);
        return maxSumOfEitherBranchOrCurrNode;
    }

    public static int maxPathSum(TreeNode root) {
        int r = maxSumPathUtil(root);
        return Math.max(maxSum,r);
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/2982/

    public static int ladderLength(String beginWord, String endWord, List<String> wordList) {
        int depth = 0;
        Set<String> hs = new HashSet<>(wordList);
        if (!hs.contains(endWord)) {
            return depth;
        }
        Queue<String> wordQueue = new LinkedList<>();
        wordQueue.add(beginWord);
        wordQueue.add(null);
        while (!wordQueue.isEmpty()) {
            String n = wordQueue.remove();
            if (n != null) {
                char[] word = n.toCharArray();
                int l = 0;
                while (l < word.length) {
                    char c = word[l];
                    for (char curr = 'a'; curr <= 'z'; curr++) {
                        if (curr != c) {
                            word[l] = curr;
                            String newWord = new String(word);
                            if (newWord.equals(endWord)) {
                                return depth + 2;
                            } else if (hs.contains(newWord)) {
                                hs.remove(newWord);
                                wordQueue.add(newWord);
                            }
                        }
                    }
                    word[l] = c;
                    l++;
                }
            } else {
                if (!wordQueue.isEmpty()) {
                    wordQueue.add(null);
                    depth++;
                }
            }
        }
        return 0;
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/483/

    public static void dfsLadder(String curr, String endWord, List<String> currPath, List<List<String>> result,
                          Map<String,Set<String>> adj) {
        currPath.add(curr);
        if (curr.equals(endWord)) {
            result.add(new ArrayList<>(currPath));
            currPath.remove(currPath.size() - 1);
            return;
        }
        for (String neighbour : adj.getOrDefault(curr, new HashSet<>())) {
            dfsLadder(neighbour, endWord, currPath, result, adj);
        }
        currPath.remove(currPath.size() - 1);
    }

    public static List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> result = new ArrayList<>();
        int depth = 0;
        Set<String> hs = new HashSet<>(wordList);
        if (!hs.contains(endWord)) {
            return result;
        }
        Map<String,Set<String>> adj = new HashMap<>();
        Map<String,Integer> wordDepth = new HashMap<>();
        Queue<String> wordQueue = new LinkedList<>();
        wordDepth.put(beginWord, depth);
        wordQueue.add(beginWord);
        wordQueue.add(null);
        while (!wordQueue.isEmpty()) {
            String n = wordQueue.remove();
            if (n != null) {
                char[] word = n.toCharArray();
                int l = 0;
                while (l < word.length) {
                    char c = word[l];
                    for (char curr = 'a'; curr <= 'z'; curr++) {
                        if (curr != c) {
                            word[l] = curr;
                            String newWord = new String(word);
                            if (hs.contains(newWord)) {
                                if (!wordDepth.containsKey(newWord)) {
                                    wordDepth.put(newWord,depth + 1);
                                    Set<String> set;
                                    if (adj.containsKey(n)) {
                                        set = adj.get(n);
                                    } else {
                                        set = new HashSet<>();
                                    }
                                    set.add(newWord);
                                    adj.put(n,set);
                                    wordQueue.add(newWord);
                                } else if (wordDepth.get(newWord).equals(wordDepth.get(n) + 1)) {
                                    Set<String> set;
                                    if (adj.containsKey(n)) {
                                        set = adj.get(n);
                                    } else {
                                        set = new HashSet<>();
                                    }
                                    set.add(newWord);
                                    adj.put(n,set);
                                    wordQueue.add(newWord);
                                }
                            }
                        }
                    }
                    word[l] = c;
                    l++;
                }
            } else {
                if (!wordQueue.isEmpty()) {
                    wordQueue.add(null);
                    depth++;
                }
            }
        }
        dfsLadder(beginWord, endWord, new ArrayList<>(),result, adj);
        return result;
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/2983/

    public static boolean canFinishUtil(Map<Integer,Set<Integer>> adj, int [] visited, int node) {
        if (visited[node] == 2) {
            return true;
        }
        visited[node] = 2;
        for (int neighbour : adj.getOrDefault(node, new HashSet<>())) {
            if (visited[neighbour] != 1) {
                if (canFinishUtil(adj, visited, neighbour)) {
                    return true;
                }
            }
        }
        visited[node] = 1;
        return false;
    }

    public static boolean canFinish(int numCourses, int[][] prerequisites) {
        int [] visited = new int[numCourses];
        Map<Integer,Set<Integer>> adj = new HashMap<>();
        for (int[] prerequisite : prerequisites) {
            Set<Integer> set;
            int node = prerequisite[0];
            if (adj.containsKey(node)) {
                set = adj.get(node);
            } else {
                set = new HashSet<>();
            }
            set.add(prerequisite[1]);
            adj.put(node, set);
        }
        for (int key : adj.keySet()) {
            if (visited[key] == 0) {
                if (canFinishUtil(adj, visited, key)) {
                    return false;
                }
            }
        }
        return true;
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/2984/

    public static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if (left != null && right != null) {
            return root;
        }
        return left != null ? left : right;
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/2985/

    static int maxDiameter = Integer.MIN_VALUE;

    public static int diameterOfBinaryTree(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = diameterOfBinaryTree(root.left);
        int right = diameterOfBinaryTree(root.right);
        maxDiameter = Math.max(maxDiameter,left + right + 1);
        return Math.max(left,right) + 1;
    }

//    https://www.geeksforgeeks.org/k-maximum-sum-combinations-two-arrays/

    static class PairData {

        int data;
        int i;
        int j;

        PairData(int data, int i, int j) {
            this.data = data;
            this.i = i;
            this.j = j;
        }
    }

    public static List<Integer> kMaxSumCombinations(int [] a, int [] b) {
        int l = a.length;
        Arrays.sort(a);
        Arrays.sort(b);
        List<Integer> result = new ArrayList<>();
        Set<Pair<Integer,Integer>> hs = new HashSet<>();
        hs.add(new Pair<>(l - 1, l -1));
        PriorityQueue<PairData> maxHeap = new PriorityQueue<>((x,y) -> Integer.compare(y.data, x.data));
        maxHeap.add(new PairData(a[l - 1] + b[l - 1], l - 1, l - 1));
        while (!maxHeap.isEmpty()) {
            PairData maxValueNow = maxHeap.remove();
            result.add(maxValueNow.data);
            int m = maxValueNow.i;
            int n = maxValueNow.j;

            if (!hs.contains(new Pair<>(m, n - 1)) && m >= 0 && n > 0) {
                maxHeap.add(new PairData(a[m] + b[n - 1], m, n - 1));
                hs.add(new Pair<>(m, n - 1));
            }
            if (!hs.contains(new Pair<>(m - 1, n)) && m > 0 && n >= 0) {
                maxHeap.add(new PairData(a[m - 1] + b[n],m - 1, n));
                hs.add(new Pair<>(m - 1, n));
            }
        }
        return result;
    }

//    https://leetcode.com/explore/interview/card/amazon/78/trees-and-graphs/2986/

    public static int bfsGolf(Map<Long,Set<Long>> adj, long startNode, long endNode) {
        int depth = 0;
        Queue<Long> queue = new LinkedList<>();
        Set<Long> visited = new HashSet<>();
        queue.add(startNode);
        queue.add(null);
        while (!queue.isEmpty()) {
            Long node = queue.remove();
            if (node != null) {
                if (!visited.contains(node)) {
                    if (node == endNode) {
                        return depth;
                    }
                    queue.addAll(adj.getOrDefault(node, new HashSet<>()));
                    visited.add(node);
                }
            } else {
                if (!queue.isEmpty()) {
                    queue.add(null);
                }
                depth++;
            }
        }
        return -1;
    }

    public static int cutOffTree(List<List<Integer>> forest) {
        int minPath = 0;
        Map<Long,Set<Long>> adj = new TreeMap<>();
        for (int i = 0; i < forest.size(); i++) {
            for (int j = 0; j < forest.get(i).size(); j++) {
                long node = forest.get(i).get(j);
                if (node != 0) {
                    Set<Long> neighbours = new HashSet<>();
                    if (i > 0 && forest.get(i - 1).get(j) != 0) {
                        neighbours.add((long) forest.get(i - 1).get(j));
                    }
                    if (i < forest.size() - 1 && forest.get(i + 1).get(j) != 0) {
                        neighbours.add((long) forest.get(i + 1).get(j));
                    }
                    if (j > 0 && forest.get(i).get(j - 1) != 0) {
                        neighbours.add((long) forest.get(i).get(j - 1));
                    }
                    if (j < forest.get(i).size() - 1 && forest.get(i).get(j + 1) != 0) {
                        neighbours.add((long) forest.get(i).get(j + 1));
                    }
                    adj.put(node, neighbours);
                }
            }
        }
        long startNode = forest.get(0).get(0);
        for (long key : adj.keySet()) {
            int bfsMinPath = bfsGolf(adj, startNode, key);
            if (bfsMinPath == -1) {
                return -1;
            }
            minPath += bfsMinPath;
            startNode = key;
        }
        return minPath;
    }

    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

}
