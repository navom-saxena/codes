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
        System.out.println(isMatch("a","ab*"));
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

    public static boolean isMatchUtils(String s, String p, int i, int j) {
        if (i == s.length() && j >= p.length()) return true;
        else if (i == s.length() || j >= p.length()) {
            while (j < p.length() - 1) {
                if (p.charAt(j + 1) == '*') j = j + 2;
                else break;
            }
            return i == s.length() && j >= p.length();
        }
        else if ((j < p.length() - 1 && p.charAt(j + 1) != '*') || j == p.length() - 1) {
            if (i < s.length() && s.charAt(i) == p.charAt(j) || p.charAt(j) == '.')
                return isMatchUtils(s, p, i + 1, j + 1);
            else return false;
        } else if (j < p.length() - 1 && p.charAt(j + 1) == '*') {
            if (i < s.length() && s.charAt(i) == p.charAt(j) || p.charAt(j) == '.') {
                return isMatchUtils(s, p, i + 1, j) || isMatchUtils(s, p, i + 1, j + 2);
            }
            else return isMatchUtils(s, p, i, j + 2);
        } else {
            return false;
        }
    }

    public static boolean isMatch(String s, String p) {
        return isMatchUtils(s, p, 0, 0);
    }

}
