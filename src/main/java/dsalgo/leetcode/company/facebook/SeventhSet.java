package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;
import javafx.util.Pair;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class SeventhSet {

    public static void main(String[] args) {
//        System.out.println(openLock(new String[]{"0201","0101","0102","1212","2002"},"0202"));
//        System.out.println(findSubstring("barfoothefoobarman", new String[]{"foo","bar"}));
//        System.out.println(minDistance("horse","hors"));
        System.out.println(("http://news.yahoo.com/news/topics/".split("/")[2]));
    }


    //    https://leetcode.com/problems/diameter-of-n-ary-tree/

    static class NNode {
        public int val;
        public List<NNode> children;


        public NNode() {
            children = new ArrayList<>();
        }

        public NNode(int _val) {
            val = _val;
            children = new ArrayList<>();
        }

        public NNode(int _val,ArrayList<NNode> _children) {
            val = _val;
            children = _children;
        }
    }

    int diameterUtils(NNode root, int [] diameter) {
        if (root == null) return 0;
        int maxDepth = 0;
        int secondMaxDepth = 0;
        for (NNode child : root.children) {
            int d = diameterUtils(child, diameter);
            if (d > maxDepth) {
                secondMaxDepth = maxDepth;
                maxDepth = d;
            } else if (d > secondMaxDepth) secondMaxDepth = d;
        }
        diameter[0] = Math.max(diameter[0], maxDepth + secondMaxDepth + 1);
        return maxDepth + 1;
    }

    public int diameter(NNode root) {
        int [] maxDiameter = new int[]{Integer.MIN_VALUE};
        int d = diameterUtils(root, maxDiameter);
        maxDiameter[0] = Math.max(maxDiameter[0], d);
        return maxDiameter[0] == Integer.MIN_VALUE ? 0 : maxDiameter[0] - 1;
    }

//    https://leetcode.com/problems/convert-a-number-to-hexadecimal/

    public String toHex(int n) {
        if (n == 0) return "0";
        StringBuilder sb = new StringBuilder();
        char [] map = new char[16];
        int i;
        for (i = 0; i < 10; i++) {
            map[i] = (char) (i + '0');
        }
        for (char c = 'a'; c <= 'f'; c++) {
            map[i] = c;
            i++;
        }
        long num = n < 0 ? n + (long) Math.pow(2,32) : n;
        while (num > 0) {
            long q = num / 16;
            int r = (int)( num % 16);
            sb.append(map[r]);
            num = q;
        }
        return sb.reverse().toString();
    }

//    https://leetcode.com/problems/minimum-area-rectangle-ii/

    double distance(int [] p1, int [] p2) {
        return Math.pow(p2[0] - p1[0],2) + Math.pow(p2[1] - p1[1],2);
    }

    public double minAreaFreeRect(int[][] points) {
        double minArea = Integer.MAX_VALUE;
        Set<Pair<Integer,Integer>> pointsSet = new HashSet<>();
        for (int [] point : points) {
            pointsSet.add(new Pair<>(point[0],point[1]));
        }
        for (int [] p1 : points) {
            for (int [] p2 : points) {
                if (p1 == p2) continue;
                for (int [] p3 : points) {
                    if (p3 == p1 || p3 == p2) continue;
                    if (distance(p1,p2) + distance(p2,p3) != distance(p1,p3)) continue;
                    int x1 = p1[0];
                    int y1 = p1[1];
                    int x2 = p2[0];
                    int y2 = p2[1];
                    int x3 = p3[0];
                    int y3 = p3[1];
                    int x4 = x1 + x3 - x2;
                    int y4 = y1 + y3 - y2;
                    if (pointsSet.contains(new Pair<>(x4,y4))) {
                        double area = Math.sqrt(distance(p1,p2) * distance(p2,p3));
                        minArea = Math.min(minArea,area);
                    }
                }
            }
        }
        return minArea == Integer.MAX_VALUE ? 0 : minArea;
    }

//    https://leetcode.com/problems/maximal-rectangle/

    int histogramMaxArea(int [] histogram) {
        int [] leftMinIndexFromCurr = new int[histogram.length];
        int [] rightMinIndexFromCurr = new int[histogram.length];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < histogram.length; i++) {
            while (!stack.isEmpty() && histogram[stack.peek()] >= histogram[i]) stack.pop();
            if (stack.isEmpty()) leftMinIndexFromCurr[i] = -1;
            else leftMinIndexFromCurr[i] = stack.peek();
            stack.push(i);
        }
        stack.clear();
        for (int i = histogram.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && histogram[stack.peek()] >= histogram[i]) stack.pop();
            if (stack.isEmpty()) rightMinIndexFromCurr[i] = histogram.length;
            else rightMinIndexFromCurr[i] = stack.peek();
            stack.push(i);
        }
        int maxArea = 0;
        for (int i = 0; i < histogram.length; i++) {
            int area = histogram[i] * (rightMinIndexFromCurr[i] - leftMinIndexFromCurr[i] - 1);
            maxArea = Math.max(maxArea,area);
        }
        return maxArea;
    }

    public int maximalRectangle(char[][] matrix) {
        int maxArea = 0;
        if (matrix.length == 0) return maxArea;
        int [] histogram = new int[matrix[0].length];
        for (char[] chars : matrix) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (chars[j] == '0') histogram[j] = 0;
                else histogram[j]++;
            }
            int area = histogramMaxArea(histogram);
            maxArea = Math.max(maxArea, area);
        }
        return maxArea;
    }

//    https://leetcode.com/problems/next-greater-element-iii/

    public int nextGreaterElement(int n) {
        char [] arr = String.valueOf(n).toCharArray();
        boolean flag = true;
        int i = arr.length - 2;
        for (; i >= 0; i--) {
            int curr = arr[i] - '0';
            int next = arr[i + 1] - '0';
            if (curr < next) {
                flag = false;
                break;
            }
        }
        if (flag) return -1;
        int nextGreaterNoIndex = i;
        int curr = arr[i] - '0';
        for (int j = arr.length - 1; j > i; j--) {
            int nextNo = arr[j] - '0';
            if (nextNo > curr) {
                nextGreaterNoIndex = j;
                break;
            }
        }
        char temp = arr[i];
        arr[i] = arr[nextGreaterNoIndex];
        arr[nextGreaterNoIndex] = temp;
        int k = i + 1;
        int l = arr.length - 1;
        while (k < l) {
            char t = arr[k];
            arr[k] = arr[l];
            arr[l] = t;
            k++;
            l--;
        }
        long m = Long.parseLong(String.valueOf(arr));
        return m <= Integer.MAX_VALUE ? (int) m : -1;
    }

//    https://leetcode.com/problems/design-search-autocomplete-system/

    static class AutoComplete {
        String sentence;
        int freq;

        AutoComplete(String sentence, int freq) {
            this.sentence = sentence;
            this.freq = freq;
        }
    }

    static class AutoCompleteTrie {
        char c;
        Map<Character, AutoCompleteTrie> children;
        Set<String> sentencesAtNode;

        AutoCompleteTrie(char c) {
            this.c = c;
            this.children = new HashMap<>();
            sentencesAtNode = new HashSet<>();
        }
    }

    static class AutocompleteSystem {

        AutoCompleteTrie root = new AutoCompleteTrie('.');
        StringBuilder sb = new StringBuilder();
        Map<String, Integer> sentencesFreq = new HashMap<>();

        void addInTrie(String sentence, int i, AutoCompleteTrie node) {
            if (i == sentence.length()) return;
            char c = sentence.charAt(i);
            AutoCompleteTrie child;
            if (node.children.containsKey(c)) {
                child = node.children.get(c);
            } else {
                child = new AutoCompleteTrie(c);
                node.children.put(c, child);
            }
            child.sentencesAtNode.add(sentence);
            addInTrie(sentence, i + 1, child);
        }

        public AutocompleteSystem(String[] sentences, int[] times) {
            for (int i = 0; i < sentences.length; i++) sentencesFreq.merge(sentences[i], times[i], Integer::sum);
            for (String sentence : sentences) {
                addInTrie(sentence, 0, root);
            }
        }

        List<String> searchInTrie(String sentence, int i, Map<String, Integer> sentencesFreq, AutoCompleteTrie node) {
            if (i == sentence.length()) return new ArrayList<>();
            char c = sentence.charAt(i);
            AutoCompleteTrie child = null;
            if (node.children.get(c) != null) {
                child = node.children.get(c);
            }
            if (child == null) return new ArrayList<>();
            if (i == sentence.length() - 1) {
                List<String> res = new ArrayList<>();
                List<String> sentencesAtNodeL = new ArrayList<>(child.sentencesAtNode);
                sentencesAtNodeL.sort((a, b) ->
                        sentencesFreq.get(b) - sentencesFreq.get(a) == 0
                                ? a.compareTo(b) : sentencesFreq.get(b) - sentencesFreq.get(a));
                for (int j = 0; j < 3; j++) {
                    if (j < sentencesAtNodeL.size()) res.add(sentencesAtNodeL.get(j));
                }
                return res;
            } else {
                return searchInTrie(sentence, i + 1, sentencesFreq, child);
            }
        }

        public List<String> input(char c) {
            if (c != '#') {
                sb.append(c);
                return searchInTrie(sb.toString(), 0, sentencesFreq, root);
            } else {
                String newSentence = sb.toString();
                sentencesFreq.merge(newSentence, 1, Integer::sum);
                addInTrie(newSentence, 0, root);
                sb = new StringBuilder();
                return new ArrayList<>();
            }
        }
    }

//    https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/

    void longestConsecutiveUtil(TreeNode node, int prev, int l, int [] maxL) {
        if (node == null) return;
        if (prev + 1 == node.val) l = l + 1;
        else l = 1;
        maxL[0] = Math.max(maxL[0],l);
        longestConsecutiveUtil(node.left, node.val, l, maxL);
        longestConsecutiveUtil(node.right, node.val, l, maxL);
    }

    public int longestConsecutive(TreeNode root) {
        int [] maxConsecutiveL = new int[]{0};
        longestConsecutiveUtil(root, Integer.MIN_VALUE, 1, maxConsecutiveL);
        return maxConsecutiveL[0];
    }

//    https://leetcode.com/problems/stream-of-characters/

    static class StreamTrie {
        char c;
        Map<Character, StreamTrie> children;
        boolean endOfWord;

        StreamTrie(char c) {
            this.c = c;
            this.children = new HashMap<>();
            this.endOfWord = false;
        }
    }

    static class StreamChecker {

        StreamTrie root;
        StringBuilder sb;

        void insert(char [] word, int i, StreamTrie node) {
            if (i < 0) return;
            char c = word[i];
            StreamTrie child = node.children.get(c);
            if (child == null) {
                child = new StreamTrie(c);
                node.children.put(c,child);
            }
            if (i == 0) child.endOfWord = true;
            else insert(word , i - 1, child);
        }

        boolean search(StringBuilder sb, int i, StreamTrie node) {
            if (i < 0) return false;
            char c = sb.charAt(i);
            StreamTrie child = node.children.get(c);
            if (child == null) {
                return false;
            }
            if (child.endOfWord) return true;
            else return search(sb , i - 1, child);
        }

        public StreamChecker(String[] words) {

            root = new StreamTrie('.');
            sb = new StringBuilder();

            for (String word : words) {
                insert(word.toCharArray(), word.length() - 1, root);
            }
        }

        public boolean query(char letter) {
            sb.append(letter);
            return search(sb, sb.length() - 1, root);
        }
    }

//    https://leetcode.com/problems/open-the-lock/

    static int charToInt(char c) {
        return c - '0';
    }

    static char intToChar(int i) {
        return (char) (i + '0');
    }

    public static int openLock(String[] deadends, String target) {
        Set<List<Character>> deadEnds = new HashSet<>();
        for (String deadEnd : deadends) {
            List<Character> l = new ArrayList<>(4);
            for (char c : deadEnd.toCharArray()) l.add(c);
            deadEnds.add(l);
        }
        List<Character> t = new ArrayList<>(4);
        for (char c : target.toCharArray()) t.add(c);
        List<Character> initial = new ArrayList<>(4);
        for (int i = 0; i < 4; i++) initial.add('0');
        int d = 0;
        Deque<List<Character>> deque = new ArrayDeque<>();
        deque.add(initial);
        while (!deque.isEmpty()) {
            int n = deque.size();
            for (int i = 0; i < n; i++) {
                List<Character> node = deque.remove();
                if (node.equals(t)) return d;
                if (deadEnds.contains(node)) continue;
                deadEnds.add(node);
                for (int j = 0; j < 4; j++) {
                    char c = node.get(j);
                    int cInt = charToInt(c);
                    int nextInt = (cInt + 1 + 10) % 10;
                    int prevInt = (cInt - 1 + 10) % 10;
                    List<Character> nextCombination = new ArrayList<>(node);
                    nextCombination.set(j, intToChar(nextInt));
                    if (!deadEnds.contains(nextCombination)) deque.add(nextCombination);
                    List<Character> prevCombination = new ArrayList<>(node);
                    prevCombination.set(j, intToChar(prevInt));
                    if (!deadEnds.contains(prevCombination)) deque.add(prevCombination);
                }
            }
            d++;
        }
        return -1;
    }

//    https://leetcode.com/problems/single-number-ii/

    public int singleNumber(int[] nums) {
        int seenOnce = 0;
        int seenTwice = 0;
        for (int no : nums) {
            seenOnce = ~seenTwice & (seenOnce ^ no);
            seenTwice = ~seenOnce & (seenTwice ^ no);
        }
        return seenOnce;
    }

//    https://leetcode.com/problems/koko-eating-bananas/

    boolean isPossible(int [] piles, int h, int k) {
        int totalHours = 0;
        for (int pile : piles) {
            totalHours += (int) Math.ceil((double) pile / k);
            if (totalHours > h) return false;
        }
        return true;
    }

    public int minEatingSpeed(int[] piles, int h) {
        int low = 1;
        int high = 1_000_000_000;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (isPossible(piles, h, mid)) high = mid;
            else low = mid + 1;
        }
        return low;
    }

//    https://leetcode.com/problems/stamping-the-sequence/

    boolean canReplace(char [] t, char [] s, int i) {
        int j = 0;
        while (j < s.length) {
            if (t[i] != '?' && t[i] != s[j]) return false;
            i++;
            j++;
        }
        return true;
    }

    int replace(char [] t, char [] s, int i) {
        int count = 0;
        for (int k = i; k < i + s.length; k++) {
            if (t[k] != '?') {
                t[k] = '?';
                count++;
            }
        }
        return count;
    }

    public int[] movesToStamp(String stamp, String target) {
        char[] t = target.toCharArray();
        char[] s = stamp.toCharArray();
        List<Integer> result = new ArrayList<>();
        boolean[] visited = new boolean[target.length()];
        int count = 0;
        while (count != target.length()) {
            boolean isChanged = false;
            for (int i = 0; i <= t.length - s.length; i++) {
                if (!visited[i] && canReplace(t, s, i)) {
                    isChanged = true;
                    int c = replace(t, s, i);
                    result.add(i);
                    visited[i] = true;
                    count += c;
                }
            }
            if (!isChanged) break;
        }
        if (count != target.length()) return new int[0];
        int[] res = new int[result.size()];
        for (int i = result.size() - 1; i >= 0; i--) {
            res[res.length - i - 1] = result.get(i);
        }
        return res;
    }

//    https://leetcode.com/problems/car-pooling/

    public boolean carPooling(int[][] trips, int capacity) {
        Map<Integer,Integer> range = new TreeMap<>();
        for (int [] trip : trips) {
            range.put(trip[1], range.getOrDefault(trip[1],0) + trip[0]);
            range.put(trip[2], range.getOrDefault(trip[2],0) - trip[0]);
        }
        int currCapacity = 0;
        for (int key : range.keySet()) {
            currCapacity += range.get(key);
            if (currCapacity > capacity) return false;
        }
        return true;
    }

//    https://leetcode.com/problems/contiguous-array/

    public int findMaxLength(int[] nums) {
        Map<Integer,Integer> lastDiff = new HashMap<>();
        lastDiff.put(0, - 1);
        int maxLength = 0;
        int runningDiff = 0;
        for (int i = 0; i < nums.length; i++) {
            runningDiff += nums[i] == 0 ? -1 : 1;
            if (lastDiff.containsKey(runningDiff)) {
                maxLength = Math.max(maxLength, i - lastDiff.get(runningDiff));
            } else lastDiff.put(runningDiff, i);
        }
        return maxLength;
    }

//    https://leetcode.com/problems/brick-wall/

    public int leastBricks(List<List<Integer>> wall) {
        Map<Integer,Integer> range = new HashMap<>();
        int wallEnd = 0;
        for (List<Integer> row : wall) {
            int counter = 0;
            for (int width : row) {
                counter += width;
                range.put(counter, range.getOrDefault(counter, 0) + 1);
            }
            wallEnd = counter;
        }
        int maxBricksEnding = 0;
        for (int k : range.keySet()) {
            if (k != wallEnd) maxBricksEnding = Math.min(maxBricksEnding, wall.size() - range.get(k));
        }
        return maxBricksEnding;
    }

//    https://leetcode.com/company/facebook/

    boolean allCharsGreaterThanK(int [] arr, int k) {
        for (int h : arr) if (h != 0 && h < k) return false;
        return true;
    }

    public int longestSubstring(String s, int k) {
       int n = s.length();
       if (n == 0 || n < k) return 0;
       if (k <= 1) return n;
       Map<Character,Integer> freq = new HashMap<>();
       for (char c : s.toCharArray()) freq.merge(c,1, Integer::sum);
       int l = 0;
       while (l < n && freq.get(s.charAt(l)) >= k) l++;
       if (l >= n - 1) return l;
       int lM = longestSubstring(s.substring(0,l), k);
       while (l < n && freq.get(s.charAt(l)) < k) l++;
       int rM = l < n  ? longestSubstring(s.substring(l), k) : 0;
       return Math.max(lM,rM);
    }

//    https://leetcode.com/problems/binary-tree-longest-consecutive-sequence-ii/

    int [] longestConsecutive2Utils(TreeNode node, int [] maxL) {
        if (node == null) return new int[]{0,0};
        int [] l = longestConsecutive2Utils(node.left, maxL);
        int [] r = longestConsecutive2Utils(node.right, maxL);
        int inc = 1;
        int dec = 1;
        if (node.left != null && node.val - node.left.val == 1) inc = l[0] + 1;
        else if (node.left != null && node.val - node.left.val == -1) dec = l[1] + 1;
        if (node.right != null && node.val - node.right.val == 1) inc = Math.max(inc,r[0] + 1);
        else if (node.right != null && node.val - node.right.val == -1) dec = Math.max(dec, r[1] + 1);
        maxL[0] = Math.max(maxL[0], inc + dec - 1);
        return new int[]{inc, dec};
    }

    public int longestConsecutive2(TreeNode root) {
        int [] maxLen = new int[]{0};
        int [] lR = longestConsecutive2Utils(root, maxLen);
        return Math.max(maxLen[0], Math.max(lR[0],lR[1]));
    }

//    https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/

    public int numSubseq(int[] nums, int target) {
        int mod = (int) (Math.pow(10, 9) + 7);
        Arrays.sort(nums);
        int [] power = new int[nums.length];
        power[0] = 1;
        for (int k = 1; k < power.length; k++) power[k] = (power[k - 1] * 2) % mod;
        int i = 0;
        int j = nums.length - 1;
        int no = 0;
        while (i <= j) {
            if (nums[i] + nums[j] <= target) {
                no = (no + power[j - i]) % mod;
                i++;
            } else j--;
        }
        return no;
    }

//    https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        if (root.val < p.val && root.val < q.val) return lowestCommonAncestor(root.right, p, q);
        else if (root.val > p.val && root.val > q.val) return lowestCommonAncestor(root.left, p, q);
        else return root;
    }

//    https://leetcode.com/problems/substring-with-concatenation-of-all-words/

    static class Trie {
        char c;
        Map<Character,Trie> children;
        boolean end;
        String w;

        Trie(char c) {
            this.c = c;
            this.children = new HashMap<>();
        }
    }

    static void insertFindSubstring(char [] word, int i, Trie node, Map<String,Integer> allWords) {
        if (i == word.length) return;
        char c = word[i];
        Trie child = node.children.get(c);
        if (child == null) {
            child = new Trie(c);
            node.children.put(c, child);
        }
        if (i == word.length - 1) {
            child.end = true;
            child.w = String.valueOf(word);
            allWords.merge(child.w, 1, Integer::sum);
        }
        else insertFindSubstring(word, i + 1, child, allWords);
    }

    public static List<Integer> findSubstringTrie(String s, String[] words) {
        List<Integer> indexes = new ArrayList<>();
        Trie root = new Trie('.');
        Map<String,Integer> allWords = new HashMap<>();
        for (String word : words) insertFindSubstring(word.toCharArray(), 0, root, allWords);
        Trie curr = root;
        int i = 0;
        char [] sen = s.toCharArray();
        Map<String,Integer> visited = new HashMap<>();
        while (i < sen.length) {
            int j = i;
            while (j < sen.length) {
                while (j < sen.length && curr.children.containsKey(sen[j])) {
                    curr = curr.children.get(sen[j]);
                    j++;
                }
                if (curr.end && visited.getOrDefault(curr.w, 0) < allWords.getOrDefault(curr.w,0)) {
                    visited.merge(curr.w, 1, Integer::sum);
                    curr = root;
                    if (visited.size() == allWords.size() && visited.equals(allWords)) {
                        indexes.add(i);
                        break;
                    }
                } else break;
            }
            visited.clear();
            curr = root;
             i++;
        }
        return indexes;
    }

    public static List<Integer> findSubstring(String s, String[] words) {
        Map<String,Integer> allWords = new HashMap<>();
        for (String word : words) allWords.put(word, allWords.getOrDefault(word, 0) + 1);
        Map<String,Integer> visited = new HashMap<>();
        char [] sentence = s.toCharArray();
        int k = words[0].length();
        int c = words.length;
        int i = 0;
        List<Integer> result = new ArrayList<>();
        while (i <= sentence.length - (k * c)) {
            int co = 0;
            for (int j = i; j <= sentence.length - k; j = j + k) {
                String sbs = s.substring(j, j + k);
                if (visited.getOrDefault(sbs, 0) < allWords.getOrDefault(sbs, 0)) {
                    visited.put(sbs, visited.getOrDefault(sbs, 0) + 1);
                    co++;
                    if (co == c) result.add(i);
                } else break;
            }
            visited.clear();
            i++;
        }
        return result;
    }

//    https://leetcode.com/problems/decode-ways-ii/

    int decodeChar(char c) {
        if (c == '*') return 9;
        if (c == '0') return 0;
        return 1;
    }

    int decodeTwoChars(char curr, char prev) {
        if (curr == '*') {
            if (prev == '*') return 15;
            else if (prev == '1') return 9;
            else if (prev == '2') return 6;
            else return 0;
        } else if (prev == '*') {
            if (curr < '7') return 2;
            else return 1;
        } else if (prev == '1') return 1;
        else if (prev == '2') return curr < '7' ? 1 : 0;
        else return 0;
    }

    public int numDecodings(String s) {
        char [] sArr = s.toCharArray();
        long [] dp = new long[sArr.length + 1];
        dp[0] = 1;
        dp[1] = decodeChar(sArr[0]);
        for (int i = 2; i < dp.length; i++) {
            char curr = sArr[i - 1];
            char prev = sArr[i - 2];
            dp[i] += dp[i - 1] * decodeChar(curr);
            dp[i] += dp[i - 2] * decodeTwoChars(curr, prev);
            dp[i] %= 1_000_000_007;
        }
        return (int) dp[dp.length - 1];
    }

//    https://leetcode.com/problems/longest-absolute-file-path/

    static class PathData {
        List<PathData> children;
        int level;
        boolean isFile;
        int len;

        PathData(int level, int len, boolean isFile) {
            this.level = level;
            this.len = len;
            this.isFile = isFile;
            children = new ArrayList<>();
        }
    }

    int maxSize(PathData node) {
        if (node == null) return 0;
        if (node.isFile) return node.len + node.level;
        int maxV = 0;
        for (PathData child : node.children) maxV = Math.max(maxV,maxSize(child));
        if (maxV == 0) return 0;
        return maxV + node.len;
    }

    public int lengthLongestPath(String input) {
//        Map<Integer,PathData> levels = new HashMap<>();
//        PathData root = new PathData(-1,0,false);
//        levels.put(-1,root);
//        for (String s : input.split("\\n")) {
//            char [] pArr = s.toCharArray();
//            int tCount = 0;
//            boolean isFile = false;
//            for (char c : pArr) {
//                if (c == '\t') tCount++;
//                else if (c == '.') isFile = true;
//            }
//            int len = pArr.length - tCount;
//            PathData pathData = new PathData(tCount, len, isFile);
//            PathData parent = levels.get(tCount - 1);
//            if (parent != null) {
//                parent.children.add(pathData);
//            }
//            levels.put(tCount,pathData);
//        }
//        return maxSize(root);

        int maxLength = 0;
        Map<Integer,Integer> levels = new HashMap<>();
        levels.put(-1,0);
        for (String p : input.split("\\n")) {
            String [] nameS = p.split("\\t");
            int len = nameS[nameS.length - 1].length();
            int level = p.length() - len;
            boolean isFile = p.contains(".");
            if (isFile) {
                maxLength = Math.max(maxLength, levels.getOrDefault(level - 1,0) + len);
            } else {
                levels.put(level, levels.getOrDefault(level - 1,0) + len + 1);
            }
        }
        return maxLength;
    }

//    https://leetcode.com/problems/edit-distance/

    public static int minDistance(String word1, String word2) {
//        if (word1.equals(word2)) return 0;
//        Deque<Pair<String, Integer>> deque = new ArrayDeque<>();
//        Map<String, Integer> visited = new HashMap<>();
//        int d = 0;
//        deque.add(new Pair<>(word1, 1));
//        deque.add(new Pair<>(word2, 2));
//        while (!deque.isEmpty()) {
//            int n = deque.size();
//            for (int i = 0; i < n; i++) {
//                Pair<String, Integer> p = deque.remove();
//                String node = p.getKey();
//                int from = p.getValue();
//                Integer foundB = visited.get(node);
//                if (foundB != null) {
//                    if (foundB == from) continue;
//                    return d;
//                }
//                visited.put(node, from);
//                int len = node.length();
//                for (int j = 0; j < len; j++) {
//                    for (char c = 'a'; c <= 'z'; c++) {
//                        String newS = node.substring(0, j) + c + node.substring(j);
//                        if (!visited.containsKey(newS)) {
//                            deque.add(new Pair<>(newS, from));
//                        } else {
//                            int f = visited.get(newS);
//                            if (from != f) return d + 1;
//                        }
//                    }
//                }
//                for (int j = 0; j < len; j++) {
//                    String newS = node.substring(0, j) + node.substring(j + 1);
//                    if (!visited.containsKey(newS)) {
//                        deque.add(new Pair<>(newS, from));
//                    } else {
//                        int f = visited.get(newS);
//                        if (from != f) return d + 1;
//                    }
//                }
//
//                char[] nodeArr = node.toCharArray();
//                for (int j = 0; j < len; j++) {
//                    char ch = nodeArr[j];
//                    for (char c = 'a'; c <= 'z'; c++) {
//                        if (c != ch) {
//                            nodeArr[j] = c;
//                            String newS = String.valueOf(nodeArr);
//                            if (!visited.containsKey(newS)) {
//                                deque.add(new Pair<>(newS, from));
//                            } else {
//                                int f = visited.get(newS);
//                                if (from != f) return d + 1;
//                            }
//                        }
//                    }
//                    nodeArr[j] = ch;
//                }
//            }
//            d++;
//        }
//        return d;

        char [] w1Arr = word1.toCharArray();
        char [] w2Arr = word2.toCharArray();
        int [][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 0; i < dp.length; i++) dp[i][0] = i;
        for (int j = 0; j < dp[0].length; j++) dp[0][j] = j;
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[i].length; j++) {
                int up = dp[i - 1][j] + 1;
                int down = dp[i][j - 1] + 1;
                int prev = dp[i - 1][j - 1];
                if (w1Arr[i - 1] != w2Arr[j - 1]) prev += 1;
                dp[i][j] = Math.min(prev, Math.min(up,down));
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }

//    https://leetcode.com/problems/swap-nodes-in-pairs/

    public ListNode swapPairs(ListNode head) {
        if (head == null) return null;
        if (head.next == null) return head;
        ListNode nextToProcess = head.next.next;
        ListNode nextN = head.next;
        nextN.next = head;
        head.next = swapPairs(nextToProcess);
        return nextN;
    }

//    https://leetcode.com/problems/intersection-of-two-linked-lists/

    int getLength(ListNode head) {
        ListNode curr = head;
        int len = 0;
        while (curr != null) {
            len++;
            curr = curr.next;
        }
        return len;
    }

    ListNode getIntersection(ListNode headA, ListNode headB, int d) {
        int i = 0;
        while (i != d) {
            headA = headA.next;
            i++;
        }
        while (headA != null && headA != headB ) {
            headA = headA.next;
            headB = headB.next;
        }
        return headA;
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        int lenA = getLength(headA);
        int lenB = getLength(headB);
        if (lenA > lenB) {
            int d = lenA - lenB;
            return getIntersection(headA, headB, d);
        } else {
            int d = lenB - lenA;
            return getIntersection(headB, headA, d);
        }
    }

//    https://leetcode.com/problems/largest-rectangle-in-histogram/

    public int largestRectangleArea(int[] heights) {
        Deque<Integer> stack = new ArrayDeque<>();
        int [] leftFirstMin = new int[heights.length];
        int [] rightFirstMin = new int[heights.length];
        for (int i = 0; i < heights.length; i++) {
            while (!stack.isEmpty() && heights[stack.peek()] > heights[i]) {
                stack.pop();
            }
            leftFirstMin[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        stack.clear();
        for (int i = heights.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && heights[stack.peek()] > heights[i]) {
                stack.pop();
            }
            rightFirstMin[i] = stack.isEmpty() ? heights.length : stack.peek();
            stack.push(i);
        }
        int maxH = 0;
        for (int i = 0; i < heights.length; i++) {
            maxH = Math.max(maxH, (rightFirstMin[i] - leftFirstMin[i] - 1) * heights[i]);
        }
        return maxH;
    }

//    https://leetcode.com/problems/nested-list-weight-sum-ii/

    public interface NestedInteger {

         public boolean isInteger();

         public Integer getInteger();

         public void setInteger(int value);

         public void add(NestedInteger ni);

         public List<NestedInteger> getList();
    }

    void findMaxDepth(List<NestedInteger> nestedList, int d, int [] maxDepth) {
        maxDepth[0] = Math.max(maxDepth[0],d);
        for (NestedInteger nestedInteger : nestedList) {
            if (nestedInteger.isInteger()) continue;
            findMaxDepth(nestedInteger.getList(), d + 1, maxDepth);
        }
    }

    int depthSumInverseUtil(List<NestedInteger> nestedList, int d, int [] maxDepth) {
        int weightedSum = 0;
        for (NestedInteger nestedInteger : nestedList) {
            if (nestedInteger.isInteger()) weightedSum += (maxDepth[0] - d + 1) * nestedInteger.getInteger();
            weightedSum += depthSumInverseUtil(nestedInteger.getList(), d + 1, maxDepth);
        }
        return weightedSum;
    }

    public int depthSumInverse(List<NestedInteger> nestedList) {
//        int [] maxDepth = new int[]{0};
//        findMaxDepth(nestedList, 1, maxDepth);
//        return depthSumInverseUtil(nestedList, 1, maxDepth);

        int levelWeight = 0;
        int totalWeight = 0;
        Deque<NestedInteger> deque = new ArrayDeque<>(nestedList);
        while (!deque.isEmpty()) {
            int n = deque.size();
            for (int i = 0; i < n; i++) {
                NestedInteger nestedInteger = deque.remove();
                if (nestedInteger.isInteger()) levelWeight += nestedInteger.getInteger();
                else deque.addAll(nestedInteger.getList());
            }
            totalWeight += levelWeight;
        }
        return totalWeight;
    }

//    https://leetcode.com/problems/web-crawler-multithreaded/

    interface HtmlParser {
        public List<String> getUrls(String url);
    }

    void crawlUtils(String startUrl, HtmlParser htmlParser, List<String> l, Set<String> visited) {
        if (visited.contains(startUrl)) return;
        l.add(startUrl);
        visited.add(startUrl);
        String hostName = startUrl.split("/")[2];
        List<String> outLinks = htmlParser.getUrls(startUrl);
        List<Thread> threads = new ArrayList<>();
        for (String outLink : outLinks) {
            if (outLink.contains(hostName) && !visited.contains(outLink)) {
                Thread t = new Thread(() -> {
                    crawlUtils(outLink, htmlParser, l, visited);
                });
                threads.add(t);
                t.start();
            }
        }
        for (Thread t : threads) {
            try {
                t.join();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public List<String> crawl(String startUrl, HtmlParser htmlParser) {
        List<String> l = Collections.synchronizedList(new ArrayList<>());
        Set<String> visited = ConcurrentHashMap.newKeySet();
        crawlUtils(startUrl, htmlParser, l, visited);
        return l;
    }

//    https://leetcode.com/problems/mirror-reflection/

    int gcd(int x, int y) {
        if (y == 0) return x;
        return gcd(y, x % y);
    }

    public int mirrorReflection(int p, int q) {
        if (q == 0) return 0;
        int gcd = gcd(p,q);
        p /= gcd;
        q /= gcd;
        if (p % 2 == 0) return 2;
        if (q % 2 == 0) return 0;
        return 1;
    }

//    https://leetcode.com/problems/text-justification/

    public List<String> fullJustify(String[] words, int maxWidth) {
       List<String> result = new ArrayList<>();
       int n = words.length;
       int index = 0;
       while (index < n) {
           int totalChars = words[index].length();
           int last = index + 1;
           while (last < n) {
               if (totalChars + 1 + words[last].length() > maxWidth) break;
               totalChars += words[last].length() + 1;
               last++;
           }
           int gaps = last - index - 1;
           StringBuilder sb = new StringBuilder();
           if (last == n || gaps == 0) {
               for (int i = index; i < last; i++) {
                   sb.append(words[i]);
                   sb.append(" ");
               }
               sb.deleteCharAt(sb.length() - 1);
               while (sb.length() < maxWidth) sb.append(" ");
           } else {
               int spaces = (maxWidth - totalChars) / gaps;
               int rest = (maxWidth - totalChars) % gaps;
               for (int i = index; i < last - 1; i++) {
                   sb.append(words[i]);
                   sb.append(" ");
                   for (int j = 0; j < spaces + (i - index < rest ? 1 : 0); j++) sb.append(" ");
               }
               sb.append(words[last - 1]);
           }
           result.add(sb.toString());
           index = last;
       }
       return result;
    }

//    https://leetcode.com/problems/palindrome-linked-list/

    ListNode curr = null;

    boolean checkPalindromeRecursively(ListNode node) {
        if (node == null) return true;
        boolean b = checkPalindromeRecursively(node.next);
        if (!b) return false;
        if (curr.val == node.val) {
            curr = curr.next;
            return true;
        }
        return false;
    }

    public boolean isPalindrome(ListNode head) {
        curr = head;
        return checkPalindromeRecursively(head);
    }

//    https://leetcode.com/problems/split-bst/

    void splitBSTUtil(TreeNode node, int target, TreeNode [] smallerGreater, TreeNode s, TreeNode g) {
       if (node == null) return;
       if (node.val > target) {
           if (smallerGreater[1] == null) smallerGreater[1] = node;
           else g.left = node;
           g = node;
           TreeNode l = node.left;
           node.left = null;
           splitBSTUtil(l, target, smallerGreater, s, g);
       } else if (node.val < target) {
           if (smallerGreater[0] == null) smallerGreater[0] = node;
           else s.right = node;
           s = node;
           TreeNode r = node.right;
           node.right = null;
           splitBSTUtil(r, target, smallerGreater, s, g);
       } else {
           TreeNode r = node.right;
           node.right = null;
           if (smallerGreater[0] == null) smallerGreater[0] = node;
           else s.right = node;
           if (smallerGreater[1] == null) smallerGreater[1] = r;
           else g.left = r;
       }
    }

    public TreeNode[] splitBST(TreeNode root, int target) {
//        TreeNode [] smallerGreater = new TreeNode[2];
//        splitBSTUtil(root, target, smallerGreater, null, null);
//        return smallerGreater;
        TreeNode [] res = new TreeNode[2];
        if (root == null) return res;
        else if (root.val <= target) {
            TreeNode [] returned = splitBST(root.right, target);
            root.right = returned[0];
            res[0] = root;
            res[1] = returned[1];
        } else {
            TreeNode [] returned = splitBST(root.left, target);
            root.left = returned[1];
            res[0] = returned[0];
            res[1] = root;
        }
        return res;
    }

//    https://leetcode.com/problems/course-schedule-ii/

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        Map<Integer,Set<Integer>> adj = new HashMap<>();
        int [] inDegrees = new int[numCourses];
        for (int [] prerequisite : prerequisites) {
            int after = prerequisite[0];
            int before = prerequisite[1];
            Set<Integer> beforeN = adj.getOrDefault(before, new HashSet<>());
            beforeN.add(after);
            adj.put(before, beforeN);
            inDegrees[after]++;
        }
        Deque<Integer> deque = new ArrayDeque<>();
        Set<Integer> visited = new HashSet<>();
        int [] res = new int[numCourses];
        int i = 0;
        for (int j = 0; j < inDegrees.length; j++) {
            if (inDegrees[j] == 0) deque.add(j);
        }
        while (!deque.isEmpty()) {
            int n = deque.size();
            for (int j = 0; j < n; j++) {
                int before = deque.remove();
                if (visited.contains(before)) continue;
                visited.add(before);
                res[i] = before;
                i++;
                for (int neighbour : adj.getOrDefault(before, new HashSet<>())) {
                    inDegrees[neighbour]--;
                    if (inDegrees[neighbour] == 0) deque.add(neighbour);
                }
            }
        }
        return i == res.length ? res : new int[0];
    }

//    https://leetcode.com/problems/validate-binary-tree-nodes/

    boolean validate(int [] leftChild, int [] rightChild, int i, int n, Set<Integer> visited) {
        if (i < 0) return true;
        if (visited.contains(i)) return false;
        visited.add(i);
        return validate(leftChild, rightChild, leftChild[i], n, visited)
        && validate(leftChild, rightChild, rightChild[i], n, visited);
    }

    public boolean validateBinaryTreeNodes(int n, int[] leftChild, int[] rightChild) {
        Set<Integer> visited = new HashSet<>();
        boolean [] rootTable = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (leftChild[i] != -1) rootTable[leftChild[i]] = true;
            if (rightChild[i] != -1) rootTable[rightChild[i]] = true;
        }
        int root = -1;
        for (int i = 0; i < n; i++) {
            if (!rootTable[i] && root == -1) root = i;
            else if (!rootTable[i]) return false;
        }
        if (root == -1) return false;
        return validate(leftChild, rightChild, root, n, visited) && visited.size() == n;
    }

}
