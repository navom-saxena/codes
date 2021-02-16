package dsalgo.leetcode.company.amazon;

import java.util.*;

public class FirstSet {

    public static void main(String[] args) {

//        System.out.println(firstUniqChar("loveleetcode"));
//        System.out.println(singleNumber(new int[]{1}));
//        System.out.println(isPowerOfTwo(-16));
//        System.out.println(judgeCircle("RRDD"));
//        System.out.println(missingNumber(new int[]{0}));
//        System.out.println(maxProfit(new int[] {7,6,4,3,1}));
//        System.out.println(Arrays.toString(reorderLogFiles(new String[]
//                {"a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo","a2 act car"})));
//        System.out.println(partitionLabels("ababcbacadefegdehijhklij"));
//        System.out.println(numIslands(new char[][]{{'1','1','0','0','0'}, {'1','1','0','0','0'},
//                {'0','0','1','0','0'}, {'0','0','0','1','1'}}));
//        System.out.println(Arrays.toString(topKFrequent(new int[]{1,2}, 2)));
//        System.out.println(mostCommonWord("Bob hit a ball, the hit BALL flew far after it was hit.",
//                new String[]{"hit"}));
//        LRUCache lRUCache = new LRUCache(2);
//        lRUCache.put(1, 1); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
//        System.out.println(lRUCache.get(2));    // returns -1 (not found)
//        lRUCache.put(4, 1); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
//        System.out.println(lRUCache.get(2));    // return -1 (not found)
//        System.out.println(lRUCache.get(3));    // return 3
//        System.out.println(lRUCache.get(4));    // return 4
//        System.out.println(Arrays.toString(prisonAfterNDays(new int[]{1,0,0,1,0,0,1,0},1000000000)));
//        System.out.println(Arrays.deepToString(kClosest(new int[][]{{3,3},{5,-1},{-2,4}}, 2)));
//        ListNode a = new ListNode(1);
//        a.next = new ListNode(4);
//        a.next.next = new ListNode(5);
//        ListNode b = new ListNode(1);
//        b.next = new ListNode(3);
//        b.next.next = new ListNode(4);
//        ListNode c = new ListNode(2);
//        c.next = new ListNode(6);
//        ListNode m = mergeKLists(new ListNode[]{a,b,c});
//        System.out.println(m);
//        System.out.println(calPoints(new String[]{"36","28","70","65","C","+","33","-46","84","C"}));
//        System.out.println(Arrays.toString(twoSum(new int[]{2, 7, 11, 15}, 9)));
//        System.out.println(findCircleNum(new int[][]{{1,1,0},{1,1,1},{0,1,1}}));
        System.out.println(findWords(new char[][]{{'a','a'}},new String[]{"aa"}));
    }

//    https://leetcode.com/problems/first-unique-character-in-a-string/

    public static int firstUniqChar(String s) {
        int [] countArr = new int[27];
        int length = s.length();
        for (int i = 0; i < length; i++) {
            countArr[s.charAt(i) - 'a']++;
        }
        for (int i = 0; i < length; i++) {
            if (countArr[s.charAt(i) - 'a'] == 1) {
                return i;
            }
        }
        return -1;
    }

//    https://leetcode.com/problems/single-number/

    public static int singleNumber(int[] nums) {
        int xorResult = 0;
        for (int num : nums) {
            xorResult ^= num;
        }
        return xorResult;
    }

//    https://leetcode.com/problems/power-of-two/

    public static boolean isPowerOfTwo(int n) {
        long i = 1;
        while (i < n) {
            i *= 2;
        }
        return i == n;
    }

//    https://leetcode.com/problems/robot-return-to-origin/

    public static boolean judgeCircle(String moves) {
        int hCount = 0;
        int vCount = 0;
        int length = moves.length();
        for (int i = 0; i < length; i++) {
            switch (moves.charAt(i)) {
                case 'U' :
                    vCount++;
                    break;
                case 'D' :
                    vCount--;
                    break;
                case 'L' :
                    hCount++;
                    break;
                case 'R' :
                    hCount--;
                    break;
                default:
                    break;
            }
        }
        return hCount == 0 && vCount == 0;
    }

//    https://leetcode.com/problems/missing-number/

    public static int missingNumber(int[] nums) {
        int nSum = ((nums.length + 1) * (nums.length)) / 2;
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        return nSum - sum;
    }

//    https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

    public static int maxProfit(int[] prices) {
        int maxProfit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] - prices[i - 1] > 0) {
                maxProfit += prices[i] - prices[i - 1];
            }
        }
        return maxProfit;
    }

//    https://leetcode.com/problems/reorder-data-in-log-files/

    public static String[] reorderLogFiles(String[] logs) {
        Arrays.sort(logs, (o1, o2) -> {
            int splitIndex1 = o1.indexOf(" ");
            int splitIndex2 = o2.indexOf(" ");;

            boolean isDigit1 = Character.isDigit(o1.charAt(splitIndex1 + 1));
            boolean isDigit2 = Character.isDigit(o2.charAt(splitIndex2 + 1));
            if (isDigit1 && isDigit2) {
                return 0;
            } else if (isDigit1) {
                return 1;
            } else if (isDigit2) {
                return -1;
            } else {
                String word1 = o1.substring(splitIndex1 + 1);
                String word2 = o2.substring(splitIndex2 + 1);
                int res = word1.compareTo(word2);
                if (res == 0) {
                    String id1 = o1.substring(0, splitIndex1);
                    String id2 = o2.substring(0, splitIndex2);
                    return id1.compareTo(id2);
                }
                return res;
            }
        });
        return logs;
    }

//    https://leetcode.com/problems/partition-labels/

    public static List<Integer> partitionLabels(String s) {
        int [] lastIndexes = new int[27];
        for (int i = 0; i < s.length(); i++) {
            lastIndexes[s.charAt(i) - 'a'] = i;
        }
        List<Integer> partitions = new ArrayList<>();
        int i = 0;
        while (i < s.length()) {
            int lastIndex = lastIndexes[s.charAt(i) - 'a'];
            int startIndex = i;
            while (i <= lastIndex) {
                if (lastIndexes[s.charAt(i) - 'a'] > lastIndex) {
                    lastIndex = lastIndexes[s.charAt(i) - 'a'];
                }
                i++;
            }
            partitions.add(i - startIndex);
        }
        return partitions;
    }

//    https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/

    public static int minDifficulty(int[] jobDifficulty, int d) {
        return Integer.MAX_VALUE;
    }

//    https://leetcode.com/problems/number-of-islands/

    static void calculateCount(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == '0' || grid[i][j] == '2') {
            return;
        }
        if (grid[i][j] == '1') {
            grid[i][j] = '2';
            calculateCount(grid, i + 1, j);
            calculateCount(grid, i - 1, j);
            calculateCount(grid, i, j + 1);
            calculateCount(grid, i, j - 1);
        }
    }

    public static int numIslands(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    calculateCount(grid, i, j);
                }
            }
        }
        return count;
    }

//    https://leetcode.com/problems/critical-connections-in-a-network/

    public static List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
       return new ArrayList<>();
    }

//    https://leetcode.com/problems/top-k-frequent-elements/

    public static int[] topKFrequent(int[] nums, int k) {
        Map<Integer,Integer> counts = new HashMap<>();
        for (int num : nums) {
            counts.merge(num, 1, Integer::sum);
        }

        PriorityQueue<Integer> minHeap = new PriorityQueue<>(Comparator.comparingInt(counts::get));
        for (int key : counts.keySet()) {
            minHeap.offer(key);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }

        int [] result = new int[k];
        for (int i = 0; i < k; i++) {
            result[i] = minHeap.remove();
        }
        return result;
    }

//    https://leetcode.com/problems/most-common-word/

    public static String mostCommonWord(String paragraph, String[] banned) {
        Set<String> bannedHash = new HashSet<>(Arrays.asList(banned));
        Map<String,Integer> frequency = new HashMap<>();
        for (String word : paragraph.split("[\\s!?',;.]")) {
            String cleanedWord = word.toLowerCase();
            if (!cleanedWord.equals("") && !bannedHash.contains(cleanedWord)) {
                frequency.merge(cleanedWord, 1, Integer::sum);
            }
        }
        String mostCommon = null;
        int maxFreq = Integer.MIN_VALUE;
        for (String key : frequency.keySet()) {
            int freq = frequency.get(key);
            if (freq > maxFreq) {
                mostCommon = key;
                maxFreq = freq;
            }
        }
        return mostCommon;
    }

//    https://leetcode.com/problems/prison-cells-after-n-days/

    public static int[] prisonAfterNDays(int[] cells, int n) {
        Set<String> hs = new HashSet<>();
        boolean isCompletelyBuild = false;
        for (int i = 1; i <= n; i++) {
            int prevCellState = cells[0];
            for (int j = 1; j < cells.length - 1; j++) {
                if ((prevCellState == 1 && cells[j + 1] == 1) || (prevCellState == 0 && cells[j + 1] == 0)) {
                    prevCellState = cells[j];
                    cells[j] = 1;
                } else {
                    prevCellState = cells[j];
                    cells[j] = 0;
                }
            }
            cells[0] = 0;
            cells[cells.length - 1] = 0;
            String stringify = Arrays.toString(cells);
            if (!hs.contains(stringify)) {
                hs.add(stringify);
            } else {
                isCompletelyBuild = true;
                break;
            }
        }
        if (isCompletelyBuild) {
            for (int i = 1; i <= ((n % hs.size()) - 1 + hs.size()) % hs.size(); i++) {
                int prevCellState = cells[0];
                for (int j = 1; j < cells.length - 1; j++) {
                    if ((prevCellState == 1 && cells[j + 1] == 1) || (prevCellState == 0 && cells[j + 1] == 0)) {
                        prevCellState = cells[j];
                        cells[j] = 1;
                    } else {
                        prevCellState = cells[j];
                        cells[j] = 0;
                    }
                }
                cells[0] = 0;
                cells[cells.length - 1] = 0;
            }
        }
        return cells;
    }

//    https://leetcode.com/problems/k-closest-points-to-origin/

    public static int[][] kClosest(int[][] points, int k) {
       PriorityQueue<int []> maxHeap = new PriorityQueue<>((a,b) -> ((b[0]*b[0]) + (b[1]*b[1])) - ((a[0]*a[0]) + (a[1]*a[1])));
        for (int [] point : points) {
            maxHeap.offer(point);
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
        }
        int [][] result = new int[k][];
        int i = 0;
        while (maxHeap.size() != 0) {
            result[i] = maxHeap.poll();
            i++;
        }
        return result;
    }

//    https://leetcode.com/problems/copy-list-with-random-pointer/

    static class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    public static Node copyRandomList(Node head) {
        Map<Node,Node> mapping = new HashMap<>();
        Node curr = head;
        while (curr != null) {
            mapping.put(curr, new Node(curr.val));
            curr = curr.next;
        }
        curr = head;
        while (curr != null) {
            if (curr.next != null) {
                mapping.get(curr).next = mapping.get(curr.next);
            }
            if (curr.random != null) {
                mapping.get(curr).random = mapping.get(curr.random);
            }
            curr = curr.next;
        }
        return mapping.get(head);
    }

//    https://leetcode.com/problems/merge-k-sorted-lists/

    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    public static ListNode mergeKLists(ListNode[] lists) {
//        ListNode head = null;
//        ListNode curr = head;
//        int k = lists.length;
//        ListNode min = null;
//        int j = 0;
//        int allNullCounter = 0;
//        while (allNullCounter < k) {
//            allNullCounter = 0;
//            for (int i = 0; i < k; i++) {
//                ListNode listNode = lists[i];
//                if (listNode != null) {
//                    if (min == null) {
//                        min = listNode;
//                        j = i;
//                    } else if (min.val > listNode.val) {
//                        min = listNode;
//                        j = i;
//                    }
//                } else {
//                    allNullCounter++;
//                }
//            }
//            if (min != null) {
//                if (curr == null) {
//                    curr = new ListNode(min.val);
//                    head = curr;
//                } else {
//                    curr.next = new ListNode(min.val);
//                    curr = curr.next;
//                }
//                min = min.next;
//                if (lists[j] != null) {
//                    lists[j] = lists[j].next;
//                }
//            }
//        }
//        return head;

//        optimised
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (ListNode listNode : lists) {
            while (listNode != null) {
                minHeap.offer(listNode.val);
                listNode = listNode.next;
            }
        }
        ListNode head = new ListNode(Integer.MIN_VALUE);
        ListNode curr = head;
        while (!minHeap.isEmpty()) {
            Integer value = minHeap.poll();
            curr.next = new ListNode(value);
            curr = curr.next;
        }
        return head.next;
    }

//    https://leetcode.com/problems/word-ladder/

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        return -1;
    }

//    https://leetcode.com/problems/subtree-of-another-tree/

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

    public static boolean check(TreeNode s, TreeNode t) {
        if (s == null || t == null) {
            return s == null && t == null;
        }
        if (s.val != t.val) {
            return false;
        }
        return check(s.left, t.left) && check(s.right, t.right);
    }

    public static boolean isSubtree(TreeNode s, TreeNode t) {
        if (s == null) {
            return false;
        }
        if (!check(s,t)) {
            return isSubtree(s.left, t) || isSubtree(s.right, t);
        }
        return true;
    }

//    https://leetcode.com/problems/maximal-square/

    public static int maximalSquare(char[][] matrix) {
        int [][] dp = new int[matrix.length + 1][matrix[0].length + 1];
        int largest = 0;
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[i].length; j++) {
                if (matrix[i - 1][j - 1] == '1') {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    if (dp[i][j] > largest) {
                        largest = dp[i][j];
                    }
                }
            }
        }
        return largest * largest;
    }

//    https://leetcode.com/problems/baseball-game/

    public static int calPoints(String[] ops) {
        int result = 0;
        Stack<Integer> stack = new Stack<>();
        for (String value : ops) {
            switch (value) {
                case "C":
                    int popped = stack.pop();
                    result -= popped;
                    break;
                case "D" :
                    int peekDouble = stack.peek() * 2;
                    stack.push(peekDouble);
                    result += peekDouble;
                    break;
                case "+" :
                    int firstValue = stack.pop();
                    int secondValue = stack.peek();
                    stack.push(firstValue);
                    int sum = firstValue + secondValue;
                    stack.push(sum);
                    result += sum;
                    break;
                default:
                    int no = Integer.parseInt(value);
                    stack.push(no);
                    result += no;
            }
        }
        return result;
    }

//    https://leetcode.com/problems/two-sum/

    public static int[] twoSum(int[] nums, int target) {
      Map<Integer,Integer> hm = new HashMap<>();
      int [] result = new int[2];
      for (int i = 0; i < nums.length; i++) {
          if (hm.get(target - nums[i]) != null) {
              result[0] = i;
              result[1] = hm.get(target - nums[i]);
          } else {
              hm.put(nums[i], i);
          }
      }
      return result;
    }

//    https://leetcode.com/problems/friend-circles/

    static void dfs(int [][] grid, int i) {
        for (int indexJ = 0; indexJ < grid[i].length; indexJ++) {
            if (grid[i][indexJ] == 1 && i == indexJ) {
                grid[i][indexJ] = 0;
            }
            if (grid[i][indexJ] == 1) {
                grid[i][indexJ] = 0;
                grid[indexJ][i] = 0;
                dfs(grid, indexJ);
            }
        }
    }

    public static int findCircleNum(int[][] m) {
        int count = 0;
        for (int i = 0; i < m.length; i++) {
            if (m[i][i] == 1) {
                dfs(m, i);
                count++;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/concatenated-words/

    public static boolean canFormWord(String word, Set<String> hs) {
        if (cache.contains(word)) {
            return true;
        }
        for (int i = 1; i < word.length(); i++) {
            String left = word.substring(0,i);
            String right = word.substring(i);
            if (hs.contains(left)) {
                if (hs.contains(right) || canFormWord(right,hs)) {
                    cache.add(word);
                    return true;
                }
            }
        }
        return false;
    }

    static Set<String> cache = new HashSet<>();

    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        Set<String> hs = new HashSet<>(Arrays.asList(words));
        List<String> result = new ArrayList<>();
        for (String word : words) {
            if (canFormWord(word, hs)) {
                result.add(word);
            }
        }
        return result;
    }

//    https://leetcode.com/problems/merge-two-sorted-lists/

    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(Integer.MIN_VALUE);
        ListNode current = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                current.next = new ListNode(l1.val);
                l1 = l1.next;
            } else {
                current.next = new ListNode(l2.val);
                l2 = l2.next;
            }
            current = current.next;
        }
        while (l1 != null) {
            current.next = new ListNode(l1.val);
            l1 = l1.next;
            current = current.next;
        }
        while (l2 != null) {
            current.next = new ListNode(l2.val);
            l2 = l2.next;
            current = current.next;
        }
        return dummy.next;
    }

//    https://leetcode.com/problems/word-search-ii/

    public static boolean wordSearchDfs(char [][] board, String word, int i, int j, int index) {
        if (index == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || word.charAt(index) != board[i][j]) {
            return false;
        }
        char temp = board[i][j];
        board[i][j] = ' ';
        boolean check = wordSearchDfs(board, word, i - 1 ,j ,index + 1) ||
                wordSearchDfs(board, word, i + 1 ,j ,index + 1) ||
                wordSearchDfs(board, word, i ,j - 1 ,index + 1) ||
                wordSearchDfs(board, word, i ,j + 1 ,index + 1);
        board[i][j] = temp;
        return check;
    }

    public static List<String> findWords(char[][] board, String[] words) {
        Set<String> result = new HashSet<>();
        for (String word : words) {
            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[i].length; j++) {
                    if (board[i][j] == word.charAt(0) && wordSearchDfs(board,word,i,j, 0)) {
                        result.add(word);
                    }
                }
            }
        }
        return new ArrayList<>(result);
    }

//    https://leetcode.com/problems/trapping-rain-water/

    public static int trap(int[] height) {
        int i = 0;
        int j = height.length - 1;
        int lMax = 0;
        int rMax = 0;
        int result = 0;
        while (i <= j) {
            lMax = Math.max(lMax,height[i]);
            rMax = Math.max(rMax,height[j]);
            if (lMax < rMax) {
                result += Math.max(lMax - height[i], 0);
                i++;
            } else {
                result += Math.max(rMax - height[j], 0);
                j--;
            }
        }
        return result;
    }

//    https://leetcode.com/problems/word-break-ii/

    static Map<String,List<String>> wordBreakCache = new HashMap<>();

    public List<String> wordBreak(String s, List<String> wordDict) {
        if (wordBreakCache.containsKey(s)) {
            return wordBreakCache.get(s);
        }
        List<String> result = new ArrayList<>();
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                if (s.length() == word.length()) {
                    result.add(word);
                } else {
                    List<String> temp = wordBreak(s.substring(word.length()),wordDict);
                    for (String t : temp) {
                        result.add(word + " " + t);
                    }
                }
            }
        }
        wordBreakCache.put(s,result);
        return result;
    }

//    https://leetcode.com/problems/lru-cache/

    static class LruNode {
        int key;
        int value;
        LruNode next;
        LruNode previous;

        LruNode(int key, int data) {
            this.key = key;
            this.value = data;
        }
    }

    static class LRUCache {

        Map<Integer, LruNode> hm;
        LruNode head;
        LruNode last;
        int count;
        int capacity;

        public LRUCache(int capacity) {
            this.hm = new HashMap<>();
            this.capacity = capacity;
            this.count = 0;
        }

        public int get(int key) {
            if (this.hm.containsKey(key)) {
                LruNode ref = this.hm.get(key);

                if (ref == this.head) {
                    return hm.get(key).value;
                }

                if (ref == this.last && this.last.next != null) {
                    this.last = this.last.next;
                }

                LruNode prevN = ref.previous;
                LruNode nextN = ref.next;

                if (prevN != null) {
                    prevN.next = nextN;
                }
                if (nextN != null) {
                    nextN.previous = prevN;
                }

                ref.next = null;
                ref.previous = null;
                this.head.next = ref;
                ref.previous = this.head;
                this.head = this.head.next;

                return ref.value;
            } else {
                return -1;
            }
        }

        public void put(int key, int value) {
            LruNode ref;
            if (this.hm.containsKey(key)) {
                ref = this.hm.get(key);
                ref.value = value;

                if (ref == this.head) {
                    return;
                }

                if (ref == this.last && this.last.next != null) {
                    this.last = this.last.next;
                }

                LruNode prevN = ref.previous;
                LruNode nextN = ref.next;

                if (prevN != null) {
                    prevN.next = nextN;
                }
                if (nextN != null) {
                    nextN.previous = prevN;
                }
            } else {
                ref = new LruNode(key, value);
                this.hm.put(key, ref);
                if (this.last == null) {
                    this.last = ref;
                }
                this.count++;
            }
            if (this.head != null) {
                this.head.next = ref;
                ref.previous = this.head;
                this.head = this.head.next;
            } else {
                this.head = ref;
            }
            if (this.count > this.capacity) {
                if (this.hm.get(this.last.key).value == this.last.value) {
                    hm.remove(this.last.key);
                }
                this.last = this.last.next;
                this.last.previous = null;
                this.count--;
            }
        }
    }
}
