package dsalgo.leetcode.company.facebook;

import dsalgo.leetcode.Models.*;

import java.util.*;

public class EighthSet {

    public static void main(String[] args) {
//        System.out.println(Integer.toBinaryString(-3));
//        MaxStack stack = new MaxStack();
//        stack.push(5);
//        stack.push(1);
//        stack.push(5);
//        stack.top();
//        stack.popMax();
//        stack.top();
        MagicDictionary magicDictionary = new MagicDictionary();
        magicDictionary.buildDict(new String[]{"judge","judgg"});
        System.out.println(magicDictionary.search("juggg"));
    }

//    https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/

    public int numSubmatrixSumTarget(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        for (int i = 0; i < m; i++) {
            int prefixSum = 0;
            for (int j = 0; j < n; j++) {
                prefixSum += matrix[i][j];
                matrix[i][j] = prefixSum;
            }
        }
        int count = 0;
        for (int c1 = 0; c1 < n; c1++) {
            for (int c2 = c1; c2 < n; c2++) {
                Map<Integer,Integer> prevMapping = new HashMap<>();
                prevMapping.put(0,1);
                int sum = 0;
                for (int[] row : matrix) {
                    sum += row[c2] - c1 > 0 ? row[c1 - 1] : 0;
                    if (prevMapping.containsKey(sum - target)) {
                        count += prevMapping.getOrDefault(sum - target, 0);
                    }
                    prevMapping.put(sum, prevMapping.getOrDefault(sum, 0) + 1);
                }
            }
        }
        return count;
    }

//    https://leetcode.com/problems/daily-temperatures/

    public int[] dailyTemperatures(int[] temperatures) {
        Deque<Integer> stack = new ArrayDeque<>();
        int [] res = new int[temperatures.length];
        for (int i = temperatures.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && temperatures[stack.peek()] <= temperatures[i]) stack.pop();
            res[i] = stack.isEmpty() ? 0 : stack.peek() - i;
            stack.push(i);
        }
        return res;
    }

//    https://leetcode.com/problems/path-sum-iii/

    void pathSumThreeUtils(TreeNode node, int target, Map<Integer,Integer> prevMapping, int runningSum, int [] count) {
        if (node == null) return;
        int currSum = runningSum + node.val;
        if (prevMapping.containsKey(currSum - target)) {
            count[0] += prevMapping.get(currSum - target);
        }
        prevMapping.put(currSum,prevMapping.getOrDefault(currSum,0) + 1);
        pathSumThreeUtils(node.left, target, prevMapping, currSum, count);
        pathSumThreeUtils(node.right, target, prevMapping, currSum, count);
        int currVFreq = prevMapping.get(currSum);
        if (currVFreq > 1) prevMapping.put(currSum, currVFreq - 1);
        else prevMapping.remove(currSum);
    }

    public int pathSum(TreeNode root, int targetSum) {
        int [] count = new int[]{0};
        Map<Integer,Integer> prevMapping = new HashMap<>();
        prevMapping.put(0,1);
        pathSumThreeUtils(root, targetSum, prevMapping, 0, count);
        return count[0];
    }

//    https://leetcode.com/problems/shortest-palindrome/

    public String shortestPalindrome(String s) {
        char [] sArr = s.toCharArray();
        int j;
        int i;
        int counter = 0;
        while (true) {
            i = 0;
            j = sArr.length - 1 - counter;
            while (i < j) {
                if (sArr[i] == sArr[j]) {
                    i++;
                    j--;
                } else break;
            }
            if (i >= j) break;
            counter++;
        }
        StringBuilder sb = new StringBuilder();
        j = sArr.length - 1;
        while (counter > 0) {
            sb.append(sArr[j]);
            counter--;
            j--;
        }
        sb.append(s);
        return sb.toString();
    }

//    https://leetcode.com/problems/coin-change-2/

    boolean changeUtils(int amount, int [] coins, int i, int [] count, boolean [][] dp) {
        if (amount == 0 || (i >= 0 && dp[amount][i])) return true;
        if (i < 0) return false;
        int maxTaken = amount / coins[i];
        for (int j = 0; j <= maxTaken; j++) {
            boolean sumZero = changeUtils(amount, coins, i - 1, count, dp);
            if (sumZero) {
                dp[amount][i] = true;
                count[0]++;
            }
            amount -= coins[i];
        }
        return false;
    }

    public int change(int amount, int[] coins) {
//        if (amount == 0) return 1;
//        int [] count = new int[]{0};
//        boolean [][] dp = new boolean[amount + 1][coins.length];
//        changeUtils(amount, coins, coins.length - 1, count, dp);
//        return count[0];
        int [][] dp = new int[coins.length + 1][amount + 1];
        dp[0][0] = 1;
        for (int i = 1; i < dp.length; i++) dp[i][0] = 1;
        for (int j = 1; j < dp[0].length; j++) dp[0][j] = 0;
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[0].length; j++) {
                dp[i][j] += dp[i - 1][j] + (j - coins[i - 1] >= 0 ? dp[i][j - coins[i - 1]] : 0);
            }
        }
        return dp[dp.length - 1][dp[0].length - 1];
    }

//    https://leetcode.com/problems/find-duplicate-file-in-system/

    public List<List<String>> findDuplicate(String[] paths) {
        Map<String,List<String>> contentMapping = new HashMap<>();
        for (String path : paths) {
            String [] dirPaths = path.split(" ");
            String dir = dirPaths[0];
            for (int i = 1; i < dirPaths.length; i++) {
                String s = dirPaths[i];
                String absolutePath = dir + "/" + s.substring(0, s.indexOf("("));
                String content = s.substring(s.indexOf("(") + 1,s.indexOf(")"));
                List<String> sameContentPaths = contentMapping.getOrDefault(content, new ArrayList<>());
                sameContentPaths.add(absolutePath);
                contentMapping.put(content, sameContentPaths);
            }
        }
        List<List<String>> res = new ArrayList<>();
        for (List<String> v : contentMapping.values()) {
            if (v.size() > 1) res.add(v);
        }
        return res;
    }

//    https://leetcode.com/problems/range-sum-query-immutable/

    static class NumArray {

        int [] prefixSumArr;

        public NumArray(int[] nums) {
            prefixSumArr = new int[nums.length];
            int prefixSum = 0;
            for (int i = 0; i < nums.length; i++) {
                prefixSum += nums[i];
                prefixSumArr[i] = prefixSum;
            }
        }

        public int sumRange(int left, int right) {
            if (left > 0) return prefixSumArr[right] - prefixSumArr[left - 1];
            else return prefixSumArr[right];
        }
    }

//    https://leetcode.com/problems/kth-largest-element-in-a-stream/

    static class KthLargest {

        PriorityQueue<Integer> minHeap;
        int k;

        public KthLargest(int k, int[] nums) {
            this.k = k;
            minHeap = new PriorityQueue<>();
            for (int num : nums) minHeap.add(num);
        }

        public int add(int val) {
            minHeap.add(val);
            while (minHeap.size() > k) minHeap.remove();
            if (minHeap.isEmpty()) return -1;
            return minHeap.peek();
        }
    }

//    https://leetcode.com/problems/uncommon-words-from-two-sentences/

    public String[] uncommonFromSentences(String s1, String s2) {
        Map<String,Integer> freqMap = new HashMap<>();
        for (String word : s1.split(" ")) freqMap.merge(word, 1, Integer::sum);
        for (String word : s2.split(" ")) freqMap.merge(word, 1, Integer::sum);
        List<String> unCommonWords = new ArrayList<>();
        for (String key : freqMap.keySet()) {
            if (freqMap.get(key) == 1) unCommonWords.add(key);
        }
        String [] res = new String[unCommonWords.size()];
        for (int i = 0; i < unCommonWords.size(); i++) {
            res[i] = unCommonWords.get(i);
        }
        return res;
    }

//    https://leetcode.com/problems/valid-anagram/

    public boolean isAnagram(String s, String t) {
        char [] s1C = new char[26];
        char [] s2C = new char[26];
        for (char c : s.toCharArray()) s1C[c - 'a']++;
        for (char c : t.toCharArray()) s2C[c - 'a']++;
        for (int i = 0; i < 26; i++) if (s1C[i] != s2C[i]) return false;
        return true;
    }

//    https://leetcode.com/problems/shortest-path-in-binary-matrix/

    boolean checkCondAndBounds(int x, int y, int [][] grid) {
        return x >= 0 && x < grid.length && y >= 0 && y < grid[x].length && grid[x][y] == 0;
    }

    public int shortestPathBinaryMatrix(int[][] grid) {
        int r = grid.length;
        int c = grid[0].length;
        if (grid[0][0] == 1) return -1;
        int [][] directions = new int[][]{{1,0}, {-1,0}, {0,1}, {0,-1}, {1,1}, {-1,-1}, {-1,1}, {1,-1}};
        Set<Integer> visited = new HashSet<>();
        Deque<int[]> deque = new ArrayDeque<>();
        deque.add(new int[]{0,0});
        int d = 0;
        while (!deque.isEmpty()) {
            int n = deque.size();
            for (int i = 0; i < n; i++) {
                int [] node = deque.remove();
                int x = node[0];
                int y = node[1];
                if (x == r - 1 && y == c - 1) return d;
                int hash = (x * r) + y;
                if (visited.contains(hash)) continue;
                visited.add(hash);
                for (int [] direction : directions) {
                    int newX = x + direction[0];
                    int newY = y + direction[1];
                    int newHash = newX * r + newY;
                    if (!visited.contains(newHash) && checkCondAndBounds(newX, newY, grid))
                        deque.add(new int[]{newX,newY});
                }
            }
            d++;
        }
        return -1;
    }

//    https://leetcode.com/problems/number-of-1-bits/

    public int hammingWeight(int n) {
        int bits = 0;
        int mask = 1;
        for (int i = 0; i < 32; i++) {
            if ((n & mask) != 0) {
                bits++;
            }
            mask <<= 1;
        }
        return bits;
    }

//    https://leetcode.com/problems/swim-in-rising-water/

    boolean checkBounds(int x, int y, int [][] grid) {
        return x >= 0 && x < grid.length && y >= 0 && y < grid[x].length;
    }

    static class RisingWater {
        int x;
        int y;
        int t;

        RisingWater(int x, int y, int t) {
            this.x = x;
            this.y = y;
            this.t = t;
        }
    }

    public int swimInWater(int[][] grid) {
        int n = grid.length;
        PriorityQueue<RisingWater> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> a.t));
        minHeap.add(new RisingWater(0, 0, grid[0][0]));
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int maxTimeElapsed = 0;
        Set<Integer> visited = new HashSet<>();
        while (!minHeap.isEmpty()) {
            RisingWater rW = minHeap.remove();
            maxTimeElapsed = Math.max(maxTimeElapsed, rW.t);
            if (rW.x == grid.length - 1 && rW.y == grid[0].length - 1) return maxTimeElapsed;
            int hash = n * rW.x + rW.y;
            if (visited.contains(hash)) continue;
            visited.add(hash);
            for (int[] direction : directions) {
                int newX = rW.x + direction[0];
                int newY = rW.y + direction[1];
                int newHash = newX * n + newY;
                if (checkBounds(newX, newY, grid) && !visited.contains(newHash))
                    minHeap.add(new RisingWater(newX,newY, grid[newX][newY]));
            }
        }
        return -1;
    }

//    https://leetcode.com/problems/valid-tic-tac-toe-state/

    public boolean validTicTacToe(String[] board) {
        int n = board.length;
        int xC = 0;
        int zC = 0;
        int [] row = new int[3];
        int [] col = new int[3];
        int d1 = 0;
        int d2 = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < board[i].length(); j++) {
                if (board[i].charAt(j) == 'X') {
                    xC++;
                    row[i]++;
                    col[j]++;
                    if (i == j) d1++;
                    if (i + j == n - 1) d2++;
                }
                else if (board[i].charAt(j) =='O') {
                    zC--;
                    row[i]--;
                    col[j]--;
                    if (i == j) d1--;
                    if (i + j == n - 1) d2--;
                }
            }
        }
        if (xC + zC > 1 || xC + zC < 0) return false;
        int xWin = 0;
        int zWin = 0;
        if (d1 == 3) xWin++;
        else if (d1 == -3) zWin++;
        if (d2 == 3) xWin++;
        else if (d2 == -3) zWin++;
        for (int r : row) {
            if (r == 3) xWin++;
            else if (r == -3) zWin++;
        }
        for (int c : col) {
            if (c == 3) xWin++;
            else if (c == -3) zWin++;
        }
        if (xWin > 0 && zWin > 0) return false;
        if (xWin > 0 && (xC != (zC * -1) + 1)) return false;
        if (zWin > 0 && xC != zC * -1) return false;
        return xWin <= 2 && zWin <= 2;
    }

//    https://leetcode.com/problems/peeking-iterator/

    static class PeekingIterator implements Iterator<Integer> {

        Iterator<Integer> it;
        Integer peekedValue;

        public PeekingIterator(Iterator<Integer> iterator) {
            // initialize any member here.
            it = iterator;
        }

        // Returns the next element in the iteration without advancing the iterator.
        public Integer peek() {
           if (peekedValue == null) {
               if (it.hasNext()) peekedValue = it.next();
               else throw new NoSuchElementException();
           }
           return peekedValue;
        }

        // hasNext() and next() should behave the same as in the Iterator interface.
        // Override them if needed.
        @Override
        public Integer next() {
            if (peekedValue != null) {
                int v = peekedValue;
                peekedValue = null;
                return v;
            }
            if (it.hasNext()) {
                return it.next();
            }
            throw new NoSuchElementException();
        }

        @Override
        public boolean hasNext() {
           return peekedValue != null || it.hasNext();
        }
    }

//    https://leetcode.com/problems/find-the-kth-smallest-sum-of-a-matrix-with-sorted-rows/

    static class Data {
        int sum;
        int [] r;

        Data(int sum, int [] r) {
            this.sum = sum;
            this.r = r;
        }

        @Override
        public String toString() {
            return "Data{" +
                    "sum=" + sum +
                    ", r=" + Arrays.toString(r) +
                    '}';
        }
    }

    public int kthSmallest(int[][] mat, int k) {
        int m = mat.length;
        int n = mat[0].length;
        Set<String> visited = new HashSet<>();
        PriorityQueue<Data> minHeap = new PriorityQueue<>(Comparator.comparingInt(x -> x.sum));
        int s = 0;
        for (int[] ints : mat) {
            s += ints[0];
        }
        minHeap.add(new Data(s, new int[m]));
        while (!minHeap.isEmpty()) {
            k--;
            System.out.println(minHeap);
            Data d = minHeap.remove();
            if (k == 0) return d.sum;
            int [] r = d.r;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    if (r[j] < n - 1) {
                        int a = mat[j][r[j]];
                        int b =  mat[j][r[j] + 1];
                        int sum = d.sum - a + b;
                        int [] newR = Arrays.copyOf(r, r.length);
                        newR[j]++;
                        String arrS = Arrays.toString(newR);
                        if (visited.contains(arrS)) continue;
                        visited.add(arrS);
                        minHeap.add(new Data(sum, newR));
                    }
                }
            }
        }
        return -1;
    }

//    https://leetcode.com/problems/queue-reconstruction-by-height/

    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people,(a,b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        int [][] res = new int[people.length][people[0].length];
        for (int [] ints : res) Arrays.fill(ints, -1);
        int n = people.length;
        for (int[] person : people) {
            int h = person[0];
            int count = person[1];
            for (int j = 0; j < n; j++) {
                if (res[j][0] == -1 && count == 0) {
                    res[j][0] = person[0];
                    res[j][1] = person[1];
                    break;
                } else if (res[j][0] == -1 || res[j][0] >= h) count--;
            }
        }
        return res;
    }

//    https://leetcode.com/problems/closest-binary-search-tree-value-ii/

    void updatePathToLeaf(TreeNode node, PriorityQueue<Integer> maxHeap, int k) {
        if (node == null) return;
        updatePathToLeaf(node.left, maxHeap, k);
        maxHeap.add(node.val);
        while (maxHeap.size() > k) maxHeap.remove();
        updatePathToLeaf(node.right, maxHeap, k);
    }

    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) ->
                Double.compare(Math.abs(b - target),Math.abs(a - target)));
        updatePathToLeaf(root, maxHeap, k);
        return new ArrayList<>(maxHeap);
    }

//    https://leetcode.com/problems/lfu-cache/

//    better sol - https://leetcode.com/problems/lfu-cache/discuss/1457717/Java-Solution-LFU

    static class LfuNode {
        int key;
        int value;
        int freq;
        LfuNode next;
        LfuNode prev;

        LfuNode(int k, int v, int f) {
            this.key = k;
            this.value = v;
            next = null;
            prev = null;
            freq = f;
        }

        @Override
        public String toString() {
            return "LfuNode{" +
                    "key=" + key +
                    ", value=" + value +
                    ", freq=" + freq +
                    '}';
        }
    }

    static class DdlLfuCache {
        LfuNode first;
        LfuNode last;

        DdlLfuCache(LfuNode f, LfuNode l) {
            first = f;
            last = l;
        }
    }

    static class LFUCache {

        Map<Integer,LfuNode> map;
        Map<Integer,DdlLfuCache> freqList;
        int capacity;
        int n;
        int minFreq;

        public LFUCache(int capacity) {
            this.capacity = capacity;
            this.n = 0;
            map = new HashMap<>();
            freqList = new HashMap<>();
            minFreq = 0;
        }

        void putInPosition(LfuNode node) {
            DdlLfuCache firstLast = freqList.get(node.freq);
            if (firstLast == null) {
                LfuNode first = new LfuNode(-1,Integer.MIN_VALUE, -1);
                LfuNode last = new LfuNode(-1,Integer.MIN_VALUE, -1);
                first.next = last;
                last.prev = first;
                freqList.put(node.freq, new DdlLfuCache(first, last));
            }
            firstLast = freqList.get(node.freq);
            LfuNode first = firstLast.first;
            LfuNode temp = first.next;
            first.next = node;
            node.prev = first;
            node.next = temp;
            temp.prev = node;
            DdlLfuCache prevFreqN = freqList.get(node.freq - 1);
            if (prevFreqN != null && prevFreqN.first.next == prevFreqN.last && minFreq == node.freq - 1)
                minFreq = node.freq;
        }

        public int get(int key) {
            if (map.containsKey(key)) {
                LfuNode node = map.get(key);
                node.prev.next = node.next;
                node.next.prev = node.prev;
                node.freq++;
                putInPosition(node);
                return node.value;
            }
            else return -1;
        }

        public void put(int key, int value) {
            if (capacity == 0) return;
            if (map.containsKey(key)) {
                map.get(key).value = value;
                get(key);
            } else {
                n++;
                if (n > capacity) {
                   DdlLfuCache firstLast = freqList.get(minFreq);
                   LfuNode last = firstLast.last;
                   LfuNode toBeRemoved = last.prev;
                   last.prev = last.prev.prev;
                   last.prev.next = last;
                   map.remove(toBeRemoved.key);
                   n--;
                }
                LfuNode node = new LfuNode(key,value,1);
                minFreq = 1;
                map.put(key, node);
                putInPosition(node);
            }
        }
    }

//    https://leetcode.com/problems/shortest-way-to-form-string/

    public int shortestWay(String source, String target) {
        char [] s = source.toCharArray();
        char [] t = target.toCharArray();
        int k = 0;
        int count = 0;
        while (k < t.length) {
            int j = 0;
            int i = k;
            while (i < t.length && j < s.length) {
                if (t[i] == s[j]) i++;
                j++;
            }
            if (k == i) return -1;
            count++;
            k = i;
        }
        return count;
    }

//    https://leetcode.com/problems/sudoku-solver/

    boolean solvedSudoku;

    void solveSudokuUtil(char [][] board, int i, int j, Map<Integer,Set<Character>> rows,
                            Map<Integer,Set<Character>> cols, Map<Integer,Set<Character>> box) {
        if (i == board.length) {
            solvedSudoku = true;
            return;
        }
        int newI;
        int newJ;
        if (j == board[0].length - 1) {
            newI = i + 1;
            newJ = 0;
        } else {
            newI = i;
            newJ = j + 1;
        }
        if (board[i][j] == '.') {
            for (char x = '1'; x <= '9'; x++) {
                int bb = ((i / 3) * 3) + (j / 3);
                Set<Character> r = rows.get(i);
                Set<Character> c = cols.get(j);
                Set<Character> b = box.get(bb);
                if (!r.contains(x) && !c.contains(x) && !b.contains(x)) board[i][j] = x;
                else continue;
                r.add(x);
                c.add(x);
                b.add(x);
                solveSudokuUtil(board, newI, newJ, rows, cols, box);
                if (solvedSudoku) return;
                board[i][j] = '.';
                r.remove(x);
                c.remove(x);
                b.remove(x);
            }
        } else {
            solveSudokuUtil(board, newI, newJ, rows, cols, box);
        }
    }

    public void solveSudoku(char[][] board) {
        Map<Integer,Set<Character>> rows = new HashMap<>();
        Map<Integer,Set<Character>> cols = new HashMap<>();
        Map<Integer,Set<Character>> box = new HashMap<>();
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                int bb = ((i / 3) * 3) + (j / 3);
                Set<Character> r = rows.getOrDefault(i, new HashSet<>());
                Set<Character> c = cols.getOrDefault(j, new HashSet<>());
                Set<Character> b = box.getOrDefault(bb, new HashSet<>());
                if (board[i][j] != '.') {
                    r.add(board[i][j]);
                    c.add(board[i][j]);
                    b.add(board[i][j]);
                }
                rows.put(i, r);
                cols.put(j, c);
                box.put(bb, b);
            }
        }
        System.out.println(rows);
        System.out.println(cols);
        System.out.println(box);
        solveSudokuUtil(board, 0, 0, rows, cols, box);
    }

//    https://leetcode.com/problems/cutting-ribbons/

    boolean canHaveKRibbons(int [] ribbons, int len, int k) {
        int count = 0;
        for (int ribbon : ribbons) count += ribbon / len;
        return count >= k;
    }

    public int maxLength(int[] ribbons, int k) {
        int minLength = 1;
        int maxLength = Integer.MIN_VALUE;
        for (int r : ribbons) maxLength = Math.max(maxLength, r);
        while (minLength <= maxLength) {
            int mid = minLength + (maxLength - minLength) / 2;
            if (canHaveKRibbons(ribbons, mid, k)) minLength = mid + 1;
            else maxLength = mid - 1;
        }
        return maxLength;
    }

//    https://leetcode.com/problems/product-of-two-run-length-encoded-arrays/

    public List<List<Integer>> findRLEArray(int[][] encoded1, int[][] encoded2) {
        List<List<Integer>> res = new ArrayList<>();
        int i = 0;
        int j = 0;
        int m = encoded1.length;
        int n = encoded2.length;
        while (i < m && j < n) {
            int [] a = encoded1[i];
            int [] b = encoded2[j];
            int minF = Math.min(a[1],b[1]);
            int mul = a[0] * b[0];
            if (res.size() > 0 && res.get(res.size() - 1).get(0) == mul) {
                res.get(res.size() - 1).set(1,res.get(res.size() - 1).get(1) + minF);
            } else res.add(Arrays.asList(mul,minF));
            if (minF == a[1] && minF == b[1]) {
                i++;
                j++;
            } else if (minF == a[1]) {
                i++;
                b[1] -= minF;
            }
            else {
                j++;
                a[1] -= minF;
            }
        }
        return res;
    }

//    https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/

    boolean canShipWithW(int [] weights, int days, int w) {
        int sum = 0;
        int i = 0;
        int n = weights.length;
        int d = 0;
        while (i < n) {
            if (sum + weights[i] <= w) {
                sum += weights[i];
            } else {
                d++;
                sum = weights[i];
            }
            i++;
        }
        return d + 1 <= days;
    }

    public int shipWithinDays(int[] weights, int days) {
       int minW = 0;
       int maxW = 0;
       for (int w : weights) {
           minW = Math.max(minW,w);
           maxW += w;
       }
       while (minW <= maxW) {
           int mid = minW + (maxW - minW) / 2;
           if (canShipWithW(weights, days, mid)) maxW = mid - 1;
           else minW = mid + 1;
       }
       return minW;
    }

//    https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/

    void maxLevelSumUtil(TreeNode node, Map<Integer,Integer> levelsSum, int depth) {
        if (node == null) return;
        int levelSum = levelsSum.getOrDefault(depth, 0) + node.val;
        levelsSum.put(depth, levelSum);
        maxLevelSumUtil(node.left, levelsSum, depth + 1);
        maxLevelSumUtil(node.right, levelsSum, depth + 1);
    }

    public int maxLevelSum(TreeNode root) {
        int maxSum = Integer.MIN_VALUE;
        int maxSumD = 0;
        int d = 1;
        Deque<TreeNode> deque = new ArrayDeque<>();
        deque.add(root);
        while (!deque.isEmpty()) {
            int n = deque.size();
            int levelSum = 0;
            for (int i = 0; i < n; i++) {
                TreeNode node = deque.remove();
                levelSum += node.val;
                if (node.left != null) deque.add(node.left);
                if (node.right != null) deque.add(node.right);
            }
            if (levelSum > maxSum) {
                maxSum = levelSum;
                maxSumD = d;
            }
            d++;
        }
        return maxSumD;
    }

//    https://leetcode.com/problems/detect-cycles-in-2d-grid/

    boolean dfsDirectedCycle(int i, boolean [] visited, Deque<Integer> path, Map<Integer,Set<Integer>> adj) {
        if (visited[i] && path.contains(i)) return true;
        visited[i] = true;
        path.push(i);
        for (int neighbour : adj.get(i)) {
            if (dfsDirectedCycle(neighbour, visited, path, adj)) return true;
        }
        path.pop();
        return false;
    }

    boolean dfsUndirectedGraph(int i, boolean [] visited, int [] parent, Map<Integer,Set<Integer>> adj) {
        if (visited[i]) return true;
        visited[i] = true;
        for (int neighbour : adj.get(i)) {
            if (neighbour == parent[i]) continue;
            if (dfsUndirectedGraph(neighbour, visited, parent, adj)) return true;
        }
        return false;
    }

    boolean containsCycleUtil(char[][] grid, int i, int j, char c,
                              String parent, int[][] directions, Set<String> visited) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] != c) return false;
        String s = i + " " + j;
        if (visited.contains(s)) return true;
        visited.add(s);
            for (int [] direction : directions) {
                int newI = i + direction[0];
                int newJ = j + direction[1];
                String newS = newI + " " + newJ;
                if (!newS.equals(parent)) {
                    if (containsCycleUtil(grid, newI, newJ, c, s, directions, visited))
                        return true;
                }
            }
        return false;
    }

    public boolean containsCycle(char[][] grid) {
        Set<String> visited = new HashSet<>();
        int [][] directions = new int[][]{{-1,0},{1,0},{0,1},{0,-1}};
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (!visited.contains(i + " " + j) &&
                        containsCycleUtil(grid, i, j, grid[i][j],"", directions, visited))
                    return true;
            }
        }
        return false;
    }

//    https://leetcode.com/problems/range-module/

//    idea is to put ranges in tm and get that subMap of range. if sub is empty, it's true because there is no
//    change in ranges. idea of using floor and ceil is to put data only if it's not within a specified range while
//    adding range, and we create new range and in removing range, use of floor and ceil is to make sure we put remove
//    range only inside an add range.

    static class RangeModule {

        TreeMap<Integer,Integer> tm;

        public RangeModule() {
            tm = new TreeMap<>();
        }

        public void addRange(int left, int right) {
            clean(left, right);
            Integer floor = tm.floorKey(left);
            Integer ceil = tm.ceilingKey(right);
            if (floor == null || tm.get(floor) == 1) tm.put(left,0);
            if (ceil == null || tm.get(ceil) == 0) tm.put(right,1);
        }

        public boolean queryRange(int left, int right) {
            Integer floor = tm.floorKey(left);
            if (floor == null || tm.get(floor) == 1) return false;
            Map<Integer,Integer> sub = tm.subMap(left, false, right, false);
            return sub.size() == 0;
        }

        public void removeRange(int left, int right) {
            clean(left, right);
            Integer floor = tm.floorKey(left);
            Integer ceil = tm.ceilingKey(right);
            if (floor != null && tm.get(floor) == 0) tm.put(left,1);
            if (ceil != null && tm.get(ceil) == 1) tm.put(right,0);
        }

        void clean(int left, int right) {
            Map<Integer,Integer> sub = tm.subMap(left,true, right, true);
            Set<Integer> keysToClean = new HashSet<>(sub.keySet());
            tm.keySet().removeAll(keysToClean);
        }

    }

//    https://leetcode.com/problems/beautiful-array/

    public int[] beautifulArray(int n) {
       List<Integer> arr = new ArrayList<>();
       arr.add(1);
       int i = 1;
       while (i <= n) {
        List<Integer> temp = new ArrayList<>();
        for (int num : arr) {
            int odd = 2 * num - 1;
            if (odd <= n) temp.add(odd);
        }
        for (int num : arr) {
            int even = num * 2;
            if (even <= n) temp.add(even);
        }
        arr = temp;
        i++;
       }
       int [] res = new int[n];
       for (int j = 0; j < n; j++) res[j] = arr.get(j);
       return res;
    }

//    https://leetcode.com/problems/strings-differ-by-one-character/

    public boolean differByOne(String[] dict) {
        Set<String> set = new HashSet<>(Arrays.asList(dict));
        for (String s : dict) {
            for (int i = 0; i < s.length(); i++) {
                String sChanged = s.substring(0,i) + "*" + s.substring(i + 1);
                if (set.contains(sChanged)) return true;
                set.add(sChanged);
            }
        }
        return false;
    }

//    https://leetcode.com/problems/next-closest-time/

    void generateAllPermutations(int [] noArr, TreeSet<Integer> hoursPerm, TreeSet<Integer> minPerm) {
        for (int k : noArr) {
            for (int i : noArr) {
                int no = k * 10 + i;
                if (no >= 60) continue;
                if (no < 24) hoursPerm.add(no);
                minPerm.add(no);
            }
        }
    }

    String patZeros(int no) {
        return (no < 10) ? "0" + no : "" + no;
    }

    public String nextClosestTime(String time) {
        char [] timeCh = time.toCharArray();
        int [] allNos = new int[]{timeCh[0] - '0', timeCh[1] - '0', timeCh[3] - '0', timeCh[4] - '0'};
        TreeSet<Integer> hoursPerm = new TreeSet<>();
        TreeSet<Integer> minPerm = new TreeSet<>();
        generateAllPermutations(allNos, hoursPerm, minPerm);
        int currH = allNos[0] * 10 + allNos[1];
        int currMin = allNos[2] * 10 + allNos[3];
        Integer minCeil = minPerm.higher(currMin);
        if (minCeil != null) {
            return patZeros(currH) + ":" + patZeros(minCeil);
        } else {
            Integer smallestMin = minPerm.first();
            Integer nextHr = hoursPerm.higher(currH);
            if (nextHr == null) {
                return patZeros(hoursPerm.first()) + ":" + patZeros(smallestMin);
            } else return patZeros(nextHr) + ":" + patZeros(smallestMin);
        }
    }

//    https://leetcode.com/problems/all-elements-in-two-binary-search-trees/

    void inorder(TreeNode node, List<Integer> arr) {
        if (node == null) return;
        inorder(node.left, arr);
        arr.add(node.val);
        inorder(node.right, arr);
    }

    public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {
//        List<Integer> t1 = new ArrayList<>();
//        List<Integer> t2 = new ArrayList<>();
//        inorder(root1, t1);
//        inorder(root2, t2);
//        int i = 0;
//        int j = 0;
//        int m = t1.size();
//        int n = t2.size();
//        List<Integer> res = new ArrayList<>();
//        while (i < m && j < n) {
//            int a = t1.get(i);
//            int b = t2.get(j);
//            if (a < b) {
//                res.add(a);
//                i++;
//            } else {
//                res.add(b);
//                j++;
//            }
//        }
//        while (i < m) {
//            res.add(t1.get(i));
//            i++;
//        }
//        while (j < n) {
//            res.add(t2.get(j));
//            j++;
//        }
//        return res;

        List<Integer> res = new ArrayList<>();
        Deque<TreeNode> s1 = new ArrayDeque<>();
        Deque<TreeNode> s2 = new ArrayDeque<>();
        while (root1 != null || root2 != null || !s1.isEmpty() || !s2.isEmpty()) {
            while (root1 != null) {
                s1.push(root1);
                root1 = root1.left;
            }
            while (root2 != null) {
                s2.push(root2);
                root2 = root2.left;
            }
            if (s2.isEmpty() || (!s1.isEmpty() && s1.peek().val <= s2.peek().val)) {
                root1 = s1.pop();
                res.add(root1.val);
                root1 = root1.right;
            } else {
                root2 = s2.pop();
                res.add(root2.val);
                root2 = root2.right;
            }
        }
        return res;
    }

//    https://leetcode.com/problems/bus-routes/

    public int numBusesToDestination(int[][] routes, int source, int target) {
        if (source == target) return 0;

        Map<Set<Integer>,Set<Set<Integer>>> adj = new HashMap<>();
        Set<Set<Integer>> visited = new HashSet<>();
        Deque<Set<Integer>> deque = new ArrayDeque<>();

        Set<Set<Integer>> allRoutes = new HashSet<>();
        for (int [] route : routes) {
            Set<Integer> cycle = new HashSet<>();
            boolean flag = false;
            for (int r : route) {
                cycle.add(r);
                if (r == source) flag = true;
            }
            if (flag) {
                deque.add(cycle);
            }
            allRoutes.add(cycle);
        }

        for (Set<Integer> cycle : allRoutes) {
            for (Set<Integer> anotherCycle : allRoutes) {
                if (anotherCycle != cycle) {
                    Set<Integer> intersection = new HashSet<>(cycle);
                    intersection.retainAll(anotherCycle);
                    if (intersection.size() > 0) {
                        Set<Set<Integer>> intersections = adj.getOrDefault(cycle, new HashSet<>());
                        intersections.add(anotherCycle);
                        adj.put(cycle, intersections);

                        Set<Set<Integer>> intersections1 = adj.getOrDefault(anotherCycle, new HashSet<>());
                        intersections1.add(cycle);
                        adj.put(anotherCycle, intersections1);
                    }
                }
            }
        }

        int d = 1;
        while (!deque.isEmpty()) {
            int n = deque.size();
            for (int i = 0; i < n; i++) {
                Set<Integer> currCycle = deque.remove();
                if (currCycle.contains(target)) return d;
                if (visited.contains(currCycle)) continue;
                visited.add(currCycle);
                for (Set<Integer> otherCycle : adj.getOrDefault(currCycle, new HashSet<>())) {
                    if (!visited.contains(otherCycle)) {
                        deque.add(otherCycle);
                        visited.add(otherCycle);
                    }
                }
            }
            d++;
        }
        return -1;
    }

//    https://leetcode.com/problems/my-calendar-iii/

    static class MyCalendarThree {

        Map<Integer,Integer> tm;

        public MyCalendarThree() {
            tm = new TreeMap<>();
        }

        public int book(int start, int end) {
            tm.put(start, tm.getOrDefault(start, 0) + 1);
            tm.put(end - 1, tm.getOrDefault(end - 1,0) -1);
            int maxC = 0;
            int c = 0;
            for (int k : tm.keySet()) {
                c += tm.get(k);
                maxC = Math.max(maxC, c);
            }
            return maxC;
        }
    }

//    https://leetcode.com/problems/egg-drop-with-2-eggs-and-n-floors/

    int twoEggDropUtil(int n, int k, int [][] dp) {
       if (dp[n][k] != -1) return dp[n][k];
       if (n <= 1) return 1;
       if (k == 1) return n;
       dp[n][k] = Integer.MAX_VALUE;
       for (int i = 1; i <= n; i++) {
           dp[n][k] = Math.min(dp[n][k],
                   Math.max(twoEggDropUtil(i - 1,k - 1,dp),twoEggDropUtil(n - i, k, dp)) + 1);
       }
       return dp[n][k];
    }

    public int twoEggDrop(int n) {
        int [][] dp = new int[1001][3];
        for (int [] d : dp) Arrays.fill(d, -1);
        return twoEggDropUtil(n, 2, dp);
    }

//    https://leetcode.com/problems/different-ways-to-add-parentheses/

    List<Integer> diffWaysToComputeUtil(String expr, Set<Character> signs) {
        List<Integer> result = new ArrayList<>();
        boolean flag = false;
        for (int i = 0; i < expr.length(); i++) {
            if (signs.contains(expr.charAt(i))) {
                flag = true;
                List<Integer> first = diffWaysToComputeUtil(expr.substring(0, i), signs);
                List<Integer> second = diffWaysToComputeUtil(expr.substring(i + 1), signs);
                for (int f : first) {
                    for (int s : second) {
                        if (expr.charAt(i) == '+') result.add(f + s);
                        else if (expr.charAt(i) == '-') result.add(f - s);
                        else if (expr.charAt(i) == '*') result.add(f * s);
                    }
                }
            }
        }
        if (!flag) result.add(Integer.parseInt(expr));
        return result;
    }

    public List<Integer> diffWaysToCompute(String expression) {
        Set<Character> signs = new HashSet<>();
        signs.add('+');
        signs.add('-');
        signs.add('*');
        return diffWaysToComputeUtil(expression, signs);
    }

//    https://leetcode.com/problems/delete-nodes-and-return-forest/

    TreeNode delNodesUtil(TreeNode node, Set<Integer> toDelete, List<TreeNode> treeNodes) {
        if (node == null) return null;
        TreeNode left = delNodesUtil(node.left, toDelete, treeNodes);
        TreeNode right = delNodesUtil(node.right, toDelete, treeNodes);
        if (toDelete.contains(node.val)) {
            if (left != null) treeNodes.add(left);
            if (right != null) treeNodes.add(right);
            return null;
        }
        node.left = left;
        node.right = right;
        return node;
    }

    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        Set<Integer> toDelete = new HashSet<>();
        for (int num : to_delete) toDelete.add(num);
        List<TreeNode> result = new ArrayList<>();
        TreeNode r = delNodesUtil(root, toDelete, result);
        if (!toDelete.contains(root.val)) result.add(r);
        return result;
    }

//    https://leetcode.com/problems/employee-free-time/

    static class Interval {
        public int start;
        public int end;

        public Interval() {}

        public Interval(int _start, int _end) {
            start = _start;
            end = _end;
        }
    };

    public List<Interval> employeeFreeTime(List<List<Interval>> schedule) {
        Map<Integer,Integer> tm = new TreeMap<>();
        for (List<Interval> s : schedule) {
            for (Interval interval : s) {
                tm.put(interval.start, tm.getOrDefault(interval.start,0) + 1);
                tm.put(interval.end, tm.getOrDefault(interval.end, 0) - 1);
            }
        }
        List<Interval> res = new ArrayList<>();
        int counter = 0;
        int start = -1;
        for (int k : tm.keySet()) {
            counter += tm.get(k);
            if (counter == 0) {
                start = k;
            } else if (counter >= 1 && start != -1 && start != k) {
                res.add(new Interval(start, k));
                start = -1;
            }
        }
        return res;
    }

//    https://leetcode.com/problems/maximum-nesting-depth-of-the-parentheses/

    public int maxDepth(String s) {
        char [] arr = s.toCharArray();
        int openClose = 0;
        int maxDepth = 0;
        for (char c : arr) {
            if (c == '(') openClose++;
            else if (c == ')') openClose--;
            maxDepth = Math.max(maxDepth, openClose);
        }
        return maxDepth;
    }

//    https://leetcode.com/problems/path-with-maximum-minimum-value/

    void dfsMinMax(int [][] grid, int i, int j, int tI, int tJ, int runningMin, int [] maxInAll,
                   boolean [][] visited, int [][] directions) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || visited[i][j]) return;
        visited[i][j] = true;
        runningMin = Math.min(runningMin, grid[i][j]);
        if (i == tI && j == tJ) {
            maxInAll[0] = Math.max(maxInAll[0], runningMin);
            return;
        }
        for (int [] direction : directions) {
            int newI = i + direction[0];
            int newJ = j + direction[1];
            dfsMinMax(grid, newI, newJ, tI, tJ, runningMin, maxInAll, visited, directions);
        }
        visited[i][j] = false;
    }

    static class DjikstraPQ {
        int v;
        int i;
        int j;

        DjikstraPQ(int v, int i, int j) {
            this.v = v;
            this.i = i;
            this.j = j;
        }

        @Override
        public String toString() {
            return "DjikstraPQ{" +
                    "v=" + v +
                    ", i=" + i +
                    ", j=" + j +
                    '}';
        }
    }

    public int maximumMinimumPath(int[][] grid) {
        int [] maxScore = new int[]{Integer.MIN_VALUE};
        int [][] directions = new int[][]{{-1,0},{1,0},{0,1},{0,-1}};
        boolean [][] visited = new boolean[grid.length][grid[0].length];
//        dfsMinMax(grid, 0, 0, grid.length - 1, grid[0].length - 1, Integer.MAX_VALUE,
//                maxScore, visited, directions);
        PriorityQueue<DjikstraPQ> minHeap = new PriorityQueue<>((a,b) -> b.v - a.v);
        minHeap.add(new DjikstraPQ(grid[0][0], 0, 0));
        while (!minHeap.isEmpty()) {
            DjikstraPQ minV = minHeap.remove();
            if (minV.i == grid.length - 1 && minV.j == grid[0].length - 1) {
                return minV.v;
            }
            for (int [] direction : directions) {
                int newI = minV.i + direction[0];
                int newJ = minV.j + direction[1];
                if (newI >= 0 && newI < grid.length && newJ >= 0 && newJ < grid[0].length && !visited[newI][newJ]) {
                    minHeap.add(new DjikstraPQ(Math.min(minV.v, grid[newI][newJ]), newI,newJ));
                }
            }
            visited[minV.i][minV.j] = true;
        }
        return maxScore[0];
    }

//    https://leetcode.com/problems/minimum-cost-to-cut-a-stick/

    int minCostUtil(int start, int end, int [] cuts, Map<String,Integer> dp) {
        if (dp.containsKey(start + " " + end)) return dp.get(start + " " + end);
        boolean noCutsInRange = false;
        int ans = Integer.MAX_VALUE;
        for (int cut : cuts) {
            if (start < cut && cut < end) {
                noCutsInRange = true;
                ans = Math.min(ans, end - start
                        + minCostUtil(start, cut, cuts, dp) + minCostUtil(cut, end, cuts, dp));
            }
        }
        if (!noCutsInRange) ans = 0;
        dp.put(start + " " + end,ans);
        return ans;
    }

    public int minCost(int n, int[] cuts) {
        Map<String,Integer> dp = new HashMap<>();
        return minCostUtil(0, n, cuts, dp);
    }

//    https://leetcode.com/problems/max-stack/

    static class MaxSNode {
        int v;
        MaxSNode next;
        MaxSNode prev;

        MaxSNode(int v) {
            this.v = v;
        }

        @Override
        public String toString() {
            return "MaxSNode{" +
                    "v=" + v +
                    '}';
        }
    }

    static class DDL {

        MaxSNode first;
        MaxSNode last;

        DDL() {
            first = new MaxSNode(Integer.MIN_VALUE);
            last = new MaxSNode(Integer.MIN_VALUE);
            first.next = last;
            last.prev = first;
        }

        void push(MaxSNode node) {
            MaxSNode temp = first.next;
            first.next = node;
            node.prev = first;
            node.next = temp;
            temp.prev = node;
        }

        MaxSNode pop() {
            MaxSNode popped = first.next;
            unlink(popped);
            return popped;
        }

        void unlink(MaxSNode node) {
            node.next.prev = node.prev;
            node.prev.next = node.next;
        }

        int peek() {
            return first.next.v;
        }

    }

    static class MaxStack {

        DDL ddl;
        TreeMap<Integer,List<MaxSNode>> map;

        public MaxStack() {
            ddl = new DDL();
            map = new TreeMap<>();
        }

        public void push(int x) {
          MaxSNode node = new MaxSNode(x);
          List<MaxSNode> maxSNodes = map.getOrDefault(x,new ArrayList<>());
          maxSNodes.add(node);
          map.put(x,maxSNodes);
          ddl.push(node);
        }

        public int pop() {
         MaxSNode popped = ddl.pop();
         List<MaxSNode> list = map.getOrDefault(popped.v, new ArrayList<>());
         list.remove(list.size() - 1);
         if (list.size() == 0) map.remove(popped.v);
         return popped.v;
        }

        public int top() {
            return ddl.peek();
        }

        public int peekMax() {
            if (map.isEmpty()) return Integer.MIN_VALUE;
          return map.lastKey();
        }

        public int popMax() {
           List<MaxSNode> maxNodes = map.get(peekMax());
           MaxSNode max = maxNodes.get(maxNodes.size() - 1);
           maxNodes.remove(maxNodes.size() - 1);
           if (maxNodes.size() == 0) map.remove(peekMax());
           ddl.unlink(max);
           return max.v;
        }
    }

//    https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii/

    TreeNode lowestCommonAncestorUtils(TreeNode node, TreeNode p, TreeNode q, int [] foundBoth) {
        if (node == null) return null;
        TreeNode left = lowestCommonAncestorUtils(node.left, p, q, foundBoth);
        TreeNode right = lowestCommonAncestorUtils(node.right, p, q, foundBoth);
        if (foundBoth[0] == 2) return left != null ? left : right;
        if (node == p || node == q) {
            if (left != null || right != null) {
                foundBoth[0] = 2;
            } else {
                foundBoth[0] = 1;
            }
            return node;
        } else {
            if (left != null && right != null) {
                foundBoth[0] = 2;
                return node;
            } else if (left != null || right != null) {
                foundBoth[0] = 1;
                return left != null ? left : right;
            }
        }
        return null;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        int [] foundBoth = new int[]{0};
        TreeNode node = lowestCommonAncestorUtils(root, p, q, foundBoth);
        if (foundBoth[0] == 2) return node;
        else return null;
    }

//    https://leetcode.com/problems/implement-magic-dictionary/

//    static class MagicDictionary {
//
//        Set<String> set;
//
//        /** Initialize your data structure here. */
//        public MagicDictionary() {
//            set = new HashSet<>();
//        }
//
//        public void buildDict(String[] dictionary) {
//            set.addAll(Arrays.asList(dictionary));
//        }
//
//        public boolean search(String searchWord) {
//            char [] arr = searchWord.toCharArray();
//            for (int i = 0; i < arr.length; i++) {
//                char curr = arr[i];
//                for (char c = 'a'; c <= 'z'; c++) {
//                    if (c != curr) {
//                        arr[i] = c;
//                        String oneCharChanged = String.valueOf(arr);
//                        if (set.contains(oneCharChanged)) return true;
//                    }
//                }
//                arr[i] = curr;
//            }
//            return false;
//        }
//    }

    static class TrieOneCh {
        char c;
        Map<Character,TrieOneCh> children;
        boolean endOfWord;

        TrieOneCh(char c) {
            this.c = c;
            children = new HashMap<>();
        }

        @Override
        public String toString() {
            return "TrieOneCh{" +
                    "c=" + c +
                    ", children=" + children +
                    ", endOfWord=" + endOfWord +
                    '}';
        }
    }

    static class MagicDictionary {

        TrieOneCh root;

        /** Initialize your data structure here. */
        public MagicDictionary() {
            root = new TrieOneCh('.');
        }

        void addInTrie(char [] arr, int i, TrieOneCh node) {
            char c = arr[i];
            TrieOneCh child = node.children.get(c);
            if (child == null) {
                child = new TrieOneCh(c);
                node.children.put(c, child);
            }
            if (i == arr.length - 1) {
                child.endOfWord = true;
            } else addInTrie(arr, i + 1, child);
        }

        public void buildDict(String[] dictionary) {
            for (String word : dictionary) {
                addInTrie(word.toCharArray(), 0, root);
            }
        }

        boolean searchInTrie(char [] arr, int i, TrieOneCh node, boolean flag) {
            if (i == arr.length) return node.endOfWord && flag;
            char c = arr[i];
            for (TrieOneCh child : node.children.values()) {
                boolean b;
                if (child.c != c) {
                    if (flag) continue;
                    b = searchInTrie(arr, i + 1, child, true);
                }
                else b = searchInTrie(arr, i + 1, child, flag);

                if (b) return true;
            }
            return false;
        }

        public boolean search(String searchWord) {
            char [] word = searchWord.toCharArray();
            return searchInTrie(word, 0, root, false);
        }
    }


}