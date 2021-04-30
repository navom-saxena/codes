package dsalgo.leetcode.arrays;

import java.util.*;

public class ArrayMedium3 {

    public static void main(String[] args) {
//        System.out.println(maxChunksToSorted(new int[]{1,0,2,3,4}));
//        System.out.println(isIdealPermutation(new int[]{2,1,0}));
//        System.out.println(numMatchingSubseq("abcde",new String[]{"a", "bb", "acd", "ace"}));
//        System.out.println(Arrays.deepToString(candyCrush(new int[][]{{110,5,112,113,114},{210,211,5,213,214},
//                {310,311,3,313,314},{410,411,412,5,414},{5,1,512,3,3},{610,4,1,613,614},{710,1,2,713,714},
//                {810,1,2,1,1},{1,1,2,2,2},{4,1,4,4,1014}})));
//        MRUQueue queue = new MRUQueue(8);
//        System.out.println(queue.fetch(3));
//        System.out.println(queue.fetch(5));
//        System.out.println(queue.fetch(2));
//        System.out.println(queue.fetch(8));
        System.out.println(findLonelyPixel(new char[][]{{'W','B','W','W'},{'W','B','B','W'},{'W','W','W','W'}}));
    }

//    https://leetcode.com/problems/max-chunks-to-make-sorted/

    public static int maxChunksToSorted(int[] arr) {
        int chunks = 0;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < arr.length; i++) {
           max = Math.max(max,arr[i]);
           if (max == i) {
               chunks++;
           }
        }
        return chunks;
    }

//    https://leetcode.com/problems/global-and-local-inversions/

    public static boolean isIdealPermutation(int[] a) {
        int globalInversions = 0;
        int localInversions = 0;
        boolean isGloballyInverted = false;
        for (int i = 0; i < a.length - 1; i++) {
            if (a[i] - i > 0) {
                globalInversions += a[i] - i;
                isGloballyInverted = true;
            }
            if (a[i] > a[i + 1]) {
                localInversions++;
                if (!isGloballyInverted) {
                    globalInversions++;
                }
            }
            isGloballyInverted = false;
        }
        System.out.println(globalInversions + " " + localInversions);
        return globalInversions == localInversions;
    }

//    https://leetcode.com/problems/number-of-matching-subsequences/

    public static boolean isSubsequenceOf(String word, String s, int wordIndex, int sIndex) {
        if (wordIndex == word.length()) {
            return true;
        } else if (sIndex == s.length()) {
            return false;
        } else {
            if (word.charAt(wordIndex) == s.charAt(sIndex)) {
                return isSubsequenceOf(word, s, wordIndex + 1, sIndex + 1);
            } else {
                return isSubsequenceOf(word, s, wordIndex, sIndex + 1);
            }
        }
    }

    public static int cielBinarySearchSubSeq(List<Integer> arr, int low, int high, int x) {
        if (x < arr.get(low)) {
            return arr.get(low);
        } else if (x >= arr.get(high)) {
            return -1;
        }
        int mid = low + (high - low) / 2;
        if (mid < high && arr.get(mid) <= x && arr.get(mid + 1) > x) {
            return arr.get(mid + 1);
        } else if (arr.get(mid) <= x) {
            return cielBinarySearchSubSeq(arr, mid + 1, high, x);
        } else {
            return cielBinarySearchSubSeq(arr, low, mid, x);
        }
    }

    public static boolean checkSubSequence(String word, Map<Character,List<Integer>> hm) {
        int prev = -1;
        int i = 0;
        while (i < word.length()) {
            if (hm.containsKey(word.charAt(i))) {
                List<Integer> arr = hm.get(word.charAt(i));
                int index = cielBinarySearchSubSeq(arr,0, arr.size() - 1, prev);
                if (index > prev) {
                    prev = index;
                } else {
                    break;
                }
            } else {
                break;
            }
            i++;
        }
        return i == word.length();
    }

    public static int numMatchingSubseq(String s, String[] words) {
        Map<Character, List<Integer>> hm = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            List<Integer> arr;
            if (hm.containsKey(s.charAt(i))) {
                arr = hm.get(s.charAt(i));
            } else {
                arr = new ArrayList<>();
            }
            arr.add(i);
            hm.put(s.charAt(i),arr);
        }
        int count = 0;
        Set<String> present = new HashSet<>();
        Set<String> absent = new HashSet<>();
        for (String word : words) {
            if (present.contains(word)) {
                count++;
            } else if (absent.contains(word)) {
            } else if (checkSubSequence(word, hm)) {
                present.add(word);
                count++;
            } else {
                absent.add(word);
            }
        }
        return count;
    }

//    https://leetcode.com/problems/number-of-subarrays-with-bounded-maximum/

    public static int numSubarrayBoundedMax(int[] a, int l, int r) {
        int end = -1;
        int start = -1;
        int result = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > r) {
                start = i;
            }
            if (a[i] >= l) {
                end = i;
            }
            result += end - start;
        }
        return result;
    }

    //    https://leetcode.com/problems/candy-crush/

    public static int[][] candyCrush(int[][] board) {
        while (true) {
            boolean flag = true;

            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j <= board[i].length - 3; j++) {
                    if (board[i][j] != 0 && Math.abs(board[i][j]) == Math.abs(board[i][j + 1])
                            && Math.abs(board[i][j]) == Math.abs(board[i][j + 2])) {
                        flag = false;
                        board[i][j] = Math.abs(board[i][j]) * -1;
                        board[i][j + 1] = Math.abs(board[i][j + 1]) * -1;
                        board[i][j + 2] = Math.abs(board[i][j + 2]) * -1;
                    }
                }
            }
            for (int j = 0; j < board[0].length; j++) {
                for (int i = 0; i <= board.length - 3; i++) {
                    if (board[i][j] != 0 && Math.abs(board[i][j]) == Math.abs(board[i + 1][j])
                            && Math.abs(board[i][j]) == Math.abs(board[i + 2][j])) {
                        flag = false;
                        board[i][j] = Math.abs(board[i][j]) * -1;
                        board[i + 1][j] = Math.abs(board[i + 1][j]) * -1;
                        board[i + 2][j] = Math.abs(board[i + 2][j]) * -1;
                    }
                }
            }
            for (int j = 0; j < board[0].length; j++) {
                int jump = 0;
                for (int i = board.length - 1; i >= 0; i--) {
                    while (i >= 0 && board[i][j] < 0) {
                        jump++;
                        i--;
                    }
                    if (i >= 0) board[i + jump][j] = board[i][j];
                }
                int i = 0;
                while (jump != 0) {
                    flag = false;
                    board[i][j] = 0;
                    jump--;
                    i++;
                }
            }
            if (flag) {
                break;
            }
        }
        return board;
    }

//    https://leetcode.com/problems/wiggle-sort/

        public static void swap(int [] arr, int i, int j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }

    public static void wiggleSort(int[] nums) {
        for (int i = 1; i < nums.length; i++) {
            if (i % 2 != 0) {
                if (nums[i] < nums[i - 1]) {
                    swap(nums, i, i - 1);
                }
                if (i < nums.length - 1 && nums[i] < nums[i + 1]) {
                    swap(nums, i, i + 1);
                }
            }
        }
    }

//    https://leetcode.com/problems/range-addition/

    public static int[] getModifiedArray(int length, int[][] updates) {
        int [] result = new int[length];
        for (int [] update : updates) {
            int start = update[0];
            int end = update[1];
            int inc = update[2];
            result[start] += inc;
            if (end + 1 < length) {
                result[end + 1] -= inc;
            }
        }
        int preSum = 0;
        for (int i = 0; i < result.length; i++) {
            result[i] += preSum;
            preSum = result[i];
        }
        return result;
    }

//    https://leetcode.com/problems/lonely-pixel-i/

    public static int findLonelyPixel(char[][] picture) {
        Map<Integer,List<Integer>> rowMap = new HashMap<>();
        Map<Integer,List<Integer>> columnMap = new HashMap<>();
        for (int i = 0; i < picture.length; i++) {
            for (int j = 0; j < picture[0].length; j++) {
                if (picture[i][j] == 'B') {
                    List<Integer> columnsList = rowMap.getOrDefault(i, new ArrayList<>());
                    columnsList.add(j);
                    rowMap.put(i, columnsList);
                    List<Integer> rowList = columnMap.getOrDefault(j, new ArrayList<>());
                    rowList.add(i);
                    columnMap.put(j, rowList);
                }
            }
        }
        int count = 0;
        for (List<Integer> columnsList : rowMap.values()) {
            if (columnsList.size() == 1 && columnMap.get(columnsList.get(0)).size() == 1) {
                count++;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/design-most-recently-used-queue/
//    idea is to use bucketing where size of each bucket = Sqrt(n) and in each bucket, store nodes as
//    circular linkedList with -1 as sentinel Node. to re-balance after removal, add next bucket's 1st node
//    to previous bucket's last node. Sentinel node's prev node is that bucket's last node.
//    this is sqrt Decomposition. Fetching time complexity is reduced by O(sqrt(n)) as we have to check
//    only circular linkedList of desired index which is calculated by division

    static class MRUQueue {

        Node [] nodes;
        int bucket;

        public MRUQueue(int n) {
            bucket = (int) Math.sqrt(n);
            nodes = new Node[(n + bucket - 1) / bucket];
            for (int i = 0; i < nodes.length; i++) {
                nodes[i] = new Node(- 1);
            }
            for (int i = 1; i <= n; i++) {
                nodes[(i - 1) / bucket].prev.append(new Node(i));
            }
        }

        public int fetch(int k) {
            int bucketIndex = (k - 1) / bucket;
            Node curr = nodes[bucketIndex];
            for (int i = (k - 1) % bucket; i >= 0; i--) {
                curr = curr.next;
            }
            curr.remove();
            for (int i = bucketIndex + 1; i < nodes.length; i++) {
                nodes[i - 1].prev.append(nodes[i].next.remove());
            }
            nodes[nodes.length - 1].prev.append(curr);
            return curr.value;
        }
    }

    static class Node {
        Node prev = this;
        Node next = this;
        int value;

        Node(int value) {
            this.value = value;
        }

        void append(Node newNode) {
            Node temp = this.next;
            this.next = newNode;
            newNode.prev = this;
            newNode.next = temp;
            temp.prev = newNode;
        }

        Node remove() {
            this.prev.next = this.next;
            this.next.prev = this.prev;
            return prev = next = this;
        }
    }

}
