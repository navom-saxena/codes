package dsalgo.leetcode.company.google;

import java.util.*;

public class FirstSet {

    public static void main(String[] args) {
//        System.out.println(minDominoRotations(new int[]{2,1,1,3,2,1,2,2,1},new int[]{3,2,3,1,3,2,3,3,2}));
//        findSecretWord(new String[]{}, null);
//        System.out.println(expressiveWords("aaa",new String[]{"aaaa"}));
//        System.out.println(jump(new int[]{1,1,1,1}));
//        System.out.println(canConvert("aabcc","ccdee"));
//        System.out.println(minWindow("abcdebdde","bde"));
//        System.out.println(validateStackSequences(new int[]{1,2,3,4,5},new int[]{4,3,5,1,2}));
//        System.out.println(maxScore(new int[]{2,79,80,1,1,1,200,1},3));
//        System.out.println(longestSubarray(new int[]{8,2,4,7},4));
        System.out.println(minSumOfLengths(new int[]{1,6,1},7));
    }

//    https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/

    public static int minDominoRotations(int[] a, int[] b) {
       int [] countArrA = new int[7];
       int [] countArrB = new int[7];
       int [] both = new int[7];
       for (int i = 0; i < a.length; i++) {
           int currA = a[i];
           int currB = b[i];
           countArrA[currA]++;
           countArrB[currB]++;
           if (currA == currB) both[currA]++;
       }
       int minSwitches = Integer.MAX_VALUE;
       for (int i = 1; i < countArrA.length; i++) {
           if (countArrA[i] + countArrB[i] - both[i] >= a.length) {
               minSwitches = Math.min(minSwitches, a.length - Math.max(countArrA[i], countArrB[i]));
           }
       }
       return minSwitches == Integer.MAX_VALUE ? -1 : minSwitches;
    }

//    https://leetcode.com/problems/guess-the-word/

    interface Master {
      int guess(String word);
    }

    public static boolean kCharMatch(String word, String currWord, int match) {
        int k = 0;
        for (int i = 0; i < 6; i++) {
            if (word.charAt(i) == currWord.charAt(i)) k++;
        }
        return k == match;
    }

    public static void findSecretWord(String[] wordlist, Master master) {
        Set<String> wordSet = new HashSet<>(Arrays.asList(wordlist));
        String currWord = null;
        int match = -1;
        while (true) {
            for (String word : wordSet) {
                match = master.guess(word);
                if (match == 6) return;
                currWord = word;
                break;
            }
            Iterator<String> it = wordSet.iterator();
            while (it.hasNext()) {
                if (!kCharMatch(it.next(), currWord, match)) {
                    it.remove();
                }
            }
            wordSet.remove(currWord);
        }
    }

//    https://leetcode.com/problems/expressive-words/

    public static int expressiveWords(String s, String[] words) {
        int count = 0;
        for (String word : words) {
            if (word.length() > s.length()) continue;
            int i = 0;
            int j = 0;
            boolean isStretchy = true;
            while (i < word.length() && j < s.length()) {
                int countW = 1;
                int countS = 1;
                while (i < word.length() - 1 && word.charAt(i) == word.charAt(i + 1)) {
                    i++;
                    countW++;
                }
                while (j < s.length() - 1 && s.charAt(j) == s.charAt(j + 1)) {
                    j++;
                    countS++;
                }
                if (countW != countS && countS < 3 || countW > countS ||
                        (i < word.length() && j < s.length() && word.charAt(i) != s.charAt(j))) {
                    isStretchy = false;
                    break;
                }
                i++;
                j++;
            }
            if (i < word.length() || j < s.length()) continue;
            if (isStretchy) count++;
        }
        return count;
    }

//    https://leetcode.com/problems/jump-game-ii/

    public static int jump(int[] nums) {
//        int n = nums.length;
//        int maxJump = nums[0];
//        int [] minJumps = new int[nums.length];
//        int i = 0;
//        while (i <= maxJump && i < n) {
//            maxJump = Math.max(maxJump, i + nums[i]);
//            for (int j = i + 1; j <= maxJump && j < n; j++) {
//                if (minJumps[j] == 0) minJumps[j] = Integer.MAX_VALUE;
//                minJumps[j] = Math.min(minJumps[j], 1 + minJumps[i]);
//            }
//            i++;
//        }
//        return minJumps[n - 1];
        int currReach = 0;
        int maxJump = 0;
        int jumps = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            if (maxJump < nums[i] + i) {
                maxJump = nums[i] + i;
            }
            if (i == currReach) {
                currReach = maxJump;
                jumps++;
            }
        }
        return jumps;
    }

//    https://leetcode.com/problems/split-array-largest-sum/

    public static int getPieces(int [] nums, int largestSum) {
        int pieces = 1;
        int currRunningSum = 0;
        for (int num : nums) {
            if (currRunningSum > largestSum) {
                pieces++;
                currRunningSum = num;
            } else {
                currRunningSum += num;
            }
        }
        return pieces;
    }

    public static int splitArray(int[] nums, int m) {
        int max = 0;
        int sum = 0;
        for (int num : nums) {
            max = Math.max(max, num);
            sum += num;
        }
        int low = max;
        int high = sum;
        while (low < high) {
            int mid = low + (high - low) / 2;
            int pieces = getPieces(nums, mid);
            if (pieces > m) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }

//    https://leetcode.com/problems/campus-bikes/

    static class UserBike {
        int worker;
        int bike;
        double minDistance;

        UserBike(int worker, int bike, int minDistance) {
            this.worker = worker;
            this.bike = bike;
            this.minDistance = minDistance;
        }

        public int getWorker() {
            return worker;
        }

        public int getBike() {
            return bike;
        }

        public double getMinDistance() {
            return minDistance;
        }
    }

    public static int[] assignBikes(int[][] workers, int[][] bikes) {
        PriorityQueue<UserBike> minHeap = new PriorityQueue<>
                (Comparator.comparingDouble(UserBike::getMinDistance)
                        .thenComparingInt(UserBike::getWorker)
                        .thenComparingInt(UserBike::getBike));

        for (int i = 0; i < workers.length; i++) {
            for (int j = 0; j < bikes.length; j++) {
                int minDistance = Math.abs(workers[i][0] - bikes[j][0]) + Math.abs(workers[i][1] - bikes[j][1]);
                minHeap.add(new UserBike(i, j, minDistance));
            }
        }
        boolean [] workerCheck = new boolean[workers.length];
        boolean [] bikeCheck = new boolean[bikes.length];
        int [] result = new int[workers.length];
        while (!minHeap.isEmpty()) {
            UserBike userBike = minHeap.remove();
            if (!workerCheck[userBike.worker] && !bikeCheck[userBike.bike]) {
                result[userBike.worker] = userBike.bike;
                workerCheck[userBike.worker] = true;
                bikeCheck[userBike.bike] = true;
            }
        }
        return result;
    }

//    https://leetcode.com/problems/string-transforms-into-another-string/

    public static boolean canConvert(String str1, String str2) {
        if (str1.equals(str2)) return true;
        Set<Character> hs = new HashSet<>();
        Map<Character,Character> hm = new HashMap<>();
        for (int i = 0; i < str1.length(); i++) {
            char key = str1.charAt(i);
            char value = str2.charAt(i);
            if (hm.containsKey(key) && hm.get(key) != value) return false;
            hm.put(key,value);
            hs.add(value);
        }
        return hs.size() < 26;
    }

//    https://leetcode.com/problems/minimum-window-subsequence/

    public static String minWindow(String s, String t) {
        int minSize = Integer.MAX_VALUE;
        String minSubSeq = "";
        int j = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == t.charAt(j)) {
               j++;
            }
            if (j == t.length()) {
               j--;
               int end = i;
               while (j >= 0) {
                   if (s.charAt(i) == t.charAt(j)) {
                       j--;
                   }
                   i--;
               }
               i++;
               if (end - i < minSize) {
                   minSize = end - i;
                   minSubSeq = s.substring(i, end);
               }
               j++;
            }
        }
        return minSubSeq;
    }

    int maxSubArraySum(int [] arr) {
        int maxSum = Integer.MIN_VALUE;
        int runningMax = 0;
        for (int num : arr) {
            runningMax += num;
            if (runningMax > maxSum) {
                maxSum = runningMax;
            }
            if (runningMax + num < 0) {
                runningMax = 0;
            }
        }
        return maxSum;
    }

//    https://leetcode.com/problems/logger-rate-limiter/

    static class Logger {

        Map<String,Integer> hm;

        public Logger() {
            hm = new HashMap<>();
        }

        public boolean shouldPrintMessage(int timestamp, String message) {
            if ((!hm.containsKey(message))
                    || hm.containsKey(message) && timestamp - hm.get(message) >= 10) {
                hm.put(message, timestamp);
                return true;
            }
            return false;
        }
    }

    //    https://leetcode.com/problems/validate-stack-sequences/

    public static boolean validateStackSequences(int[] pushed, int[] popped) {
//        int i = 0;
//        int j = 0;
//        while (i < pushed.length) {
//            if (pushed[i] == popped[j]) {
//                pushed[i] = -1;
//                if (i > 0) i--;
//                else i++;
//                j++;
//            } else {
//                i++;
//            }
//        }
        Deque<Integer> stack = new ArrayDeque<>();
        int i = 0;
        int j = 0;
        while (i < pushed.length) {
            stack.push(pushed[i]);
            while (!stack.isEmpty() && j < popped.length && stack.peek() == popped[j]) {
                stack.pop();
                j++;
            }
            i++;
        }
        return i == j;
    }

//    https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/

    public static int maxScore(int[] cardPoints, int k) {
        int runningSum = 0;
        int j = 0;
        int i = 0;
        for (; i < k && i < cardPoints.length; i++) {
            runningSum += cardPoints[i];
        }
        i--;
        int maxSum = runningSum;
        while (j < k && j < cardPoints.length) {
            runningSum = runningSum - cardPoints[i] + cardPoints[cardPoints.length - 1 - j];
            maxSum = Math.max(maxSum, runningSum);
            i--;
            j++;
        }
        return maxSum;
    }

//    https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/

    public static int longestSubarray(int[] nums, int limit) {
        int maxLength = Integer.MIN_VALUE;
        int start = 0;
        int end = 0;
        Deque<Integer> min = new ArrayDeque<>();
        Deque<Integer> max = new ArrayDeque<>();
        min.offer(nums[end]);
        max.offer(nums[end]);
        while (end < nums.length) {
            while (!min.isEmpty() && nums[end] < min.peekLast()) {
                min.removeLast();
            }
            min.offerLast(nums[end]);
            while (!max.isEmpty() && nums[end] > max.peekLast()) {
                max.removeLast();
            }
            max.offerLast(nums[end]);
            while (!max.isEmpty() && !min.isEmpty() && max.peek() - min.peek() > limit) {
                if (nums[start] == min.peek()) {
                    min.removeFirst();
                }
                if (!max.isEmpty() && nums[start] == max.peek()) {
                    max.removeFirst();
                }
                start++;
            }
            maxLength = Math.max(maxLength, end - start + 1);
            end++;
        }
        return maxLength;
    }

//    https://leetcode.com/problems/find-two-non-overlapping-sub-arrays-each-with-target-sum/

    public static int minSumOfLengths(int[] arr, int target) {
        int [] dp = new int[arr.length];
        Arrays.fill(dp, Integer.MAX_VALUE);
        int start = 0;
        int end = 0;
        int runningSum = 0;
        int result = Integer.MAX_VALUE;
        int dpRunningMin = Integer.MAX_VALUE;
        while (end < arr.length) {
            runningSum += arr[end];
            while (start < arr.length && runningSum > target) {
                runningSum -= arr[start];
                start++;
            }
            if (runningSum == target) {
                if (start > 0 && dp[start - 1] != Integer.MAX_VALUE) {
                    result = Math.min(result, dp[start - 1] + end - start + 1);
                }
                dpRunningMin = Math.min(dpRunningMin, end - start + 1);
            }
            dp[end] = dpRunningMin;
            end++;
        }
        return result == Integer.MAX_VALUE ? -1 : result;
    }

}
