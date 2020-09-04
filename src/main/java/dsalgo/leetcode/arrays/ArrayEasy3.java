package dsalgo.leetcode.arrays;

import java.util.*;

public class ArrayEasy3 {

    public static void main(String[] args) {
//        System.out.println(addToArrayForm(new int[]{1,2,0,0},34));
//        System.out.println(numRookCaptures(new char[][]{{'.','.','.','.','.','.','.','.'},{'.','.','.','p','.','.','.','.'}
//        ,{'.','.','.','p','.','.','.','.'},{'p','p','.','R','.','p','B','.'},{'.','.','.','.','.','.','.','.'}
//        ,{'.','.','.','B','.','.','.','.'},{'.','.','.','p','.','.','.','.'},{'.','.','.','.','.','.','.','.'}}));
//        System.out.println(commonChars(new String[]{"cool","lock","cook"}));
//        System.out.println(numPairsDivisibleBy60(new int[]{30,20,150,100,40}));
//        System.out.println(canThreePartsEqualSum(new int[]{1,-1,1,-1}));
//        System.out.println(prefixesDivBy5(new int[]{1,0,1,1}));
//        System.out.println(countCharacters(new String[]{"hello","world","leetcode"}, "welldonehoneyr"));
//        duplicateZeros(new int[]{1,2,3});
//        System.out.println(Arrays.toString(relativeSortArray(new int[]{2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19},
//                new int[]{2, 1, 4, 3, 9, 6})));
//        System.out.println(findSpecialInteger(new int[]{1,2,2,6,6,6,6,7,10}));
//        System.out.println(numEquivDominoPairs(new int[][]{{1,2},{1,2},{1,1},{1,2},{2,2}}));
//        System.out.println(Arrays.toString(replaceElements(new int[]{17, 18, -5, -4, 0, -11})));
//        System.out.println(Arrays.toString(decompressRLElist(new int[]{1, 2, 3, 4})));
//        System.out.println(Arrays.toString(arrayRankTransform(new int[]{40, 10, 20, 30})));
//        System.out.println(Arrays.toString(numSmallerByFrequencyOptimised(new String[]{"aabbabbb","abbbabaa","aabbbabaa","aabba","abb","a","ba","aa","ba","baabbbaaaa","babaa","bbbbabaa"}
//        ,new String[]{"b","aaaba","aaaabba","aa","aabaabab","aabbaaabbb","ababb","bbb","aabbbabb","aab","bbaaababba","baaaaa"})));
//        System.out.println(distanceBetweenBusStops(new int[]{7,10,1,12,11,14,5,0},7,2));
//        System.out.println(threeConsecutiveOdds(new int[]{1,2,34,3,4,5,7,23,12}));
//        System.out.println(minimumAbsDifference(new int[]{3,8,-10,23,19,-4,-14,27}));
//        System.out.println(minCostToMoveChips(new int[]{1,1000000000}));
//        System.out.println(checkStraightLine(new int[][]{{-4,-3},{1,0},{3,-1},{0,-1},{-5,2}}));
//        System.out.println(shiftGrid(new int[][]{{-353, 853, 839, 122, -337}, {819, 356, 116, 0, 791}, {-516, -502, -681, -427, -314}, {-386, -400, 597, 740, 836}, {129, 598, 40, -875, -968}, {495, -604, 79, 414, -104}, {237, 121, 211, 4, 677}, {-712, 351, -947, -203, 361}}
//                , 7));
//        System.out.println(tictactoe(new int[][]{{2, 0}, {1, 1}, {0, 2}, {2, 1}, {1, 2}, {1, 0}, {0, 0}, {0, 1}}));
        System.out.println(Arrays.toString(sumZero(3)));
    }

//    https://leetcode.com/problems/add-to-array-form-of-integer/

    public static List<Integer> addToArrayForm(int[] a, int k) {
        LinkedList<Integer> result = new LinkedList<>();
        int i = a.length - 1;
        int carryOver = 0;
        while (k != 0 || carryOver != 0 || i >= 0) {
            int currentValue = 0;
            if (i >= 0) {
                currentValue = a[i];
            }
            int modulo = k % 10;
            int value = (currentValue + modulo + carryOver);
            result.addFirst(value % 10);
            carryOver = value / 10;
            i--;
            k /= 10;
        }
        return result;
    }

//    https://leetcode.com/problems/available-captures-for-rook/

    public static int numRookCaptures(char[][] board) {
        int rookIPos = 0;
        int rookJPos = 0;
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == 'R') {
                    rookIPos = i;
                    rookJPos = j;
                    break;
                }
            }
        }
        int count = 0;
        int i = rookIPos;
        int j = rookJPos;
        while (i >= 0 && i < board.length && j < board.length) {
            if (board[i][j] == '.' || board[i][j] == 'R') {
                i++;
                continue;
            }
            if (board[i][j] == 'p') {
                count++;
                break;
            }
            if (board[i][j] != 'p') {
                break;
            }
            i++;
        }
        i = rookIPos;
        j = rookJPos;
        while (i >= 0 && i < board.length && j < board.length) {
            if (board[i][j] == '.' || board[i][j] == 'R') {
                i--;
                continue;
            }
            if (board[i][j] == 'p') {
                count++;
                break;
            }
            if (board[i][j] != 'p') {
                break;
            }
            i--;
        }
        i = rookIPos;
        j = rookJPos;
        while (i < board.length && j < board.length) {
            if (board[i][j] == '.' || board[i][j] == 'R') {
                j++;
                continue;
            }
            if (board[i][j] == 'p') {
                count++;
                break;
            }
            if (board[i][j] != 'p') {
                break;
            }
            j++;
        }
        i = rookIPos;
        j = rookJPos;
        while (i < board.length && j >= 0 && j < board.length) {
            if (board[i][j] == '.' || board[i][j] == 'R') {
                j--;
                continue;
            }
            if (board[i][j] == 'p') {
                count++;
                break;
            }
            if (board[i][j] != 'p') {
                break;
            }
            j--;
        }
        return count;
    }

//    https://leetcode.com/problems/find-common-characters/

    public static List<String> commonChars(String[] a) {
        int[] count = new int[26];
        Arrays.fill(count, Integer.MAX_VALUE);

        for (String s : a) {
            int[] eachStringCount = new int[26];
            s.chars().forEach(c -> eachStringCount[c - 'a']++);

            for (int i = 0; i < 26; i++) {
                count[i] = Math.min(count[i], eachStringCount[i]);
            }
        }
        List<String> result = new ArrayList<>();
        for (int i = 0; i < 26; i++) {
            while (count[i] > 0) {
                result.add(Character.toString((char) (i + 'a')));
                count[i]--;
            }
        }
        return result;
    }

//    https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/

    public static int numPairsDivisibleBy60(int[] time) {
        int result = 0;
        int[] countArr = new int[60];
        for (int value : time) {
            int key = value % 60;
            int remainingValue = (60 - key) % 60;
            result += countArr[remainingValue];
            countArr[key]++;
        }
        return result;
    }

//    https://leetcode.com/problems/partition-array-into-three-parts-with-equal-sum/

    public static boolean canThreePartsEqualSum(int[] a) {
        int totalSum = 0;
        for (int number : a) {
            totalSum += number;
        }
        if (totalSum % 3 == 0) {
            totalSum /= 3;
            int individualSum = 0;
            int individualCheckCount = 0;
            for (int number : a) {
                individualSum += number;
                if (individualSum == totalSum) {
                    individualCheckCount++;
                    individualSum = 0;
                }
            }
            return individualSum == 0 && individualCheckCount >= 3 || individualCheckCount == 3;
        } else {
            return false;
        }
    }

//    https://leetcode.com/problems/binary-prefix-divisible-by-5/

    public static List<Boolean> prefixesDivBy5(int[] a) {
        int currentNo = 0;
        List<Boolean> resultArr = new ArrayList<>();
        for (int number : a) {
            int currentValue = ((currentNo << 1) + number) % 10;
            resultArr.add(currentValue % 5 == 0);
            currentNo = currentValue;
        }
        return resultArr;
    }

//    https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/

    public static int countCharacters(String[] words, String chars) {
        int[] charsArr = new int[26];
        for (int i = 0; i < chars.length(); i++) {
            charsArr[chars.charAt(i) - 'a']++;
        }
        int count = 0;
        for (String word : words) {
            int[] wordArr = new int[26];
            int alphabetCount = 0;
            for (int i = 0; i < word.length(); i++) {
                int alphabetPos = word.charAt(i) - 'a';
                wordArr[alphabetPos]++;
                if (wordArr[alphabetPos] <= charsArr[alphabetPos]) {
                    alphabetCount++;
                }
            }
            if (alphabetCount == word.length()) {
                count += word.length();
            }
        }
        return count;
    }

//    https://leetcode.com/problems/height-checker/

    public static int heightChecker(int[] heights) {
        int[] copyArr = new int[heights.length];
        System.arraycopy(heights, 0, copyArr, 0, heights.length);
        Arrays.sort(copyArr);
        int count = 0;
        for (int i = 0; i < heights.length; i++) {
            if (heights[i] != copyArr[i]) {
                count++;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/duplicate-zeros/

    public static void duplicateZeros(int[] arr) {
        int zeroCount = 0;
        boolean oneZeroFlag = false;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == 0) {
                int check = arr.length - ((2 * zeroCount) + i - zeroCount);
                if (check > 1) {
                    zeroCount++;
                } else if (check == 1) {
                    oneZeroFlag = true;
                }
            }
        }
        int i = arr.length - 1 - zeroCount;
        int j = arr.length - 1;
        if (oneZeroFlag) {
            arr[j] = arr[i];
            j = j - 1;
            i = i - 1;
        }
        while (i >= 0 && j > 0) {
            if (arr[i] == 0 && i < arr.length - 1) {
                arr[j] = 0;
                arr[j - 1] = 0;
                j = j - 2;
            } else {
                arr[j] = arr[i];
                j--;
            }
            i--;
        }
        System.out.println(Arrays.toString(arr));
    }

//    https://leetcode.com/problems/relative-sort-array/

    public static int[] relativeSortArray(int[] arr1, int[] arr2) {
        int[] countArr = new int[1001];
        for (int num : arr2) {
            countArr[num]++;
        }
        for (int num : arr1) {
            if (countArr[num] >= 1) {
                countArr[num]++;
            } else {
                countArr[num]--;
            }
        }
        int i = 0;
        int k = 0;
        while (k < arr2.length) {
            int count = countArr[arr2[k]];
            int j = 0;
            while (j < count - 1) {
                arr1[i + j] = arr2[k];
                j++;
            }
            i = i + j;
            k++;
        }
        for (int l = 0; l < countArr.length; l++) {
            if (countArr[l] < 0) {
                int j = 0;
                int absValue = Math.abs(countArr[l]);
                while (j < absValue) {
                    arr1[i + j] = l;
                    j++;
                }
                i = i + j;
            }
        }
        return arr1;
    }

//    https://leetcode.com/problems/element-appearing-more-than-25-in-sorted-array/

    public static int findSpecialInteger(int[] arr) {
        int noCount = arr.length / 4;
        int count = 1;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[i - 1]) {
                count = 1;
            } else {
                if (count > noCount) {
                    return arr[i];
                }
                count++;
            }
        }
        return arr[0];
    }

//    https://leetcode.com/problems/number-of-equivalent-domino-pairs/

    public static int numEquivDominoPairs(int[][] dominoes) {
        int count = 0;
        int[][] countMatrix = new int[10][10];
        for (int[] domino : dominoes) {
            int firstValue = domino[0];
            int secondValue = domino[1];
            if (countMatrix[firstValue][secondValue] > 0) {
                count += countMatrix[firstValue][secondValue];
            } else if (countMatrix[secondValue][firstValue] > 0) {
                count += countMatrix[secondValue][firstValue];
            }
            if (firstValue == secondValue) {
                countMatrix[firstValue][secondValue]++;
            } else {
                countMatrix[firstValue][secondValue]++;
                countMatrix[secondValue][firstValue]++;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/replace-elements-with-greatest-element-on-right-side/

    public static int[] replaceElements(int[] arr) {
        int maxOnRight = arr[arr.length - 1];
        arr[arr.length - 1] = -1;
        for (int i = arr.length - 2; i >= 0; i--) {
            int temp = arr[i];
            arr[i] = maxOnRight;
            maxOnRight = Math.max(maxOnRight, temp);
        }
        return arr;
    }

//    https://leetcode.com/problems/decompress-run-length-encoded-list/

    public static int[] decompressRLElist(int[] nums) {
        int length = 0;
        for (int i = 0; i < nums.length; i = i + 2) {
            length += nums[i];
        }
        int[] result = new int[length];
        int k = 0;
        for (int i = 0; i < nums.length; i = i + 2) {
            int freq = nums[i];
            while (freq > 0) {
                result[k] = nums[i + 1];
                freq--;
                k++;
            }
        }
        return result;
    }

//    https://leetcode.com/problems/rank-transform-of-an-array/

    public static int[] arrayRankTransform(int[] arr) {
        int[] copyArr = Arrays.copyOfRange(arr, 0, arr.length);
        Arrays.sort(copyArr);
        Map<Integer, Integer> hm = new HashMap<>();
        int rank = 1;
        for (int num : copyArr) {
            if (hm.get(num) == null) {
                hm.put(num, rank);
                rank++;
            }
        }
        for (int i = 0; i < arr.length; i++) {
            arr[i] = hm.get(arr[i]);
        }
        return arr;
    }

//    https://leetcode.com/problems/compare-strings-by-frequency-of-the-smallest-character/

    public static int[] numSmallerByFrequency(String[] queries, String[] words) {
        int[] wordsArr = new int[words.length];
        for (int w = 0; w < words.length; w++) {
            int[] alphabets = new int[26];
            for (char i = 0; i < words[w].length(); i++) {
                alphabets[words[w].charAt(i) - 'a']++;
            }
            int smallestValue = 0;
            for (int num : alphabets) {
                if (num > 0) {
                    smallestValue = num;
                    break;
                }
            }
            wordsArr[w] = smallestValue;
        }
        Arrays.sort(wordsArr);

        int[] queriesArr = new int[queries.length];
        for (int w = 0; w < queries.length; w++) {
            int[] alphabets = new int[26];
            for (char i = 0; i < queries[w].length(); i++) {
                alphabets[queries[w].charAt(i) - 'a']++;
            }
            int smallestValue = 0;
            for (int num : alphabets) {
                if (num > 0) {
                    smallestValue = num;
                    break;
                }
            }
            queriesArr[w] = smallestValue;
        }
        int[] answers = new int[queries.length];
        for (int i = 0; i < queriesArr.length; i++) {
            int value = reverseCeilSearch(wordsArr, 0, wordsArr.length - 1, wordsArr.length, queriesArr[i]);
            answers[i] = value;
        }
        return answers;
    }


    public static int reverseCeilSearch(int[] arr, int low, int high, int length, int x) {
        int mid;

        if (x < arr[low])
            return high + 1;

        if (x == arr[low]) {
            int l = low;
            while (l < length && x == arr[l]) {
                l++;
            }
            return length - l;
        }

        if (x > arr[high])
            return 0;

        mid = (low + high) / 2;  /* low + (high - low)/2 */

        if (arr[mid] == x) {
            int m = mid;
            while (m < length && x == arr[m]) {
                m++;
            }
            return length - 1 - m + 1;
        } else if (arr[mid] < x) {
            if (mid + 1 <= high && x < arr[mid + 1])
                return length - 1 - (mid + 1) + 1;
            else
                return reverseCeilSearch(arr, mid + 1, high, length, x);
        } else {
            if (mid - 1 >= low && x > arr[mid - 1])
                return length - 1 - mid + 1;
            else
                return reverseCeilSearch(arr, low, mid - 1, length, x);
        }
    }

    public static int[] numSmallerByFrequencyOptimised(String[] queries, String[] words) {
        int[] wd = new int[11];
        for (String s : words) {
            int c = count(s);
            ++wd[c];
        }

        int[] ans = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            int c = count(queries[i]);
            int total = 0;
            for (int j = c + 1; j < wd.length; ++j) {
                total += wd[j];
            }
            ans[i] = total;
        }
        return ans;
    }

    public static int count(String s) {
        if (s.length() < 1) return 0;
        int cnt = 1;
        char c = s.charAt(0);
        for (int i = 1; i < s.length(); ++i) {
            if (c > s.charAt(i)) {
                cnt = 1;
                c = s.charAt(i);
            } else if (c == s.charAt(i)) {
                cnt++;
            }
        }

        return cnt;
    }

//    https://leetcode.com/problems/distance-between-bus-stops/

    public static int distanceBetweenBusStops(int[] distance, int start, int destination) {
        if (start > destination) {
            int temp = start;
            start = destination;
            destination = temp;
        }
        int sum = 0;
        int directSum = 0;
        for (int i = 0; i < distance.length; i++) {
            sum += distance[i];
            if (i >= start && i < destination) {
                directSum += distance[i];
            }
        }
        return Math.min(directSum, sum - directSum);
    }

//    https://leetcode.com/problems/day-of-the-week/submissions/

    public static String dayOfTheWeek(int day, int month, int year) {
        int[] months = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        String[] days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};

        //Set February days to 29 for leap years
        if (year % 4 == 0) months[1] = 29;

        //01/01/1971 day (Thursday)
        int dayCount = 4;

        for (int i = 1971; i < year; i++) {
            //if leap year add two days else one
            dayCount = (i % 4 == 0) ? dayCount + 2 : dayCount + 1;
        }

        for (int i = 0; i < month; i++) {
            for (int j = 0; j < months[i]; j++) {
                if (i + 1 == month && j + 1 == day) break;
                dayCount++;
            }
        }
        return days[(dayCount) % 7];
    }

//    https://leetcode.com/problems/three-consecutive-odds/

    public static boolean threeConsecutiveOdds(int[] arr) {
        int count = 0;
        for (int num : arr) {
            if (num % 2 != 0) {
                count++;
            } else {
                count = 0;
            }
            if (count >= 3) {
                return true;
            }
        }
        return false;
    }

//    https://leetcode.com/problems/minimum-absolute-difference/

    public static List<List<Integer>> minimumAbsDifference(int[] arr) {
        Arrays.sort(arr);
        int minDiff = arr[1] - arr[0];
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 1; i < arr.length; i++) {
            int currentDiff = arr[i] - arr[i - 1];
            if (currentDiff == minDiff) {
                List<Integer> pair = new ArrayList<>();
                pair.add(arr[i - 1]);
                pair.add(arr[i]);
                result.add(pair);
            } else if (currentDiff < minDiff) {
                minDiff = currentDiff;
                result.clear();
                List<Integer> pair = new ArrayList<>();
                pair.add(arr[i - 1]);
                pair.add(arr[i]);
                result.add(pair);
            }
        }
        return result;
    }

//    https://leetcode.com/problems/minimum-cost-to-move-chips-to-the-same-position/

    public static int minCostToMoveChips(int[] position) {
        int oddCount = 0;
        int evenCount = 0;
        for (int num : position) {
            if (num % 2 == 0) {
                evenCount++;
            } else {
                oddCount++;
            }
        }
        return Math.min(oddCount, evenCount);
    }

//    https://leetcode.com/problems/check-if-it-is-a-straight-line/

    public static boolean checkStraightLine(int[][] coordinates) {
        float d = (coordinates[1][0] - coordinates[0][0]);
        boolean x = true;
        float m = 0;
        if (d != 0) {
            m = (coordinates[1][1] - coordinates[0][1]) / d;
            x = false;
        }
        for (int i = 2; i < coordinates.length; i++) {
            float d1 = (coordinates[i][0] - coordinates[i - 1][0]);
            boolean x1 = true;
            float m1 = 0;
            if (d1 != 0) {
                m1 = (coordinates[i][1] - coordinates[i - 1][1]) / d1;
                x1 = false;
            }
            if (x != x1 || m != m1) {
                return false;
            }
            m = m1;
        }
        return true;
    }

//    https://leetcode.com/problems/cells-with-odd-values-in-a-matrix/

    public static int oddCells(int n, int m, int[][] indices) {
        int[][] matrix = new int[n][m];
        for (int[] index : indices) {
            for (int i = 0; i < m; i++) {
                matrix[index[0]][i] += 1;
            }
            for (int i = 0; i < n; i++) {
                matrix[i][index[1]] += 1;
            }
        }
        int oddCount = 0;
        for (int[] row : matrix) {
            for (int value : row) {
                if (value % 2 != 0) {
                    oddCount++;
                }
            }
        }
        return oddCount;
    }

//    https://www.youtube.com/watch?v=2kgEc6oH9J0&list=RDNSQLc55wSuk&index=4

    public static List<List<Integer>> shiftGrid(int[][] grid, int k) {
        int rowsCount = grid.length;
        int columnsCount = grid[0].length;
        int gridLength = rowsCount * columnsCount;
        int start = ((gridLength - (k % gridLength)) % gridLength);
        int startRow = start / columnsCount;
        int startColumn = start % columnsCount;
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> arr = new ArrayList<>();
        for (int c = startColumn; c < columnsCount; c = c + 1) {
            if (arr.size() == grid[0].length) {
                result.add(arr);
                arr = new ArrayList<>();
            }
            arr.add(grid[startRow][c]);
        }
        for (int r = startRow + 1; r < rowsCount; r++) {
            for (int c = 0; c < columnsCount; c = c + 1) {
                if (arr.size() == grid[0].length) {
                    result.add(arr);
                    arr = new ArrayList<>();
                }
                arr.add(grid[r][c]);
            }
        }
        int counterStart = 0;
        outerLoop:
        for (int r = 0; r <= rowsCount; r++) {
            for (int c = 0; c < columnsCount; c++) {
                if (counterStart < start) {
                    if (arr.size() == grid[0].length) {
                        result.add(arr);
                        arr = new ArrayList<>();
                    }
                    arr.add(grid[r][c]);
                } else {
                    break outerLoop;
                }
                counterStart++;
            }
        }
        result.add(arr);
        return result;
    }

//    https://leetcode.com/problems/minimum-time-visiting-all-points/

    public static int minTimeToVisitAllPoints(int[][] points) {
        int totalDistance = 0;
        for (int i = 1; i < points.length; i++) {
            totalDistance += Math.max(Math.abs(points[i][1] - points[i - 1][1]),
                    Math.abs(points[i][0] - points[i - 1][0]));
        }
        return totalDistance;
    }

//    https://leetcode.com/problems/find-winner-on-a-tic-tac-toe-game/

    public static String tictactoe(int[][] moves) {
        int[][] matrix = new int[3][3];
        for (int i = 0; i < moves.length; i++) {
            if (i % 2 == 0) {
                matrix[moves[i][0]][moves[i][1]] = 2;
                if (checkMatrix(matrix, 2)) {
                    return "A";
                }
            } else {
                matrix[moves[i][0]][moves[i][1]] = 1;
                if (checkMatrix(matrix, 1)) {
                    return "B";
                }
            }
        }
        return moves.length == 9 ? "Draw" : "Pending";
    }

    public static boolean checkMatrix(int[][] matrix, int value) {
        return (matrix[0][0] == value && matrix[0][1] == value && matrix[0][2] == value)
                || (matrix[1][0] == value && matrix[1][1] == value && matrix[1][2] == value)
                || (matrix[2][0] == value && matrix[2][1] == value && matrix[2][2] == value)
                || (matrix[0][0] == value && matrix[1][0] == value && matrix[2][0] == value)
                || (matrix[0][1] == value && matrix[1][1] == value && matrix[2][1] == value)
                || (matrix[0][2] == value && matrix[1][2] == value && matrix[2][2] == value)
                || (matrix[0][0] == value && matrix[1][1] == value && matrix[2][2] == value)
                || (matrix[0][2] == value && matrix[1][1] == value && matrix[2][0] == value);
    }

//    https://leetcode.com/problems/find-numbers-with-even-number-of-digits/

    public static int findNumbers(int[] nums) {
        int count = 0;
        for (int num : nums) {
            int digitsCount = 0;
            while (num != 0) {
                num /= 10;
                digitsCount++;
            }
            if (digitsCount % 2 == 0) {
                count++;
            }
        }
        return count;
    }

//    https://leetcode.com/problems/find-n-unique-integers-sum-up-to-zero/

    public static int[] sumZero(int n) {
        int[] arr = new int[n];
        int start = -n / 2;
        int end = n / 2;
        int j = 0;
        for (int i = start; i <= end; i++) {
            if (i == 0) {
                if (n % 2 != 0) {
                    arr[j] = i;
                    j++;
                }
            } else {
                arr[j] = i;
                j++;
            }
        }
        return arr;
    }

}