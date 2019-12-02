package dsalgo.practice;

import java.io.*;
import java.math.BigInteger;
import java.util.*;

public class HackerRankAux2 {
    public static void main(String[] args) throws IOException {
//        breakingRecords(new int[] {3, 4, 21, 36, 10, 28, 35, 5, 24, 42});
//        System.out.println(divionSumPairs(6, 3, new int[] {1, 3, 2, 6, 1, 2}));
//        List<Integer> arr = new ArrayList<>();
//        arr.add(1);
//        arr.add(4);
//        arr.add(4);
//        arr.add(4);
//        arr.add(5);
//        System.out.println(migratoryBirds(arr));
//        List<Integer> bill = new ArrayList<>();
//        bill.add(3);
//        bill.add(10);
//        bill.add(2);
//        bill.add(9);
//        bonAppetit(bill, 1, 7);
//        System.out.println(sockMerchant(9, new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0}));
//        System.out.println(pageCount(7,2));
//        System.out.println(countingValleys(12,"DDUUDDUDUUUD"));
//        daigonalTraversal(new int[][] {{-2,-3,-6,-5,50,3},{8,7,10,-5,-3,30},{6,3,70,9,-20,-7},{-9,9,-6,7,3,2},{-1,7,7,6,-4,3},{8,5,6,-9,40,8}});
//        spiralTraversal(new int[][] {{-44,25,-52,69,-5},{17,22,51,27,-44},{-79,28,-78,1,-47},{65,-77,-14,-21,-6},{-96,43,-21,-20,90}},5);
//        matrixProduct(2,2, new int[][] {{1,2},{3,-1}}, 2,3,new int[][] {{1,-2,3},{2,3,-1}});
//        BigInteger n = bigFactorial(BigInteger.valueOf(100000));
//        BigInteger nIterative = bigFactorialIterative(100000);
//        System.out.println(nIterative);
//        System.out.println(trailingZeros(nIterative));
//        System.out.println(trailingZerosModified(5));
//        LCMandHCF(4,710);
//        System.out.println(findMissingNumber(10, new int[] {8,11,10,2,7,4,3,5,1,6}));
//        List<Integer> l = new ArrayList<>();
//        l.add(1);
//        l.add(3);
//        l.add(2);
//        l.add(3);
//        l.add(4);
//        l.add(6);
//        l.add(5);
//        l.add(5);
//        repeatedNumbers(l);
//        List<Integer> tripleT = new ArrayList<>();
//        tripleT.add(4);
//        tripleT.add(4);
//        tripleT.add(4);
//        tripleT.add(3);
//        tripleTrouble(tripleT);
//        List<Integer> cont = new ArrayList<>();
//        cont.add(1);
//        cont.add(2);
//        cont.add(2);
//        cont.add(2);
//        cont.add(3);
//        cont.add(4);
//        cont.add(4);
//        cont.add(3);
//        maximumContigiousSubsequence(cont);
//        lcmAndhcfUsingEuclid(605904,996510762);
//        decimalToBinary(4);
//        bitFlips(549,24);
//        reverseBits(15);
//        swapBits(100);
//        System.out.println(power(10,10));
//        checkPallindrome("m");
//        buff();
//        printTwoSetBitNums(4);
//        twoSetBits(4);
//        long n = 1;
//        System.out.println(cubeRootBinarySearchIter(-Double.valueOf(Math.pow(10,6)).longValue(),Double.valueOf(Math.pow(10,6)).longValue(),n));
//        System.out.println(cubeRootBInarySearchRecur(-Double.valueOf(Math.pow(10,6)).longValue(),Double.valueOf(Math.pow(10,6)).longValue(),n));
//        frequencySort(new int[] {4,-2,10,12,-8,4});
//        System.out.println(anagrams("aaca","aca"));
//        System.out.println(rabinKarpAlgo("rrrrr","",999999937));
//        System.out.println(rabinKarp("smart","yekicmsmartplrplsmartrplplmrpsmartrpsmartwmrmsmartsmart"));
//        generateSubArrays(new int[] {-5,10,-3});
//        validSubArrays(new int[] { 1, 0, 0, 1, 0, 1, 1});
//        countSubarrWithEqualZeroAndOne(new int[] { 1,0,1,0 });
//        rangeSumSubarrays(new int[] {-5,10,-3},-10,5);
//        generateSubsequence(new int[] {3,5,15});
//        subsequenceSum(new int[] {1,-2,3},1,3);
//        System.out.println(overlappingRectangles(2,5,4,6,6,7,8,9));
    }

    public static void buff() throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
        bw.write("");
        bw.flush();
        int firstInt = Integer.parseInt(br.readLine());
        StringTokenizer st = new StringTokenizer(br.readLine());
        int strToken = Integer.parseInt(st.nextToken());
        for (int z = 0; z < firstInt; z++) {
            int n = Integer.parseInt(br.readLine());
            String input = br.readLine();
        }
    }

    public static int[] breakingRecords(int[] arr) {
        int max = arr[0];
        int min = arr[0];
        int maxCount = 0;
        int minCount = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                maxCount++;
            } else if (arr[i] < min) {
                min = arr[i];
                minCount++;
            }
        }
        return new int[]{maxCount, minCount};
    }

    public static int divionSumPairs(int n, int k, int[] arr) {
        int divisionCount = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((i < j) && ((arr[i] + arr[j]) % k == 0)) {
                    divisionCount++;
                }
            }
        }
        return divisionCount;
    }

    public static int migratoryBirds(List<Integer> arr) {
        int[] freqArr = new int[arr.size() + 1];
        for (int i = 0; i < freqArr.length - 1; i++) {
            freqArr[arr.get(i)]++;
        }
        int maxValue = 0;
        int maxfreq = 0;
        for (int j = 0; j < freqArr.length - 1; j++) {
            if (freqArr[j] > maxfreq) {
                maxfreq = freqArr[j];
                maxValue = j;
            }
        }
        return maxValue;
    }

    public static void bonAppetit(List<Integer> bill, int k, int b) {
        int actualSum = 0;
        int wrongSum = 0;
        for (int i = 0; i < bill.size(); i++) {
            wrongSum += bill.get(i);
        }
        actualSum = wrongSum - bill.get(k);
        int actualShare = actualSum / 2;
        if (b == actualShare) {
            System.out.println("Bon Appetit");
        } else {
            System.out.println(b - actualShare);
        }
    }

    public static int sockMerchant(int n, int[] ar) {
        int count = 0;
        int[] freqAr = new int[101];
        for (int i = 0; i < ar.length; i++) {
            freqAr[ar[i]]++;
        }
        for (int j = 0; j < freqAr.length; j++) {
            if (freqAr[j] != 0 && freqAr[j] / 2 > 0) {
                int pairs = freqAr[j] / 2;
                count += pairs;
            }
        }
        return count;
    }

    public static int pageCount(int n, int p) {
        return Math.min(p / 2, n / 2 - p / 2);
    }

    public static int countingValleys(int n, String s) {
        char[] charArray = s.toCharArray();
        int valleyCount = 0;
        int valleySteps = 0;
        int mountainSteps = 0;
        boolean negativeFlag = false;
        for (int i = 0; i < charArray.length; i++) {
            if (charArray[i] == 'U') {
                mountainSteps++;
            } else {
                valleySteps++;
            }
            int diff = valleySteps - mountainSteps;
            if (diff > 0) {
                negativeFlag = true;
            }
            if (diff == 0 && negativeFlag) {
                valleyCount++;
                negativeFlag = false;
            }
        }
        return valleyCount;
    }

    public static void daigonalTraversal(int[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = matrix.length - 1; j > -1; j--) {
                int sum = 0;
                int ix = i;
                int jx = j;
                if (ix == 0 || jx == 0) {
                    while (ix < matrix.length && jx < matrix.length) {
                        sum += matrix[ix][jx];
                        ix++;
                        jx++;
                    }
                    System.out.print(sum + " ");
                }
            }
        }
    }

    public static void spiralTraversal(int[][] matrix, int x) {
        int i = 0; // iterator
        int k = 0; // starting index of row
        int m = x; // ending index of row
        int l = 0; // starting index of column
        int n = x; // ending index of column
        while (k < m && l < n) {
            for (i = l; i < n; i++) {
                System.out.print(matrix[k][i] + " ");
            }
            k++;
            for (i = k; i < m; i++) {
                System.out.print(matrix[i][n - 1] + " ");
            }
            n--;
            if (k < m) {
                for (i = n - 1; i >= l; i--) {
                    System.out.print(matrix[m - 1][i] + " ");
                }
                m--;
            }
            if (l < n) {
                for (i = m - 1; i >= k; i--) {
                    System.out.print(matrix[i][l] + " ");
                }
                l++;
            }
        }
    }

    public static void matrixProduct(int n1, int m1, int[][] matrix1, int n2, int m2, int[][] matrix2) {
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < m2; j++) {
                int sum = 0;
                for (int k = 0; k < m1; k++) {
                    sum += matrix1[i][k] * matrix2[k][j];
                }
                System.out.print(sum + " ");
            }
            System.out.println();
        }
    }

    public static BigInteger bigFactorial(BigInteger n) {
        if (n.longValue() == 1) {
            return BigInteger.valueOf(1);
        } else {
            BigInteger o = BigInteger.valueOf(n.longValue() - 1);
            return n.multiply(bigFactorial(o));
        }
    }

    public static BigInteger bigFactorialIterative(long n) {
        BigInteger factorialValue = BigInteger.valueOf(1);
        for (int i = 1; i <= n; i++) {
            factorialValue = factorialValue.multiply(BigInteger.valueOf(i));
        }
        return factorialValue;
    }

    public static int trailingZeros(BigInteger n) {
        int count = 0;
        while (n.remainder(BigInteger.valueOf(5)).equals(BigInteger.valueOf(0))) {
            count++;
            n = n.divide(BigInteger.valueOf(5));
        }
        return count;
    }

    public static long trailingZerosModified(long n) {
        long count = 0;
        for (long i = 5; n / i >= 1; i *= 5) {
            count += n / i;
        }
        return count;
    }

    public static void lcmAndhcf(long a, long b) {
        List<Long> listA = new ArrayList<>();
        List<Long> listB = new ArrayList<>();
        for (long i = 1; i <= Math.sqrt(a); i++) {
            if (a % i == 0) {
                listA.add(i);
                listA.add(a / i);
            }
        }
        for (long j = 1; j <= Math.sqrt(b); j++) {
            if (b % j == 0) {
                listB.add(j);
                listB.add(b / j);
            }
        }
        Collections.sort(listA);
        Collections.sort(listB);
        long hcf = 1;
        int sizeA = listA.size();
        int sizeB = listB.size();
        if (sizeA < sizeB) {
            for (int x = 0; x < listA.size(); x++) {
                long searchValue = Collections.binarySearch(listB, listA.get(x));
                if (searchValue >= 0) {
                    hcf = listA.get(x);
                }
            }
        } else {
            for (int x = 0; x < listB.size(); x++) {
                long searchValue = Collections.binarySearch(listA, listB.get(x));
                if (searchValue >= 0) {
                    hcf = listB.get(x);
                }
            }
        }
        System.out.println(listA);
        System.out.println(listB);
        long lcm = (a * b) / hcf;
        System.out.println(lcm + " " + hcf);
    }

    public static void lcmAndhcfUsingEuclid(long a, long b) {
        long hcf = gcdUsingEuclid(a, b);
        long lcm = (a / hcf) * b;
        System.out.println(lcm + " " + hcf);
    }

    public static long gcdUsingEuclid(long a, long b) {
        long tempLarger = 0;
        long tempSmaller = 0;
        if (a > b) {
            tempLarger = a;
            tempSmaller = b;
        } else {
            tempLarger = b;
            tempSmaller = a;
        }
        while (tempSmaller != 0) {
            long temp = tempSmaller;
            tempSmaller = tempLarger % tempSmaller;
            tempLarger = temp;
        }
        return tempLarger;
    }

    public static int findMissingNumber(int n, int[] arr) {
        int actualSum = ((n + 1) * (n + 2)) / 2;
        int wrongSum = 0;
        for (int i = 0; i < arr.length; i++) {
            wrongSum += arr[i];
        }
        return actualSum - wrongSum;
    }

    public static void repeatedNumbers(List<Integer> arr) {
        Collections.sort(arr);
        for (int i = 0; i < arr.size() - 1; i++) {
            if (arr.get(i).equals(arr.get(i + 1))) {
                System.out.print(arr.get(i) + " ");
            }
        }
    }

    public static void tripleTrouble(List<Integer> arr) {
        Collections.sort(arr);
        int i = 0;
        boolean flag = false;
        while (i < arr.size() - 2) {
            if (arr.get(i).equals(arr.get(i + 1)) && arr.get(i).equals(arr.get(i + 2))) {
                i += 3;
            } else {
                System.out.println(arr.get(i));
                flag = true;
                break;
            }
        }
        if (!flag) {
            System.out.println(arr.get(arr.size() - 1));
        }
    }

    public static void maximumContigiousSubsequence(List<Integer> arr) {
        Collections.sort(arr);
        TreeSet<Integer> ts = new TreeSet<>(arr);
        List<Integer> arr1 = new ArrayList<>(ts);
        Iterator<Integer> it = ts.iterator();
//        System.out.println(arr);
        int counter = 0;
        int max = 0;
        int i = 0;
        while (i < arr1.size()) {
            if (i != arr1.size() - 1 && arr1.get(i) + 1 == arr1.get(i + 1)) {
                counter = counter + 1;
            } else {
                counter += 1;
                if (counter > max) {
                    max = counter;
                }
                counter = 0;
            }
            i++;
        }
        System.out.println(max);
    }

    public static void decimalToBinary(int num) {
        String binary = "";
        if (num == 0) {
            binary = "0";
        }
        while (num != 0) {
            int x = num % 2;
            binary = x + binary;
            num = num >> 1;
        }
        System.out.println(binary);
    }

    public static void bitFlips(int num1, int num2) {
        int counter = 0;
        int res = num1 ^ num2;
        while (res != 0) {
            if (res % 2 != 0) {
                counter++;
            }
            res = res >> 1;
        }
        System.out.println(counter);
    }

    public static void reverseBits(int n) {
        long result = 0;
        int counter = 0;
        while (n > 0) {
            int x = n % 2;
            if (x != 0) {
                result += Math.pow(2, 31 - counter);
            }
            n >>= 1;
            counter++;
        }
        System.out.println(result);
    }

    public static void swapBits(int n) {
        long result = 0;
        int counter = 0;
        while (n > 0) {
            int x = n % 2;
            n >>= 1;
            int y = n % 2;
            n >>= 1;
            if (x != 0) {
                result += Math.pow(2, counter + 1);
            }
            if (y != 0) {
                result += Math.pow(2, counter);
            }
            counter = counter + 2;
        }
        System.out.println(result);
    }

    public static long power(long x, long y) {
        long res = 1;
        while (y > 0) {
            if ((y & 1) == 1) {
                res = (res * x) % 1000000007;
            }
            y = y >> 1;
            x = (x * x) % 1000000007;
        }
        return res;
    }

    public static void checkPallindrome(String input) {
        int len = input.length();
        boolean flag = true;
        for (int i = 0; i < len / 2; ++i) {
            if ((input.charAt(i) != input.charAt(len - i - 1))) {
                flag = false;
            }
        }
        if (flag) {
            System.out.println("Yes");
        } else {
            System.out.println("No");
        }
    }

    public static void evenSplit(BigInteger n) {
        if (n.longValue() != 0 && n.longValue() != 2 && n.remainder(BigInteger.valueOf(2)).equals(BigInteger.valueOf(0))) {
            System.out.println("Yes");
        } else {
            System.out.println("No");
        }
    }

    public static void twoSetBits(long n) {
        long r = 0;
        long i = Double.valueOf(Math.floor((-1 + Math.sqrt(1 + 8 * n)) / 2)).longValue();
        int counter = 0;
        System.out.println(i);
        while ((r = i * (i + 1) / 2) < n) {
            i++;
            counter++;
        }
        long res = (power(2, i) + power(2, (i - (r - n) - 1))) % 1000000007;
        System.out.println(res);
    }

    public static void printTwoSetBitNums(long n) {
        long y1 = 1;
        long x1 = 1;
        long x = 1;
        while (n > 0) {
            long y = 0;
            while (y < x) {
                x1 = x;
                y1 = y;
                n--;
                y++;
                if (n == 0)
                    break;
            }
            x++;
        }
        long res = power(2, x1) + power(2, y1);
        System.out.println(res);
    }

    public static long cubeRootBInarySearchRecur(long low, long high, long n) {
        long mid = low + (high - low) / 2;
        long midCube = mid * mid * mid;
        if (midCube == n) {
            return mid;
        }
        if (midCube > n) {
            return cubeRootBInarySearchRecur(low, mid - 1, n);
        } else {
            return cubeRootBInarySearchRecur(mid + 1, high, n);
        }
    }

    public static long cubeRootBinarySearchIter(long low, long high, long n) {
        long mid = 1;
        long midCube = 1;
        while (midCube != n) {
            mid = low + (high - low) / 2;
            midCube = mid * mid * mid;
            if (midCube == n) {
                return mid;
            } else if (midCube < n) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return 1;
    }

    public static void frequencySort(int[] arr) {
        Map<Integer, Integer> tm = new TreeMap<>();
        for (int i = 0; i < arr.length; i++) {
            tm.merge(arr[i], 1, Integer::sum);
        }
        tm.entrySet().stream().sorted(Map.Entry.comparingByValue()).forEach(entrySet -> {
            int v = entrySet.getValue();
            for (int i = 1; i <= v; i++) {
                System.out.print(entrySet.getKey() + " ");
            }
        });
        System.out.println();
    }

    public static boolean anagrams(String a, String b) {
        if (a.length() != b.length()) {
            return false;
        }
        char[] cA1 = a.toCharArray();
        char[] cA2 = b.toCharArray();
        Arrays.sort(cA1);
        Arrays.sort(cA2);
        for (int i = 0; i < cA1.length; i++) {
            if (cA1[i] != cA2[i]) {
                return false;
            }
        }
        return true;
    }

    public static long rabinKarpAlgo(String pattern, String text, long primeNo) {
        long textHash = 0;
        long patternHash = 0;
        int m = text.length();
        int n = pattern.length();
        long matchCount = 0;
        if (m == 0 || n == 0 || n > m) {
            return 0;
        }
        for (int i = 0; (i < n); i++) {
            patternHash = (patternHash + pattern.charAt(i) * Double.valueOf(Math.pow(2, n - i - 1)).longValue()) % primeNo;
            textHash = (textHash + text.charAt(i) * Double.valueOf(Math.pow(2, n - i - 1)).longValue()) % primeNo;
        }
        for (int j = 0; j <= m - n; j++) {
            if (patternHash == textHash) {
                boolean stringMatchFlag = true;
                for (int k = 0; k < n; k++) {
                    if (text.charAt(j + k) != pattern.charAt(k)) {
                        stringMatchFlag = false;
                        break;
                    }
                }
                if (stringMatchFlag) {
                    matchCount++;
                }
            }
            if (j != m - n) {
                textHash = (((textHash - (text.charAt(j) * Double.valueOf(Math.pow(2, n - 1)).longValue())) % primeNo) * 2 + text.charAt(j + n)) % primeNo;
            }
        }
        return matchCount;
    }

    public static int rabinKarp(String pattern, String text) {
        int matchCount = 0;
        long primeNo = Double.valueOf(Math.pow(10, 9) + 7).longValue();
        int b = 30;
        int m = text.length();
        int n = pattern.length();
        long textHash = 0;
        long patternHash = 0;
        for (int i = 0; (i < n); i++) {
            patternHash = (patternHash * b + (pattern.charAt(i))) % primeNo;
            textHash = (textHash * b + (text.charAt(i))) % primeNo;
        }
        long power = 1;
        for (int i = 0; i < n - 1; i++) {
            power = power * b;
        }
        if (patternHash == textHash) {
            matchCount++;
        }
        for (int j = 1; j < m - n + 1; j++) {
            textHash = (textHash - (text.charAt(j - 1)) * power) % primeNo;
            textHash = (textHash + primeNo) % primeNo;
            textHash = (textHash * b + (text.charAt(j + n - 1))) % primeNo;
            if (patternHash == textHash) {
                boolean stringMatchFlag = true;
                for (int k = 0; k < n; k++) {
                    if (text.charAt(j + k) != pattern.charAt(k)) {
                        stringMatchFlag = false;
                        break;
                    }
                }
                if (stringMatchFlag) {
                    matchCount++;
                }
            }
        }
        return matchCount;
    }

    public static void generateSubArrays(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            for (int j = i; j < arr.length; j++) {
                for (int k = i; k <= j; k++) {
                    System.out.print(arr[k] + " ");
                }
                System.out.println();
            }
        }
    }

    public static void rangeSumSubarrays(int[] arr, int r1, int r2) {
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = i; j < arr.length; j++) {
                int sum = 0;
                for (int k = i; k <= j; k++) {
                    sum += arr[k];
                }
                if (r1 <= sum && sum <= r2) {
                    count++;
                }
            }
        }
        System.out.println(count);
    }

    public static void validSubArrays(int[] arr) {
        int counter = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = i; j < arr.length; j++) {
                int sum = 0;
                for (int k = i; k <= j; k++) {
                    if (arr[k] == 0) {
                        sum = sum + -1;
                    } else {
                        sum = sum + 1;
                    }
                }
                if (sum == 0) {
                    counter++;
                }
            }
        }
        System.out.println(counter);
    }

    static void countSubarrWithEqualZeroAndOne(int[] arr) {
        Map<Integer, Integer> hm = new HashMap<>();
        int sum = 0;
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            //Replacing 0's in array with -1
            if (arr[i] == 0)
                arr[i] = -1;

            sum += arr[i];

            if (sum == 0)
                count++;

            if (hm.containsKey(sum))
                count += hm.get(sum);

            hm.merge(sum, 1, Integer::sum);
        }
        System.out.println(count);
    }

    public static void generateSubsequence(int[] arr) {
        for (int i = 1; i < 1 << arr.length; i++) {
            for (int j = 0; j < arr.length; j++) {
                if ((i >> j) % 2 != 0) {
                    System.out.print(arr[j] + " ");
                }
            }
            System.out.println();
        }
    }

    public static void subsequenceSum(int[] arr, int l, int r) {
        int count = 0;
        for (int i = 0; i < 1 << arr.length; i++) {
            int sum = 0;
            for (int j = 0; j < arr.length; j++) {
                if ((i >> j) % 2 != 0) {
                    System.out.print(arr[j] + " ");
                    sum += arr[j];
                }
            }
            System.out.println();
            if ((l <= sum) && (sum <= r)) {
                count++;
            }
        }
        System.out.println(count);
    }

    public static long overlappingRectangles(long r1blx, long r1bly, long r1trx, long r1try, long r2blx, long r2bly, long r2trx, long r2try) {
        long area1 = Math.abs(r1blx - r1trx) * Math.abs(r1bly - r1try);
        long area2 = Math.abs(r2blx - r2trx) * Math.abs(r2bly - r2try);
        long intersectingArea = (Math.min(r1trx, r2trx) - Math.max(r1blx, r2blx)) * (Math.min(r1try, r2try) - Math.max(r1bly, r2bly));
        if ((r1blx > r2trx || r2blx > r1trx) || (r2bly > r1try || r1bly > r2try)) {
            return area1 + area2;
        } else {
            return area1 + area2 - intersectingArea;
        }
    }
}
