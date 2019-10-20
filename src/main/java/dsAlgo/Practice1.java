package dsAlgo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class Practice1 {
    private static int[] nbitArray = new int[]{1, 0};

    public static void main(String[] args) throws IOException {
//        List<Integer> a = new ArrayList<>();
//        a.add(5);
//        a.add(6);
//        a.add(7);
//        List<Integer> b = new ArrayList<>();
//        b.add(3);
//        b.add(6);
//        b.add(10);
//        System.out.println(compareTriplets(a,b));
//        System.out.println(aVeryBigSum(new long[]{1000000001, 1000000002, 1000000003, 1000000004, 1000000005}));
//        plusMinus(new int[]{-4, 3, -9, 0, 4, 1});
//        staircase(6);
//        miniMax(new int[] {-1, -1, -1, -1, -1});
//        birthdayCakeCandles(new int[] {3, 2, 1, 3});
//        System.out.println(timeConversion("12:45:54PM"));
//        int [][] mat = new int[][]{{11, 2, 4}, {4,5,6}, {10,8,-12}};
//        System.out.println(diagonalDifference(mat));
//        int [] op = gradingStudents(new int[] {73, 67, 38, 33});
//        for (int a: op){
//            System.out.println(a);
//        }
//        countApplesAndOranges(7,10, 4 ,12, new int[]{2, 3, -4}, new int[]{3, -2, -4});
//        System.out.println(kangaroo(0, 5, 3, 2));
//        System.out.println(getTotalX(new int[] {2,6}, new int[]{24,36}));
//        List<Integer> l1 = new ArrayList<>();
//        l1.add(4);
//        System.out.println(birthday(l1,4,1));
//        Scanner sc = new Scanner(System.in);
//        int testCasesNo = sc.nextInt();
//        int [] testcases = new int[testCasesNo];
//        for (int i = 0;i < testCasesNo; i++){
//            testcases[i] = sc.nextInt();
//        hollowDaimond(7);
//        System.out.println(factorial(5));
//        System.out.println(summation(9));
//        towersOfHenoi(3,'A','B', 'C');
//        System.out.println(arraySortCheckRecusion(new int []{1,2,3,4,5}));
//        int [] nbitArray = new int[] {1,2,3,4};
//        binaryNo(2);
//        kString(2,3);
//        List<Integer> sumpairsList = new ArrayList<>();
//        sumpairsList.add(112);
//        diffOfPairs(sumpairsList, 11);
//        checkPowerOfTwo(1);
//        List<Integer> tripletSumList = new ArrayList<>();
//        tripletSumList.add(1);
//        tripletSumList.add(20);
//        tripletSumList.add(40);
//        tripletSumList.add(100);
//        tripletSumList.add(80);
//        tripletWithSumK(tripletSumList,60);
//        tripletWithSumKImproved(tripletSumList,60);
//        noOfDivisors(97158);
//        maximumSetBits(13);
//        List<Integer> cielArr = new ArrayList<>();
//        cielArr.add(-6);
//        cielArr.add(10);
//        cielArr.add(-1);
//        cielArr.add(20);
//        cielArr.add(15);
//        cielArr.add(5);
//        ceil(cielArr, 13);
//        Collections.sort(cielArr);
//        System.out.println("now binarySearch");
//        System.out.println(cielArr);
//        System.out.println(cielBinarySearch(cielArr,0, cielArr.size() -1,13));
//        sumAndxor(5);
//        sumAndxor(7);
//        sumAndxorModified(7);
//        for (int i = 0; i <= 90; i++){
//            long a = sumAndxor(i);
//            long b = sumAndxorModified(i);
//            if (a != b){
//                System.out.println("a - " + a);
//                System.out.println("b - " + b);
//                System.out.println(i);
//            }
//        }
//        List<Integer> floorArr = new ArrayList<>();
//        floorArr.add(-6);
//        floorArr.add(10);
//        floorArr.add(-1);
//        floorArr.add(20);
//        floorArr.add(15);
//        floorArr.add(5);
//        Collections.sort(floorArr);
//        System.out.println(floorArr);
//        System.out.println(floorBinarySearch(floorArr,0,floorArr.size() -1,10));
//        List<Integer> a = new ArrayList<>();
//        a.add(20);
//        a.add(20);
//        List<Integer> b = new ArrayList<>();
//        b.add(20);
//        b.add(20);
//        Collections.sort(a);
//        Collections.sort(b);
//        powerGame(a,b);
        smallerElements(new int[]{15, 35, 10, 15, 15, 15});
    }

    public static List<Integer> compareTriplets(List<Integer> a, List<Integer> b) {
        int aliceScore = 0;
        int bobScore = 0;
        for (int i = 0; i < a.size(); i++) {
            if (a.get(i) > b.get(i)) {
                aliceScore++;
            } else if (a.get(i) < b.get(i)) {
                bobScore++;
            }
        }
        List<Integer> returningList = new ArrayList<>();
        returningList.add(aliceScore);
        returningList.add(bobScore);
        return returningList;
    }

    public static long aVeryBigSum(long[] ar) {
        long sum = 0;
        for (long element : ar) {
            sum += element;
        }
        return sum;
    }

    public static void plusMinus(int[] arr) {
        float positiveNumbers = 0;
        float negativeNumbers = 0;
        float zeroNumbers = 0;
        for (int element : arr) {
            if (element > 0) {
                positiveNumbers++;
            } else if (element < 0) {
                negativeNumbers++;
            } else {
                zeroNumbers++;
            }
        }
        System.out.printf("%.6f", positiveNumbers / arr.length);
        System.out.println();
        System.out.printf("%.6f", negativeNumbers / arr.length);
        System.out.println();
        System.out.printf("%.6f", zeroNumbers / arr.length);
        System.out.println();
    }

    public static void staircase(int n) {
        for (int i = 1; i <= n; i++) {
            StringBuilder sb = new StringBuilder(i);
            for (int j = 1; j <= i; j++) {
                sb.append("#");
            }
            System.out.printf("%" + n + "s", sb);
            System.out.println();
        }
    }

    public static boolean checkBit(int n, int i) {
        return ((n >> i) % 2) != 0;
    }

    public static void miniMax(int[] arr) {
        long min = Integer.MIN_VALUE;
        long max = Integer.MAX_VALUE;
        for (int i = 0; i < (1 << arr.length); i++) {
            int count = 0;
            long sum = 0;
            for (int j = 0; j < arr.length; j++) {
                if (((i >> j) % 2) != 0) {
                    count++;
                    sum += arr[j];
                }
            }
            if (count == 4) {
                if (min == Integer.MIN_VALUE && max == Integer.MAX_VALUE) {
                    min = sum;
                    max = sum;
                } else {
                    if (sum < min) {
                        min = sum;
                    } else if (sum > max) {
                        max = sum;
                    }
                }
            }
        }
        System.out.println(min + " " + max);
    }

    public static int birthdayCakeCandles(int[] ar) {
        int height = ar[0];
        int count = 0;
        for (int value : ar) {
            if (height <= value) {
                height = value;
            }
        }
        for (int i : ar) {
            if (i == height) {
                count++;
            }
        }
        System.out.println(height);
        System.out.println(count);
        return count;
    }

    public static String timeConversion(String s) {
        if (s.equals("12:00:00AM")) {
            return "00:00:00";
        } else if (s.equals("12:00:00PM")) {
            return "12:00:00";
        } else {
            String[] splittedString = s.split(":");
            int twelveHourTime = Integer.parseInt(splittedString[0]);
            if (splittedString[2].contains("AM")) {
                if (twelveHourTime == 11) {
                    return twelveHourTime + ":" + splittedString[1] + ":" + splittedString[2].replaceAll("[A-Z]", "");
                } else if (twelveHourTime == 12) {
                    return "00" + ":" + splittedString[1] + ":" + splittedString[2].replaceAll("[A-Z]", "");
                } else {
                    return "0" + twelveHourTime + ":" + splittedString[1] + ":" + splittedString[2].replaceAll("[A-Z]", "");
                }
            } else {
                if (twelveHourTime < 12) {
                    return (twelveHourTime + 12) + ":" + splittedString[1] + ":" + splittedString[2].replaceAll("[A-Z]", "");
                } else {
                    return "12" + ":" + splittedString[1] + ":" + splittedString[2].replaceAll("[A-Z]", "");
                }
            }
        }
    }

    public static int diagonalDifference(int[][] arr) {
        int primaryDaigonalSum = 0;
        int secondaryDaigonalSum = 0;
        for (int i = 0; i < arr.length; i++) {
            primaryDaigonalSum += arr[i][i];
            secondaryDaigonalSum += arr[i][arr.length - 1 - i];
        }
        return Math.abs(primaryDaigonalSum - secondaryDaigonalSum);
    }

    public static int[] gradingStudents(int[] grades) {
        int[] roundedGrades = new int[grades.length];
        for (int i = 0; i < grades.length; i++) {
            if (grades[i] < 38) {
                roundedGrades[i] = grades[i];
            } else if ((5 - (grades[i] % 5)) < 3) {
                roundedGrades[i] = grades[i] + 5 - (grades[i] % 5);
            } else {
                roundedGrades[i] = grades[i];
            }
        }
        return roundedGrades;
    }

    public static void countApplesAndOranges(int s, int t, int a, int b, int[] apples, int[] oranges) {
        int applesCount = 0;
        int orangesCount = 0;
        for (int apple : apples) {
            if (apple + a >= s && apple + a <= t) {
                applesCount++;
            }
        }
        for (int orange : oranges) {
            if (orange + b >= s && orange + b <= t) {
                orangesCount++;
            }
        }
        System.out.println(applesCount);
        System.out.println(orangesCount);
    }

    public static String kangaroo(int x1, int v1, int x2, int v2) {
        long a = v2 - v1;
        long b = x1 - x2;
        if (a != 0 && b % a == 0 && b / a > 0) {
            return "YES";
        } else {
            return "NO";
        }
    }

    public static int getTotalX(int[] a, int[] b) {
        int count = 0;
        for (int i = a[a.length - 1]; i <= b[0]; i++) {
            int countA = 0;
            int countB = 0;
            for (int value : a) {
                if (i % value == 0) {
                    countA++;
                }
            }
            for (int value : b) {
                if (value % i == 0) {
                    countB++;
                }
            }
            if (countA == a.length && countB == b.length) {
                count++;
            }
        }
        return count;
    }

    public static int birthday(List<Integer> s, int d, int m) {
        int count = 0;
        for (int i = 0; i <= s.size() - m; i++) {
            int sum = 0;
            for (int j = i; j < i + m; j++) {
                sum += s.get(j);
            }
            if (sum == d) {
                count++;
            }
        }
        return count;
    }

    public static void hollowDaimond(int n) {
        int num = (n + 1) / 2;
        int numConstant = num;
        int space = 2;
        for (int i = 1; i <= numConstant; i++) {
            if (i == 1) {
                System.out.printf("%" + num + "s", "*");
            } else {
                System.out.printf("%" + num + "s", "*");
                System.out.printf("%" + space + "s", "*");
                space += 2;
            }
            num -= 1;
            System.out.println();
        }
        space -= 4;
        num += 2;
        for (int j = numConstant - 1; j > 0; j--) {
            if (j == 1) {
                System.out.printf("%" + num + "s", "*");
            } else {
                System.out.printf("%" + num + "s", "*");
                System.out.printf("%" + space + "s", "*");
                space -= 2;
            }
            num += 1;
            System.out.println();
        }
    }

    public static void matrix90rotate() {
        Scanner sc = new Scanner(System.in);
        int testcasesNo = sc.nextInt();
        for (int x = 1; x <= testcasesNo; x++) {
            int size = sc.nextInt();
            int[][] matrix = new int[size][size];
            for (int y = 0; y < size; y++) {
                for (int z = 0; z < size; z++) {
                    matrix[y][z] = sc.nextInt();
                }
            }
            System.out.println("Test Case #" + x + ":");
            int[][] matrixRotated = new int[matrix.length][matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix.length; j++) {
                    matrixRotated[i][j] = matrix[matrix.length - 1 - j][i];
                }
            }
            for (int k = 0; k < matrix.length; k++) {
                for (int l = 0; l < matrix.length; l++) {
                    System.out.print(matrixRotated[k][l] + " ");
                }
                System.out.println();
            }
        }
    }

    private static int factorial(int n) {
        if (n == 1) {
            return 1;
        }
        return factorial(n - 1) * n;
    }

    private static int summation(int n) {
        if (n == 1) {
            return 1;
        } else {
            return summation(n - 1) + n;
        }
    }

    private static void towersOfHenoi(int n, char from, char to, char aux) {
        if (n == 1) {
            System.out.println("moving 1 disk from source to destination");
            return;
        }
        towersOfHenoi(n - 1, from, aux, to);
        System.out.println("moving " + n + "th from from source to destination");
        towersOfHenoi(n - 1, aux, to, from);
    }

    private static boolean arraySortCheckRecusion(int[] arr) {
        if (arr.length == 2 && arr[0] <= arr[1]) {
            return true;
        } else {
            if (arr[0] <= arr[1]) {
                int[] newArr = new int[arr.length - 1];
                System.arraycopy(arr, 1, newArr, 0, arr.length - 1);
                return arraySortCheckRecusion(newArr);
            } else {
                return false;
            }
        }
    }

    private static void binaryNo(int n) {
        if (n < 1) {
            for (int i = 0; i < nbitArray.length; i++) {
                System.out.print(nbitArray[i] + " ");
            }
            System.out.println();
        } else {
            nbitArray[n - 1] = 0;
            binaryNo(n - 1);
            nbitArray[n - 1] = 1;
            binaryNo(n - 1);
        }
    }

    private static void kString(int n, int k) {
        if (n < 1) {
            for (int value : nbitArray) {
                System.out.print(value + " ");
            }
        } else {
            for (int j = 0; j < k; j++) {
                nbitArray[n - 1] = j;
                kString(n - 1, k);
            }
        }
    }

    public static void sumOfPairs(List<Integer> arr, int k) {
        int firstPointer = 0;
        int secondPointer = arr.size() - 1;
        Collections.sort(arr);
        boolean flag = false;
        if (arr.size() == 1 && arr.get(0) == k) {
            flag = true;
        }
        while (firstPointer != secondPointer) {
            int sum = arr.get(firstPointer) + arr.get(secondPointer);
            if (sum == k) {
                flag = true;
                break;
            } else if (sum < k) {
                firstPointer++;
            } else {
                secondPointer--;
            }
        }
        if (flag) {
            System.out.println("True");
        } else {
            System.out.println("False");
        }
    }

    public static void diffOfPairs(List<Integer> arr, int k) {
        boolean flag = false;
        Collections.sort(arr);
        for (int i = 0; i < arr.size() - 1; i++) {
            int searchValue = arr.get(i) + k;
            List<Integer> subArr = arr.subList(i + 1, arr.size());
            int searchresult = Collections.binarySearch(subArr, searchValue);
            if (searchresult >= 0) flag = true;
        }
        if (flag) {
            System.out.println("true");
        } else {
            System.out.println("false");
        }
    }

    public static void checkPowerOfTwo(long number) {
        boolean flag = false;
        int counter = 0;
        while (number != 0) {
            if ((number & 1) == 1) {
                counter++;
            }
            number = number >> 1;
        }
        if (counter == 1) {
            flag = true;
        }
        if (flag) {
            System.out.println("True");
        } else {
            System.out.println("False");
        }
    }

    public static void tripletWithSumK(List<Integer> arr, int k) {
        boolean flag = false;
        Collections.sort(arr);
        for (int i = 0; i < arr.size(); i++) {
            for (int j = 0; j < arr.size(); j++) {
                if (i != j) {
                    int searchValue = k - (arr.get(i) + arr.get(j));
                    List<Integer> tempCopy = new ArrayList<>(arr);
                    tempCopy.remove(i);
                    if (j != arr.size() - 1) {
                        List<Integer> subArr = tempCopy.subList(j + 1, arr.size() - 1);
                        int searchresult = Collections.binarySearch(subArr, searchValue);
                        if (searchresult >= 0) flag = true;
                    }
                }
            }
        }
        if (flag) {
            System.out.println("true");
        } else {
            System.out.println("false");
        }
    }

    public static void tripletWithSumKImproved(List<Integer> arr, int k) {
        Collections.sort(arr);
        boolean flag = false;
        for (int i = 0; i < arr.size() - 1; i++) {
            int firstPointer = i + 1;
            int secondPointer = arr.size() - 1;
            int value = k - arr.get(i);
            while (firstPointer != secondPointer) {
                if (arr.get(firstPointer) + arr.get(secondPointer) == value) {
                    flag = true;
                    break;
                } else if (arr.get(firstPointer) + arr.get(secondPointer) < value) {
                    firstPointer++;
                } else {
                    secondPointer--;
                }
            }
        }
        if (flag) {
            System.out.println("true");
        } else {
            System.out.println("false");
        }
    }

    public static void noOfDivisors(long n) {
        long count = 0;
        for (long i = 1; i <= Math.sqrt(n); i++) {
            if (Math.sqrt(n) == i) {
                count++;
            } else if (n % i == 0) {
                count = count + 2;
            }
        }
        System.out.println(count);
    }

    public static void maximumSetBits(long n) {
        String bin = Long.toBinaryString(n);
        long max = 0;
        long counter = 0;
        int length = bin.length();
        for (int i = length - 1; i >= 0; i--) {
            if (bin.charAt(i) == '1') {
                counter++;
            } else {
                counter = 0;
            }
            if (counter > max) {
                max = counter;
            }
        }
        System.out.println(max);
    }

    public static void ceil(List<Integer> arr, int n) {
        int minDiff = Integer.MAX_VALUE;
        int diffNo = Integer.MAX_VALUE;
        int length = arr.size();
        for (Integer integer : arr) {
            int diff = integer - n;
            if (diff >= 0 && diff < minDiff) {
                minDiff = diff;
                diffNo = integer;
            }
        }
        System.out.println(diffNo);
    }

    private static int cielBinarySearch(List<Integer> arr, int low, int high, int n) {
        int mid = low + (high - low) / 2;
        if (n <= arr.get(low)) {
            return arr.get(low);
        }
        if (n > arr.get(high)) {
            return Integer.MAX_VALUE;
        }
        if (arr.get(mid) == n) {
            return n;
        } else if (arr.get(mid) < n) {
            if (mid + 1 <= high && n <= arr.get(mid + 1)) {
                return arr.get(mid + 1);
            } else
                return cielBinarySearch(arr, mid + 1, high, n);
        } else {
            if (mid - 1 >= low && n > arr.get(mid - 1)) {
                return arr.get(mid);
            } else {
                return cielBinarySearch(arr, low, mid - 1, n);
            }
        }
    }

    private static int floorBinarySearch(List<Integer> arr, int low, int high, int n) {
        int mid = low + (high - low) / 2;
        if (n < arr.get(low)) {
            return Integer.MIN_VALUE;
        }
        if (n > arr.get(high)) {
            return arr.get(high);
        }
        if (n == mid) {
            return n;
        } else if (n < arr.get(mid)) {
            if (mid - 1 >= low && n > arr.get(mid - 1)) {
                return arr.get(mid - 1);
            } else {
                return floorBinarySearch(arr, low, mid - 1, n);
            }
        } else {
            if (n < arr.get(mid + 1)) {
                return arr.get(mid);
            } else {
                return floorBinarySearch(arr, mid + 1, high, n);
            }
        }
    }

    private static int floorBinarySearchReturnIndex(List<Integer> arr, int low, int high, int n) {
        int mid;
        while (low < high) {
            mid = low + (high - low + 1) / 2;
            if (arr.get(mid) >= n)
                high = mid - 1;
            else
                low = mid;
        }
        return arr.get(low) < n ? low : -1;
    }

    public static void sumOfTwoNumbers(List<Integer> arr, int k) {
        boolean flag = false;
        int len = arr.size();
        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                if (i != j && (2 * (arr.get(i) + arr.get(j)) == k)) {
                    flag = true;
                    break;
                }
            }
        }
        if (flag) {
            System.out.println("Yes");
        } else {
            System.out.println("No");
        }
    }

    public static void sumOfTwoNumbersWithTwoPointer(List<Integer> arr, int k) {
        boolean flag = false;
        int len = arr.size();
        int firstPointer = 0;
        int secondPointer = arr.size() - 1;
        Collections.sort(arr);
        while (firstPointer != secondPointer) {
            int sum = arr.get(firstPointer) + arr.get(secondPointer);
            if (2 * sum == k) {
                flag = true;
                break;
            } else if (2 * sum < k) {
                firstPointer++;
            } else {
                secondPointer--;
            }
        }
        if (flag) {
            System.out.println("Yes");
        } else {
            System.out.println("No");
        }
    }

    public static long sumAndxor(long n) {
        long counter = 0;
        for (long i = 1; i <= n; i++) {
            if ((n + i) == (n ^ i)) {
                counter++;
            }
        }
        return counter;
    }

    public static long sumAndxorModified(long n) {
        long counter = 0;
//        for (long i = 1; i <= n; i = i*2){
//            if ((i & 1) == 0){
//                counter++;
//            }
//        }
        while (n > 0) {
            if ((n & 1) == 0) {
                counter++;
            }
            n = n >> 1;
        }
        if (counter == 0) {
            return counter;
        } else {
            return ((1 << counter) - 1);
        }
    }

    public static void powerGame(List<Integer> a, List<Integer> b) {
        int counter = 0;
        int aLength = a.size();
        for (int x = aLength - 1; x >= 0; x--) {
            int floor = floorBinarySearchReturnIndex(b, 0, b.size() - 1, a.get(x));
            if (floor >= 0) {
                b.remove(floor);
                counter++;
            }
        }
        System.out.println(counter);
    }

    private static void smallerElements(int[] arr) {
        int counter = 0;
        for (int i = 0; i < arr.length; i++) {
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[i] > arr[j]) {
                    counter++;
                }
            }
        }
        System.out.println(counter);
    }

    public void matrixRotationBy90(int[][] matrix) {
        int[][] matrixRotated = new int[matrix.length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                matrixRotated[i][j] = matrix[matrix.length - 1 - j][i];
            }
        }
        for (int k = 0; k < matrix.length; k++) {
            for (int l = 0; l < matrix.length; l++) {
                System.out.print(matrixRotated[k][l] + " ");
            }
            System.out.println();
        }
    }
}
