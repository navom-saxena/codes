package dsAlgo.hackerRank.datastructures;

import java.io.*;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

public class Stack {
    public static void main(String[] args) throws IOException {

        java.util.Stack<Integer> stack = new java.util.Stack<>();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        System.out.println(stack);
        reverseStack(stack);
        System.out.println(stack);
//        maxElementInStack();
//        System.out.println(isBalancedBrackets("()()("));
//        System.out.println(equalStacks(new int[] {3,2,1,1,1}, new int[] {4,3,2}, new int[] {1,1,4,1}));
        System.out.println(twoStacksGame(67, new int[]{19, 9, 8, 13, 1, 7, 18, 0, 19, 19, 10, 5, 15, 19, 0, 0, 16, 12, 5, 10},
                new int[]{11, 17, 1, 18, 14, 12, 9, 18, 14, 3, 4, 13, 4, 12, 6, 5, 12, 16, 5, 11, 16, 8, 16, 3, 7, 8, 3, 3, 0, 1, 13, 4, 10, 7, 14}));
    }

    private static void maxElementInStack() throws IOException {
        Deque<Integer> stack = new ArrayDeque<>();
        Deque<Integer> auxiliaryStack = new ArrayDeque<>();

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
        int firstInt = Integer.parseInt(br.readLine());
        for (int z = 0; z < firstInt; z++) {
            String input = br.readLine();
            String[] inputArr = input.split(" ");
            if (Integer.parseInt(inputArr[0]) == 1) {
                Integer pushValue = Integer.parseInt(inputArr[1]);
                stack.push(pushValue);
                if (auxiliaryStack.isEmpty()) {
                    auxiliaryStack.push(pushValue);
                } else {
                    int topValue = auxiliaryStack.peek();
                    if (topValue < pushValue) {
                        auxiliaryStack.push(pushValue);
                    } else {
                        auxiliaryStack.push(topValue);
                    }
                }
            } else if (Integer.parseInt(inputArr[0]) == 2) {
                stack.pop();
                auxiliaryStack.pop();
            } else if (Integer.parseInt(inputArr[0]) == 3) {
                if (!auxiliaryStack.isEmpty()) {
                    bw.write(auxiliaryStack.peek().toString());
                    bw.flush();
                }
            }
        }
    }

    private static String isBalancedBrackets(String s) {
        if (s.length() % 2 != 0) {
            return "NO";
        }
        Deque<Character> stack = new ArrayDeque<>();
        for (int i = 0; i < s.length(); i++) {
            char characterAti = s.charAt(i);
            if (characterAti == '(' || characterAti == '[' || characterAti == '{') {
                stack.push(characterAti);
            } else {
                if (stack.isEmpty()) {
                    return "NO";
                } else {
                    char poppedValue = stack.pop();
                    if (!(characterAti == ')' && poppedValue == '(' ||
                            characterAti == ']' && poppedValue == '[' ||
                            characterAti == '}' && poppedValue == '{')) {
                        return "NO";
                    }
                }
            }
        }
        if (!stack.isEmpty()) {
            return "NO";
        }
        return "YES";
    }

    private static int equalStacks(int[] h1, int[] h2, int[] h3) {
        int height1 = h1.length;
        int height2 = h2.length;
        int height3 = h3.length;
        int[] suffixSumH1 = new int[height1];
        int[] suffixSumH2 = new int[height2];
        int[] suffixSumH3 = new int[height3];

        suffixSumH1[height1 - 1] = h1[height1 - 1];
        for (int i = height1 - 2; i >= 0; i--) {
            suffixSumH1[i] = suffixSumH1[i + 1] + h1[i];
        }
        suffixSumH2[height2 - 1] = h2[height2 - 1];
        for (int i = height2 - 2; i >= 0; i--) {
            suffixSumH2[i] = suffixSumH2[i + 1] + h2[i];
        }
        suffixSumH3[height3 - 1] = h3[height3 - 1];
        for (int i = height3 - 2; i >= 0; i--) {
            suffixSumH3[i] = suffixSumH3[i + 1] + h3[i];
        }
        int min = Math.min(Math.min(height1, height2), height3);
        if (min == height1) {
            Arrays.sort(suffixSumH2);
            Arrays.sort(suffixSumH3);
            for (int value : suffixSumH1) {
                if (Arrays.binarySearch(suffixSumH2, value) >= 0 &&
                        Arrays.binarySearch(suffixSumH3, value) >= 0) {
                    return value;
                }
            }
        } else if (min == height2) {
            Arrays.sort(suffixSumH1);
            Arrays.sort(suffixSumH3);
            for (int value : suffixSumH2) {
                if (Arrays.binarySearch(suffixSumH1, value) >= 0 &&
                        Arrays.binarySearch(suffixSumH3, value) >= 0) {
                    return value;
                }
            }
        } else {
            for (int value : suffixSumH3) {
                Arrays.sort(suffixSumH2);
                Arrays.sort(suffixSumH1);
                if (Arrays.binarySearch(suffixSumH1, value) >= 0 &&
                        Arrays.binarySearch(suffixSumH2, value) >= 0) {
                    return value;
                }
            }
        }
        return 0;
    }

    private static int twoStacksGame(int x, int[] a, int[] b) {
        int lengthA = 0;
        int sum = 0;
        while (lengthA < a.length && sum + a[lengthA] <= x) {
            sum += a[lengthA];
            lengthA++;
        }
        int maxScore = lengthA;
        for (int lengthB = 1; lengthB <= b.length; lengthB++) {
            sum += b[lengthB - 1];
            while (sum > x && lengthA > 0) {
                lengthA--;
                sum -= a[lengthA];
            }
            if (sum > x) {
                break;
            }
            maxScore = Math.max(maxScore, lengthA + lengthB);
        }
        return maxScore;
    }

    // stack java testing

    private static void reverseStack(java.util.Stack<Integer> stack) {
        if (stack.isEmpty()) {
            return;
        }
        int temp = stack.pop();
        reverseStack(stack);
        insertAtBottom(stack, temp);
    }

    private static void insertAtBottom(java.util.Stack<Integer> stack, int data) {
        if (stack.isEmpty()) {
            stack.push(data);
            return;
        }
        int temp = stack.pop();
        insertAtBottom(stack, data);
        stack.push(temp);
    }
}
