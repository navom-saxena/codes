package dsalgo.hackerrank.datastructures;

import dsalgo.implementations.LinkedListBasedNodes;
import dsalgo.practice.HackerRankAux3;
import javafx.util.Pair;

import java.io.*;
import java.util.*;
import java.util.Arrays;
import java.util.stream.Collectors;

public class Stacks {

    private static List<Integer> primes;

    static {
//        primes = HackerRankAux3.sieveOfEratosthenes(1300);
    }

    public static void main(String[] args) throws IOException {

//        Deque<Integer> stack = new ArrayDeque<>();
//        stack.push(1);
//        stack.push(2);
//        stack.push(3);
//        System.out.println(stack);
//        reverseStack(stack);
//        System.out.println(stack);
//        maxElementInStack();
//        System.out.println(isBalancedBrackets("()()("));
//        System.out.println(equalStacks(new int[] {3,2,1,1,1}, new int[] {4,3,2}, new int[] {1,1,4,1}));
//        System.out.println(twoStacksGame(67, new int[]{19, 9, 8, 13, 1, 7, 18, 0, 19, 19, 10, 5, 15, 19, 0, 0, 16, 12, 5, 10},
//                new int[]{11, 17, 1, 18, 14, 12, 9, 18, 14, 3, 4, 13, 4, 12, 6, 5, 12, 16, 5, 11, 16, 8, 16, 3, 7, 8, 3, 3, 0, 1, 13, 4, 10, 7, 14}));
//        System.out.println(largestRectangle(new int[]{1,2,3,4,5}));
//        simpleTextEditor();
//        simpleTextEditorOptimised();
//        int [] finalNumbers = waiter(new int[] {3,3,4,4,9},2);
//        Arrays.stream(finalNumbers).boxed().collect(Collectors.toList()).forEach(System.out::println);
//        System.out.println(andXorOr(new int[]{9, 6, 3, 5, 2}));
        System.out.println(poisonousPlants(new int[]{3,2,5,4}));
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

    private static void reverseStack(Deque<Integer> stack) {
        if (stack.isEmpty()) {
            return;
        }
        int temp = stack.pop();
        reverseStack(stack);
        insertAtBottom(stack, temp);
    }

    private static void insertAtBottom(Deque<Integer> stack, int data) {
        if (stack.isEmpty()) {
            stack.push(data);
            return;
        }
        int temp = stack.pop();
        insertAtBottom(stack, data);
        stack.push(temp);
    }

    private static long largestRectangle(int[] h) {
        Deque<Integer> stack = new ArrayDeque<>();
        int i;
        long maxArea = Long.MIN_VALUE;
        for (i = 0; i < h.length; i++) {
            if (stack.isEmpty() || h[stack.peek()] <= h[i]) {
                stack.push(i);
            } else {
                while (!stack.isEmpty() && h[stack.peek()] > h[i]) {
                    int poppedElement = stack.pop();
                    if (stack.isEmpty()) {
                        long area = h[poppedElement] * (i);
                        maxArea = Math.max(maxArea, area);
                    } else {
                        long area = h[poppedElement] * (i - poppedElement);
                        maxArea = Math.max(maxArea, area);
                    }
                }
                stack.push(i);
            }
        }
        while (!stack.isEmpty()) {
            int poppedElement = stack.pop();
            if (stack.isEmpty()) {
                long area = h[poppedElement] * (i);
                maxArea = Math.max(maxArea, area);
            } else {
                long area = h[poppedElement] * (i - poppedElement);
                maxArea = Math.max(maxArea, area);
            }
        }
        return maxArea;
    }

    private static void simpleTextEditor() throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
        int firstInt = Integer.parseInt(br.readLine());
        List<Character> editorString = new ArrayList<>();
        Deque<Pair<String, String>> operationsStack = new ArrayDeque<>();
        for (int z = 0; z < firstInt; z++) {
            String[] input = br.readLine().split(" ");
            String operation = input[0];
            switch (operation) {
                case "1": {
                    String value = input[1];
                    for (int i = 0; i < value.length(); i++) {
                        editorString.add(value.charAt(i));
                    }
                    operationsStack.push(new Pair<>(operation, value));
                    break;
                }
                case "2": {
                    int deleteCount = Integer.parseInt(input[1]);
                    StringBuilder sb = new StringBuilder();
                    while (deleteCount > 0) {
                        sb.append(editorString.remove(editorString.size() - 1));
                        deleteCount--;
                    }
                    operationsStack.push(new Pair<>(operation, sb.reverse().toString()));
                    break;
                }
                case "3": {
                    int printIndex = Integer.parseInt(input[1]);
                    char value = editorString.get(printIndex - 1);
                    bw.write(value + "\n");
                    bw.flush();
                    break;
                }
                case "4": {
                    Pair<String, String> poppedOperation = operationsStack.pop();
                    if (poppedOperation.getKey().equals("1")) {
                        int appendedValue = poppedOperation.getValue().length();
                        while (appendedValue > 0) {
                            editorString.remove(editorString.size() - 1);
                            appendedValue--;
                        }
                    } else if (poppedOperation.getKey().equals("2")) {
                        String value = poppedOperation.getValue();
                        for (int i = 0; i < value.length(); i++) {
                            editorString.add(value.charAt(i));
                        }
                    }
                    break;
                }
            }
        }
    }

    private static void simpleTextEditorOptimised() throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
        Deque<String> stack = new ArrayDeque<>();
        String currentValue = "";
        stack.push(currentValue);
        int firstInt = Integer.parseInt(br.readLine());
        for (int z = 0; z < firstInt; z++) {
            String[] input = br.readLine().split(" ");
            String operation = input[0];
            switch (operation) {
                case "1": {
                    String value = input[1];
                    for (int i = 0; i < value.length(); i++) {
                        currentValue += (value.charAt(i));
                    }
                    stack.push(currentValue);
                    break;
                }
                case "2": {
                    int deleteCount = Integer.parseInt(input[1]);
                    while (deleteCount > 0) {
                        currentValue = currentValue.substring(0, currentValue.length() - 1);
                        deleteCount--;
                    }
                    stack.push(currentValue);
                    break;
                }
                case "3": {
                    int printIndex = Integer.parseInt(input[1]);
                    char value = currentValue.charAt(printIndex - 1);
                    bw.write(value + "\n");
                    bw.flush();
                    break;
                }
                case "4": {
                    stack.pop();
                    if (!stack.isEmpty()) {
                        currentValue = stack.peek();
                    }
                }
            }
        }
        br.close();
        bw.close();
    }

    private static int[] waiter(int[] number, int q) {
        List<Deque<Integer>> listOfStacks = new ArrayList<>();
        Deque<Integer> finalNumbers = new ArrayDeque<>();
        Deque<Integer> initialNumbers = new ArrayDeque<>();
        for (int num : number) {
            initialNumbers.push(num);
        }
        Deque<Integer> nonDivisible = new ArrayDeque<>();
        for (int i = 0; i < q; i++) {
            Deque<Integer> divisible = new ArrayDeque<>();
            int primeNo = primes.get(i);
            while (!initialNumbers.isEmpty()) {
                int num = initialNumbers.pop();
                if (num % primeNo == 0) {
                    divisible.push(num);
                } else {
                    nonDivisible.push(num);
                }
            }
            listOfStacks.add(divisible);
        }
        if (!initialNumbers.isEmpty()) {
            listOfStacks.add(initialNumbers);
        }
        int[] finalArray = new int[finalNumbers.size()];
        int counter = 0;
        while (!finalNumbers.isEmpty()) {
            finalArray[counter] = finalNumbers.removeLast();
            counter++;
        }
        return finalArray;
    }

    private static int andXorOr(int[] a) {
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(a[0]);
        int max = Integer.MIN_VALUE;
        for (int i = 1; i < a.length; i++) {
            int currentNo = a[i];
            while (!stack.isEmpty()) {
                int top = stack.peek();
                int value = (currentNo ^ top);
                max = Math.max(max, value);
                if (currentNo < top) {
                    stack.pop();
                } else {
                    break;
                }
            }
            stack.push(currentNo);
        }
        return max;
    }

    private static int poisonousPlants(int[] p) {
        List<LinkedListBasedNodes> arr = new ArrayList<>();
        LinkedListBasedNodes l0 = new LinkedListBasedNodes();
        l0.addNode(p[0]);
        arr.add(l0);
        for (int i = 1; i < p.length; i++) {
            int currentValue = p[i];
            if (currentValue <= p[i - 1]) {
                LinkedListBasedNodes l = arr.get(arr.size() - 1);
                l.addNode(currentValue);
            } else {
                LinkedListBasedNodes l = new LinkedListBasedNodes();
                l.addNode(currentValue);
                arr.add(l);
            }
        }
        int whileCount = 0;
        while (arr.size() > 0 && arr.size() != 1) {
            for (int arrCounter = 1; arrCounter < arr.size(); arrCounter++) {
                LinkedListBasedNodes l = arr.get(arrCounter);
                if (arrCounter >= 1) {
                    LinkedListBasedNodes lBefore = arr.get(arrCounter - 1);
                    if (l.getFirstNode() != null && lBefore.canMerge(l)) {
                        lBefore.merge(l);
                        arr.remove(arrCounter);
                        arrCounter--;
                    } else {
                        l.deleteFirst();
                        if (l.getFirstNode() == null) {
                            arr.remove(arrCounter);
                            arrCounter--;
                        } else if (l.getFirstNode() != null && lBefore.canMerge(l)) {
                            lBefore.merge(l);
                            arr.remove(arrCounter);
                            arrCounter--;
                        }
                    }
                }
            }
            whileCount++;
        }
        return whileCount;
    }
}
