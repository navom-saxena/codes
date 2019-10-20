package dsAlgo;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Practice3 {

    private static Map<Integer, Set<Integer>> tm = new TreeMap<>();
    private static int daimeterValue = Integer.MIN_VALUE;
    private static int CHECK_VALUE = Integer.MIN_VALUE;

    public static void main(String[] args) throws IOException {
//        ArrayList<Integer> arr = new ArrayList<>();
//        arr.add(57);
//        arr.add(96);
//        subsetsLexographical(arr, arr.size(), -1, new ArrayList<>());
//        System.out.println(findingFrequency(new int[] {-6,10,-1,20,-1,15,5,-1,15},20));
//        distinctElementsInWindow(new int[] {-5,-1,4,8,-5,-3,-4,7,4,-4,0,8,0,-2,3,2,5},13);
//        distinctElementsInWindowModified(new int[] {-5,-1,4,8,-5,-3,-4,7,4,-4,0,8,0,-2,3,2,5},13);
//        aPowerB(961);
//        System.out.println(firstRepeatingCharacter("datastructures"));
//        wordsVowels("a     u    b");
//        System.out.println(sieveOfEratosthenes(30));
//        sieveOfEratosthenesBwRange(5,11);
//        System.out.println(subsequenceCheck("d","datrrrrarttttvvv"));
//        System.out.println(countSub(new int [] {-5,10,-3},3,5));
//        System.out.println(perfectSquaresInRange(3,9));
//        System.out.println(farewellParty(new String[] {"16 58","4 16","12 71","57 72"}));
//        collectingWater(new int[] {5,4,3,2,1});
//        cumArr(new long[]{1,30,13,-4,-5,12,-53,-12,43,100});
//        int [] arr = new int[Integer.MAX_VALUE];
//        System.out.println(primeFear(30));
//        collectingMangoes();
//        stockSpan(new int[] {100,80,60,70,60,75,85 });
//        System.out.println(histogramMaxArea(new int[] { 6, 2, 5, 4, 5}));
//        Node root  = new Node(1);
//        insertNodeInBST(root, 2);
//        insertNodeInBST(root, 3);
//        insertNodeInBST(root, 4);
//        insertNodeInBST(root, 5);
//        bottomUpLevelOrder(root);
        Node root = new Node(3);
        insertNodeInBST(root, 1);
        insertNodeInBST(root, 2);
        insertNodeInBST(root, 5);
        insertNodeInBST(root, 4);

//        rootToLeafTrigger(root);
//        System.out.println(sizeOfBST(root));
//        System.out.println(height(root));
//        System.out.println(daimeterValue);
//        System.out.println(searchInBST(root,8));
//        verticalLevelTraversal(root,0);
//        System.out.println(tm);
//        tm.forEach((key,value) -> {
//            value.stream().map(x -> x + " ").forEach(System.out::print);
//            System.out.println();});
//        System.out.println(countNodes(root));
    }

    private static void subsetsLexographical(ArrayList<Integer> arr, int n, int index, ArrayList<Integer> curr) {
        if (n == index) {
            return;
        }
        if (!curr.isEmpty()) {
            curr.stream().map(x -> x + " ").forEach(System.out::print);
            System.out.println();
        }
        for (int i = index + 1; i < n; i++) {
            curr.add(arr.get(i));
            subsetsLexographical(arr, arr.size(), i, curr);
            curr.remove(curr.size() - 1);
        }
    }

    public static int findingFrequency(int[] arr, int n) {
        Map<Integer, Integer> hm = new ConcurrentHashMap<>();
        Arrays.stream(arr).parallel().forEach(x -> hm.merge(x, 1, Integer::sum));
        return hm.getOrDefault(n, 0);
    }

    public static void distinctElementsInWindow(int[] arr, int k) {
        for (int i = 0; i < arr.length - k + 1; i++) {
            Set<Integer> se = new HashSet<>();
            for (int j = i; j < k + i; j++) {
                se.add(arr[j]);
            }
            System.out.println(se.size());
        }
    }

    public static void distinctElementsInWindowModified(int[] arr, int k) {
        Map<Integer, Integer> hm = new HashMap<>();
        int counter = 0;
        for (int i = 0; i < arr.length; i++) {
            counter++;
            if (i > k - 1) {
                int value = hm.get(arr[i - k]);
                if (value == 1) {
                    hm.remove(arr[i - k]);
                } else {
                    hm.put(arr[i - k], value - 1);
                }
            }
            hm.merge(arr[i], 1, Integer::sum);
            if (counter >= k) {
                System.out.println(hm.size());
            }
        }
    }

    public static void aPowerB(int n) {
        boolean flag = false;
        for (int i = 2; i <= 10000; i++) {
            long j = i;
            while (j <= n) {
                if (j == n) {
                    flag = true;
                    break;
                }
                j = j * i;
            }
        }
        System.out.println(flag);
    }

    public static char firstRepeatingCharacter(String seq) {
        Map<Character, Integer> hm = new HashMap<>();
        int len = seq.length();
        for (int i = 0; i < len; i++) {
            hm.merge(seq.charAt(i), 1, Integer::sum);
        }
        for (int i = 0; i < len; i++) {
            char charValue = seq.charAt(i);
            if (hm.get(charValue) > 1) {
                return charValue;
            }
        }
        return '.';
    }

    public static void wordsVowels(String input) {
        Map<Character, Integer> hm = new HashMap<>();
        Map<Character, Integer> hm2 = new HashMap<>();
        String trimmedInput = input.trim();
        String[] inputArr = trimmedInput.split(" ");
        int wordsCount = 0;
        int consonentsCount = 0;
        int vowelsCount = 0;
        for (String s : inputArr) {
            String temp = s.trim();
            for (int k = 0; k < temp.length(); k++) {
                char valueAtPosition = temp.charAt(k);
                if ("AEIOUaeiou".indexOf(valueAtPosition) >= 0) {
                    hm.merge(valueAtPosition, 1, Integer::sum);
                } else {
                    hm2.merge(valueAtPosition, 1, Integer::sum);
                }

            }
            if (!temp.equals("")) {
                wordsCount++;
            }
        }
        vowelsCount = hm.values().parallelStream().reduce(Integer::sum).orElse(0);
        consonentsCount = hm2.values().parallelStream().reduce(Integer::sum).orElse(0);
        if (input.equals("")) {
            wordsCount = 0;
        }
        System.out.println(wordsCount + " " + vowelsCount + " " + consonentsCount);
    }
//    interleavings py3
//    def recur(str1, str2, istr, m, n, i):
//            if m == 0 and n == 0:
//            output_list.append("".join(istr))
//            if m != 0:
//    istr[i] = str1[0]
//    recur(str1[1:], str2, istr, m-1, n, i+1)
//    if n != 0:
//    istr[i] = str2[0]
//    recur(str1, str2[1:], istr, m, n-1, i+1)
//
//for x in range(0,int(input())):
//    input_str = str(input()).split(" ")
//    str1 = input_str[0]
//    str2 = input_str[1]
//    m = len(str1)
//    n = len(str2)
//
//    istr = [''] * (m+n)
//    output_list = []
//    print("Case #{}:".format(x + 1))
//            if(str1[0] < str2[0]):
//    recur(str1, str2, istr,m,n,0)
//    else:
//    recur(str2, str1, istr,n,m,0)
//    print(*sorted(output_list),sep='\n')

    private static void interLeavings(String a, String b, int positionA, int positionB) {
        while (positionA != (a.length() - 1) && positionB != (b.length() - 1)) {
            interLeavings(a, b, positionA + 1, positionB);
            interLeavings(a, b, positionA, positionB + 1);
        }
    }

    public static void largestConcatNumber(String[] arr) {
        String op = Arrays.stream(arr).sorted((String x, String y) -> {
            String xy = x + y;
            String yx = y + x;
            return xy.compareTo(yx) > 0 ? -1 : 1;
        }).reduce((x, y) -> x + y).get();
        System.out.println(op);
    }

    public static long sieveOfEratosthenes(int n) {
        boolean[] arr = new boolean[n + 1];
        int count = 0;
        for (int i = 2; i <= Math.sqrt(n); i++) {
            for (int j = i * i; j <= n; j += i) {
                arr[j] = true;
//                System.out.println("j - " + j);
            }
        }
        for (int k = 2; k < arr.length; k++) {
            if (!arr[k]) {
                System.out.print(k + " ");
            }
        }
        System.out.println();
        return count;
    }

    public static long sieveOfEratosthenesBwRange(int l, int r) {
        long[] arr1 = new long[Double.valueOf(Math.pow(10, 6)).intValue()];
        boolean[] arr = new boolean[r + 1];
        int count = 0;
        for (int i = 2; i <= Math.sqrt(r); i++) {
            for (int j = i * i; j <= r; j += i) {
                arr[j] = true;
//                System.out.println("j - " + j);
            }
        }
        long sum = 0;
        for (int k = 2; k < arr.length; k++) {
            if (!arr[k]) {
                System.out.print(k + " ");
                sum = sum + 1;
                arr1[k] = sum;
                count++;
            } else {
                arr1[k] = sum;
            }
        }
//        System.out.println();
//        for (int u = 0; u < 15; u++){
//            System.out.print(arr1[u] + " ");
//        }
//        System.out.println();
        return arr1[r] - arr1[l - 1];
    }

    public static long primeFear(int n) {
        Set<String> hs = new HashSet<>();
        Set<Integer> ts = new TreeSet<>();
        boolean[] arr = new boolean[n + 1];
        int count = 0;
        for (int i = 2; i <= Math.sqrt(n); i++) {
            for (int j = i * i; j <= n; j += i) {
                arr[j] = true;
//                System.out.println("j - " + j);
            }
        }
        int nonZeroContainingPrime = 0;
        for (int k = 2; k < arr.length; k++) {
            if (!arr[k]) {
//                System.out.print(k + " ");
                String primeStr = Integer.toString(k);
                hs.add(primeStr);
                boolean checkflag = false;
                if (!primeStr.contains("0")) {
                    System.out.print(primeStr + " ");
                    while (primeStr.length() != 0) {
                        if (hs.contains(primeStr)) {
                            checkflag = true;
                        } else {
                            checkflag = false;
                            break;
                        }
                        primeStr = primeStr.substring(1);
                    }
                    System.out.println();
//                    System.out.println(k);
                    if (checkflag) {
                        ts.add(k);
                        System.out.println(k + " ---");
                        nonZeroContainingPrime++;
                    }
                }
            }
        }
        Integer[] tsArr = new Integer[ts.size()];
        tsArr = ts.toArray(tsArr);
        System.out.println(hs);
        System.out.println(ts);
        return nonZeroContainingPrime;
    }

    public static boolean subsequenceCheck(String a, String b) {
        int pointerA = 0;
        int pointerB = 0;
        int lengthA = a.length();
        int lengthB = b.length();
        int counterB = 0;
        int counterA = 0;
        while (counterB < lengthB && lengthA > 0) {
            if (b.charAt(pointerB) != a.charAt(pointerA)) {
                pointerB++;
            } else {
                pointerB++;
                pointerA++;
            }
            if (pointerA == lengthA) {
                return true;
            }
            counterB++;
            counterA++;
        }
        return false;
    }

    static int anagramsRearrange(String str1, String str2) {
        int[] count1 = new int[26];
        int[] count2 = new int[26];

        for (int i = 0; i < str1.length(); i++)
            count1[str1.charAt(i) - 'a']++;

        for (int i = 0; i < str2.length(); i++)
            count2[str2.charAt(i) - 'a']++;

        int result = 0;
        for (int i = 0; i < 26; i++) {
            result += Math.abs(count1[i] - count2[i]);
        }
        return result;
    }

    static int countSub(int[] arr, int n, int x) {
        int st = 0;
        int end = 0;
        int sum = 0;
        int cnt = 0;
        while (end < n) {
            sum += arr[end];
            while (st <= end && sum > x) {
                sum -= arr[st];
                st++;
            }
            cnt += (end - st + 1);
            end++;
        }
        return cnt;
    }

    public static int perfectSquaresInRange(long l, long r) {
        int count = 0;
        for (long i = l; i <= r; i++) {
            double sqrt = Math.sqrt(i);
            int c = 0;
            for (int j = 1; j <= sqrt; j++) {
                if (sqrt == j) {
                    c++;
                } else if (i % j == 0) {
                    c = c + 2;
                }
            }
            if (c % 2 != 0) {
                count++;
            }
        }
        return count;
    }

    public static int farewellParty(String[] arr) {
        int[] countArr = new int[86401];
        for (String elem : arr) {
            String[] elemArr = elem.split(" ");
            for (int i = Integer.parseInt(elemArr[0]); i <= Integer.parseInt(elemArr[1]); i++) {
                countArr[i]++;
            }
        }
        int max = Short.MIN_VALUE;
        for (int currentElem : countArr) {
            if (currentElem >= max) {
                max = currentElem;
            }
        }
        return max;
    }

    public static void collectingWater(int[] arr) {
        int[] left = new int[arr.length];
        left[0] = arr[0];
        for (int i = 1; i < arr.length; i++) {
            left[i] = Math.max(arr[i], left[i - 1]);
        }
        int[] right = new int[arr.length];
        right[arr.length - 1] = arr[arr.length - 1];
        for (int i = arr.length - 2; i >= 0; i--) {
            right[i] = Math.max(arr[i], right[i + 1]);
        }
        int sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += Math.min(left[i], right[i]) - arr[i];
        }
        System.out.println(sum);
    }

    public static void cumArr(long[] arr) {
        long[] cumArr = new long[arr.length];
        long cumArrSum = 0;
        for (int i = 0; i < arr.length; i++) {
            cumArrSum = arr[i] + cumArrSum;
            cumArr[i] = cumArrSum;
        }
        for (int y = 0; y < arr.length; y++) {
            System.out.print(cumArr[y] + " ");
        }
    }

    public static void collectingMangoes() {
        Stack<Integer> stack = new Stack<>();
        Stack<Integer> maxStack = new Stack<>();
        String[] input = new String[]{"A 10", "A 10", "R", "Q"};
        for (String inputElem : input) {
            String[] inputArr = inputElem.split(" ");
            if (inputArr[0].equals("A")) {
                int pushValue = Integer.parseInt(inputArr[1]);
                stack.push(pushValue);
                if (!maxStack.isEmpty() && maxStack.peek() <= pushValue) {
                    maxStack.push(pushValue);
                } else if (maxStack.isEmpty()) {
                    maxStack.push(pushValue);
                }
            }
            if (inputArr[0].equals("Q")) {
                if (!maxStack.isEmpty()) {
                    System.out.println(maxStack.peek());
                } else {
                    System.out.println("Empty");
                }
            }
            if (inputArr[0].equals("R")) {
                if (!stack.isEmpty()) {
                    int poppedValue = stack.pop();
                    if (!maxStack.isEmpty() && poppedValue == maxStack.peek()) {
                        maxStack.pop();
                    }
                }
            }
        }
    }

    static int[] stockSpan(int[] arr) {
        int len = arr.length;
        int[] newArr = new int[len];
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        for (int i = 1; i < len; i++) {
            while (!stack.isEmpty() && arr[stack.peek()] <= arr[i]) {
                stack.pop();
            }
            newArr[i] = (!stack.isEmpty()) ? i - stack.peek() : i + 1;
            stack.push(i);
        }
        return newArr;
    }

    public static int histogramMaxArea(int[] arr) {
        Stack<Integer> stack = new Stack<>();
        int max = 0;
        int i = 0;
        int n = arr.length;
        while (i < n) {
            if (stack.isEmpty() || arr[stack.peek()] <= arr[i]) {
                stack.push(i);
                i++;
            } else {
                int top = stack.pop();
                int area = arr[top] * (stack.isEmpty() ? i : i - stack.peek() - 1);
                if (area > max) {
                    max = area;
                }
            }
        }
        while (!stack.isEmpty()) {
            int top = stack.pop();
            int area = arr[top] * (stack.isEmpty() ? i : i - stack.peek() - 1);
            if (area > max) {
                max = area;
            }
        }
        return max;
    }

    private static Node insertNodeInBST(Node root, int data) {
        if (root == null) {
            root = new Node(data);
        } else if (data < root.data) {
            root.setLeftNode(insertNodeInBST(root.getLeftNode(), data));
        } else if (data > root.data) {
            root.setRightNode(insertNodeInBST(root.getRightNode(), data));
        }
        return root;
    }

    private static void preOrderBST(Node root) {
        if (root == null) {
            return;
        }
        System.out.print(root.data + " ");
        preOrderBST(root.getLeftNode());
        preOrderBST(root.getRightNode());
    }

    private static void inOrderBST(Node root) {
        if (root == null) {
            return;
        }
        inOrderBST(root.getLeftNode());
        System.out.print(root.data + " ");
        inOrderBST(root.getRightNode());
    }

    private static void postOrderBST(Node root) {
        if (root == null) {
            return;
        }
        postOrderBST(root.getLeftNode());
        postOrderBST(root.getRightNode());
        System.out.print(root.data + " ");
    }

    private static int height(Node root) {
        if (root == null) {
            return 0;
        }
        int leftHeight = height(root.getLeftNode());
        int rightHeight = height(root.getRightNode());
        daimeterValue = Math.max(daimeterValue, leftHeight + rightHeight + 1);
        return (1 + Math.max(leftHeight, rightHeight));
    }

    private static int countNodes(Node root) {
        if (root == null) {
            return 0;
        }
        return 1 + countNodes(root.getLeftNode()) + countNodes(root.getRightNode());
    }

    public static Map<Integer, Integer> depth(Node root) {
        Queue<Node> queue = new LinkedList<>();
        Map<Integer, Integer> hm = new HashMap<>();
        queue.add(root);
        hm.put(root.data, 0);
        queue.add(null);
        int counter = 1;
        while (!queue.isEmpty()) {
            Node dequeuedNode = queue.remove();
            if (dequeuedNode != null) {
                if (dequeuedNode.getLeftNode() != null) {
                    Node leftChild = dequeuedNode.getLeftNode();
                    hm.put(leftChild.data, counter);
                    queue.add(leftChild);
                }
                if (dequeuedNode.getRightNode() != null) {
                    Node rightChild = dequeuedNode.getRightNode();
                    hm.put(rightChild.data, counter);
                    queue.add(rightChild);
                }
            } else {
                counter++;
                if (!queue.isEmpty()) {
                    queue.add(null);
                }
            }
        }
        return hm;
    }

    public static void zigzagTraversal(Node root) {
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        System.out.println(root.data + " ");
        queue.add(null);
        int counter = 2;
        List<Integer> arr1 = new ArrayList<>();
        while (!queue.isEmpty()) {
            Node dequeuedNode = queue.remove();
            if (dequeuedNode != null) {
                if (dequeuedNode.getLeftNode() != null) {
                    Node leftChild = dequeuedNode.getLeftNode();
                    arr1.add(leftChild.data);
                    queue.add(leftChild);
                }
                if (dequeuedNode.getRightNode() != null) {
                    Node rightChild = dequeuedNode.getRightNode();
                    arr1.add(rightChild.data);
                    queue.add(rightChild);
                }
            } else {
                if (counter % 2 == 0) {
                    for (Integer integer : arr1) {
                        System.out.println(integer + " ");
                    }
                }
                if (counter % 2 != 0) {
                    for (int r = arr1.size() - 1; r >= 0; r--) {
                        System.out.println(arr1.get(r) + " ");
                    }
                }
                arr1.clear();
                if (!queue.isEmpty()) {
                    queue.add(null);
                }
                counter++;
            }
        }
        System.out.println();
    }

    public static void bottomUpLevelOrder(Node root) {
        Queue<Node> queue = new LinkedList<>();
        Map<Integer, List<Integer>> hm = new HashMap<>();
        queue.add(root);
        List<Integer> levelDataList = new ArrayList<>();
        levelDataList.add(root.data);
        hm.put(1, levelDataList);
        levelDataList.clear();
        queue.add(null);
        int counter = 1;
        while (!queue.isEmpty()) {
            Node deQueuedNode = queue.remove();
            if (deQueuedNode != null) {
                levelDataList.add(deQueuedNode.getData());
                if (deQueuedNode.getLeftNode() != null) {
                    queue.add(deQueuedNode.getLeftNode());
                }
                if (deQueuedNode.getRightNode() != null) {
                    queue.add(deQueuedNode.getRightNode());
                }
            } else {
                List<Integer> hmValueList = new ArrayList<>(levelDataList);
                hm.put(counter, hmValueList);
                levelDataList.clear();
                if (!queue.isEmpty()) {
                    queue.add(null);
                    counter++;
                }
            }
        }
        while (counter != 0) {
            List<Integer> hmValueList = hm.get(counter);
            for (Integer integer : hmValueList) {
                System.out.print(integer + " ");
            }
            System.out.println();
            counter--;
        }
    }

    private static void verticalLevelTraversal(Node root, int pos) {
        if (root == null) {
            return;
        }
        int nodeData = root.getData();
        Set<Integer> value = tm.get(pos);
        if (value == null) {
            Set<Integer> arr = new TreeSet<>();
            arr.add(nodeData);
            tm.put(pos, arr);
        } else {
            value.add(nodeData);
            tm.put(pos, value);
        }
        verticalLevelTraversal(root.getLeftNode(), pos - 1);
        verticalLevelTraversal(root.getRightNode(), pos + 1);
    }

    private static boolean isFullBinaryTree(Node root) {
        if (root == null) {
            return true;
        }
        if (root.getLeftNode() == null && root.getRightNode() == null) {
            return true;
        }
        if (root.getLeftNode() != null && root.getRightNode() != null) {
            return isFullBinaryTree(root.getLeftNode()) && isFullBinaryTree(root.getRightNode());
        }
        return false;
    }

    private static boolean isCompleteBinaryTree(Node root, int index, int nodesCount) {
        if (root == null) {
            return true;
        }
        if (index >= nodesCount) {
            return false;
        }
        return isCompleteBinaryTree(root.getLeftNode(), 2 * index + 1, nodesCount) &&
                isCompleteBinaryTree(root.getRightNode(), 2 * index + 2, nodesCount);
    }

    private static boolean isBalanced(Node root) {
        if (root == null) {
            return true;
        }
        int heightLeft = height(root.getLeftNode());
        int heightRight = height(root.getRightNode());
        return Math.abs(heightLeft - heightRight) <= 1 && isBalanced(root.getLeftNode()) && isBalanced(root.getRightNode());
    }

    public static void leftView(Node root) {
        Queue<Node> queue = new LinkedList<>();
        List<Integer> arr = new ArrayList<>();
        queue.add(root);
        queue.add(null);
        while (!queue.isEmpty()) {
            Node dequeuedNode = queue.remove();
            if (dequeuedNode != null) {
                arr.add(dequeuedNode.getData());
                if (dequeuedNode.getLeftNode() != null) {
                    queue.add(dequeuedNode.getLeftNode());
                }
                if (dequeuedNode.getRightNode() != null) {
                    queue.add(dequeuedNode.getRightNode());
                }
            } else {
                if (!queue.isEmpty()) {
                    queue.add(null);
                    arr.add(null);
                }
            }
        }
        System.out.println(arr.get(0));
        for (int i = 1; i < arr.size(); i++) {
            Integer element = arr.get(i);
            if (element == null && i < arr.size() - 1) {
                System.out.println(arr.get(i + 1));
            }
        }
    }

    private static boolean searchInBST(Node root, int value) {
        if (root == null) {
            return false;
        } else if (root.getData() == value) {
            return true;
        } else if (searchInBST(root.getLeftNode(), value)) {
            return true;
        } else {
            return searchInBST(root.getRightNode(), value);
        }
    }

    private static int sizeOfBST(Node root) {
        if (root == null) {
            return 0;
        }
        return sizeOfBST(root.getLeftNode()) + sizeOfBST(root.getRightNode()) + 1;
    }

    private static void rootToLeaf(Node root, int[] arr, int index) {
        if (root == null) {
            return;
        }
        arr[index] = root.getData();
        index++;
        if (root.getLeftNode() == null && root.getRightNode() == null) {
            IntStream intStream = Arrays.stream(arr).filter(x -> x > 0);
            Stream.of(intStream).map(x -> Arrays.toString(x.toArray()) + " ").forEach(System.out::println);
        } else {
            rootToLeaf(root.getLeftNode(), arr, index);
            rootToLeaf(root.getRightNode(), arr, index);
        }
    }

    public static void rootToLeafTrigger(Node root) {
        int[] arr = new int[256];
        rootToLeaf(root, arr, 0);
    }

    private static boolean pathSumCheck(Node root, int sum) {
        if (root == null) {
            return sum == 0;
        }
        int subtractedSum = sum - root.getData();
        return pathSumCheck(root.getLeftNode(), subtractedSum) || pathSumCheck(root.getRightNode(), subtractedSum);
    }

    private static Node mirrorTree(Node root) {
        if (root != null) {
            mirrorTree(root.getLeftNode());
            mirrorTree(root.getRightNode());
            Node temp = root.getRightNode();
            root.setRightNode(root.getLeftNode());
            root.setLeftNode(temp);
        }
        return root;
    }

    private static boolean areMirrorTrees(Node root1, Node root2) {
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null || root2 == null) {
            return false;
        }
        if (root1.getData() != root2.getData()) {
            return false;
        }
        return (areMirrorTrees(root1.getLeftNode(), root2.getRightNode()) && areMirrorTrees(root1.getRightNode(), root2.getLeftNode()));
    }

    private static Node buildBtPreOrderInOrder(int[] preOrder, int preStart, int preEnd, int[] inOrder, int inStart, int inEnd) {
        if (preStart >= preEnd || inStart >= inEnd) {
            return null;
        }
        int data = preOrder[preStart];
        Node root = new Node(data);
        int offSet = 0;
        for (; offSet < inEnd; offSet++) {
            if (inOrder[offSet] == data) {
                break;
            }
        }
        buildBtPreOrderInOrder(preOrder, preStart + 1, preStart + offSet - inStart, inOrder, inStart, offSet - 1);
        buildBtPreOrderInOrder(preOrder, preStart + offSet - inStart + 1, preEnd, inOrder, offSet + 1, inEnd);
        return root;
    }

    private static Node buildBtPostOrderInOrder(int[] postOrder, int postStart, int postEnd, int[] inOrder, int inStart, int inEnd) {
        if (postStart >= postEnd || inStart >= inEnd) {
            return null;
        }
        int data = postOrder[postEnd];
        Node root = new Node(data);
        int offSet = 0;
        for (; offSet < inEnd; offSet++) {
            if (inOrder[offSet] == data) {
                break;
            }
        }
        buildBtPostOrderInOrder(postOrder, postStart, postStart + offSet - inStart - 1, inOrder, inStart, offSet - 1);
        buildBtPostOrderInOrder(postOrder, postStart + offSet - inStart, postEnd - 1, inOrder, offSet + 1, inEnd);
        return root;
    }

    public static Node buildBt(int[] preOrder, int[] inOrder) {
        if (preOrder.length == 0 || inOrder.length == 0) {
            return null;
        }
        return buildBtPreOrderInOrder(preOrder, 0, preOrder.length - 1, inOrder, 0, inOrder.length - 1);
    }

    private static boolean printAllAncestors(Node root, Node child) {
        if (root == null) {
            return false;
        }
        if (root == child || root.getLeftNode() == child || root.getRightNode() == child ||
                printAllAncestors(root.getLeftNode(), child) || printAllAncestors(root.getRightNode(), child)) {
            System.out.println(root.getData());
            return true;
        }
        return false;
    }

    private static Node getMax(Node root) {
        if (root.getRightNode() == null) {
            return root;
        }
        return getMax(root.getRightNode());
    }

    private static Node getMin(Node root) {
        if (root.getLeftNode() == null) {
            return root;
        }
        return getMin(root.getRightNode());
    }

    private static Node deleteMax(Node root) {
        if (root.getRightNode() == null) {
            return null;
        }
        return root.setRightNode(deleteMax(root.getRightNode()));
    }

    private static Node deleteNode(Node root, int value) {
        if (root == null) {
            return null;
        } else if (root.getData() > value) {
            deleteNode(root.getLeftNode(), value);
        } else if (root.getData() < value) {
            deleteNode(root.getRightNode(), value);
        } else if (root.getData() == value) {
            if (root.getLeftNode() == null && root.getRightNode() == null) {
                return null;
            } else if (root.getLeftNode() == null && root.getRightNode() != null) {
                return root.getRightNode();
            } else if (root.getLeftNode() != null && root.getRightNode() == null) {
                return root.getLeftNode();
            } else if (root.getLeftNode() != null && root.getRightNode() != null) {
                int leftMax = getMax(root.getLeftNode()).getData();
                Node updatedNode = new Node(leftMax);
                Node leftNode = deleteMax(root.getLeftNode());
                updatedNode.setRightNode(root.getRightNode());
                updatedNode.setLeftNode(leftNode);
                return updatedNode;
            }
        }
        return root;
    }

    private static Node LCA(Node root, int a, int b) {
        if (root == null) {
            return null;
        }
        if (root.getData() == a || root.getData() == b) {
            return root;
        } else if (Math.max(a, b) < root.getData()) {
            LCA(root.getLeftNode(), a, b);
        } else if (Math.max(a, b) > root.getData()) {
            LCA(root.getRightNode(), a, b);
        }
        return root;
    }

    private static boolean isBST(Node root) {
        if (root == null) {
            return true;
        }
        if (root.getLeftNode() != null && getMax(root.getLeftNode()).getData() > root.getData()) {
            return false;
        } else if (root.getRightNode() != null && getMin(root.getRightNode()).getData() < root.getData()) {
            return false;
        } else return isBST(root.getLeftNode()) && isBST(root.getRightNode());
    }

    private static boolean isBSTv1(Node root, int max, int min) {
        if (root == null) {
            return true;
        }
        return isBSTv1(root.getLeftNode(), root.getData(), min) &&
                isBSTv1(root.getRightNode(), max, root.getData()) &&
                min < root.getData() && root.getData() < max;
    }

    private static boolean iSBSTv2(Node root) {
        if (root == null) {
            return true;
        }
        if (!iSBSTv2(root.getLeftNode())) {
            return false;
        }
        if (!(root.getData() > CHECK_VALUE)) {
            return false;
        }
        CHECK_VALUE = root.getData();
        return iSBSTv2(root.getRightNode());
    }

    private static Node BSTtoDLL(Node root) {
        if (root == null) {
            return null;
        }
        Node left = BSTtoDLL(root.getLeftNode());
        Node right = BSTtoDLL(root.getRightNode());
        if (left == null && right == null) {
            return root;
        } else if (left != null && right == null) {
            left.setNext(root);
            root.setPrev(left);
            root.setLeftNode(null);
            root.setRightNode(null);
            return root;
        } else if (left == null) {
            root.setNext(right);
            right.setPrev(root);
            root.setLeftNode(null);
            root.setRightNode(null);
            return right;
        }
        left.setNext(root);
        root.setPrev(left);
        root.setNext(right);
        right.setPrev(root);
        root.setLeftNode(null);
        root.setRightNode(null);
        return right;
    }

    public static Node DLLtoBST(Node head) {
        int length = 0;
        Node currentNode = head;
        while (currentNode != null) {
            currentNode = currentNode.getNext();
            length++;
        }
        return constructBST(head, 0, length - 1);
    }

    private static Node constructBST(Node head, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        Node leftNode = constructBST(head, start, mid - 1);
        Node root = new Node(head.getData());
        root.setLeftNode(leftNode);
        if (head.getNext() != null) {
            head.data = head.getNext().data;
            head = head.getNext().getNext();
        }
        Node rightNode = constructBST(head, mid + 1, end);
        root.setRightNode(rightNode);
        return root;
    }

    private static Node kThSmallest(Node root, int k, int count) {
        if (root == null) {
            return null;
        }
        Node leftNode = kThSmallest(root.getLeftNode(), k, count);
        if (leftNode != null) {
            return leftNode;
        }
        if (++count == k) {
            return root;
        }
        return kThSmallest(root.getRightNode(), k, count);
    }

    private static int floorSearchBST(Node root, int value) {
        if (root == null) {
            return Integer.MAX_VALUE;
        }
        if (root.getData() == value) {
            return value;
        }
        if (root.getData() > value) {
            return floorSearchBST(root.getLeftNode(), value);
        }
        int rightFloorValue = floorSearchBST(root.getRightNode(), value);
        return (rightFloorValue <= value) ? rightFloorValue : root.getData();
    }

    private static int ceilSearchBST(Node root, int value) {
        if (root == null) {
            return Integer.MIN_VALUE;
        }
        if (root.getData() == value) {
            return value;
        }
        if (root.getData() < value) {
            return ceilSearchBST(root.getRightNode(), value);
        }
        int leftCeilValue = ceilSearchBST(root.getLeftNode(), value);
        return (leftCeilValue <= value) ? root.getData() : leftCeilValue;
    }

    private static Node leftRotation(Node root) {
        Node temp = root.getLeftNode();
        if (temp != null) {
            root.setLeftNode(temp.getRightNode());
            temp.setRightNode(root);
            root.setNodeHeight(Math.max(height(root.getLeftNode()), height(root.getRightNode())));
            temp.setNodeHeight(Math.max(height(temp.getLeftNode()), height(temp.getRightNode())));
        }
        return temp;
    }

    private static Node rightRotation(Node root) {
        Node temp = root.getRightNode();
        if (temp != null) {
            root.setRightNode(temp.getLeftNode());
            temp.setLeftNode(root);
            root.setNodeHeight(Math.max(height(root.getLeftNode()), height(root.getRightNode())));
            temp.setNodeHeight(Math.max(height(temp.getLeftNode()), height(temp.getRightNode())));
        }
        return temp;
    }

    private static int getAVLBalance(Node root) {
        if (root == null) {
            return 0;
        }
        return height(root.getLeftNode()) - height(root.getRightNode());
    }

    private static Node insertInAVL(Node root, int data) {
        if (root == null) {
            return new Node(data);
        }
        if (root.getData() > data) {
            root.setLeftNode(insertInAVL(root.getLeftNode(), data));
        } else if (root.getData() < data) {
            root.setRightNode(insertInAVL(root.getRightNode(), data));
        } else {
            return root;
        }
        int balance = getAVLBalance(root);
        if (balance > 1 && data < root.getLeftNode().getData()) {
            return rightRotation(root);
        } else if (balance < -1 && data > root.getRightNode().getData()) {
            leftRotation(root);
        } else if (balance > 1 && data > root.getLeftNode().getData()) {
            root.setLeftNode(leftRotation(root.getLeftNode()));
            return rightRotation(root);
        } else if (balance < -1 && data < root.getRightNode().getData()) {
            root.setRightNode(rightRotation(root.getRightNode()));
            return leftRotation(root);
        }
        return root;
    }

    private Node constructBSTfromArray(int[] arr, int start, int end) {
        if (arr.length == 0) {
            return null;
        }
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        Node root = new Node(arr[mid]);
        Node leftNode = constructBSTfromArray(arr, start, mid - 1);
        Node rightNode = constructBSTfromArray(arr, mid + 1, end);
        root.setLeftNode(leftNode);
        root.setRightNode(rightNode);
        return root;
    }

    static class Node {
        int data;
        Node leftNode, rightNode;
        Node prev, next;
        int nodeHeight;

        Node(int data) {
            this.data = data;
        }

        public int getData() {
            return this.data;
        }

        Node getLeftNode() {
            return this.leftNode;
        }

        Node setLeftNode(Node node) {
            this.leftNode = node;
            return this;
        }

        Node getRightNode() {
            return this.rightNode;
        }

        Node setRightNode(Node node) {
            this.rightNode = node;
            return this;
        }

        public Node getPrev() {
            return this.prev;
        }

        Node setPrev(Node node) {
            this.prev = node;
            return this;
        }

        Node getNext() {
            return this.next;
        }

        Node setNext(Node node) {
            this.next = node;
            return this;
        }

        public int getNodeHeight() {
            return nodeHeight;
        }

        void setNodeHeight(int nodeHeight) {
            this.nodeHeight = nodeHeight;
        }
    }
}