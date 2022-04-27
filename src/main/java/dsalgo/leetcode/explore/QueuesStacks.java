package dsalgo.leetcode.explore;

import org.apache.spark.sql.sources.In;

import java.util.*;
import java.util.Arrays;

public class QueuesStacks {

    public static void main(String[] args) {
        char c = '5';
        System.out.println((c -'0') +'0');
    }

    static class MyCircularQueue {

        int [] arr;
        int head;
        int tail;
        int k;
        int n;

        public MyCircularQueue(int k) {
            arr = new int[k];
            this.k = k;
            head = 0;
            tail = -1;
        }

        public boolean enQueue(int value) {
            if (n >= k) return false;
            tail = (tail + 1) % k;
            arr[tail] = value;
            n++;
            return true;
        }

        public boolean deQueue() {
            if (n <= 0) return false;
            head = (head + 1) % k;
            n--;
            return true;
        }

        public int Front() {
            if (n <= 0) return -1;
            return arr[head];
        }

        public int Rear() {
            if (n <= 0) return -1;
            return arr[tail];
        }

        public boolean isEmpty() {
            return n == 0;
        }

        public boolean isFull() {
            return n == k;
        }
    }

//    https://leetcode.com/explore/learn/card/queue-stack/228/first-in-first-out-data-structure/1368/

    static class MovingAverage {

        Queue<Integer> queue;
        int n;
        double runningSum;

        public MovingAverage(int size) {
            queue = new LinkedList<>();
            n = size;
            runningSum = 0;
        }

        public double next(int val) {
            queue.offer(val);
            runningSum += val;
            if (!queue.isEmpty() && queue.size() > n) {
                int rem = queue.poll();
                runningSum -= rem;
            }
            return runningSum / queue.size();
        }
    }

//    https://leetcode.com/explore/learn/card/queue-stack/231/practical-application-queue/1373/

    public void wallsAndGates(int[][] rooms) {
        int [][] directions = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
        Deque<int []> deque = new ArrayDeque<>();

        for (int i = 0; i < rooms.length; i++) {
            for (int j = 0; j < rooms[0].length; j++) {
                if (rooms[i][j] == 0) deque.add(new int[]{i,j});
            }
        }

        while (!deque.isEmpty()) {
            int n = deque.size();
            for (int i = 0; i < n; i++) {
                int [] node = deque.remove();
                int x = node[0];
                int y = node[1];
                int v = rooms[x][y] + 1;
                for (int [] direction : directions) {
                    int xD = x + direction[0];
                    int yD = y + direction[1];
                    if (xD < 0 || xD >= rooms.length || yD < 0 || yD >= rooms[0].length || rooms[xD][yD] <= v) continue;
                    rooms[xD][yD] = v;
                    deque.add(new int[] {xD,yD});
                }
            }
        }
    }

//    https://leetcode.com/explore/learn/card/queue-stack/231/practical-application-queue/1374/

    void dfsIslands(char [][] grid, int i, int j, int [][] directions) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != '1') return;
        grid[i][j] = '0';
        for (int [] direction : directions) {
            int x = i + direction[0];
            int y = j + direction[1];
            dfsIslands(grid, x, y, directions);
        }
    }

    public int numIslands(char[][] grid) {
        int [][] directions = new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfsIslands(grid, i, j, directions);
                    count++;
                }
            }
        }
        return count;
    }

//    https://leetcode.com/explore/learn/card/queue-stack/231/practical-application-queue/1375/

    public int openLock(String[] deadends, String target) {
        Set<String> set = new HashSet<>(Arrays.asList(deadends));
        Deque<String> deque = new ArrayDeque<>();
        deque.add("0000");
        int d = 0;
        while (!deque.isEmpty()) {
            int n = deque.size();
            for (int i = 0; i < n; i++) {
                String s = deque.remove();
                if (s.equals(target)) return d;
                if (set.contains(s)) continue;
                set.add(s);

                char [] sArr = s.toCharArray();
                for (int j = 0; j < 4; j++) {

                    char v = sArr[j];
                    sArr[j] = (char)(((sArr[j] - '0') + 1 + 10) % 10 + '0');
                    String p = String.valueOf(sArr);
                    if (!set.contains(p)) deque.add(p);

                    sArr[j] = v;
                    sArr[j] = (char)(((sArr[j] - '0') - 1 + 10) % 10 + '0');
                    String m = String.valueOf(sArr);
                    if (!set.contains(m)) deque.add(m);

                    sArr[j] = v;
                }
            }
            d++;
        }
        return -1;
    }

//    https://leetcode.com/explore/learn/card/queue-stack/231/practical-application-queue/1371/

    public int numSquares(int n) {
        int sqrt = (int) Math.sqrt(n);

        Set<Integer> visited = new HashSet<>();
        Deque<Integer> deque = new ArrayDeque<>();
        deque.add(n);

        int d = 0;
        while (!deque.isEmpty()) {
            int size = deque.size();

            for (int i = 0; i < size; i++) {
                int node = deque.remove();
                if (node == 0) return d;
                if (visited.contains(node) || node < 0) continue;
                visited.add(node);
                for (int j = 1; j <= sqrt; j++) {
                    int newNo = node - (j*j);
                    if (!visited.contains(newNo)) deque.add(newNo);
                }
            }

            d++;
        }
        return d;
    }

//    https://leetcode.com/explore/learn/card/queue-stack/230/usage-stack/1360/

    static class MinStack {

        Deque<Integer> stack;
        Deque<Integer> minStack;

        public MinStack() {
            stack = new ArrayDeque<>();
            minStack = new ArrayDeque<>();
        }

        public void push(int val) {
            stack.push(val);
            if (!minStack.isEmpty() && minStack.peek() < val) minStack.push(minStack.peek());
            else minStack.push(val);
        }

        public void pop() {
            if (!stack.isEmpty()) stack.pop();
            if (!minStack.isEmpty()) minStack.pop();
        }

        public int top() {
            if (!stack.isEmpty()) return stack.peek();
            return -1;
        }

        public int getMin() {
            if (!minStack.isEmpty()) return minStack.peek();
            return -1;
        }
    }

//    https://leetcode.com/explore/learn/card/queue-stack/230/usage-stack/1361/

    public boolean isValid(String s) {
        Deque<Character> stack = new ArrayDeque<>();
        for (char c : s.toCharArray()) {
            if (c == '[' || c == '{' || c == '(') stack.push(c);
            else {
                if (stack.isEmpty()) return false;
                if (c == ']' && stack.peek() == '[') stack.pop();
                else if (c == '}' && stack.peek() == '{') stack.pop();
                else if (c == ')' && stack.peek() == '(') stack.pop();
                else return false;
            }
        }
        return stack.size() == 0;
    }

//    https://leetcode.com/explore/learn/card/queue-stack/230/usage-stack/1363/

    public int[] dailyTemperatures(int[] temperatures) {
        Deque<Integer> stack = new ArrayDeque<>();
        int n = temperatures.length;
        int [] res = new int[n];
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && temperatures[stack.peek()] <= temperatures[i]) stack.pop();
            res[i] = stack.isEmpty() ? res[i] = 0 : stack.peek() - i;
            stack.push(i);
        }
        return res;
    }

//    https://leetcode.com/explore/learn/card/queue-stack/230/usage-stack/1394/

    public int evalRPN(String[] tokens) {
        Deque<Integer> stack = new ArrayDeque<>();
        for (String s : tokens) {
            switch (s) {
                case "+":
                    if (!stack.isEmpty()) {
                        int a = stack.pop();
                        int b = stack.pop();
                        stack.push(b + a);
                    }
                    break;
                case "-":
                    if (!stack.isEmpty()) {
                        int a = stack.pop();
                        int b = stack.pop();
                        stack.push(b - a);
                    }
                    break;
                case "*":
                    if (!stack.isEmpty()) {
                        int a = stack.pop();
                        int b = stack.pop();
                        stack.push(b * a);
                    }
                    break;
                case "/":
                    if (!stack.isEmpty()) {
                        int a = stack.pop();
                        int b = stack.pop();
                        stack.push(b / a);
                    }
                    break;
                default:
                    stack.push(Integer.parseInt(s));
            }
        }
        return !stack.isEmpty() ? stack.pop() : Integer.MIN_VALUE;
    }

//    https://leetcode.com/explore/learn/card/queue-stack/232/practical-application-stack/1392/

    static class Node {
        public int val;
        public List<Node> neighbors;
        public Node() {
            val = 0;
            neighbors = new ArrayList<Node>();
        }
        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<Node>();
        }
        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    }

    Node cloneGraphUtils(Node node, Map<Node,Node> visited) {
        if (node == null) return null;
        if (visited.containsKey(node)) return visited.get(node);

        Node n = new Node(node.val);
        visited.put(node,n);

        for (Node neighbour : node.neighbors) {
            Node cloneN = cloneGraphUtils(neighbour, visited);
            n.neighbors.add(cloneN);
        }
        return n;
    }

    public Node cloneGraph(Node node) {
        Map<Node,Node> visited = new HashMap<>();
        return cloneGraphUtils(node, visited);
    }

//    https://leetcode.com/explore/learn/card/queue-stack/232/practical-application-stack/1389/

    int findTargetSumWaysUtil(int [] nums, int target, int sum, int index, int [][] dp, int total) {
        if (index == nums.length) return target == sum ? 1 : 0;
        if (dp[index][total + sum] != Integer.MIN_VALUE) return dp[index][total + sum];
        int add = findTargetSumWaysUtil(nums, target, sum + nums[index], index + 1, dp, total);
        int sub = findTargetSumWaysUtil(nums, target, sum - nums[index], index + 1, dp, total);
        dp[index][total + sum] = add + sub;
        return dp[index][total + sum];
    }

    public int findTargetSumWays(int[] nums, int target) {
        int total = Arrays.stream(nums).sum();
        int [][] dp = new int[nums.length][2 * total + 1];
        for (int [] ints : dp) Arrays.fill(ints, Integer.MIN_VALUE);
        return findTargetSumWaysUtil(nums, target, 0, 0, dp, total);
    }

//    https://leetcode.com/explore/learn/card/queue-stack/239/conclusion/1386/

    static class MyQueue {

        Deque<Integer> s1;
        Deque<Integer> s2;

        public MyQueue() {
            s1 = new ArrayDeque<>();
            s2 = new ArrayDeque<>();
        }

        public void push(int x) {
            s1.push(x);
        }

        public int pop() {
            if (!s2.isEmpty()) return s2.pop();
            while (!s1.isEmpty()) s2.push(s1.pop());
            return !s2.isEmpty() ? s2.pop() : Integer.MIN_VALUE;
        }

        public int peek() {
            if (!s2.isEmpty()) return s2.peek();
            while (!s1.isEmpty()) s2.push(s1.pop());
            return !s2.isEmpty() ? s2.peek() : Integer.MIN_VALUE;
        }

        public boolean empty() {
            return s1.isEmpty() && s2.isEmpty();
        }
    }

//    https://leetcode.com/explore/learn/card/queue-stack/239/conclusion/1387/

    static class MyStack {

        Deque<Integer> queue;

        public MyStack() {
            queue = new ArrayDeque<>();
        }

        public void push(int x) {
            queue.add(x);
            int n = queue.size();
            while (n > 1) {
                queue.add(queue.remove());
                n--;
            }
        }

        public int pop() {
            return queue.isEmpty() ? Integer.MIN_VALUE : queue.remove();
        }

        public int top() {
            return queue.isEmpty() ? Integer.MIN_VALUE : queue.peek();
        }

        public boolean empty() {
            return queue.isEmpty();
        }
    }

//    https://leetcode.com/explore/learn/card/queue-stack/239/conclusion/1379/

    String decodeStringUtils(char [] arr, int [] i) {
        StringBuilder sb = new StringBuilder();

        while (i[0] < arr.length) {
            int n = 0;
            while (i[0] < arr.length && Character.isDigit(arr[i[0]])) {
                n = n * 10 + (arr[i[0]] - '0');
                i[0]++;
            }

            if (arr[i[0]] == '[') {
                i[0]++;
                String r = decodeStringUtils(arr, i);
                for (int j = 0; j < n; j++) sb.append(r);
            } else if (arr[i[0]] == ']') {
                i[0]++;
                return sb.toString();
            } else {
                sb.append(arr[i[0]]);
                i[0]++;
            }
        }

        return sb.toString();
    }


    public String decodeString(String s) {
        int [] i = new int[]{0};
        return decodeStringUtils(s.toCharArray(), i);
    }

//    https://leetcode.com/explore/learn/card/queue-stack/239/conclusion/1393/

    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int [][] directions = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
        boolean [][] visited = new boolean[image.length][image[0].length];
        Deque<int []> deque = new ArrayDeque<>();
        int oldColor = image[sr][sc];
        deque.add(new int[]{sr,sc});

        while (!deque.isEmpty()) {
            int [] node = deque.remove();
            int x = node[0];
            int y = node[1];
            image[x][y] = newColor;
            visited[x][y] = true;
            for (int [] direction : directions) {
                int r = x + direction[0];
                int c = y + direction[1];
                if (r < 0 || r >= image.length || c < 0 || c >= image[0].length
                         || image[r][c] != oldColor || visited[r][c]) continue;
                deque.add(new int[]{r,c});
            }
        }
        return image;
    }

//    https://leetcode.com/explore/learn/card/queue-stack/239/conclusion/1388/

    public int[][] updateMatrix(int[][] mat) {
        int [][] directions = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
        Deque<int []> deque = new ArrayDeque<>();
        boolean [][] visited = new boolean[mat.length][mat[0].length];

        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[0].length; j++) {
                if (mat[i][j] == 0) deque.add(new int[]{i,j});
            }
        }

        int d = 0;
        while (!deque.isEmpty()) {
            int n = deque.size();
            for (int i = 0; i < n; i++) {
                int [] node = deque.remove();
                int x = node[0];
                int y = node[1];
                if (visited[x][y]) continue;
                mat[x][y] = d;
                visited[x][y] = true;
                for (int [] direction : directions) {
                    int a = x + direction[0];
                    int b = y + direction[1];
                    if (a < 0 || a >= mat.length || b < 0 || b >= mat[0].length || visited[a][b]) continue;
                    deque.add(new int[]{a,b});
                }
            }
            d++;
        }
        return mat;
    }

//    https://leetcode.com/explore/learn/card/queue-stack/239/conclusion/1391/

    void dfsVisit(int i, List<List<Integer>> rooms, boolean [] count, int [] res) {
        if (count[i]) return;
        count[i] = true;
        res[0]++;
        for (int keys : rooms.get(i)) dfsVisit(keys, rooms, count, res);
    }

    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        boolean [] count = new boolean[rooms.size()];
        int [] res = new int[]{0};
        for (int i = 0; i < count.length; i++) {
            dfsVisit(i, rooms, count, res);
        }
        return res[0] == count.length;
    }

}
