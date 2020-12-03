package dsalgo.hackerrank.companyrounds;

import org.spark_project.jetty.util.ArrayQueue;

import java.util.*;

public class Trees {

    static Node head;
    static Node prev;

    static class Node {
        Node left;
        Node right;
        Node previous;
        Node next;
        int data;

        Node(int data) {
            this.data = data;
        }
    }


    static class Person {
    }

    static class Student extends Person {
    }

    public static void main(String[] args) {

        List<? super Person> t = new ArrayList<>();
        t.add(new Person());
        System.out.println(t);

        System.out.println("---------------- " + lengthOfLIS(new int[]{7,1,2,4,6,5,3,8,9,10}));

//        int [] arr = new int[]{-2, -3, 4, -1, -2, 1, 5, -3};
//        System.out.println(subArraySum(arr));
//        int [] arr1 = new int[]{6,5,4,3,2,1,0};
//        System.out.println(binarySearch(arr1, 0, arr1.length - 1, 1));
//        Tree root = buildBST(arr1, 0, arr1.length - 1);
//        printPreOrderTree(root);
//        printLevelOrder(root);

        Node rootTree = new Node(10);
        rootTree.left = new Node(12);
        rootTree.left.left = new Node(25);
        rootTree.left.right = new Node(30);
        rootTree.right = new Node(15);
        rootTree.right.left = new Node(36);
        printPreOrderTree(rootTree);
        System.out.println();
        printLevelOrder(rootTree);

        createCDLLFromTree(rootTree);
        Node curr = head;
        while (curr != null) {
            System.out.println(curr.data);
            curr = curr.next;
        }
    }

    static int subArraySum(int [] arr) {
        int maxSum = Integer.MIN_VALUE;
        List<Integer> currentSumElements = new ArrayList<>();
        List<Integer> maxSumElements = new ArrayList<>();
        int start = 0;
        int end = 0;
        int currentMax = 0;
        for (int i = 0; i < arr.length; i++) {
            int num = arr[i];
            if (currentMax < 0) {
                currentMax = num;
                currentSumElements.clear();
                currentSumElements.add(num);
                start = i;
            } else {
                currentMax += num;
                currentSumElements.add(num);
            }
            if (currentMax > maxSum) {
                maxSum = currentMax;
                maxSumElements.clear();
                maxSumElements.addAll(currentSumElements);
                end = i;
            }
        }
        System.out.println(maxSumElements);
        System.out.println(start + " " + end);
        return maxSum;
    }

    static int binarySearch(int [] arr, int low, int high, int x) {
        if (x > arr[low]) {
            return -1;
        } else if (x < arr[high]) {
            return -1;
        }
        int mid = low + (high - low) / 2;
        if (arr[mid] == x) {
            return mid;
        } else if (arr[mid] > x) {
            return binarySearch(arr, mid + 1, high, x);
        } else {
            return binarySearch(arr, low, mid - 1, x);
        }
    }

    static Node buildBST(int [] arr, int low, int high) {
        if (low > high) {
            return null;
        }
        int mid = low + (high - low)/2;
        Node current = new Node(arr[mid]);
        current.left = buildBST(arr, mid + 1, high);
        current.right = buildBST(arr, low, mid - 1);
        return current;
    }

    static void printPreOrderTree(Node current) {
        if (current == null) {
            return;
        }
        System.out.println(current.data);
        printPreOrderTree(current.left);
        printPreOrderTree(current.right);
    }

    static void printLevelOrder(Node root) {
        Queue<Node> q = new ArrayQueue<>();
        q.add(root);
        q.add(null);
        List<Integer> arr = new ArrayList<>();
        while (!q.isEmpty()) {
            Node current = q.remove();
            if (current != null) {
                arr.add(current.data);
                if (current.left != null) {
                    q.add(current.left);
                }
                if (current.right != null) {
                    q.add(current.right);
                }
            } else {
                System.out.println(arr);
                arr.clear();
                if (!q.isEmpty()) {
                    q.add(null);
                }
            }
        }
    }

    static Node createCDLLFromTree(Node curr) {
        if (curr == null) {
            return null;
        }
        Node left = createCDLLFromTree(curr.left);
        if (left == null && head == null) {
            head = curr;
        }
        if (left == null && prev == null) {
            prev = curr;
        } else if (prev != null) {
            prev.next = curr;
            curr.previous = prev;
            prev = prev.next;
        }
        return createCDLLFromTree(curr.right);
    }

    public static int lengthOfLIS(int[] nums) {
        if(nums == null || nums.length == 0) { return 0; }
        int n = nums.length;

        Integer[] lis = new Integer[n];

//    /* Initialize LIS values for all indexes
    for ( int i = 0; i < n; i++ ) {
        lis[i] = 1;
    }

//    /* Compute optimized LIS values in bottom up manner
    for (int i = 1; i < n; i++ ) {
        for ( int j = 0; j < i; j++ )  {
            if ( nums[i] > nums[j] && lis[i] < lis[j] + 1) {
                lis[i] = lis[j] + 1;
            }
        }
    }
    return Collections.max(Arrays.asList(lis));

    }

}
