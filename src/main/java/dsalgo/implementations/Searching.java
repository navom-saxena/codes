package dsalgo.implementations;

public class Searching {
    public static void main(String[] args) {
        System.out.println(binarySearchRecursive(new int[]{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}, 20, 0, 9));
        System.out.println(binarySearchIterative(new int[]{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}, 20));
    }

    public static int binarySearchRecursive(int[] arr, int k, int low, int high) {
        if (low > high) return -1;
        int middle = low + ((high - low) / 2);
        if (arr[middle] == k) {
            return middle;
        } else if (arr[middle] > k) {
            return binarySearchRecursive(arr, k, low, middle - 1);
        } else {
            return binarySearchRecursive(arr, k, middle + 1, high);
        }
    }

    public static int binarySearchIterative(int[] arr, int k) {
        int low = 0;
        int high = arr.length - 1;
        int medium = Integer.MIN_VALUE;
        while (low <= high) {
            medium = low + ((high - low) / 2);
            if (arr[medium] == k) {
                return medium;
            } else if (arr[medium] > k) {
                high = medium - 1;
            } else {
                low = medium + 1;
            }
        }
        return -1;
    }
}
