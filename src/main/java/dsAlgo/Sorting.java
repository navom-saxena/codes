package dsAlgo;

import java.util.Arrays;

public class Sorting {
    public static void main(String[] args) {
//        int[] arr = new int[]{274, 204, -161, 481, -606, -767, -351};
        int[] arr = new int[]{4, 6, 1, 9, 2, 10, 5};
//        bubbleSort(arr);
//        selectionSortReverseLogic(arr);
//        insertionSort(arr);
//        merge2ArraysAndSort(new int[] {2,3,6,8}, new int[]{1,4,5,9},new int[] {5,9,3,2,8,6,1,4});
//        mergeSort(new int[] {2,4,1,6,8,5,3,7});
//        quickSort(arr, 0, arr.length-1);
//        System.out.println(mergeSortAndInversionCount(new int[]{15,35,10,15,15,15}));
        countSort(arr);
    }

    public static void bubbleSort(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            boolean flag = false;
            for (int j = 0; j < arr.length - 1 - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                    flag = true;
                }
            }
            if (!flag) {
                break;
            }
            System.out.print(i + " -> ");
            for (int x = 0; x < arr.length; x++) {
                System.out.print(arr[x] + " ");
            }
            System.out.println();
        }
    }

    public static void selectionSortReverseLogic(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            int maximum = arr[0];
            int maximumIndex = 0;
            for (int j = 0; j < arr.length - i; j++) {
                if (maximum < arr[j]) {
                    maximum = arr[j];
                    maximumIndex = j;
                }
            }
            int temp = arr[arr.length - 1 - i];
            arr[arr.length - 1 - i] = maximum;
            arr[maximumIndex] = temp;
            for (int x = 0; x < arr.length; x++) {
                System.out.print(arr[x] + " ");
            }
            System.out.println();
        }
    }

    public static void selectionSort(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            int minimum = arr[i];
            int minimumIndex = i;
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j] < minimum) {
                    minimum = arr[j];
                    minimumIndex = j;
                }
            }
            int temp = arr[i];
            arr[i] = minimum;
            arr[minimumIndex] = temp;
            for (int x = 0; x < arr.length; x++) {
                System.out.print(arr[x] + " ");
            }
            System.out.println();
        }
    }

    public static void insertionSort(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            int j = i;
            int value = arr[i];
            int loop = 0;
            while (j > 0 && value < arr[j - 1]) {
                arr[j] = arr[j - 1];
//                System.out.println("inserted position - " + (int) (i));
                loop++;
//                System.out.println("loop " + loop);
                j--;
            }
            arr[j] = value;
            for (int x = 0; x < arr.length; x++) {
                System.out.print(arr[x] + " ");
            }
            System.out.println();
        }
    }

    public static void merge2ArraysAndSort(int[] left, int[] right, int[] main) {
        int leftPointer = 0;
        int rightPointer = 0;
        int mainArrPointer = 0;
        while (leftPointer < left.length && rightPointer < right.length) {
            if (left[leftPointer] < right[rightPointer]) {
                main[mainArrPointer] = left[leftPointer];
                leftPointer++;
                mainArrPointer++;
            } else {
                main[mainArrPointer] = right[rightPointer];
                rightPointer++;
                mainArrPointer++;
            }
        }
        while (leftPointer < left.length) {
            main[mainArrPointer] = left[leftPointer];
            leftPointer++;
            mainArrPointer++;
        }
        while (rightPointer < right.length) {
            main[mainArrPointer] = right[rightPointer];
            rightPointer++;
            mainArrPointer++;
        }
        for (int x = 0; x < main.length; x++) {
            System.out.print(main[x] + " ");
        }
        System.out.println();
    }

    public static void mergeSort(int[] arr) {
        if (arr.length < 2) {
            return;
        }
        int[] left = new int[arr.length / 2];
        int[] right = new int[arr.length - (arr.length / 2)];
        for (int i = 0; i < arr.length / 2; i++) {
            left[i] = arr[i];
        }
        for (int j = 0; j < arr.length - (arr.length / 2); j++) {
            right[j] = arr[arr.length / 2 + j];
        }
        mergeSort(left);
        mergeSort(right);
        merge2ArraysAndSort(left, right, arr);
    }

    public static long merge2ArraysAndSortAndInversionCount(int[] left, int[] right, int[] main) {
        int leftPointer = 0;
        int rightPointer = 0;
        int mainArrPointer = 0;
        long inversionCount = 0;
        int midPointer = left.length;
        while (leftPointer < left.length && rightPointer < right.length) {
            if (left[leftPointer] > right[rightPointer]) {
                inversionCount = inversionCount + midPointer - leftPointer;
            }
            if (left[leftPointer] <= right[rightPointer]) {
                main[mainArrPointer] = left[leftPointer];
                leftPointer++;
                mainArrPointer++;
            } else {
                main[mainArrPointer] = right[rightPointer];
                rightPointer++;
                mainArrPointer++;
            }
        }
        while (leftPointer < left.length) {
            main[mainArrPointer] = left[leftPointer];
            leftPointer++;
            mainArrPointer++;
        }
        while (rightPointer < right.length) {
            main[mainArrPointer] = right[rightPointer];
            rightPointer++;
            mainArrPointer++;
        }
//        for (int x = 0; x < main.length; x++) {
//            System.out.print(main[x] + " ");
//        }
//        System.out.println();
//        System.out.println("inversion - " + inversionCount);
        return inversionCount;
    }

    public static long mergeSortAndInversionCount(int[] arr) {
        if (arr.length < 2) {
            return 0;
        }

        int[] left = new int[arr.length / 2];
        int[] right = new int[arr.length - (arr.length / 2)];
        for (int i = 0; i < arr.length / 2; i++) {
            left[i] = arr[i];
        }
        for (int j = 0; j < arr.length - (arr.length / 2); j++) {
            right[j] = arr[arr.length / 2 + j];
        }
        long inversionC = mergeSortAndInversionCount(left);
        inversionC += mergeSortAndInversionCount(right);
        inversionC += merge2ArraysAndSortAndInversionCount(left, right, arr);
        return inversionC;
    }

    public static int pivotPartition(int[] arr, int start, int end) {
        int pivot = arr[end];
        int pivotIndex = start;
        for (int i = start; i < end; i++) {
            if (arr[i] <= pivot) {
                int temp = arr[pivotIndex];
                arr[pivotIndex] = arr[i];
                arr[i] = temp;
                pivotIndex++;
            }
        }
        int temp2 = arr[pivotIndex];
        arr[pivotIndex] = pivot;
        arr[end] = temp2;

        for (int j = start; j <= end; j++) {
            System.out.print(arr[j] + " ");
        }
        System.out.println();

        return pivotIndex;
    }

    public static void quickSort(int[] arr, int start, int end) {
        if (start < end) {
            int pivotIndex = pivotPartition(arr, start, end);

            quickSort(arr, start, pivotIndex - 1);
            quickSort(arr, pivotIndex + 1, end);

            System.out.println("quicksort - ");
            for (int x = 0; x < arr.length; x++) {
                System.out.print(arr[x] + " ");
            }
            System.out.println();
        }
    }

    public static void countSort(int[] arr) {
        int[] countArray = new int[256];
        for (int j = 0; j < arr.length; j++) {
            countArray[arr[j]]++;
        }
        for (int k = 1; k < 255; k++) {
            countArray[k] += countArray[k - 1];
        }
        int[] sortedArray = new int[arr.length];
        for (int l = 0; l < arr.length; l++) {
            sortedArray[countArray[arr[l]] - 1] = arr[l];
            countArray[arr[l]]--;
        }
        Arrays.stream(sortedArray).forEach(x -> System.out.print(x + " "));
    }
}
