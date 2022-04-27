package dsalgo.leetcode.explore;

import java.util.*;

public class Arrays {

    public static void main(String[] args) {
        int y = 0x1010;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/521/introduction/3238/

    public int findMaxConsecutiveOnes(int[] nums) {
        int maxC = 0;
        int c = 0;

        for (int no : nums) {
            if (no == 1) c++;
            else c = 0;

            maxC = Math.max(maxC, c);
        }

        return maxC;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/521/introduction/3237/

    public int findNumbers(int[] nums) {
        int c = 0;

        for (int no : nums) {
            int digitsNo = 0;

            while (no > 0) {
                no /= 10;
                digitsNo++;
            }

            if (digitsNo % 2 == 0) c++;
        }

        return c;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/521/introduction/3240/

    public int[] sortedSquares(int[] nums) {
        int n = nums.length;
        int [] sqArr = new int[n];

        int low = 0;
        int high = n - 1;
        int smallestP = n - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] >= 0) {
                smallestP = mid;
                high = mid - 1;
            } else low = mid + 1;
        }

        int left = smallestP - 1;
        int right = smallestP;

        int k = 0;
        while (left >= 0 || right < n) {
            int no;

            if (left >= 0 && right < n) {
                if (Math.abs(nums[left]) <= Math.abs(nums[right])) {
                    no = nums[left];
                    left--;
                } else {
                    no = nums[right];
                    right++;
                }
            } else if (left >= 0) {
                no = nums[left];
                left--;
            } else {
                no = nums[right];
                right++;
            }

            sqArr[k] = no * no;
            k++;
        }

        return sqArr;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/525/inserting-items-into-an-array/3245/

    public void duplicateZeros(int[] arr) {
        int n = arr.length;
        int zerosCount = 0;
        for (int no : arr) if (no == 0) zerosCount++;

        for (int i = n - 1; i >= 0; i--) {
            if (i + zerosCount < n) arr[i + zerosCount] = arr[i];

            if (arr[i] == 0) {
                zerosCount--;
                if (i + zerosCount < n) arr[i + zerosCount] = arr[i];
            }
        }
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/525/inserting-items-into-an-array/3253/

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;

        int k = nums1.length - 1;
        while (i >= 0 && j >= 0) {
            if (nums1[i] >= nums2[j]) {
                nums1[k] = nums1[i];
                i--;
            } else {
                nums1[k] = nums2[j];
                j--;
            }
            k--;
        }

        while (j >= 0) {
            nums1[k] = nums2[j];
            j--;
            k--;
        }
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/526/deleting-items-from-an-array/3247/

    public int removeElement(int[] nums, int val) {
        int n = nums.length;
        int k = 0;

        for (int i = 0; i < n; i++) {
            if (nums[i] != val) {
                nums[k] = nums[i];
                k++;
            }
        }

        return k;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/526/deleting-items-from-an-array/3248/

    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        int k = 1;

        for (int i = 1; i < n; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[k] = nums[i];
                k++;
            }
        }

        return k;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/527/searching-for-items-in-an-array/3250/

    public boolean checkIfExist(int[] arr) {
        Set<Integer> set = new HashSet<>();

        for (int v : arr) {
            if ((v % 2 == 0 && set.contains(v / 2)) || set.contains(v * 2)) {
                return true;
            }
            set.add(v);
        }

        return false;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/527/searching-for-items-in-an-array/3251/

    public boolean validMountainArray(int[] arr) {
        int n = arr.length;
        int i = 0;

        while (i < n - 1 && arr[i] < arr[i + 1]) i++;

        if (i == 0 || i == n - 1) return false;

        while (i < n - 1 && arr[i] > arr[i + 1]) i++;

        return i == n - 1;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/511/in-place-operations/3259/

    public int[] replaceElements(int[] arr) {
        int n = arr.length;
        int maxFromR = -1;

        for (int i = n - 1; i>= 0; i--) {
            int temp = arr[i];
            arr[i] = maxFromR;
            maxFromR = Math.max(maxFromR, temp);
        }

        return arr;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/511/in-place-operations/3157/

    public void moveZeroes(int[] nums) {
        int zeroPointer = 0;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[zeroPointer] = nums[i];
                zeroPointer++;
            }
        }
        for (int i = zeroPointer; i < nums.length; i++) {
            nums[i] = 0;
        }
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/511/in-place-operations/3260/

    public int[] sortArrayByParity(int[] nums) {
        int n = nums.length;
        int evenPointer = 0;

        for (int i = 0; i < n; i++) {
            if (nums[i] % 2 == 0) {
                int temp = nums[evenPointer];
                nums[evenPointer] = nums[i];
                nums[i] = temp;

                evenPointer++;
            }
        }

        return nums;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/523/conclusion/3228/

    public int heightChecker(int[] heights) {
        int [] freqArr = new int[101];

        for (int height : heights) {
            freqArr[height]++;
        }

        int currOrderedH = 0;
        int mismatched = 0;

        for (int height : heights) {
            while (freqArr[currOrderedH] == 0) currOrderedH++;

            if (currOrderedH != height) mismatched++;
            freqArr[currOrderedH]--;
        }

        return mismatched;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/523/conclusion/3230/

    public int findMaxConsecutiveOnesTwo(int[] nums) {
        int n = nums.length;
        int zerosC = 0;
        int j = 0;
        int maxOnes = 0;

        for (int i = 0; i < n; i++) {
            zerosC += nums[i] == 0 ? 1 : 0;

            while (j < i && zerosC > 1) {
                zerosC -= nums[j] == 0 ? 1 : 0;
                j++;
            }

            maxOnes = Math.max(i - j + 1, maxOnes);
        }

        return maxOnes;
    }

//    https://leetcode.com/explore/learn/card/fun-with-arrays/523/conclusion/3270/

    public List<Integer> findDisappearedNumbers(int[] nums) {
        int n = nums.length;
        List<Integer> missing = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            int v = Math.abs(nums[i]);
            if (v == n) v = 0;
            if (nums[v] > 0) nums[v] = -nums[v];
        }

        for (int i = 0; i < n; i++) {
            if (nums[i] > 0) {
                if (i == 0) missing.add(n);
                else missing.add(i);
            }
        }

        return missing;
    }

}
