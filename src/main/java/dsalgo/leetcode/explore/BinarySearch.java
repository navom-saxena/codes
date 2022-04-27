package dsalgo.leetcode.explore;

import dsalgo.leetcode.Models.*;

import java.util.*;
import java.util.Arrays;

public class BinarySearch {

    public static void main(String[] args) {
       // System.out.println(Math.pow(2147395599,0.5));
        System.out.println(getPivot(new int[]{0,1,2,3,4,5,6,7}, 0, 7));
    }

//    https://leetcode.com/explore/learn/card/binary-search/138/background/1038/

    public int search(int[] nums, int target) {
        int low = 0;
        int high = nums.length - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] < target) low = mid + 1;
            else high = mid - 1;
        }

        return -1;
    }

//    https://leetcode.com/explore/learn/card/binary-search/125/template-i/950/

    public int mySqrt(int x) {
        if (x < 2) return x;
        int low = 0;
        int high = x / 2;
        int res = high;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            long sq = (long) mid * mid;

            if (sq <= x) {
                res = mid;
                low = mid + 1;
            }
            else high = mid - 1;
        }
        return res;
    }

//    https://leetcode.com/explore/learn/card/binary-search/125/template-i/951/

    int guess(int num) {
        return -1;
    }

    int guessNumberUtil(int low, int high) {
        int mid = low + (high - low) / 2;
        int no = guess(mid);
        if (no == 0) return mid;
        else if (no > 0) return guessNumberUtil(mid + 1, high);
        else return guessNumberUtil(low, mid - 1);
    }

    public int guessNumber(int n) {
        return guessNumberUtil(1, n);
    }

//    https://leetcode.com/explore/learn/card/binary-search/125/template-i/952/

    static int getPivot(int [] nums, int low, int high) {
        if (low == high) return low;
        int mid = low + (high - low) / 2;
        if (mid < high && nums[mid] > nums[mid + 1]) return mid;
        else if (nums[low] <= nums[mid]) return getPivot(nums, mid + 1, high);
        else return getPivot(nums, low, mid - 1);
    }

    static int searchUtils(int [] nums, int target, int low, int high) {
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] == target) return mid;
            if (nums[mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }

    public int searchP(int[] nums, int target) {
        int pivotIndex = getPivot(nums, 0, nums.length - 1);
        int left = searchUtils(nums, target, 0, pivotIndex);
        return left != -1 ? left : searchUtils(nums, target, pivotIndex + 1, nums.length - 1);
    }

//    https://leetcode.com/explore/learn/card/binary-search/126/template-ii/947/

    boolean isBadVersion(int version) { return true;}

    public int firstBadVersion(int n) {
        int low = 1;
        int high = n;
        int firstBad = -1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            boolean isBad = isBadVersion(mid);
            if (isBad) {
                firstBad = mid;
                high = mid - 1;
            } else low = mid + 1;
        }
        return firstBad;
    }

//    https://leetcode.com/explore/learn/card/binary-search/126/template-ii/948/

    public int findPeakElement(int[] nums) {
        long l = Long.MIN_VALUE;
        long h = Long.MIN_VALUE;
        int low = 0;
        int high = nums.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (mid - 1 >= 0) l = nums[mid - 1];
            if (mid + 1 < nums.length) h = nums[mid + 1];
            int m = nums[mid];
            if (l < m && m > h) return mid;
            else if (l < m) low = mid + 1;
            else high = mid - 1;
            l = Integer.MIN_VALUE;
            h = Integer.MIN_VALUE;
        }
        return -1;
    }

//    https://leetcode.com/explore/learn/card/binary-search/126/template-ii/949/

    public int findMin(int[] nums) {
        int pivot = getPivot(nums, 0, nums.length - 1);
        if (pivot == nums.length - 1) return nums[0];
        else return nums[pivot + 1];
    }

//    https://leetcode.com/explore/learn/card/binary-search/135/template-iii/944/

    int getLeftRange(int [] nums, int target, int low, int high) {
        int res = low;
        boolean isFound = false;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] >= target) {
                if (nums[mid] == target) isFound = true;
                res = mid;
                high = mid - 1;
            } else low = mid + 1;
        }
        return isFound ? res : -1;
    }

    int getRightRange(int [] nums, int target, int low, int high) {
        int res = low;
        boolean isFound = false;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] <= target) {
                if (nums[mid] == target) isFound = true;
                res = mid;
                low = mid + 1;
            } else high = mid - 1;
        }
        return isFound ? res : -1;
    }

    public int[] searchRange(int[] nums, int target) {
        int [] range = new int[2];
        int left = getLeftRange(nums, target, 0, nums.length - 1);
        int right = getRightRange(nums, target, 0, nums.length - 1);
        range[0] = left;
        range[1] = right;
        return range;
    }

//    https://leetcode.com/explore/learn/card/binary-search/135/template-iii/945/

    int getFloorInc(int [] nums, int x) {
        int low = 0;
        int high = nums.length - 1;
        if (x < nums[low]) return -1;
        int res = low;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] <= x) {
                res = mid;
                low = mid + 1;
            } else high = mid - 1;
        }
        return res;
    }

    int getCeil(int [] nums, int x) {
        int low = 0;
        int high = nums.length - 1;
        if (x > nums[high]) return nums.length;
        int res = high;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] > x) {
                res = mid;
                high = mid - 1;
            } else low = mid + 1;
        }
        return res;
    }

    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        List<Integer> res = new ArrayList<>();
        int n = arr.length;
        int i = getFloorInc(arr, x);
        int j = i + 1;
        while ((i >= 0 || j < n) && k > 0) {
            if (i >= 0 && j < n) {
                if (x - arr[i] <= arr[j] - x) i--;
                else j++;
            } else if (i >= 0) i--;
            else j++;
            k--;
        }
        i++;
        j--;
        for (int m = i; m <= j; m++) res.add(arr[m]);
        return res;
    }

//    https://leetcode.com/explore/learn/card/binary-search/136/template-analysis/1028/

    int currClosest = Integer.MAX_VALUE;
    double minDiff = Integer.MAX_VALUE;

    public int closestValue(TreeNode root, double target) {
        if (root == null) return Integer.MAX_VALUE;
        closestValueUtil(root, target);
        return currClosest;
    }

    void closestValueUtil(TreeNode node, double target) {
        if (node == null) return;
        if (Math.abs(node.val - target) < minDiff) {
            currClosest = node.val;
            minDiff = Math.abs(node.val - target);
        }
        if (node.val < target) closestValueUtil(node.right, target);
        else closestValueUtil(node.left, target);
    }

//    https://leetcode.com/explore/learn/card/binary-search/136/template-analysis/1061/

    interface ArrayReader {
        int get(int index);
    }

    public int search(ArrayReader reader, int target) {
       int i = 0;
       while (true) {
           int atI = reader.get(i);
           if (atI >= target) break;
           i = i == 0 ? 1 : i * 2;
       }

       int low = 0;
       int high = i;
       while (low <= high) {
        int mid = low + (high - low) / 2;
        int v = reader.get(mid);
        if (v == target) return mid;
        else if (v < target) low = mid + 1;
        else high = mid - 1;
       }
       return -1;
    }

//    https://leetcode.com/explore/learn/card/binary-search/137/conclusion/978/

    public boolean isPerfectSquare(int num) {
        if (num < 2) return true;
        int low = 0;
        int high = num / 2;
        while (low <= high) {
         int mid = low + (high - low) / 2;
         long sq = (long) mid * mid;
         if (sq == num) return true;
         else if (sq < num) low = mid + 1;
         else high = mid - 1;
        }
        return false;
    }

//    https://leetcode.com/explore/learn/card/binary-search/137/conclusion/977/

    int toInt(char c) {
        return c - 'a';
    }

    public char nextGreatestLetter(char[] letters, char target) {
        int n = letters.length;
        int targetInt = toInt(target);
        if (targetInt >= toInt(letters[n - 1])) return letters[0];
        int low = 0;
        int high = n - 1;
        char res = letters[n - 1];
        while (low <= high) {
            int mid = low + (high - low) / 2;
            char midC = letters[mid];
            int midI = toInt(midC);
            if (midI > targetInt) {
                res = midC;
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return res;
    }

//    https://leetcode.com/explore/learn/card/binary-search/144/more-practices/1031/

    public int findMin2(int[] nums) {
        int low = 0;
        int high = nums.length - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] < nums[high]) high = mid;
            else if (nums[mid] > nums[high]) low = mid + 1;
            else high--;
        }
        return nums[low];
    }

//    https://leetcode.com/explore/learn/card/binary-search/144/more-practices/1034/

    public int[] intersection(int[] nums1, int[] nums2) {
        int [] cNums1 = new int[1001];
        int [] res = new int[2002];
        int k = 0;
        for (int no : nums1) cNums1[no] = 1;
        for (int no : nums2) {
            if (cNums1[no] > 0) {
                res[k] = no;
                k++;
                cNums1[no] = 0;
            }
        }
        int [] kRes = new int[k];
        System.arraycopy(res, 0, kRes, 0, k);
        return kRes;
    }

//    https://leetcode.com/explore/learn/card/binary-search/144/more-practices/1029/

    public int[] intersect(int[] nums1, int[] nums2) {
        int [] cNums1 = new int[1001];
        for (int no : nums1) cNums1[no]++;
        int [] res = new int[2002];
        int k = 0;
        for (int no : nums2) {
            if (cNums1[no] > 0) {
                res[k] = no;
                k++;
                cNums1[no]--;
            }
        }
        int [] kRes = new int[k];
        System.arraycopy(res, 0, kRes, 0, k);
        return kRes;
    }

//    https://leetcode.com/explore/learn/card/binary-search/144/more-practices/1035/

    public int[] twoSum(int[] numbers, int target) {
        int low = 0;
        int high = numbers.length - 1;
        int  [] res = new int[2];
        while (low < high) {
            int sum = numbers[low] + numbers[high];
            if (sum == target) {
                res[0] = low + 1;
                res[1] = high + 1;
                break;
            } else if (sum < target) low++;
            else high--;
        }
        return res;
    }

//    https://leetcode.com/explore/learn/card/binary-search/146/more-practices-ii/1039/

    public int findDuplicate(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int absV = Math.abs(nums[i]);
            int v = absV == n ? 0 : absV;
            if (nums[v] < 0) return absV;
            nums[v] = -nums[v];
        }
        return -1;
    }

//    https://leetcode.com/explore/learn/card/binary-search/146/more-practices-ii/1040/

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) return findMedianSortedArraysUtil(nums2, nums1);
        else return findMedianSortedArraysUtil(nums1, nums2);
    }

    double findMedianSortedArraysUtil(int[] nums1, int[] nums2) {
        int lx = nums1.length;
        int ly = nums2.length;

        int low = 0;
        int high = nums1.length;

        while (low <= high) {
            int partitionX = low + (high - low) / 2;
            int partitionY = (lx + ly + 1) / 2 - partitionX;

            int leftMaxX = partitionX == 0 ? Integer.MIN_VALUE : nums1[partitionX - 1];
            int rightMinX = partitionX == lx ? Integer.MAX_VALUE : nums1[partitionX];

            int leftMaxY = partitionY == 0 ? Integer.MIN_VALUE : nums2[partitionY - 1];
            int rightMinY = partitionY == ly ? Integer.MAX_VALUE : nums2[partitionY];

            if (leftMaxX <= rightMinY && leftMaxY <= rightMinX) {
                if ((lx + ly) % 2 == 0) {
                    return (double) (Math.max(leftMaxX, leftMaxY) + Math.min(rightMinX, rightMinY)) / 2;
                } else {
                    return Math.max(leftMaxX, leftMaxY);
                }
            } else if (leftMaxX > rightMinY) high = partitionX - 1;
            else low = partitionX + 1;
        }
        return -1;
    }

//    https://leetcode.com/explore/learn/card/binary-search/146/more-practices-ii/1041/

    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        int low = 0;
        int high = nums[nums.length - 1] - nums[0];
        int res = low;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (isSmallerPairs(nums, k, mid)) {
                res = mid;
                high = mid - 1;
            } else low = mid + 1;
        }
        return res;
    }

    private boolean isSmallerPairs(int [] nums, int k, int mid) {
        int count = 0;
        int left = 0;
        for (int right = 1; right < nums.length; right++) {
            while (nums[right] - nums[left] > mid) left ++;
            count += right - left;
        }
        return count >= k;
    }

//    https://leetcode.com/explore/learn/card/binary-search/146/more-practices-ii/1042/

    public int splitArray(int[] nums, int m) {
        int low = 0, high = 0;
        for (int num : nums) {
            low = Math.max(low, num);
            high += num;
        }
        int res = low;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (isSplittable(nums, mid, m)) {
                res = mid;
                high = mid - 1;
            } else low = mid + 1;
        }
        return res;
    }

    private boolean isSplittable(int [] nums, int maxSum, int m) {
        int breaks = 1;
        int sum = 0;
        for (int num : nums) {
            sum += num;
            if (sum > maxSum) {
                sum = num;
                breaks++;
            }
            if (breaks > m) return false;
        }
        return true;
    }

}