package dsalgo.leetcode.explore;

import org.apache.spark.ml.clustering.KMeans;

import java.util.*;

public class Heap {

    public static void main(String[] args) {

    }

//    https://leetcode.com/explore/featured/card/heap/646/practices/4014/

    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int num : nums) {
            minHeap.add(num);
            if (minHeap.size() > k) minHeap.remove();
        }
        return !minHeap.isEmpty() ? minHeap.peek() : Integer.MIN_VALUE;
    }

//    https://leetcode.com/explore/featured/card/heap/646/practices/4015/

    static class Pair {
        int v;
        int f;

        Pair(int v, int f) {
            this.v = v;
            this.f = f;
        }
    }

    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        PriorityQueue<Pair> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> a.f));
        for (int key : map.keySet()) {
            minHeap.add(new Pair(key, map.get(key)));
            if (minHeap.size() > k) minHeap.remove();
        }
        int [] res = new int[k];
        int i = 0;
        while (!minHeap.isEmpty()) {
            res[i] = minHeap.remove().v;
            i++;
        }
        return res;
    }

//    https://leetcode.com/explore/featured/card/heap/646/practices/4016/

    static class KthLargest {

        PriorityQueue<Integer> minHeap;
        int k;

        public KthLargest(int k, int[] nums) {
            this.minHeap = new PriorityQueue<>();
            this.k = k;
            for (int num : nums) {
                minHeap.add(num);
                if (minHeap.size() > k) minHeap.remove();
            }
        }

        public int add(int val) {
            minHeap.add(val);
            if (minHeap.size() > k) minHeap.remove();
            return minHeap.isEmpty() ? Integer.MIN_VALUE : minHeap.peek();
        }
    }

//    https://leetcode.com/explore/featured/card/heap/646/practices/4084/

    public int lastStoneWeight(int[] stones) {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a,b) -> b - a);
        for (int stone : stones) {
            maxHeap.add(stone);
        }
        while (maxHeap.size() >= 2) {
            int a = maxHeap.remove();
            int b = maxHeap.remove();
            maxHeap.add(a - b);
        }
        return maxHeap.isEmpty() ? 0 : maxHeap.remove();
    }

//    https://leetcode.com/explore/featured/card/heap/646/practices/4085/

    static class Weakest {
        int no;
        int index;

        Weakest(int no, int index) {
            this.no = no;
            this.index = index;
        }
    }

    static class WeakestComp implements Comparator<Weakest> {

        @Override
        public int compare(Weakest o1, Weakest o2) {
            if (o1.no == o2.no) return o1.index - o2.index;
            else return o1.no - o2.no;
        }
    }

    public int[] kWeakestRows(int[][] mat, int k) {
        PriorityQueue<Weakest> minHeap = new PriorityQueue<>(new WeakestComp());
        for (int i = 0; i < mat.length; i++) {
            int [] row = mat[i];
            int no = 0;
            int j = 0;
            while (j < row.length && row[j] == 1) {
                no++;
                j++;
            }
            minHeap.add(new Weakest(no, i));
        }
        int [] res = new int[k];
        for (int i = 0; i < res.length; i++) {
            if (!minHeap.isEmpty()) {
                res[i] = minHeap.remove().index;
            }
        }
        return res;
    }

//    https://leetcode.com/explore/featured/card/heap/646/practices/4086/

    static class KSmall {
        int v;
        int index;
        int [] arr;

        KSmall(int v, int index, int [] arr) {
            this.v = v;
            this.index = index;
            this.arr = arr;
        }
    }

    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<KSmall> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> a.v));
        for (int[] ints : matrix) {
            minHeap.add(new KSmall(ints[0], 0, ints));
        }
        while (!minHeap.isEmpty() && k > 1) {
            KSmall kSmall = minHeap.remove();
            int index = kSmall.index;
            int v = kSmall.v;
            int [] arr = kSmall.arr;
            if (index + 1 < arr.length) minHeap.add(new KSmall(arr[index + 1],kSmall.index + 1, arr));
            k--;
        }
        return minHeap.isEmpty() ? Integer.MIN_VALUE : minHeap.peek().v;
    }

//    https://leetcode.com/explore/featured/card/heap/646/practices/4087/

    public int minMeetingRooms(int[][] intervals) {
        TreeMap<Integer,Integer> tm = new TreeMap<>();
        for (int [] interval : intervals) {
            int start = interval[0];
            int end = interval[1];
            tm.put(start, tm.getOrDefault(start, 0) + 1);
            tm.put(end, tm.getOrDefault(end, 0) - 1);
        }
        int min = 0;
        int counter = 0;
        for (int k : tm.keySet()) {
            counter += tm.get(k);
            min = Math.max(counter, min);
        }
        return min;
    }

//    https://leetcode.com/explore/featured/card/heap/646/practices/4088/

    public int[][] kClosest(int[][] points, int k) {
        PriorityQueue<int []> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> (a[1] * a[1] + a[0] * a[0])));
        for (int [] point : points) {
            int x = point[0];
            int y = point[1];
            minHeap.add(new int[]{x,y});
        }
        int [][] kPoints = new int[k][2];
        int i = 0;
        while (!minHeap.isEmpty() && k > 0) {
            kPoints[i] = minHeap.remove();
            i++;
            k--;
        }
        return kPoints;
    }

//    https://leetcode.com/explore/learn/card/heap/646/practices/4090/

    public int connectSticks(int[] sticks) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int stick : sticks) {
            minHeap.add(stick);
        }
        int cost = 0;
        while (minHeap.size() > 1) {
            int a = minHeap.remove();
            int b = minHeap.remove();
            int sum = a + b;
            cost += sum;
            minHeap.add(sum);
        }
        return cost;
    }

//    https://leetcode.com/problems/furthest-building-you-can-reach/

    public int furthestBuilding(int[] heights, int bricks, int ladders) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int i = 0; i < heights.length - 1; i++) {
            int d = heights[i + 1] - heights[i];
            if (d <= 0) continue;
            minHeap.add(d);
            if (minHeap.size() > ladders) {
                bricks -= minHeap.remove();
            }
            if (bricks < 0) return i;
        }
        return heights.length - 1;
    }

//    https://leetcode.com/explore/learn/card/heap/646/practices/4092/

    static class MedianFinder {

        PriorityQueue<Integer> minHeap;
        PriorityQueue<Integer> maxHeap;

        public MedianFinder() {
            minHeap = new PriorityQueue<>();
            maxHeap = new PriorityQueue<>((a,b) -> b - a);
        }

        public void addNum(int num) {
            if (maxHeap.isEmpty() || maxHeap.peek() <= num) minHeap.add(num);
            else maxHeap.add(num);
            if (Math.abs(maxHeap.size() - minHeap.size()) > 1) {
                if (maxHeap.size() > minHeap.size()) minHeap.add(maxHeap.remove());
                else maxHeap.add(minHeap.remove());
            }
        }

        public double findMedian() {
            if (maxHeap.size() + minHeap.size() % 2 == 0) {
                return minHeap.isEmpty() || maxHeap.isEmpty() ? 0 : minHeap.peek() + maxHeap.peek() / 2.0;
            } else if (maxHeap.size() > minHeap.size()) return maxHeap.peek();
            else if (!minHeap.isEmpty()) return minHeap.peek();
            else return 0;
        }
    }

}

