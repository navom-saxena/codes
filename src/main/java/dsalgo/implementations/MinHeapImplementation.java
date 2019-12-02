package dsalgo.implementations;

import java.util.ArrayList;
import java.util.List;

public class MinHeapImplementation {
    List<Integer> arr;

    MinHeapImplementation() {
        this.arr = new ArrayList<>();
    }

    public int getSize() {
        return this.arr.size();
    }

    public void displayHeap() {
        System.out.println(this.arr);
    }

    public void insertDataInHeap(int data) {
        this.arr.add(data);
        stabilizeHeapBottomUp();
    }

    public void stabilizeHeapBottomUp() {
        int i = this.arr.size() - 1;
        int parentIndex = (i - 1) / 2;
        while (this.arr.get(parentIndex) > this.arr.get(i)) {
            int temp = this.arr.get(parentIndex);
            this.arr.set(parentIndex, this.arr.get(i));
            this.arr.set(i, temp);
            i = i / 2;
            parentIndex = (i - 1) / 2;
        }
    }

    public void stabilizeHeapTopDown() {
        int lastElementIndex = this.arr.size() - 1;
        this.arr.set(0, this.arr.get(lastElementIndex));
        this.arr.remove(lastElementIndex);
        int parentIndex = 0;
        int leftChildIndex = (2 * parentIndex) + 1;
        int rightChildIndex = (2 * parentIndex) + 2;
        while (this.arr.size() > rightChildIndex &&
                (this.arr.get(parentIndex) > this.arr.get(leftChildIndex)
                        || this.arr.get(parentIndex) > this.arr.get(rightChildIndex))) {
            int temp = this.arr.get(parentIndex);
            if (this.arr.get(leftChildIndex) < this.arr.get(rightChildIndex)) {
                this.arr.set(parentIndex, this.arr.get(leftChildIndex));
                this.arr.set(leftChildIndex, temp);
                parentIndex = leftChildIndex;
            } else {
                this.arr.set(parentIndex, this.arr.get(rightChildIndex));
                this.arr.set(rightChildIndex, temp);
                parentIndex = rightChildIndex;
            }
            leftChildIndex = (2 * parentIndex) + 1;
            rightChildIndex = (2 * parentIndex) + 2;
        }
    }

    public int deleteMin() {
        int data = this.arr.get(0);
        stabilizeHeapTopDown();
        return data;
    }

    public int peekMin() {
        return this.arr.get(0);
    }
}
