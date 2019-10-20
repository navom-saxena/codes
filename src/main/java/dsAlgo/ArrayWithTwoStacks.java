package dsAlgo;

import java.util.ArrayList;
import java.util.EmptyStackException;

public class ArrayWithTwoStacks {
    int[] dataArr;
    int size, topOne, topTwo;

    ArrayWithTwoStacks(int size) {
        if (size < 2) {
            throw new IllegalStateException("size less than 2");
        }
        this.size = size;
        this.topOne = 0;
        this.topTwo = size - 1;
        this.dataArr = new int[size];
    }

    public void displayArrayWithTwoStacks() {
        ArrayList<Integer> a1 = new ArrayList<Integer>();
        for (int i = 0; i < this.topOne; i++) {
            a1.add(this.dataArr[i]);
        }
        ArrayList<Integer> a2 = new ArrayList<Integer>();
        for (int i = this.size - 1; i > this.topTwo; i--) {
            a2.add(this.dataArr[i]);
        }
        ArrayList<Integer> aux = new ArrayList<Integer>();
        for (int i = 0; i < this.size; i++) {
            aux.add(this.dataArr[i]);
        }
        System.out.println(a1);
        System.out.println(a2);
        System.out.println(aux);
    }

    public void push(int stackId, int data) {
        if (this.topOne == this.topTwo + 1) {
            throw new StackOverflowError("Array is full");
        }
        if (stackId == 1) {
            this.dataArr[this.topOne] = data;
            this.topOne++;
        } else if (stackId == 2) {
            this.dataArr[this.topTwo] = data;
            this.topTwo--;
        } else {
            throw new IllegalArgumentException("stack id illegal");
        }
    }

    public int pop(int stackId) {
        int poppedValue = 0;
        if (this.topOne == -1 || this.topTwo == this.size) {
            throw new EmptyStackException();
        }
        if (stackId == 1) {
            poppedValue = this.dataArr[this.topOne];
            this.topOne--;
            this.dataArr[this.topOne] = 0;
        } else if (stackId == 2) {
            poppedValue = this.dataArr[this.topTwo];
            this.topTwo++;
            this.dataArr[this.topTwo] = 0;
        } else {
            throw new IllegalArgumentException("stack id illegal");
        }
        return poppedValue;
    }

    public int top(int stackId) {
        int topValue = 0;
        if (this.topOne == -1 || this.topTwo == this.size) {
            throw new EmptyStackException();
        }
        if (stackId == 1) {
            topValue = this.dataArr[this.topOne];
        } else if (stackId == 2) {
            topValue = this.dataArr[this.topTwo];
        } else {
            throw new IllegalArgumentException("stack id illegal");
        }
        return topValue;
    }

    public boolean isEmpty(int stackId) {
        boolean flag = false;
        if (stackId == 1) {
            if (this.topOne == 0) {
                flag = true;
            }
        } else if (stackId == 2) {
            if (this.topTwo == this.size - 1) {
                flag = true;
            }
        } else {
            throw new IllegalArgumentException("stack id illegal");
        }
        return flag;
    }
}
