package dsAlgo;

import java.util.ArrayList;
import java.util.EmptyStackException;

public class ArrayWithThreeStack {
    int[] dataArr;
    int size, topOne, topTwo, topThree, topThreeThres;

    public ArrayWithThreeStack(int size, int thirdPoint) {
        if (size < 2) {
            throw new IllegalStateException("Size of array less than 2");
        }
        dataArr = new int[size];
        this.size = size;
        this.topOne = 0;
        this.topTwo = this.size - 1;
        this.topThree = thirdPoint - 1;
        this.topThreeThres = thirdPoint - 1;
    }

    public void displayArrayWithThreeStacks() {
        ArrayList<Integer> a1 = new ArrayList<Integer>();
        for (int i = 0; i < this.topOne; i++) {
            a1.add(this.dataArr[i]);
        }
        ArrayList<Integer> a2 = new ArrayList<Integer>();
        for (int i = this.size - 1; i > this.topTwo; i--) {
            a2.add(this.dataArr[i]);
        }
        ArrayList<Integer> a3 = new ArrayList<Integer>();
        for (int i = this.topThreeThres; i < this.topThree; i++) {
            a3.add(this.dataArr[i]);
        }
        ArrayList<Integer> aux = new ArrayList<Integer>();
        for (int i = 0; i < this.size; i++) {
            aux.add(this.dataArr[i]);
        }
        System.out.println(a1);
        System.out.println(a2);
        System.out.println(a3);
        System.out.println(aux);
    }

    public void push(int stackId, int data) {
        if (stackId == 1) {
            if (this.topOne >= this.topThreeThres) {
                if (this.topThree + 1 < this.topTwo) {
                    for (int i = this.topThree; i >= this.topThreeThres; i--) {
                        dataArr[i + 1] = dataArr[i];
                    }
                    dataArr[this.topThreeThres] = 0;
                    this.topThreeThres++;
                    this.topThree++;
                } else {
                    throw new StackOverflowError("Array full");
                }
            }
            this.dataArr[this.topOne] = data;
            this.topOne++;
        } else if (stackId == 2) {
            if (this.topTwo < this.topThree) {
                if (this.topThreeThres > this.topOne) {
                    for (int i = this.topThreeThres; i <= this.topThree; i++) {
                        this.dataArr[i - 1] = this.dataArr[i];
                    }
                    this.dataArr[this.topThree] = 0;
                    this.topThreeThres--;
                    this.topThree--;
                } else {
                    throw new StackOverflowError("Array full");
                }
            }
            this.dataArr[this.topTwo] = data;
            this.topTwo--;
        } else if (stackId == 3) {
            if (this.topThree > this.topTwo) {
                for (int i = this.topThreeThres; i <= this.topThree; i++) {
                    this.dataArr[i - 1] = this.dataArr[i];
                }
                this.dataArr[this.topThree] = 0;
                this.topThreeThres--;
                this.topThree--;
            } else {
                throw new StackOverflowError("Array full");
            }
            this.dataArr[this.topThree] = data;
            this.topThree++;
        } else {
            throw new IllegalArgumentException("wrong stackId");
        }
    }

    public int pop(int stackId) {
        if (this.topOne == 0 || this.topTwo == this.size || this.topThree == this.topThreeThres) {
            throw new EmptyStackException();
        }
        int poppedValue = 0;
        if (stackId == 1) {
            this.topOne--;
            poppedValue = this.dataArr[this.topOne];
            this.dataArr[this.topOne] = 0;
        } else if (stackId == 2) {
            this.topTwo++;
            poppedValue = this.dataArr[this.topTwo];
            this.dataArr[this.topTwo] = 0;
        } else if (stackId == 3) {
            this.topThree--;
            poppedValue = this.dataArr[this.topThree];
            this.dataArr[this.topThree] = 0;
        } else {
            throw new IllegalArgumentException("wrong stackId");
        }
        return poppedValue;
    }

    public int top(int stackId) {
        if (this.topOne == 0 || this.topTwo == this.size || this.topThree == this.topThreeThres) {
            throw new EmptyStackException();
        }
        int topValue = 0;
        if (stackId == 1) {
            topValue = this.dataArr[this.topOne - 1];
        } else if (stackId == 2) {
            topValue = this.dataArr[this.topTwo + 1];
        } else if (stackId == 3) {
            topValue = this.dataArr[this.topThree - 1];
        } else {
            throw new IllegalArgumentException("wrong stackId");
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
        } else if (stackId == 3) {
            if (this.topTwo == this.topThreeThres) {
                flag = true;
            }
        } else {
            throw new IllegalArgumentException("stack id illegal");
        }
        return flag;
    }
}
