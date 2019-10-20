package dsAlgo;

import java.util.ArrayList;
import java.util.EmptyStackException;

import static java.lang.Math.abs;

public class ArrayStack {
    private int[] arrayStack = null;
    private int topElementIndex = -1;
    private int size = 0;

    ArrayStack(int size) {
        this.size = size;
        arrayStack = new int[this.size];
    }

    void displayArrayStack() {
        ArrayList<Integer> a1 = new ArrayList<>();
        for (int i = 0; i < this.size; i++) {
            a1.add(this.arrayStack[i]);
        }
        System.out.println(a1);
    }

    private int sizeUsed() {
        return this.topElementIndex + 1;
    }

    public void push(int data) {
        if (this.topElementIndex + 1 < this.size) {
            arrayStack[this.topElementIndex + 1] = data;
            this.topElementIndex++;
        } else {
            throw new StackOverflowError();
        }
    }

    public int pop() {
        int poppedValue = 0;
        if (this.topElementIndex >= 0) {
            poppedValue = this.arrayStack[topElementIndex];
            arrayStack[topElementIndex] = 0;
            topElementIndex--;
        } else {
            throw new EmptyStackException();
        }
        return poppedValue;
    }

    int top() {
        int topValue = 0;
        if (this.topElementIndex >= 0) {
            topValue = this.arrayStack[topElementIndex];
        } else {
            throw new EmptyStackException();
        }
        return topValue;
    }

    boolean isEmptyStack() {
        boolean flag = false;
        if (this.topElementIndex == -1) {
            flag = true;
        }
        return flag;
    }

    public boolean isFullStack() {
        boolean flag = false;
        if (this.topElementIndex + 1 == this.size) {
            flag = true;
        }
        return flag;
    }

    public void getMaxRectangleArea(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            if (arr[i + 1] >= arr[i]) {
                int ax = arr[i];
                int j = i;
                while (j < arr.length - 1 && arr[j + 1] >= arr[j]) {
                    ax += arr[i];
                    j++;
                    if (this.isEmptyStack()) {
                        this.push(ax);
                    } else {
                        if (this.top() < ax) {
                            this.pop();
                            this.push(ax);
                        }
                    }
                }
            }
        }
        System.out.println(this.top());
    }

    public void getSpans(int[] arr) {
        ArrayList<Integer> al1 = new ArrayList<Integer>();
        int poppedCount = 0;
        for (int value : arr) {
            int count = 1;
            if (!this.isEmptyStack() && this.top() >= value) {
                this.push(value);
                al1.add(count);
            } else {
                if (this.isEmptyStack()) {
                    this.push(value);
                    al1.add(count);
                } else {
                    while (this.top() < value) {
                        this.pop();
                        poppedCount++;
                        count = count + poppedCount;
                    }
                    al1.add(count);
                    this.push(value);
                }
            }
        }
        System.out.println(al1);
    }

    public void histogramMaxArea(int[] arr) {
        ArrayList<Integer> a1 = new ArrayList<>();
        for (int value : arr) {
            if (!this.isEmptyStack() && value >= this.top()) {
                this.push(value);
            } else {
                if (this.isEmptyStack()) {
                    this.push(value);
                } else {
                    int popCounter = 1;
                    while (!this.isEmptyStack()) {
                        a1.add(this.pop() * popCounter);
                        popCounter++;
                    }
                    a1.add(value * popCounter);
                    this.push(value);
                }
            }
        }
        int popCounter = 1;
        while (!this.isEmptyStack()) {
            a1.add(this.pop() * popCounter);
            popCounter++;
        }
        System.out.println(a1);
    }

    public void sortStack() {
        ArrayStack auxStack = new ArrayStack(this.size);
        auxStack.push(this.pop());
        while (!this.isEmptyStack()) {
            if (this.top() >= auxStack.top()) {
                auxStack.push(this.pop());
            } else if (auxStack.isEmptyStack()) {
                auxStack.push(this.pop());
            } else {
                int temp = this.pop();
                while (!auxStack.isEmptyStack() && temp < auxStack.top()) {
                    this.push(auxStack.pop());
                }
                auxStack.push(temp);
            }
        }
        auxStack.displayArrayStack();
    }

    public void removeAdjacantDuplicatesFromStack() {
        ArrayStack auxStack = new ArrayStack(this.size);
        while (!this.isEmptyStack()) {
            int topValue = this.pop();
            if (!this.isEmptyStack() && this.top() == topValue) {
                while (this.top() == topValue) {
                    this.pop();
                }
            } else {
                if (!auxStack.isEmptyStack() && topValue != auxStack.top()) {
                    auxStack.push(topValue);
                } else if (!auxStack.isEmptyStack() && topValue == auxStack.top()) {
                    auxStack.pop();
                } else {
                    auxStack.push(topValue);
                }
            }
        }
        auxStack.displayArrayStack();
    }

    public void removeAdjacantDuplicatesFromArray(int[] arr) {
        boolean flag = false;
        int i = 0;
        while (i < arr.length) {
            if (i == 0) {
                this.push(arr[i]);
            } else {
                if (!this.isEmptyStack() && this.top() != arr[i]) {
                    if (flag) {
                        this.pop();
                        flag = false;
                    }
                    if (this.top() != arr[i]) {
                        this.push(arr[i]);
                    } else {
                        this.pop();
                    }

                } else if (!this.isEmptyStack() && this.top() == arr[i]) {
                    flag = true;
                }
            }
            i++;
        }
        this.displayArrayStack();
    }

    public void postfixEval(String[] arr) {
        for (String s : arr) {
            if (s.matches(".*\\d+.*")) {
                this.push(Integer.parseInt(s));
            } else {
                int a = this.pop();
                int b = this.pop();
                if (s.contains("+")) {
                    int c = b + a;
                    this.push(c);
                }
                if (s.contains("-")) {
                    int c = b - a;
                    this.push(c);
                }
                if (s.contains("*")) {
                    int c = b * a;
                    this.push(c);
                }
            }
        }
        System.out.println(this.pop());
    }

    public void infixEval(String[] arr) {
        ArrayStack auxStack = new ArrayStack(arr.length);
        for (String s : arr) {
            if (s.matches(".*\\d+.*")) {
                this.push(Integer.parseInt(s));
            } else if (s.contains("(")) {

            } else if (s.contains(")")) {
                int a = this.pop();
                int b = this.pop();
                int operator = auxStack.pop();
                if (operator == 1) {
                    int c = b + a;
                    this.push(c);
                } else if (operator == 2) {
                    int c = b - a;
                    this.push(c);
                } else if (operator == 3) {
                    int c = b * a;
                    this.push(c);
                }
            } else {
                if (s.contains("+")) {
                    auxStack.push(1);
                } else if (s.contains("-")) {
                    auxStack.push(2);
                } else if (s.contains("*")) {
                    auxStack.push(3);
                }
            }
        }
        System.out.println(this.pop());
    }

    public void stackUsingTwoQueues(int data, int flag) {
        ArrayQueue aq1 = new ArrayQueue(this.size);
        ArrayQueue aq2 = new ArrayQueue(this.size);
        if (flag == 0) {
            aq1.enQueue(data);
        } else {
            int temp = Integer.MIN_VALUE;
            while (aq1.getSize() != 1) {
                temp = aq1.deQueue();
                aq2.enQueue(temp);
            }
            System.out.println(temp);
            while (aq2.isEmptyQueue()) {
                int temp1 = aq2.deQueue();
                aq1.enQueue(temp1);
            }
        }
    }

    public void checkConsecutive() {
        boolean flag = false;
        if (this.sizeUsed() % 2 == 0) {
            while (!this.isEmptyStack()) {
                int temp1 = this.pop();
                int temp2 = this.pop();
                if (abs(temp1 - temp2) == 1) {
                    flag = true;
                } else {
                    flag = false;
                    break;
                }
            }
        } else {
            this.pop();
            while (!this.isEmptyStack()) {
                int temp1 = this.pop();
                int temp2 = this.pop();
                if (abs(temp1 - temp2) == 1) {
                    flag = true;
                } else {
                    flag = false;
                    break;
                }
            }
        }
        System.out.println(flag);
        Integer.parseInt("");
    }
}