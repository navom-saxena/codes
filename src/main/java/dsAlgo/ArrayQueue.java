package dsAlgo;

import java.util.ArrayList;

public class ArrayQueue {
    int[] queueArr;
    int size, front, rear;

    public ArrayQueue(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("size less than 1");
        }
        this.size = size;
        this.rear = -1;
        this.front = -1;
        this.queueArr = new int[this.size];
    }

    public int getSize() {
        return this.size;
    }

    public void displayQueue() {
        ArrayList<Integer> a1 = new ArrayList<>();
        ArrayList<Integer> a2 = new ArrayList<>();
        for (int i = 0; i < this.size; i++) {
            a1.add(queueArr[i]);
            if (i >= this.front && i <= this.rear) {
                a2.add(queueArr[i]);
            }
        }
        System.out.println(a1);
        System.out.println(a2);
    }

    public void enQueue(int data) {
        if (this.rear + 1 != this.size) {
            if (this.rear == -1) {
                this.front++;
            }
            this.rear++;
            this.queueArr[this.rear] = data;
        } else {
            throw new RuntimeException("Queue full. Cannot enQueue");
        }
    }

    public int deQueue() {
        int deQueuedVal = Integer.MIN_VALUE;
        if (this.front != -1 && this.front != this.size) {
            if (this.front == this.rear) {
                deQueuedVal = this.queueArr[this.front];
                this.queueArr[this.front] = 0;
                this.front = -1;
                this.rear = -1;
            } else {
                deQueuedVal = this.queueArr[this.front];
                this.queueArr[this.front] = 0;
                this.front++;
            }
        } else {
            throw new RuntimeException("Queue empty. Cannot deQueue");
        }
        return deQueuedVal;
    }

    public int peek() {
        return this.queueArr[this.front];
    }

    public boolean isEmptyQueue() {
        boolean flag = false;
        if (this.rear == -1 && this.front == -1) {
            flag = true;
        }
        return flag;
    }

    public boolean isFullQueue() {
        boolean flag = false;
        if (this.rear + 1 == this.size) {
            flag = true;
        }
        return flag;
    }

    public void reverseQueue() {
        ArrayStack as = new ArrayStack(this.getSize());
        while (!this.isEmptyQueue()) {
            as.push(this.deQueue());
        }
        while (!as.isEmptyStack()) {
            this.enQueue(as.pop());
        }
        this.displayQueue();
    }

    public void queueUsingTwoStacks() {
        ArrayList<Integer> a1 = new ArrayList<Integer>();
        ArrayStack as1 = new ArrayStack(this.getSize());
        ArrayStack as2 = new ArrayStack(this.getSize());
        while (!this.isEmptyQueue()) {
            as1.push(this.deQueue());
        }
        while (!as1.isEmptyStack()) {
            as2.push(as1.pop());
        }
        while (!as2.isEmptyStack()) {
            a1.add(as2.pop());
        }
        System.out.println(a1);
    }
}
