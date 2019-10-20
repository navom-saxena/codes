package dsAlgo;

import java.util.ArrayList;


public class CircularArrayQueue {
    int[] queueCircularArr;
    int size, rear, front;

    CircularArrayQueue(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("size less than 1");
        }
        this.size = size;
        this.rear = -1;
        this.front = -1;
        this.queueCircularArr = new int[this.size];
    }

    public void displayCircularQueue() {
        ArrayList<Integer> a1 = new ArrayList<Integer>();
        ArrayList<Integer> a2 = new ArrayList<Integer>();
        for (int i = 0; i < this.size; i++) {
            a1.add(this.queueCircularArr[i]);
        }
        System.out.println(a1);
    }

    public void enQueue(int data) {
        if ((this.rear + 1) % this.size != (this.front) % this.size) {
            if (this.rear == -1 && this.front == -1) {
                this.rear = (this.rear + 1) % this.size;
                this.front = (this.front + 1) % this.size;
                this.queueCircularArr[this.rear] = data;
            } else {
                this.rear = (this.rear + 1) % this.size;
                this.queueCircularArr[this.rear] = data;
            }
        } else {
            throw new RuntimeException("Circular queue full. Cannot enqueue");
        }
    }

    public int deQueue() {
        int deQueueVal = Integer.MIN_VALUE;
        if (this.front != -1) {
            if (this.front == this.rear) {
                deQueueVal = this.queueCircularArr[this.front];
                this.queueCircularArr[this.front] = 0;
                this.front = -1;
                this.rear = -1;
            } else {
                deQueueVal = this.queueCircularArr[this.front];
                this.queueCircularArr[this.front] = 0;
                this.front = (this.front + 1) % this.size;
            }
        } else {
            throw new RuntimeException("Circular queue full. Cannot dequeue");
        }
        return deQueueVal;
    }

    public int peek() {
        return this.queueCircularArr[this.front];
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
        if ((this.rear + 1) % this.size == this.front) {
            flag = true;
        }
        return flag;
    }
}
