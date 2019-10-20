package dsAlgo;

import java.util.ArrayList;

public class DynamicCircularArrayQueue {
    int[] dynamicQueueCircularArr;
    int size, front, rear;

    DynamicCircularArrayQueue(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("size less than 1");
        }
        this.size = size;
        this.rear = -1;
        this.front = -1;
        this.dynamicQueueCircularArr = new int[this.size];
    }

    public void displayDynamicCircularQueue() {
        ArrayList<Integer> a1 = new ArrayList<Integer>();
        for (int i = 0; i < this.size; i++) {
            a1.add(dynamicQueueCircularArr[i]);
        }
        System.out.println(a1);
    }

    public void enqueue(int data) {
        if ((this.rear + 1) % this.size != (this.front) % this.size) {
            if (this.rear == -1 && this.front == -1) {
                this.front = (this.front + 1) % this.size;
                this.rear = (this.rear + 1) % this.size;
                this.dynamicQueueCircularArr[this.rear] = data;
            } else {
                this.rear = (this.rear + 1) % this.size;
                this.dynamicQueueCircularArr[this.rear] = data;
            }
        } else {
            int[] dynamicQueueCircularArrayDoubled = new int[this.size * 2];
            dynamicQueueCircularArrayDoubled[this.front] = this.dynamicQueueCircularArr[this.front];
            int j = this.front + 1;
            while (this.dynamicQueueCircularArr[j] != this.dynamicQueueCircularArr[this.front]) {
                dynamicQueueCircularArrayDoubled[j] = this.dynamicQueueCircularArr[j];
                j = (j + 1) % this.size;
            }
            this.dynamicQueueCircularArr = dynamicQueueCircularArrayDoubled;
            this.size = this.size * 2;
            this.rear = (this.rear + 1) % this.size;
            this.dynamicQueueCircularArr[this.rear] = data;
        }
    }

    public int deQueue() {
        int deQueueVal = Integer.MIN_VALUE;
        if (this.front != -1) {
            if (this.front == this.rear) {
                deQueueVal = this.dynamicQueueCircularArr[this.front];
                this.dynamicQueueCircularArr[this.front] = 0;
                this.front = -1;
                this.rear = -1;
            } else {
                deQueueVal = this.dynamicQueueCircularArr[this.front];
                this.dynamicQueueCircularArr[this.front] = 0;
                this.front = (this.front + 1) % this.size;
                int length = 0;
                for (int i = this.front; i <= this.rear; i = (i + 1) % this.size) {
                    length++;
                }
                if (length < this.size / 2) {
                    int[] dynamicQueueCircularArrayHalved = new int[this.size / 2];
                    for (int i = this.front; i <= this.rear; i = (i + 1) % this.size) {
                        dynamicQueueCircularArrayHalved[i] = this.dynamicQueueCircularArr[i];
                    }
                    this.dynamicQueueCircularArr = dynamicQueueCircularArrayHalved;
                    this.size = this.size / 2;
                }
            }
        }
        return deQueueVal;
    }

    public int peek() {
        return this.dynamicQueueCircularArr[this.front];
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

