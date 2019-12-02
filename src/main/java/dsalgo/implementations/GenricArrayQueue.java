package dsalgo.implementations;

public class GenricArrayQueue<T> {
    T[] queueArr;
    int size, front, rear;

    GenricArrayQueue(T data) {
        if (size < 1) {
            throw new IllegalArgumentException("size less than 1");
        }
        this.size = size;
        this.rear = -1;
        this.front = -1;
        this.queueArr = (T[]) new Object[size];
    }
}
