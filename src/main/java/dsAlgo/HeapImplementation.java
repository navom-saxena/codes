package dsAlgo;

public class HeapImplementation {
    public static void main(String[] args) {
        MinHeapImplementation heapObj = new MinHeapImplementation();
        heapObj.displayHeap();
        heapObj.insertDataInHeap(0);
        heapObj.insertDataInHeap(1);
        heapObj.insertDataInHeap(2);
        heapObj.insertDataInHeap(3);
        heapObj.insertDataInHeap(4);
        heapObj.insertDataInHeap(5);
        heapObj.insertDataInHeap(6);
        heapObj.insertDataInHeap(7);
        heapObj.displayHeap();
        heapObj.deleteMin();
        heapObj.displayHeap();
        heapObj.insertDataInHeap(0);
        heapObj.displayHeap();
    }
}
