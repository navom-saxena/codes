package dsalgo.implementations;

public class ListNode {
    int data;
    ListNode nextNode;

    ListNode(int data) {
        this.data = data;
    }

    public int getData() {
        return this.data;
    }

    public void setData(int data) {
        this.data = data;
    }

    public ListNode getNextNode() {
        return this.nextNode;
    }

    public void setNextNode(ListNode node) {
        this.nextNode = node;
    }
}