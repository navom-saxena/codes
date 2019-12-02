package dsalgo.implementations;

public class DoublyLinkedListNode {
    int data;
    DoublyLinkedListNode nextNode;
    DoublyLinkedListNode prevNode;

    DoublyLinkedListNode(int data) {
        this.data = data;
    }

    DoublyLinkedListNode(int data, DoublyLinkedListNode nextNode, DoublyLinkedListNode prevNode) {
        this.data = data;
        this.nextNode = nextNode;
        this.prevNode = prevNode;
    }

    public int getData() {
        return this.data;
    }

    public void setData(int data) {
        this.data = data;
    }

    public DoublyLinkedListNode getNextNode() {
        return this.nextNode;
    }

    public void setNextNode(DoublyLinkedListNode node) {
        this.nextNode = node;
    }

    public DoublyLinkedListNode getPrevNode() {
        return this.prevNode;
    }

    public void setPrevNode(DoublyLinkedListNode node) {
        this.prevNode = node;
    }
}