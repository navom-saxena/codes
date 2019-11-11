package dsAlgo.hackerRank.datastructures;

public class DoublyLinkedLists {

    public static void main(String[] args) {
        int i = 3;
        DoublyLinkedList head = new DoublyLinkedList(1);
        while (i < 20) {
            head = insertNodeAtTail(head, i);
            i = i + 2;
        }
        printDoublyLinkedList(head);
        head = sortedInsert(head, 10);
        printDoublyLinkedList(head);
    }

    private static DoublyLinkedList sortedInsert(DoublyLinkedList node, int value) {
        if (node == null) {
            return new DoublyLinkedList(value);
        }
        DoublyLinkedList head = node;
        DoublyLinkedList newNode = new DoublyLinkedList(value);
        if (value < node.getValue()) {
            newNode.setNextNode(node);
            node.setPrevNode(newNode);
            return newNode;
        }
        while (node != null && node.getNextNode() != null) {
            DoublyLinkedList nextNode = node.getNextNode();
            if (nextNode.getValue() > value) {
                node.setNextNode(newNode);
                newNode.setPrevNode(node);
                newNode.setNextNode(nextNode);
                nextNode.setPrevNode(newNode);
                return head;
            }
            node = node.getNextNode();
        }
        if (node != null) {
            node.setNextNode(newNode);
        }
        newNode.setPrevNode(node);
        return head;
    }

    private static DoublyLinkedList insertNodeAtTail(DoublyLinkedList node, int value) {
        if (node == null) {
            return new DoublyLinkedList(value);
        }
        DoublyLinkedList head = node;
        while (node.getNextNode() != null) {
            node = node.getNextNode();
        }
        DoublyLinkedList newNode = new DoublyLinkedList(value);
        node.setNextNode(newNode);
        newNode.setPrevNode(node);
        return head;
    }

    private static void printDoublyLinkedList(DoublyLinkedList node) {
        if (node == null) {
            return;
        }
        while (node != null) {
            System.out.println(node.getValue());
            node = node.getNextNode();
        }
    }

    private DoublyLinkedList reverse(DoublyLinkedList node) {
        if (node == null) {
            return null;
        }
        while (node != null && node.getNextNode() != null) {
            DoublyLinkedList nextNode = node.getNextNode();
            DoublyLinkedList preNode = node.getPrevNode();
            node.setPrevNode(nextNode);
            node.setNextNode(preNode);
            node = nextNode;
        }
        if (node != null) {
            node.setNextNode(node.getPrevNode());
            node.setPrevNode(null);
        }
        return node;
    }

    public static class DoublyLinkedList {

        int value;
        DoublyLinkedList nextNode;
        DoublyLinkedList prevNode;

        DoublyLinkedList() {

        }

        DoublyLinkedList(int value) {
            this.value = value;
        }

        int getValue() {
            return value;
        }

        void setValue(int value) {
            this.value = value;
        }

        DoublyLinkedList getNextNode() {
            return nextNode;
        }

        void setNextNode(DoublyLinkedList nextNode) {
            this.nextNode = nextNode;
        }

        DoublyLinkedList getPrevNode() {
            return prevNode;
        }

        void setPrevNode(DoublyLinkedList prevNode) {
            this.prevNode = prevNode;
        }
    }

}
