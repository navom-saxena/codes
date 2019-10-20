package dsAlgo;

import java.util.ArrayList;

public class DoublyLinkedListImplementation {
    DoublyLinkedListNode head;

    DoublyLinkedListImplementation(int data) {
        this.head = new DoublyLinkedListNode(data);
    }

    public void displayDoublyLinkedList() {
        ArrayList<Integer> a1 = new ArrayList<>();
        DoublyLinkedListNode currentNode = this.head;
        while (currentNode != null) {
            a1.add(currentNode.getData());
            currentNode = currentNode.getNextNode();
        }
        System.out.println(a1);
    }

    public void displayDoublyLinkedListReverse() {
        ArrayList<Integer> a1 = new ArrayList<>();
        DoublyLinkedListNode currentNode = this.head;
        while (currentNode.getNextNode() != null) {
            currentNode = currentNode.getNextNode();
        }
        while (currentNode != null) {
            a1.add(currentNode.getData());
            currentNode = currentNode.getPrevNode();
        }
        System.out.println(a1);
    }

    public void insertNodeAtStart(int nodeData) {
        DoublyLinkedListNode newNode = new DoublyLinkedListNode(nodeData);
        DoublyLinkedListNode currentHead = this.head;
        this.head = newNode;
        this.head.setNextNode(currentHead);
        this.head.getNextNode().setPrevNode(this.head);
    }

    public DoublyLinkedListNode insertNodeAtEnd(int nodeData) {
        DoublyLinkedListNode newnode = new DoublyLinkedListNode(nodeData);
        DoublyLinkedListNode currentNode = this.head;
        while (currentNode.getNextNode() != null) {
            currentNode = currentNode.getNextNode();
        }
        currentNode.setNextNode(newnode);
        currentNode.getNextNode().setPrevNode(currentNode);
        return currentNode.getNextNode();
    }

    public void insertNodeAtIndex(int nodeData, int index) {
        int counter = 0;
        DoublyLinkedListNode newNode = new DoublyLinkedListNode(nodeData);
        DoublyLinkedListNode currentNode = this.head;
        while (currentNode.getNextNode() != null) {
            if (counter + 1 == index) {
                DoublyLinkedListNode origionalNextNode = currentNode.getNextNode();
                currentNode.setNextNode(newNode);
                currentNode.getNextNode().setPrevNode(currentNode);
                currentNode.getNextNode().setNextNode(origionalNextNode);
                currentNode.getNextNode().getNextNode().setPrevNode(currentNode.getNextNode());
            }
            currentNode = currentNode.getNextNode();
            counter++;
        }
    }

    public int getNodeDataAtIndex(int index) {
        int counter = 0;
        int data = Integer.MIN_VALUE;
        DoublyLinkedListNode currentNode = this.head;
        while (currentNode != null) {
            if (counter == index) {
                data = currentNode.getData();
            }
            currentNode = currentNode.getNextNode();
            counter++;
        }
        return data;
    }

    public void deleteNodeAtStart() {
        this.head = this.head.getNextNode();
        this.head.setPrevNode(null);
    }

    public void deleteNodeAtEnd() {
        DoublyLinkedListNode currentNode = this.head;
        while (currentNode.getNextNode().getNextNode() != null) {
            currentNode = currentNode.getNextNode();
        }
        currentNode.setNextNode(null);
    }

    public void deleteNodeAtIndex(int index) {
        int counter = 0;
        DoublyLinkedListNode currentNode = this.head;
        while (currentNode.getNextNode() != null) {
            if (counter + 1 == index) {
                currentNode.setNextNode(currentNode.getNextNode().getNextNode());
                currentNode.getNextNode().setPrevNode(currentNode);
            }
            currentNode = currentNode.getNextNode();
            counter++;
        }
    }
}