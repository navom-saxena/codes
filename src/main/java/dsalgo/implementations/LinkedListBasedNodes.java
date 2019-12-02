package dsalgo.implementations;

public class LinkedListBasedNodes {
    private Node currentNode;
    private Node firstNode;
    boolean isNull;

    public LinkedListBasedNodes() {
        currentNode = null;
        firstNode = null;
    }

    static class Node {
        int value;
        Node nextNode;
        Node(int nodeValue) {
            this.value = nodeValue;
        }
    }

    public void addNode(int data) {
        Node node = new Node(data);
        if (this.currentNode == null && this.firstNode == null) {
            this.firstNode = node;
            this.currentNode = node;
        } else if (currentNode != null) {
            this.currentNode.nextNode = node;
            currentNode = currentNode.nextNode;
        }
        this.isNull = false;
    }

    public int getFirst() {
        return this.firstNode.value;
    }

    public Node getFirstNode() {
        return this.firstNode;
    }

    public void deleteFirst() {
        if (firstNode == null) {
            currentNode = null;
            isNull = true;
            return;
        }
        if (this.firstNode.nextNode != null) {
            firstNode = this.firstNode.nextNode;
        } else {
            firstNode = null;
            currentNode = null;
            isNull = true;
        }
    }

    public Node getCurrentNode() {
        return this.currentNode;
    }

    public void setCurrentNode(Node currentNode) {
        this.currentNode = currentNode;
    }

    public void merge(LinkedListBasedNodes l2) {
        if (this.getCurrentNode() != null) {
            this.getCurrentNode().nextNode = l2.getFirstNode();
            this.setCurrentNode(l2.getCurrentNode());
        }
    }

    public boolean canMerge(LinkedListBasedNodes l2) {
        return (this.getCurrentNode() != null && l2.getFirstNode() != null
                && this.getCurrentNode().value >= l2.getFirstNode().value);
    }

}