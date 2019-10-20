package dsAlgo;

import java.util.ArrayList;

public class CircularLinkedListImplementation {
    ListNode head;

    CircularLinkedListImplementation(int headNodeData) {
        this.head = new ListNode(headNodeData);
    }

    public void insertNodeAtEnd(int nodeData) {
        ListNode newNode = new ListNode(nodeData);
        ListNode currentNode = this.head;
        while (currentNode.getNextNode() != this.head && currentNode.getNextNode() != null) {
            currentNode = currentNode.getNextNode();
        }
        currentNode.setNextNode(newNode);
        currentNode.getNextNode().setNextNode(this.head);
    }

    public void insertNodeAtStart(int nodeData) {
        ListNode newNode = new ListNode(nodeData);
        ListNode origionalHeadNode = this.head;
        ListNode currentNode = this.head;
        while (currentNode.getNextNode() != this.head) {
            currentNode = currentNode.getNextNode();
        }
        currentNode.setNextNode(newNode);
        this.head = newNode;
        this.head.setNextNode(origionalHeadNode);
    }

    public void insertNodeAtIndex(int nodeData, int index) {
        int counter = 0;
        ListNode currentNode = this.head;
        ListNode newNode = new ListNode(nodeData);
        while (currentNode.getNextNode() != this.head) {
            if (counter + 1 == index) {
                ListNode origionalNextNode = currentNode.getNextNode();
                currentNode.setNextNode(newNode);
                currentNode.getNextNode().setNextNode(origionalNextNode);
            }
            currentNode = currentNode.getNextNode();
            counter++;
        }
    }

    public void deleteNodeAtStart() {
        ListNode currentNode = this.head;
        while (currentNode.getNextNode() != null) {
            currentNode = currentNode.getNextNode();
        }
        currentNode.setNextNode(this.head.getNextNode());
    }

    public void deleteNodeAtEnd() {
        ListNode currentNode = this.head;
        while (currentNode.getNextNode().getNextNode() != null) {
            currentNode = currentNode.getNextNode();
        }
        currentNode.setNextNode(this.head);
    }

    public void deleteNodeAtIndex(int index) {
        int counter = 0;
        ListNode currentNode = this.head;
        while (currentNode.getNextNode() != this.head) {
            if (counter + 1 == index) {
                ListNode origionalNextNode = currentNode.getNextNode().getNextNode();
                currentNode.setNextNode(origionalNextNode);
            }
            currentNode = currentNode.getNextNode();
            counter++;
        }
    }

    public void displayCircularLinkedList() {
        ArrayList<Integer> a1 = new ArrayList<Integer>();
        ListNode currentNode = this.head;
        while (currentNode.getNextNode() != this.head) {
            a1.add(currentNode.getData());
            currentNode = currentNode.getNextNode();
        }
        a1.add(currentNode.getData());
        a1.add(currentNode.getNextNode().getData());
        System.out.println(a1);
    }

    public void loopChecker() {
        ListNode fastPointr = this.head;
        ListNode slowPointr = this.head;
        ListNode slowPointrHead = this.head;
        Boolean isLoopExists = false;
        while (fastPointr.getNextNode() != null) {
            fastPointr = fastPointr.getNextNode().getNextNode();
            slowPointr = slowPointr.getNextNode();
            if (fastPointr == slowPointr) {
                System.out.println("fast and slow pointer data same - loop implied");
                isLoopExists = true;
                break;
            }
        }
    }

    public int length() {
        ListNode currentNode = this.head;
        int counter = 1;
        while (currentNode.getNextNode() != this.head) {
            currentNode = currentNode.getNextNode();
            counter++;
        }
        return counter;
    }

    public int findHeadInLoop() {
        ListNode fastPointr = this.head;
        ListNode slowPointr = this.head;
        ListNode slowPointrHead = this.head;
        int loopHeadData = Integer.MIN_VALUE;
        Boolean isLoopExists = false;
        while (fastPointr.getNextNode() != null) {
            fastPointr = fastPointr.getNextNode().getNextNode();
            slowPointr = slowPointr.getNextNode();
            if (fastPointr == slowPointr) {
                System.out.println("fast and slow pointer data same - loop implied");
                isLoopExists = true;
                break;
            }
        }
        if (isLoopExists) {
            slowPointr = this.head;
            while (slowPointr != fastPointr) {
                slowPointr = slowPointr.getNextNode();
                fastPointr = fastPointr.getNextNode();
            }
            loopHeadData = fastPointr.getData();
        }
        return loopHeadData;
    }

    public int lengthOfLoop() {
        ListNode fastPointr = this.head;
        ListNode slowPointr = this.head;
        int length = 1;
        Boolean isLoopExists = false;
        while (fastPointr.getNextNode() != null) {
            fastPointr = fastPointr.getNextNode().getNextNode();
            slowPointr = slowPointr.getNextNode();
            if (fastPointr == slowPointr) {
                System.out.println("fast and slow pointer data same - loop implied");
                isLoopExists = true;
                break;
            }
        }
        if (isLoopExists) {
            while (slowPointr.getNextNode() != slowPointr) {
                length++;
                slowPointr = slowPointr.getNextNode();
            }
        }
        return length;
    }

    public CircularLinkedListImplementation splitCircularList() {
        ListNode circularHeadFastPtr = this.head.getNextNode();
        ListNode circularHeadSlowPtr = this.head;
        ListNode circularHeadSlowestPtr = null;
        while (circularHeadFastPtr.getNextNode() != this.head && circularHeadFastPtr != this.head) {
            if (circularHeadSlowestPtr != null) {
                circularHeadSlowestPtr = circularHeadSlowestPtr.getNextNode();
            }
            if (circularHeadSlowPtr == this.head) {
                circularHeadSlowestPtr = circularHeadSlowPtr;
            }
            circularHeadSlowPtr = circularHeadSlowPtr.getNextNode();
            circularHeadFastPtr = circularHeadFastPtr.getNextNode().getNextNode();
            if (circularHeadFastPtr.getNextNode() == this.head) {
                circularHeadSlowPtr = circularHeadSlowPtr.getNextNode();
                circularHeadSlowestPtr = circularHeadSlowestPtr.getNextNode();
            }
        }
        circularHeadSlowestPtr.setNextNode(this.head);
        CircularLinkedListImplementation c3 = new CircularLinkedListImplementation(0);
        c3.head = circularHeadSlowPtr;
        while (circularHeadSlowPtr.getNextNode() != this.head) {
            circularHeadSlowPtr.setNextNode(circularHeadSlowPtr.getNextNode());
            circularHeadSlowPtr = circularHeadSlowPtr.getNextNode();
        }
        circularHeadSlowPtr.setNextNode(c3.head);
        return c3;
    }
}
