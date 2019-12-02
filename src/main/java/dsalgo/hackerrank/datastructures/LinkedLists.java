package dsalgo.hackerrank.datastructures;

public class LinkedLists {

    public static void main(String[] args) {
//        LinkedList head = new LinkedList(9);
//        for (int i = 10; i < 13; i++) {
//            head = insertNodeAtTail(head, i);
//        }
//        printLinkedList(head);
//        printLinkedListReverse(head);
//        LinkedList reverseNodeRecursive = reverseLinkedListRecursively(head);
//        printLinkedList(reverseNodeRecursive);
//        LinkedList reverseNode = reverseLinkedList(head);
//        printLinkedList(reverseNode);

//        LinkedList nodeA = new LinkedList(9);
//        for (int i = 10; i < 20; i = i + 2) {
//            nodeA = insertNodeAtTail(nodeA, i);
//        }
//
//        LinkedList nodeB = new LinkedList(8);
//        for (int i = 9; i < 21; i = i + 2) {
//            nodeB = insertNodeAtTail(nodeB, i);
//        }
//
//        printLinkedList(nodeA);
//        printLinkedList(nodeB);
//
//        LinkedList mergedHead = mergeTwoSortedLists(nodeA, nodeB);
//        printLinkedList(mergedHead);

//        System.out.println(getPositionFromTail(head,2));

//        LinkedList head = new LinkedList(3);
//        head = insertNodeAtTail(head, 3);
//        head = insertNodeAtTail(head, 3);
//        head = insertNodeAtTail(head, 4);
//        head = insertNodeAtTail(head, 5);
//        head = insertNodeAtTail(head, 5);
//        head = deDuplicateNodes(head);
//        printLinkedList(head);

//        LinkedList headA = new LinkedList(1);
//        headA = insertNodeAtTail(headA, 2);
//        headA = insertNodeAtTail(headA, 3);
//        headA = insertNodeAtTail(headA, 8);
//        headA = insertNodeAtTail(headA, 10);
//        headA = insertNodeAtTail(headA, 11);
//        LinkedList sameNode = new LinkedList(15);
//        headA = insertNodeObjectAtTail(headA, sameNode);
//        headA = insertNodeAtTail(headA, 18);
//        headA = insertNodeAtTail(headA, 19);
//        LinkedList headB = new LinkedList(1);
//        headB = insertNodeAtTail(headB, 2);
//        headB = insertNodeAtTail(headB, 3);
//        headB = insertNodeObjectAtTail(headB, sameNode);
//        System.out.println(findMergeNode(headB, headA));
    }

    private static int findLength(LinkedList node) {
        int counter = 0;
        if (node == null) {
            return counter;
        }
        while (node != null) {
            counter++;
            node = node.getNextNode();
        }
        return counter;
    }

    private static LinkedList insertNodeAtTail(LinkedList node, int value) {
        if (node == null) {
            return new LinkedList(value);
        }
        LinkedList head = node;
        while (node.getNextNode() != null) {
            node = node.getNextNode();
        }
        LinkedList newNode = new LinkedList(value);
        node.setNextNode(newNode);
        return head;
    }

    private static LinkedList insertNodeObjectAtTail(LinkedList head, LinkedList node) {
        if (head == null) {
            return node;
        }
        LinkedList current = head;
        while (current.getNextNode() != null) {
            current = current.getNextNode();
        }
        current.setNextNode(node);
        return head;
    }

    private static LinkedList insertNodeAtHead(LinkedList node, int value) {
        if (node == null) {
            return new LinkedList(value);
        }
        LinkedList newNode = new LinkedList(value);
        newNode.setNextNode(node);
        return newNode;
    }

    private static LinkedList insertNodeAtPosition(LinkedList node, int value, int position) {
        if (node == null) {
            return new LinkedList(value);
        }
        LinkedList head = node;
        int index = 0;
        while (index != position - 1) {
            node = node.getNextNode();
            index++;
        }
        LinkedList nextNode = node.getNextNode();
        LinkedList newNode = new LinkedList(value);
        node.setNextNode(newNode);
        newNode.setNextNode(nextNode);

        return head;
    }

    private static LinkedList deleteNode(LinkedList node, int position) {
        if (node == null) {
            return null;
        }
        LinkedList head = node;
        int index = 0;
        if (position == 0) {
            return node.getNextNode();
        }
        while (index != position - 1) {
            node = node.getNextNode();
            index++;
        }
        LinkedList nextNodeAfterPosition = node.getNextNode().getNextNode();
        node.setNextNode(nextNodeAfterPosition);

        return head;
    }

    private static LinkedList reverseLinkedListRecursively(LinkedList node) {
        if (node == null || node.getNextNode() == null) {
            return node;
        }
        LinkedList nextNode = node.getNextNode();
        LinkedList nextNodeReverse = reverseLinkedListRecursively(nextNode);
        nextNode.setNextNode(node);
        node.setNextNode(null);
        return nextNodeReverse;
    }

    private static LinkedList reverseLinkedList(LinkedList node) {
        if (node == null) {
            return null;
        }
        if (node.getNextNode() == null) {
            return node;
        }
        LinkedList prevNode = node;
        LinkedList current = node.getNextNode();
        prevNode.setNextNode(null);
        while (current.getNextNode() != null) {
            LinkedList nextNode = current.getNextNode();
            current.setNextNode(prevNode);
            prevNode = current;
            current = nextNode;
        }
        current.setNextNode(prevNode);
        return current;
    }

    private static void printLinkedListReverse(LinkedList node) {
        if (node == null) {
            return;
        }
        LinkedList nextNode = node.getNextNode();
        printLinkedListReverse(nextNode);
        System.out.println(node.getValue());
    }

    private static void printLinkedList(LinkedList node) {
        if (node == null) {
            return;
        }
        while (node != null) {
            System.out.println(node.getValue());
            node = node.getNextNode();
        }
    }

    private static boolean compareList(LinkedList nodeA, LinkedList nodeB) {
        if (nodeA == null && nodeB == null) {
            return true;
        }
        while (nodeA != null && nodeB != null) {
            if (nodeA.getValue() == nodeB.getValue()) {
                nodeA = nodeA.getNextNode();
                nodeB = nodeB.getNextNode();
            } else {
                return false;
            }
        }
        return nodeA == null && nodeB == null;
    }

    private static LinkedList mergeTwoSortedLists(LinkedList nodeA, LinkedList nodeB) {
        LinkedList head = null;
        LinkedList prev = null;
        while (nodeA != null && nodeB != null) {
            if (nodeA.getValue() < nodeB.getValue()) {
                if (head == null) {
                    prev = nodeA;
                    head = prev;
                } else {
                    prev.setNextNode(nodeA);
                    prev = prev.getNextNode();
                }
                nodeA = nodeA.getNextNode();
            } else {
                if (head == null) {
                    prev = nodeB;
                    head = prev;
                } else {
                    prev.setNextNode(nodeB);
                    prev = prev.getNextNode();
                }
                nodeB = nodeB.getNextNode();
            }
        }
        while (nodeA != null) {
            if (head == null) {
                prev = nodeA;
                head = prev;
            } else {
                prev.setNextNode(nodeA);
                prev = prev.getNextNode();
            }
            nodeA = nodeA.getNextNode();
        }
        while (nodeB != null) {
            if (head == null) {
                prev = nodeB;
                head = prev;
            } else {
                prev.setNextNode(nodeB);
                prev = prev.getNextNode();
            }
            nodeB = nodeB.getNextNode();
        }
        return head;
    }

    private static int getPositionFromTail(LinkedList node, int positionFromTail) {
        LinkedList delayedNodePointer = node;
        int indexCounter = 0;
        while (node != null) {
            if (indexCounter > positionFromTail) {
                delayedNodePointer = delayedNodePointer.getNextNode();
            }
            indexCounter++;
            node = node.getNextNode();
        }
        return delayedNodePointer.getValue();
    }

    private static LinkedList deDuplicateNodes(LinkedList node) {
        LinkedList head = node;
        while (node != null) {
            LinkedList tempNode = node;
            while (tempNode.getNextNode() != null && tempNode.getValue() == tempNode.getNextNode().getValue()) {
                tempNode = tempNode.getNextNode();
            }
            node.setNextNode(tempNode.getNextNode());
            node = node.getNextNode();
        }
        return head;
    }

    private static int findMergeNode(LinkedList nodeA, LinkedList nodeB) {
        int lengthA = findLength(nodeA);
        int lengthB = findLength(nodeB);
        int diff = lengthA - lengthB;
        if (nodeA == null || nodeB == null) {
            return Integer.MIN_VALUE;
        }
        if (diff > 0) {
            int tempCounter = diff;
            while (tempCounter != 0 && nodeA != null) {
                nodeA = nodeA.getNextNode();
                tempCounter--;
            }
        } else if (diff < 0) {
            int tempCounter = diff;
            while (tempCounter != 0 && nodeB != null) {
                nodeB = nodeB.getNextNode();
                tempCounter++;
            }
        }
        while (nodeA != null && nodeB != null) {
            if (nodeA == nodeB) {
                return nodeA.getValue();
            }
            nodeA = nodeA.getNextNode();
            nodeB = nodeB.getNextNode();
        }
        return Integer.MIN_VALUE;
    }

    private static boolean cycleDetection(LinkedList node) {
        if (node == null) {
            return false;
        }
        LinkedList slowPointer = node;
        LinkedList fastPointer = node;
        while (slowPointer != null && fastPointer != null && fastPointer.getNextNode() != null) {
            slowPointer = slowPointer.getNextNode();
            fastPointer = fastPointer.getNextNode().getNextNode();
            if (slowPointer == fastPointer) {
                return true;
            }
        }
        return false;
    }

    static class LinkedList {
        int value;
        LinkedList nextNode;

        LinkedList() {
        }

        LinkedList(int value) {
            this.value = value;
        }

        LinkedList getNextNode() {
            return nextNode;
        }

        void setNextNode(LinkedList nextNode) {
            this.nextNode = nextNode;
        }

        int getValue() {
            return value;
        }

        void setValue(int value) {
            this.value = value;
        }
    }
}
