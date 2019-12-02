package dsalgo.implementations;

import java.util.ArrayList;

public class MainClassLinkedList {
    public static void main(String[] args) {

        LinkedListImplementation l1 = new LinkedListImplementation(11);
        l1.insertNodeAtEnd(12);
        l1.insertNodeAtEnd(13);
        l1.insertNodeAtEnd(14);
//        l1.insertNodeAtEnd(15);
//        l1.insertNodeAtEnd(16);
//        l1.insertNodeAtEnd(17);
//        l1.insertNodeAtEnd(18);
//        l1.insertNodeAtEnd(19);
//        l1.insertNodeAtEnd(20);
//        l1.insertNodeAtEnd(21);
//        l1.insertNodeAtEnd(22);
//        l1.displayLinkedList();
//        l1.splitOddEven();
//        l1.checkPallindrome();
//        l1.displayLinkedList();
//        l1.rotateByK(2);
//        l1.alternatingLinkedList();
//        l1.reversePrint(l1.head);
//        System.out.println(l1.getMiddleNodeData());
//        l1.insertNodeAtIndex(300,3);
//        System.out.println(l1.listLength());
//        l1.displayLinkedList();
//        System.out.println(l1.getNodeDataAtIndex(3));
//        System.out.println(l1.getNodeDataFromLast(3));
//        l1.getNodeDataFromLastRec(l1.getHead(),3);
//        l1.reverse();
//        l1.displayLinkedList();
//        l1.reverseRecn(l1.head);
//        l1.pairReverse();
//        l1.displayLinkedList();
//        dsAlgo.implementations.LinkedListImplementation l2 = new dsAlgo.implementations.LinkedListImplementation(101);
//        l2.insertNodeAtEnd(110);
//        l2.insertNodeAtEnd(120);
//        l2.insertNodeAtEnd(130);
//        l2.insertNodeAtEnd(140);
//        l2.insertNodeAtEnd(180);
//        l2.insertNodeAtEnd(190);
//        l2.insertNodeAtEnd(260);
//        l2.displayLinkedList();
//        System.out.println(getMergedNodeData(l1,l2));
//        dsAlgo.implementations.LinkedListImplementation l3 = new dsAlgo.implementations.LinkedListImplementation(1);
//        l3.insertNodeAtEnd(3);
//        l3.insertNodeAtEnd(5);
//        l3.insertNodeAtEnd(7);
//        l3.insertNodeAtEnd(9);
//        l3.insertNodeAtEnd(11);
//        l3.displayLinkedList();
//        dsAlgo.implementations.LinkedListImplementation l4 = new dsAlgo.implementations.LinkedListImplementation(2);
//        l4.insertNodeAtEnd(4);
//        l4.insertNodeAtEnd(6);
//        l4.insertNodeAtEnd(8);
//        l4.insertNodeAtEnd(10);
//        l4.insertNodeAtEnd(12);
//        l4.displayLinkedList();
//        mergeAndSort(l3,l4);

//        l1.deleteNodeAtStart();
//        l1.displayLinkedList();
//        l1.deleteNodeAtEnd();
//        l1.displayLinkedList();
//        l1.deleteNodeAtIndex(2);
//        l1.displayLinkedList();
//        l1.deleteLinkedlist(l1.getHead());
//        l1.displayLinkedList();
//        l1.delete();
//        l1.displayLinkedList();
//        dsAlgo.implementations.LinkedListImplementation l1 = new dsAlgo.implementations.LinkedListImplementation(3);
//        l1.insertNodeAtEnd(4);
//        l1.insertNodeAtEnd(3);
//        l1.displayLinkedList();
//
//        dsAlgo.implementations.LinkedListImplementation l2 = new dsAlgo.implementations.LinkedListImplementation(5);
//        l2.insertNodeAtEnd(6);
//        l2.insertNodeAtEnd(4);
//        l2.displayLinkedList();
//
//        addSingleDigit(l1,l2);

//        dsAlgo.implementations.LinkedListImplementation l4 = new dsAlgo.implementations.LinkedListImplementation(1);
//        l4.insertNodeAtEnd(4);
//        l4.insertNodeAtEnd(3);
//        l4.insertNodeAtEnd(2);
//        l4.insertNodeAtEnd(3);
//        l4.insertNodeAtEnd(4);
//        l4.insertNodeAtEnd(5);
//        l4.insertNodeAtEnd(2);
//        l4.sortByK(3);
//        l4.displayLinkedList();
//        l4.splitOddEven();
//        l4.displayLinkedList();

//        dsAlgo.implementations.LinkedListImplementation l5 = new dsAlgo.implementations.LinkedListImplementation(1);
//        l5.insertNodeAtEnd(2);
//        l5.insertNodeAtEnd(3);
//        l5.insertNodeAtEnd(5);
//        l5.insertNodeAtEnd(8);
//        l5.insertNodeAtEnd(9);
//        l5.insertNodeAtEnd(10);
//        l5.displayLinkedList();
//
//        dsAlgo.implementations.LinkedListImplementation l6 = new dsAlgo.implementations.LinkedListImplementation(0);
//        l6.insertNodeAtEnd(2);
//        l6.insertNodeAtEnd(4);
//        l6.insertNodeAtEnd(5);
//        l6.insertNodeAtEnd(8);
//        l6.insertNodeAtEnd(9);
//        l6.insertNodeAtEnd(10);
//        l6.displayLinkedList();
//
//        findCommon(l5,l6);

//        dsAlgo.implementations.DoublyLinkedListImplementation d1 = new dsAlgo.implementations.DoublyLinkedListImplementation(10);
//        d1.insertNodeAtEnd(11);
//        d1.insertNodeAtEnd(12);
//        d1.insertNodeAtEnd(13);
//        d1.insertNodeAtEnd(14);
//        d1.insertNodeAtEnd(15);
//        d1.insertNodeAtEnd(16);
//        d1.insertNodeAtEnd(17);
//        d1.insertNodeAtEnd(18);
//        d1.displayDoublyLinkedList();
//        d1.displayDoublyLinkedListReverse();
//        d1.insertNodeAtStart(100);
//        d1.displayDoublyLinkedList();
//        d1.displayDoublyLinkedListReverse();
//        d1.insertNodeAtIndex(300,3);
//        d1.displayDoublyLinkedList();
//        d1.displayDoublyLinkedListReverse();
//        System.out.println(d1.getNodeDataAtIndex(3));
//        d1.deleteNodeAtStart();
//        d1.displayDoublyLinkedList();
//        d1.displayDoublyLinkedListReverse();
//        d1.deleteNodeAtEnd();
//        d1.displayDoublyLinkedList();
//        d1.displayDoublyLinkedListReverse();
//        d1.deleteNodeAtIndex(2);
//        d1.displayDoublyLinkedList();
//        d1.displayDoublyLinkedListReverse();

//        dsAlgo.implementations.CircularLinkedListImplementation c1 = new dsAlgo.implementations.CircularLinkedListImplementation(10);
//        c1.insertNodeAtEnd(11);
//        c1.insertNodeAtEnd(12);
//        c1.insertNodeAtEnd(13);
//        c1.insertNodeAtEnd(14);
//        c1.insertNodeAtEnd(15);
//        c1.insertNodeAtStart(100);
//        c1.insertNodeAtIndex(200,3);
//        c1.insertNodeAtEnd(16);
//        c1.insertNodeAtEnd(17);
//        c1.insertNodeAtEnd(18);
//        c1.displayCircularLinkedList();
//        System.out.println(c1.length());
//        while (c1.length() > 1){
//            c1.deleteNodeAtIndex(1);
//            c1.displayCircularLinkedList();
//        }
//        c1.displayCircularLinkedList();
//        c1.deleteNodeAtIndex(3);
//        c1.displayCircularLinkedList();
//        c1.loopChecker();
//        dsAlgo.implementations.CircularLinkedListImplementation c3 = c1.splitCircularList();
//        c1.displayCircularLinkedList();
//        c3.displayCircularLinkedList();
    }

    public static int getMergedNodeData(LinkedListImplementation lis1, LinkedListImplementation lis2) {
        int mergedNData = Integer.MIN_VALUE;
        int lis1Length = lis1.listLength();
        int lis2Length = lis2.listLength();
        int diff = lis1Length - lis2Length;
        ListNode trimmedListHead = null;
        ListNode otherListhead = null;
        if (diff > 0) {
            int counter = 0;
            ListNode currentNode = lis1.head;
            while (currentNode != null) {
                if (counter == diff) {
                    trimmedListHead = currentNode;
                    otherListhead = lis2.head;
                    break;
                }
                currentNode = currentNode.getNextNode();
                counter++;
            }
        } else {
            int counter = 0;
            ListNode currentNode = lis2.head;
            while (currentNode != null) {
                if (counter == diff) {
                    trimmedListHead = currentNode;
                    otherListhead = lis1.head;
                    break;
                }
                currentNode = currentNode.getNextNode();
                counter++;
            }
        }
        while (trimmedListHead != null && otherListhead != null) {
            if (trimmedListHead.getData() == otherListhead.getData()) {
                mergedNData = trimmedListHead.getData();
                break;
            }
            trimmedListHead = trimmedListHead.getNextNode();
            otherListhead = otherListhead.getNextNode();
        }
        return mergedNData;
    }

    public static void mergeAndSort(LinkedListImplementation l1, LinkedListImplementation l2) {
        ArrayList<Integer> a1 = new ArrayList<>();
        ListNode currentNode1 = l1.getHead();
        ListNode currentNode2 = l2.getHead();
        while (currentNode1 != null && currentNode2 != null) {
            if (currentNode1.getData() >= currentNode2.getData()) {
                a1.add(currentNode2.getData());
                currentNode2 = currentNode2.getNextNode();
            } else {
                a1.add(currentNode1.getData());
                currentNode1 = currentNode1.getNextNode();
            }
            if (currentNode1 != null && currentNode2 == null) {
                while (currentNode1 != null) {
                    a1.add(currentNode1.getData());
                    currentNode1 = currentNode1.getNextNode();
                }
            } else if (currentNode2 != null && currentNode1 == null) {
                while (currentNode2 != null) {
                    a1.add(currentNode2.getData());
                    currentNode2 = currentNode2.getNextNode();
                }
            }
        }
        System.out.println(a1);
    }

    public static void addSingleDigit(LinkedListImplementation l1, LinkedListImplementation l2) {
        ListNode l1CurrentNode = l1.head;
        ListNode l2CurrentNode = l2.head;
        LinkedListImplementation updatedL = null;
        int carryover = 0;
        while (l1CurrentNode != null && l2CurrentNode != null) {
            if (l1CurrentNode == l1.head && l2CurrentNode == l2.head) {
                int sum = l1CurrentNode.getData() + l2CurrentNode.getData() + carryover;
                if (sum > 9) {
                    carryover = sum / 10;
                    sum = sum % 10;
                }
                updatedL = new LinkedListImplementation(sum);
            } else {
                int sum = l1CurrentNode.getData() + l2CurrentNode.getData() + carryover;
                if (sum > 9) {
                    carryover = sum / 10;
                    sum = sum % 10;
                }
                updatedL.insertNodeAtEnd(sum);
            }
            l1CurrentNode = l1CurrentNode.getNextNode();
            l2CurrentNode = l2CurrentNode.getNextNode();
        }
        updatedL.displayLinkedList();
    }

    public static void findCommon(LinkedListImplementation l1, LinkedListImplementation l2) {
        ListNode l1Node = l1.head;
        ListNode l2Node = l2.head;
        ArrayList<Integer> a1 = new ArrayList<Integer>();
        while (l1Node != null && l2Node != null) {
            if (l1Node.getData() > l2Node.getData()) {
                l2Node = l2Node.getNextNode();
            } else if (l1Node.getData() < l2Node.getData()) {
                l1Node = l1Node.getNextNode();
            } else {
                a1.add(l1Node.getData());
                l1Node = l1Node.getNextNode();
                l2Node = l2Node.getNextNode();
            }
        }
        System.out.println(a1);
    }
}