package dsalgo.implementations;

import java.util.ArrayList;
import java.util.HashMap;

public class LinkedListImplementation {
    ListNode head;
    int counter = 0;

    LinkedListImplementation(int data) {
        this.head = new ListNode(data);
    }

    public ListNode getHead() {
        return this.head;
    }

    public int listLength() {
        ListNode currentNode = this.head;
        int counter = 0;
        while (currentNode != null) {
            currentNode = currentNode.getNextNode();
            counter++;
        }
        return counter;
    }

    public void displayLinkedList() {
        ArrayList<Integer> a1 = new ArrayList<>();
        ListNode currentNode = this.head;
        while (currentNode != null) {
            a1.add(currentNode.getData());
            currentNode = currentNode.getNextNode();
        }
        System.out.println(a1);
    }

    public void insertNodeAtStart(int newNodeData) {
        ListNode newnode = new ListNode(newNodeData);
        ListNode origionalFirstNode = this.head;
        this.head = newnode;
        this.head.setNextNode(origionalFirstNode);
    }

    public void insertNodeAtEnd(int newNodeData) {
        ListNode newNode = new ListNode(newNodeData);
        ListNode currentNode = this.head;
        while (currentNode.getNextNode() != null) {
            currentNode = currentNode.getNextNode();
        }
        currentNode.setNextNode(newNode);
    }

    public void insertNodeAtIndex(int newNodeData, int index) {
        int counter = 0;
        ListNode currentNode = this.head;
        ListNode newNode = new ListNode(newNodeData);
        while (currentNode.getNextNode() != null) {
            if (counter + 1 == index) {
                ListNode origionalNextNode = currentNode.getNextNode();
                currentNode.setNextNode(newNode);
                currentNode.getNextNode().setNextNode(origionalNextNode);
            }
            currentNode = currentNode.getNextNode();
            counter++;
        }
    }

    public int getNodeDataAtIndex(int index) {
        ListNode currentNode = head;
        int data = Integer.MIN_VALUE;
        int counter = 0;
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
    }

    public void deleteNodeAtEnd() {
        ListNode currentNode = this.head;
        while (currentNode.getNextNode().getNextNode() != null) {
            currentNode = currentNode.getNextNode();
        }
        currentNode.setNextNode(null);
    }

    public void deleteNodeAtIndex(int index) {
        ListNode currentNode = this.head;
        int counter = 0;
        while (currentNode.getNextNode() != null) {
            if (counter + 1 == index) {
                ListNode origionalNextNode = currentNode.getNextNode().getNextNode();
                currentNode.setNextNode(origionalNextNode);
            }
            currentNode = currentNode.getNextNode();
            counter++;
        }
    }

    public void deleteLinkedlist(ListNode node) {
        if (node == null) return;
        deleteLinkedlist(node.nextNode);
        node.nextNode = null;
    }

    public void delete() {
        this.head = null;
    }

    public int getNodeDataFromLast(int indexFromLast) {
        ListNode currentNode = this.head;
        ListNode currentNodeDelayed = this.head;
        int indexCheckCounter = 0;
        while (currentNode != null) {
            if (indexCheckCounter >= indexFromLast) {
                currentNodeDelayed = currentNodeDelayed.getNextNode();
            }
            currentNode = currentNode.getNextNode();
            indexCheckCounter++;
        }
        return currentNodeDelayed.getData();
    }

    public void getNodeDataFromLastRec(ListNode node, int indexFromLast) {
        if (node != null) {
            getNodeDataFromLastRec(node.getNextNode(), indexFromLast);
            counter++;
            if (indexFromLast == counter) {
                System.out.println(node.getData());
            }
        }
    }

    public void reverse() {
        ListNode currentNode = this.head;
        ListNode previousNode = null;
        while (currentNode != null) {
            ListNode nextNode = currentNode.getNextNode();
            currentNode.setNextNode(previousNode);
            previousNode = currentNode;
            currentNode = nextNode;
            if (currentNode != null) {
                this.head = currentNode;
            }
        }
    }

    public void reverseRecn(ListNode node) {
        ListNode nextNode = node.getNextNode();
        if (nextNode == null) {
            this.head = node;
            return;
        }
        reverseRecn(nextNode);
        nextNode.setNextNode(node);
        node.setNextNode(null);
    }

    public int getMiddleNodeData() {
        ListNode fastPointr = this.head;
        ListNode slowPointr = this.head;
        int counter = 0;
        while (fastPointr.getNextNode() != null) {
            ListNode intermittentNode = fastPointr.getNextNode();
            fastPointr = fastPointr.getNextNode().getNextNode();
            slowPointr = slowPointr.getNextNode();
            if (fastPointr == null) {
                fastPointr = intermittentNode;
            }
        }
        return slowPointr.getData();
    }

    public void reversePrint(ListNode node) {
        if (node != null) {
            reversePrint(node.getNextNode());
            System.out.print(node.getData() + " ");
        }
    }

    public void pairReverse() {
        ListNode currentNode = this.head;
        ListNode temp1 = null;
        ListNode temp2 = null;
        while (currentNode != null && currentNode.getNextNode() != null) {
            currentNode.getData();
            if (temp1 != null) {
                temp1.getData();
                temp1.getNextNode().setNextNode(currentNode.getNextNode());
                temp1.getNextNode().getData();
                temp1.getNextNode().getNextNode().getData();
            }
            temp1 = currentNode.getNextNode();
            currentNode.setNextNode(currentNode.getNextNode().getNextNode());
            temp1.setNextNode(currentNode);
            if (temp2 == null) {
                temp2 = temp1;
            }
            currentNode = currentNode.getNextNode();
        }
        this.head = temp2;
    }

    public boolean checkPallindrome() {
        ListNode currentNode = this.head;
        ListNode slowerNode = this.head;
        ArrayList<Integer> a2 = new ArrayList<Integer>();
        while (currentNode != null && currentNode.getNextNode() != null) {
            currentNode = currentNode.getNextNode().getNextNode();
            slowerNode = slowerNode.getNextNode();
            a2.add(slowerNode.getData());
        }
//        if currentNode = null, check 2nd last element of a2 with last element of a2 and then iterate currentNode
//        and compare 3rd last onwards values with currentNode.getData . if all match, palindrome

        return false;
    }

    public void alternatingLinkedList() {
        ListNode currentNode = this.head.getNextNode();
        LinkedListImplementation altLinkedList = new LinkedListImplementation(this.head.getData());
        DoublyLinkedListNode d1 = null;
        DoublyLinkedListImplementation dl1 = new DoublyLinkedListImplementation(Integer.MIN_VALUE);
        while (currentNode != null && currentNode.getNextNode() != null) {
            d1 = dl1.insertNodeAtEnd(currentNode.getData());
            currentNode = currentNode.getNextNode().getNextNode();
            if (currentNode != null && currentNode.getNextNode() == null) {
                d1 = dl1.insertNodeAtEnd(currentNode.getData());
            }
        }
        dl1.deleteNodeAtStart();
        int counter = 0;
        currentNode = this.head;
        altLinkedList.insertNodeAtEnd(d1.getData());
        while (currentNode != null && currentNode.getNextNode() != null && currentNode.getNextNode().getNextNode() != null) {
            if (counter % 2 == 0) {
                currentNode = currentNode.getNextNode().getNextNode();
                altLinkedList.insertNodeAtEnd(currentNode.getData());
            } else {
                altLinkedList.insertNodeAtEnd(d1.getPrevNode().getData());
                d1 = d1.getPrevNode();
            }
            counter++;
        }
        if (d1.getPrevNode() != null) {
            altLinkedList.insertNodeAtEnd(d1.getPrevNode().getData());
        }
        altLinkedList.displayLinkedList();
    }

    public void rotateByK(int k) {
        ArrayList<Integer> a1 = new ArrayList<Integer>();
        ListNode fastPtr = this.head;
        ListNode slowPtr = this.head;
        int counter = 0;
        while (fastPtr.getNextNode() != null) {
            if (counter >= k) {
                slowPtr = slowPtr.getNextNode();
            }
            fastPtr = fastPtr.getNextNode();
            counter++;
        }
        fastPtr.setNextNode(this.head);
        this.head = slowPtr.getNextNode();
        slowPtr.setNextNode(null);

        ListNode newCurrentNode = this.head;
        while (newCurrentNode != null) {
            a1.add(newCurrentNode.getData());
            newCurrentNode = newCurrentNode.getNextNode();
        }
        System.out.println(a1);
    }

    public void sortByK(int k) {
        ListNode currentNode = this.head;
        int minDiff = 0;
        ListNode tempHeadNode = null;
        ListNode tempNode = null;
        LinkedListImplementation l0 = new LinkedListImplementation(Integer.MIN_VALUE);
        while (currentNode != null && currentNode.getNextNode() != null) {
            if (currentNode.getData() < k) {
                int diff = k - currentNode.getData();
                if (diff > minDiff) {
                    tempNode = currentNode;
                }
            }
            if (currentNode.getNextNode().getData() < k) {
                l0.insertNodeAtEnd(currentNode.getNextNode().getData());
                currentNode.setNextNode(currentNode.getNextNode().getNextNode());
            }
            currentNode = currentNode.getNextNode();
        }
        tempHeadNode = tempNode;
        l0.deleteNodeAtStart();
        ListNode nextToTempNode = null;
        if (tempNode.getNextNode() != null) {
            nextToTempNode = tempNode.getNextNode();
        }
        tempNode.setNextNode(l0.head);
        while (tempNode.getNextNode() != null) {
            tempNode = tempNode.getNextNode();
        }
        tempNode.setNextNode(nextToTempNode);
        while (tempHeadNode != null) {
            System.out.println(tempHeadNode.getData());
            tempHeadNode = tempHeadNode.getNextNode();
        }
    }

    public void removeDuplicates() {
        ListNode currentNode = this.head;
        HashMap<Integer, Boolean> hm = new HashMap<Integer, Boolean>();
        while (currentNode != null && currentNode.getNextNode() != null) {
            if (hm.get(currentNode.getNextNode().getData()) != null) {
                currentNode.setNextNode(currentNode.getNextNode().getNextNode());
            } else {
                hm.put(currentNode.getNextNode().getData(), Boolean.TRUE);
            }
            currentNode = currentNode.getNextNode();
        }
    }

    public void splitOddEven() {
        ListNode evenNode = null;
        ListNode oddNode = null;
        ListNode tempheadOddNode = null;
        ListNode tempheadEvenNode = null;
        ListNode currentNode = this.head;
        while (currentNode != null) {
            if (currentNode.getData() % 2 == 0) {
                if (evenNode == null) {
                    evenNode = new ListNode(currentNode.getData());
                    tempheadEvenNode = evenNode;
                } else {
                    evenNode.setNextNode(new ListNode(currentNode.getData()));
                    evenNode = evenNode.getNextNode();

                }
            } else {
                if (oddNode == null) {
                    oddNode = new ListNode(currentNode.getData());
                    tempheadOddNode = oddNode;
                } else {
                    oddNode.setNextNode(new ListNode(currentNode.getData()));
                    oddNode = oddNode.getNextNode();
                }
            }
            currentNode = currentNode.getNextNode();
        }
        evenNode.setNextNode(tempheadOddNode);
        ListNode c1 = tempheadEvenNode;
        ArrayList<Integer> a1 = new ArrayList<Integer>();
        while (c1 != null) {
            a1.add(c1.getData());
            c1 = c1.getNextNode();
        }
        System.out.println(a1);
    }
}