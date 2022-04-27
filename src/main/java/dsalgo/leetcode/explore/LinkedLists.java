package dsalgo.leetcode.explore;

import dsalgo.leetcode.Models.*;

public class LinkedLists {

    public static void main(String[] args) {

    }

//    https://leetcode.com/explore/learn/card/linked-list/209/singly-linked-list/1290/

    static class Node {

        int val;
        Node next;
        Node prev;
        Node child;

        Node(int val) {
            this.val = val;
        }
    }

    static class MyLinkedList {

        Node head;
        Node tail;
        int n;

        public MyLinkedList() {
            head = new Node(Integer.MIN_VALUE);
            tail = new Node(Integer.MAX_VALUE);
            head.next = tail;
            tail.prev = head;
            n = 0;
        }

        public int get(int index) {
            int c = -1;
            Node curr = head;
            while (curr != null && c != index) {
                curr = curr.next;
                c++;
            }
            if (curr == null || c >= n) return -1;
            return curr.val;
        }

        public void addAtHead(int val) {
            Node temp = head.next;
            Node newNode = new Node(val);
            head.next = newNode;
            newNode.prev = head;
            newNode.next = temp;
            temp.prev = newNode;
            n++;
        }

        public void addAtTail(int val) {
            Node temp = tail.prev;
            Node newNode = new Node(val);
            tail.prev = newNode;
            newNode.next = tail;
            newNode.prev = temp;
            temp.next = newNode;
            n++;
        }

        public void addAtIndex(int index, int val) {
            if (index > n) return;
            index--;
            int c = -1;
            Node curr = head;
            while (curr != null && c != index) {
                curr = curr.next;
                c++;
            }
            if (curr == null) return;
            Node temp = curr.next;
            Node newNode = new Node(val);
            curr.next = newNode;
            newNode.prev = curr;
            newNode.next = temp;
            temp.prev = newNode;
            n++;
        }

        public void deleteAtIndex(int index) {
            if (index >= n) return;
            index--;
            int c = -1;
            Node curr = head;
            while (curr != null && c != index) {
                curr = curr.next;
                c++;
            }
            if (curr == null) return;
            curr.next = curr.next.next;
            curr.next.prev = curr;
            n--;
        }
    }

//    https://leetcode.com/explore/learn/card/linked-list/214/two-pointer-technique/1212/

    public boolean hasCycle(ListNode head) {
        ListNode fastP = head;
        ListNode slowP = head;

        while (fastP != null && fastP.next != null) {
            fastP = fastP.next.next;
            slowP = slowP.next;

            if (slowP == fastP) return true;
        }
        return false;
    }

//    https://leetcode.com/explore/learn/card/linked-list/214/two-pointer-technique/1214/

    public ListNode detectCycle(ListNode head) {
        ListNode fastP = head;
        ListNode slowP = head;

        while (fastP != null && fastP.next != null) {
            fastP = fastP.next.next;
            slowP = slowP.next;

            if (slowP == fastP) {
                ListNode p = head;
                while (p != slowP) {
                    p = p.next;
                    slowP = slowP.next;
                }
                return slowP;
            }
        }
        return null;
    }

//    https://leetcode.com/explore/learn/card/linked-list/214/two-pointer-technique/1215/

    int getLength(ListNode head) {
        int l = 0;
        while (head != null) {
            head = head.next;
            l++;
        }
        return l;
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int lenA = getLength(headA);
        int lenB = getLength(headB);

        if (lenA > lenB) {
            int d = lenA - lenB;
            while (d > 0) {
                headA = headA.next;
                d--;
            }
        } else {
            int d = lenB - lenA;
            while (d > 0) {
                headB = headB.next;
                d--;
            }
        }

        while (headA != headB) {
            headA = headA.next;
            headB = headB.next;
        }
        return headA;
    }

//    https://leetcode.com/explore/learn/card/linked-list/214/two-pointer-technique/1296/

    public ListNode removeNthFromEnd(ListNode head, int n) {
       ListNode sentinel = new ListNode(Integer.MIN_VALUE);
       sentinel.next = head;

       ListNode fastP = sentinel;
       ListNode slowP = sentinel;

       int i = 0;
       while (fastP != null) {

           if (i > n) slowP = slowP.next;
           fastP = fastP.next;
           i++;
       }

       if (slowP != null) slowP.next = slowP.next.next;
       return sentinel.next;
    }

//    https://leetcode.com/explore/learn/card/linked-list/219/classic-problems/1205/

    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode reversed = reverseList(head.next);
        ListNode nextN = head.next;
        head.next = null;
        nextN.next = head;

        return reversed;
    }

//    https://leetcode.com/explore/learn/card/linked-list/219/classic-problems/1207/

    public ListNode removeElements(ListNode head, int val) {
        ListNode sentinel = new ListNode(Integer.MIN_VALUE);
        sentinel.next = head;

        ListNode prev = sentinel;
        ListNode curr = head;

        while (curr!= null) {
            if (curr.val != val) {
                prev.next = curr;
                prev = prev.next;
            }
            curr = curr.next;
        }
        prev.next = null;

        return sentinel.next;
    }

//    https://leetcode.com/explore/learn/card/linked-list/219/classic-problems/1208/

    public ListNode oddEvenList(ListNode head) {
        ListNode sentinelOdd = new ListNode(Integer.MIN_VALUE);
        ListNode sentinelEven = new ListNode(Integer.MIN_VALUE);

        ListNode curr = head;
        ListNode currOdd = sentinelOdd;
        ListNode currEven = sentinelEven;

        int c = 1;
        while (curr != null) {
            ListNode nextN = curr.next;
            curr.next = null;

            if (c % 2 == 0) {
                currEven.next = curr;
                currEven = currEven.next;
            } else {
                currOdd.next = curr;
                currOdd = currOdd.next;
            }
            c++;
            curr = nextN;
        }

        currOdd.next = sentinelEven.next;
        return sentinelOdd.next;
    }

//    https://leetcode.com/explore/learn/card/linked-list/219/classic-problems/1209/

    public boolean isPalindrome(ListNode head) {
        ListNode fastP = head;
        ListNode slowP = head;

        while (fastP != null && fastP.next != null) {
            fastP = fastP.next.next;
            slowP = slowP.next;
        }

        if (fastP != null) slowP = slowP.next;
        ListNode reversed = reverseList(slowP);

        while (reversed != null && head != null) {
            if (reversed.val != head.val) return false;
            reversed = reversed.next;
            head = head.next;
        }

        return true;
    }

//    https://leetcode.com/explore/learn/card/linked-list/210/doubly-linked-list/1294/

    static class MyDLinkedList {

        Node head;
        Node tail;
        int n;

        public MyDLinkedList() {
            head = new Node(Integer.MIN_VALUE);
            tail = new Node(Integer.MIN_VALUE);
            head.next = tail;
            tail.prev = head;
            n = 0;
        }

        public int get(int index) {
            int c = -1;
            Node curr = head;

            while (curr != null && c != index) {
                curr = curr.next;
                c++;
            }
            if (curr == null || c >= n) return -1;
            return curr.val;
        }

        public void addAtHead(int val) {
            Node newNode = new Node(val);
            Node temp = head.next;
            head.next = newNode;
            newNode.prev = head;
            newNode.next = temp;
            temp.prev = newNode;
            n++;
        }

        public void addAtTail(int val) {
            Node newNode = new Node(val);
            Node temp = tail.prev;
            tail.prev = newNode;
            newNode.next = tail;
            newNode.prev = temp;
            temp.next = newNode;
            n++;
        }

        public void addAtIndex(int index, int val) {
            if (index > n) return;
            index--;
            int c = -1;
            Node curr = head;

            while (curr != null && c != index) {
                curr = curr.next;
                c++;
            }

            if (curr != null) {
                Node temp = curr.next;
                Node newNode = new Node(val);
                curr.next = newNode;
                newNode.prev = curr;
                newNode.next = temp;
                temp.prev = newNode;
                n++;
            }
        }

        public void deleteAtIndex(int index) {
            if (index >= n) return;
            index--;
            int c = -1;
            Node curr = head;

            while (curr != null && c != index) {
                curr = curr.next;
                int y = curr.val;
                c++;
            }

            if (curr != null) {
                curr.next = curr.next.next;
                curr.next.prev = curr;
                n--;
            }
        }

    }

//    https://leetcode.com/explore/learn/card/linked-list/213/conclusion/1225/

    Node flattenUtil(Node node) {
        Node curr = node;
        Node prev = node;

        while (curr != null) {
            if (curr.child != null) {
                Node nextN = curr.next;
                Node last = flattenUtil(curr.child);
                curr.next = curr.child;
                curr.next.prev = curr;
                curr.child = null;
                last.next = nextN;
                if (nextN != null) nextN.prev = last;
                prev = last;
                curr = nextN;
            } else {
                prev = curr;
                curr = curr.next;
            }
        }

        return prev;
    }

    public Node flatten(Node head) {
        flattenUtil(head);
        return head;
    }

//    https://leetcode.com/explore/learn/card/linked-list/213/conclusion/1226/

    public Node insert(Node head, int insertVal) {
        if (head == null) {
            Node newN = new Node(insertVal);
            newN.next = newN;
            return newN;
        }

        Node prev = null;
        Node curr = head.next;
        Node newN = new Node(insertVal);
        Node greatest = head;

        while (prev != head) {

            if (prev == null) prev = head;

            if (prev.val <= insertVal && curr.val > insertVal) {
                prev.next = newN;
                newN.next = curr;
                return head;
            }

            if (greatest.val <= curr.val) greatest = prev;
            prev = curr;
            curr = curr.next;
        }

        Node temp = greatest.next;
        greatest.next = newN;
        newN.next = temp;
        return head;
    }

//    https://leetcode.com/explore/learn/card/linked-list/213/conclusion/1295/

    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return null;
        int l = getLength(head);
        int m = k % l;

        ListNode slowP = head;
        ListNode fastP = head;
        int i = 0;

        while (fastP.next != null) {
            fastP = fastP.next;
            if (i >= m) slowP = slowP.next;
            i++;
        }
        if (slowP == null || slowP.next == null) return head;
        ListNode nextN = slowP.next;
        slowP.next = null;
        fastP.next = head;
        return nextN;
    }

}