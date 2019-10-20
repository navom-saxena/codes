package dsAlgo;


import javax.validation.constraints.NotNull;
import java.util.Stack;

public class MainClassStack {
    public static void main(String[] args) {
//        dsAlgo.ArrayStack s1 = new dsAlgo.ArrayStack(5);
//        s1.displayArrayStack();
//        s1.push(1);
//        s1.push(2);
//        s1.push(3);
//        s1.push(4);
//        s1.push(5);
//        System.out.println(s1.isFullStack());
//        s1.displayArrayStack();
//        s1.pop();
//        System.out.println(s1.isFullStack());
//        s1.pop();
//        System.out.println(s1.top());
//        System.out.println(s1.sizeUsed());
//        s1.displayArrayStack();
//        s1.pop();
//        s1.pop();
//        System.out.println(s1.isEmptyStack());
//        s1.pop();
//        s1.displayArrayStack();
//        System.out.println(s1.isEmptyStack());

//        dsAlgo.DynamicArrayStack d1 = new dsAlgo.DynamicArrayStack(5);
//        d1.displayDynamicArrayStack();
//        d1.push(1);
//        d1.push(2);
//        d1.displayDynamicArrayStack();
//        System.out.println(d1.sizeUsed());
//        System.out.println(d1.totalSize());
//        d1.push(3);
//        d1.push(4);
//        d1.push(5);
//        d1.push(6);
//        d1.displayDynamicArrayStack();
//        System.out.println(d1.sizeUsed());
//        System.out.println(d1.totalSize());
//        d1.pop();
//        d1.pop();
//        d1.pop();
//        d1.pop();
//        d1.pop();
//        d1.pop();
//        d1.displayDynamicArrayStack();
//        System.out.println(d1.sizeUsed());
//        System.out.println(d1.totalSize());
//        d1.symbolsCheck("()(()[()])".toCharArray());
//        int [] arr = {2,5,1,4,6,2,0,2,6,4,1,5,2};
//        d1.getMinimum(arr);
//        d1.checkPallindrome(arr);
//        d1.push(1);
//        d1.push(2);
//        d1.push(3);
//        d1.push(4);
//        d1.push(5);
//        d1.displayDynamicArrayStack();
//        d1.reverseStack(d1);
//        d1.displayDynamicArrayStack();
//        dsAlgo.ArrayWithTwoStacks a1 = new dsAlgo.ArrayWithTwoStacks(6);
//        a1.push(1,10);
//        a1.push(2,20);
//        a1.push(1,11);
//        a1.push(2,21);
//        a1.push(1,12);
//        a1.push(2,22);
//        a1.displayArrayWithTwoStacks();
//        a1.pop(1);
//        a1.pop(2);
//        a1.pop(1);
//        a1.pop(2);
//        a1.pop(1);
//        a1.pop(2);
//        a1.displayArrayWithTwoStacks();
//        dsAlgo.ArrayWithThreeStack at1 = new dsAlgo.ArrayWithThreeStack(9,4);
//        at1.push(1,11);
//        at1.push(2,21);
//        at1.push(3,31);
//        at1.push(1,12);
//        at1.push(1,13);
//        at1.displayArrayWithThreeStacks();
//        at1.push(1,14);
//        at1.displayArrayWithThreeStacks();
//        at1.push(2,22);
//        at1.push(3,32);
//        at1.push(1,13);
//        at1.push(2,23);
//        at1.push(3,33);
//        at1.pop(1);
//        at1.pop(2);
//        at1.pop(3);
//        System.out.println(at1.top(1));
//        System.out.println(at1.top(2));
//        System.out.println(at1.top(3));
//        System.out.println(at1.isEmpty(1));
//        int [] a1 = {1,2,3,4,5,10,11,12,13,14};
//        int [] a2 = {6,7,8,9,10,11,12,13,14};
//        getMergedInt(a1,a2);
//        int [] a3 = {6,3,4,5,2};
//        int [] a4 = {3,2,5,6,1,4,4};
//        dsAlgo.ArrayStack a1 = new dsAlgo.ArrayStack(a4.length);
//        a1.histogramMaxArea(a4);
//        a1.getSpans(a3);
//        int [] arr1 = {3,2,5,6,1,4,4};
//        dsAlgo.ArrayStack a1 = new dsAlgo.ArrayStack(arr1.length);
//        a1.getMaxRectangleArea(arr1);
//        dsAlgo.ArrayStack a1 = new dsAlgo.ArrayStack(5);
//        a1.push(3);
//        a1.push(1);
//        a1.push(5);
//        a1.push(4);
//        a1.push(2);
//        a1.sortStack();
//        dsAlgo.ArrayStack a2 = new dsAlgo.ArrayStack(12);
//        a2.push(1);
//        a2.push(5);
//        a2.push(6);
//        a2.push(8);
//        a2.push(8);
//        a2.push(8);
//        a2.push(0);
//        a2.push(1);
//        a2.push(1);
//        a2.push(0);
//        a2.push(6);
//        a2.push(5);
//        a2.removeAdjacantDuplicatesFromStack();
//        int [] arr = {1,9,6,8,8,8,10,1,1,10,6,5};
//        dsAlgo.ArrayStack a1 = new dsAlgo.ArrayStack(arr.length);
//        a1.removeAdjacantDuplicatesFromArray(arr);
//        String [] arr = {"1","2","3","*","+","5","-"};
//        String [] arr = {"(","2","+","(","3","*","4",")",")"};
//        dsAlgo.ArrayStack a1 = new dsAlgo.ArrayStack(arr.length);
//        a1.postfixEval(arr);
//        a1.infixEval(arr);
//        ArrayStack a1 = new ArrayStack(8);
//        a1.push(6);
//        a1.push(5);
//        a1.push(10);
//        a1.push(11);
//        a1.push(-3);
//        a1.push(-2);
//        a1.push(5);
//        a1.push(4);
//        a1.checkConsecutive();
    }

    public static void getMergedInt(int[] a1, int[] a2) {
        ArrayStack das1 = new ArrayStack(a1.length);
        ArrayStack das2 = new ArrayStack(a2.length);
        for (int i = 0; i < a1.length; i++) {
            das1.push(a1[i]);
        }
        for (int i = 0; i < a2.length; i++) {
            das2.push(a2[i]);
        }
        das1.displayArrayStack();
        das2.displayArrayStack();
        int d = 0;
        int e = 0;
        while (d == 0) {
            d = das1.top() - das2.top();
            if (d != 0) {
                break;
            }
            e = das1.top();
            das1.pop();
            das2.pop();
        }
        System.out.println(e);
    }

    public static void reverseStack(@NotNull Stack<Integer> stack) {
        if (stack.isEmpty()) {
            return;
        }
        int temp = stack.pop();
        reverseStack(stack);
        insertAtBottom(stack, temp);
    }

    public static void insertAtBottom(Stack<Integer> stack, int data) {
        if (stack.isEmpty()) {
            stack.push(data);
            return;
        }
        int temp = stack.pop();
        insertAtBottom(stack, data);
        stack.push(temp);
    }
}
