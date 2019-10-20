package dsAlgo;

import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.HashMap;

public class DynamicArrayStack {
    int[] dynamicArrayStack = null;
    int size = 0;
    int topElementIndex = -1;

    DynamicArrayStack(int size) {
        this.size = size;
        dynamicArrayStack = new int[this.size];
    }

    public void displayDynamicArrayStack() {
        ArrayList<Integer> a1 = new ArrayList<Integer>();
        for (int i = 0; i < this.size; i++) {
            a1.add(this.dynamicArrayStack[i]);
        }
        System.out.println(a1);
    }

    public int sizeUsed() {
        return this.topElementIndex + 1;
    }

    public int totalSize() {
        return this.size;
    }

    public void push(int data) {
        if (this.topElementIndex + 1 < this.size) {
            if (this.topElementIndex + 2 == this.size) {
                int[] increasedArray = new int[this.size * 2];
                for (int i = 0; i <= this.topElementIndex; i++) {
                    increasedArray[i] = this.dynamicArrayStack[i];
                }
                this.dynamicArrayStack = increasedArray;
                this.size = this.size * 2;
            }
            this.dynamicArrayStack[this.topElementIndex + 1] = data;
            this.topElementIndex++;
        } else {
            throw new StackOverflowError();
        }
    }

    public int pop() {
        int poppedValue = 0;
        if (this.topElementIndex != -1) {
            poppedValue = this.dynamicArrayStack[this.topElementIndex];
            this.dynamicArrayStack[this.topElementIndex] = 0;
            this.topElementIndex--;
            if (this.topElementIndex + 1 < this.size / 2) {
                int[] decreasedArray = new int[this.size / 2];
                for (int i = 0; i <= this.topElementIndex; i++) {
                    decreasedArray[i] = this.dynamicArrayStack[i];
                }
                this.dynamicArrayStack = decreasedArray;
                this.size = this.size / 2;
            }
        } else {
            throw new EmptyStackException();
        }
        return poppedValue;
    }

    public int top() {
        int topValue = 0;
        if (this.topElementIndex >= 0) {
            topValue = this.dynamicArrayStack[topElementIndex];
        } else {
            throw new EmptyStackException();
        }
        return topValue;
    }

    public boolean isEmptyStack() {
        boolean flag = false;
        if (this.topElementIndex == -1) {
            flag = true;
        }
        return flag;
    }

    public boolean isFullStack() {
        boolean flag = false;
        if (this.topElementIndex + 1 == this.size) {
            flag = true;
        }
        return flag;
    }

    public void symbolsCheck(char[] input) {
        boolean flag = false;
        HashMap<Integer, Boolean> openingSymbolsMap = new HashMap<Integer, Boolean>();
        openingSymbolsMap.put((int) '{', Boolean.TRUE);
        openingSymbolsMap.put((int) '(', Boolean.TRUE);
        openingSymbolsMap.put((int) '[', Boolean.TRUE);
        int size = input.length;
        for (int i = 0; i < size; i++) {
            int returnedVal = 0;
            if (openingSymbolsMap.get((int) input[i]) != null) {
                this.push((int) input[i]);
            } else {
                try {
                    returnedVal = this.pop();
                } catch (EmptyStackException e) {
                    flag = false;
                    break;
                }
                if (returnedVal == 123 && (int) input[i] == 125) {
                    flag = true;
                } else if (returnedVal == 91 && (int) input[i] == 93) {
                    flag = true;
                } else if (returnedVal == 40 && (int) input[i] == 41) {
                    flag = true;
                } else {
                    flag = false;
                    break;
                }
            }
        }
        System.out.println(flag);
    }

    public void getMinimum(int[] inputArr) {
        int size = inputArr.length;
        DynamicArrayStack auxStack = new DynamicArrayStack(size);
        for (int i : inputArr) {
            if (auxStack.isEmptyStack()) {
                auxStack.push(i);
                this.push(i);
            } else {
                if (i < auxStack.top()) {
                    auxStack.push(i);
                }
                this.push(i);
            }
        }
        auxStack.displayDynamicArrayStack();
        int poppedValue = auxStack.pop();
        System.out.println(poppedValue);
    }

    public void checkPallindrome(int[] inputArr) {
        boolean flag = false;
        int size = inputArr.length;
        int i = 0;
        while (inputArr[i] != 0) {
            this.push(inputArr[i]);
            i++;
        }
        i++;
        while (i < size) {
            if (inputArr[i] == this.pop()) {
                flag = true;
                i++;
            } else {
                flag = false;
                break;
            }
        }
        System.out.println(flag);
    }

    public void insertAtBottom(DynamicArrayStack stack, int data) {
        if (stack.isEmptyStack()) {
            stack.push(data);
            return;
        }
        int temp = stack.pop();
        insertAtBottom(stack, data);
        stack.push(temp);
    }

    public void reverseStack(DynamicArrayStack stack) {
        if (stack.isEmptyStack()) {
            return;
        }
        int temp = stack.pop();
        reverseStack(stack);
        insertAtBottom(stack, temp);
    }
}
