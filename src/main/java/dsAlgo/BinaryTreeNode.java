package dsAlgo;

public class BinaryTreeNode {
    int data;
    BinaryTreeNode leftNode;
    BinaryTreeNode rightNode;

    BinaryTreeNode(int data) {
        this.data = data;
        this.leftNode = null;
        this.rightNode = null;
    }

    public int getData() {
        return this.data;
    }

    public void setData(int data) {
        this.data = data;
    }

    public BinaryTreeNode getLeftNode() {
        return this.leftNode;
    }

    public BinaryTreeNode setLeftNode(BinaryTreeNode leftNode) {
        this.leftNode = leftNode;
        return this;
    }

    public BinaryTreeNode getRightNode() {
        return this.rightNode;
    }

    public BinaryTreeNode setRightNode(BinaryTreeNode rightNode) {
        this.rightNode = rightNode;
        return this;
    }

    public boolean hasLeftNode() {
        if (this.leftNode != null) {
            return true;
        } else {
            return false;
        }
    }

    public boolean hasRightNode() {
        if (this.rightNode != null) {
            return true;
        } else {
            return false;
        }
    }
}