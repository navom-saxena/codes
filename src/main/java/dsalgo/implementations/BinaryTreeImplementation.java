package dsalgo.implementations;

import java.util.*;

public class BinaryTreeImplementation {
    public int daimeter = 0;
    BinaryTreeNode rootbinaryTreeNode;

    BinaryTreeImplementation(int data) {
        this.rootbinaryTreeNode = new BinaryTreeNode(data);
    }

    public BinaryTreeNode getRoot() {
        return this.rootbinaryTreeNode;
    }

    public BinaryTreeNode insertNodeInBInaryTree(BinaryTreeNode root, int data) {
        if (root == null) {
            root = new BinaryTreeNode(data);
        } else if (data < root.getData()) {
            root.setLeftNode(insertNodeInBInaryTree(root.getLeftNode(), data));
        } else if (data > root.getData()) {
            root.setRightNode(insertNodeInBInaryTree(root.getRightNode(), data));
        }
        return root;
    }

    public void breathFirstTraversalOrInorder() {
        ArrayList<Integer> a1 = new ArrayList<>();
        int max = Integer.MIN_VALUE;
        BinaryTreeNode root = this.rootbinaryTreeNode;
        Queue<BinaryTreeNode> binaryTreeNodeQueue = new LinkedList<>();
        binaryTreeNodeQueue.add(root);
        while (!binaryTreeNodeQueue.isEmpty()) {
            BinaryTreeNode dequeuedNode = binaryTreeNodeQueue.remove();
            a1.add(dequeuedNode.getData());
            if (dequeuedNode.getData() > max) {
                max = dequeuedNode.getData();
            }
            BinaryTreeNode leftChild = dequeuedNode.getLeftNode();
            BinaryTreeNode rightNode = dequeuedNode.getRightNode();
            if (leftChild != null) {
                binaryTreeNodeQueue.add(leftChild);
            }
            if (rightNode != null)
                binaryTreeNodeQueue.add(rightNode);
        }
        System.out.println(a1);
        System.out.println("max is " + max);
    }

    public void depthFirstTraversalOrPreorder(BinaryTreeNode root) {
        if (root == null) {
            return;
        }
        System.out.println(root.getData());
        depthFirstTraversalOrPreorder(root.getLeftNode());
        depthFirstTraversalOrPreorder(root.getRightNode());
    }

    public void depthFirstTraversalOrInorder(BinaryTreeNode root) {
        if (root == null) {
            return;
        }
        depthFirstTraversalOrInorder(root.getLeftNode());
        System.out.println(root.getData());
        depthFirstTraversalOrInorder(root.getRightNode());
    }

    public void depthFirstTraversalOrPostorder(BinaryTreeNode root) {
        if (root == null) {
            return;
        }
        depthFirstTraversalOrPostorder(root.getLeftNode());
        depthFirstTraversalOrPostorder(root.getRightNode());
        System.out.println(root.getData());
    }

    public void iterativeDepthFirstTraversalOrPreorder(BinaryTreeNode root) {
        List<Integer> a1 = new ArrayList<>();
        Stack<BinaryTreeNode> s1 = new Stack<>();
        s1.push(root);
        while (!s1.isEmpty()) {
            BinaryTreeNode temp = s1.pop();
            a1.add(temp.getData());
            if (temp.hasRightNode()) {
                s1.push(temp.getRightNode());
            }
            if (temp.hasLeftNode()) {
                s1.push(temp.getLeftNode());
            }
        }
        System.out.println(a1);
    }

    public void iterativeDepthFirstTraversalOrInorder(BinaryTreeNode root) {
        List<Integer> a1 = new ArrayList<>();
        Stack<BinaryTreeNode> s1 = new Stack<>();
        BinaryTreeNode currentNode = root;
        boolean done = false;
        while (!done) {
            if (currentNode != null) {
                s1.push(currentNode);
                currentNode = currentNode.getLeftNode();
            } else {
                if (s1.isEmpty()) {
                    done = true;
                } else {
                    currentNode = s1.pop();
                    a1.add(currentNode.getData());
                    currentNode = currentNode.getRightNode();
                }
            }
        }
        System.out.println(a1);
    }

    public void iterativeDepthFirstTraversalOrPostorder(BinaryTreeNode root) {
        List<Integer> a1 = new ArrayList<>();
        Stack<BinaryTreeNode> s1 = new Stack<>();
        s1.push(root);
        BinaryTreeNode previous = null;
        int maxDepth = 0;
        while (!s1.isEmpty()) {
            BinaryTreeNode current = s1.peek();
            if (previous == null || previous.getLeftNode() == current || previous.getRightNode() == current) {
                if (current.hasLeftNode()) {
                    s1.push(current.getLeftNode());
                } else if (current.hasRightNode()) {
                    s1.push(current.getRightNode());
                }
            } else if (current.getLeftNode() == previous) {
                if (current.hasRightNode()) {
                    s1.push(current.getRightNode());
                }
            } else {
                a1.add(current.getData());
                s1.pop();
            }
            previous = current;
            if (s1.size() > maxDepth) {
                maxDepth = s1.size();
            }
        }
        System.out.println("height = " + maxDepth);
        System.out.println(a1);
    }

    public int getHeightRecursive(BinaryTreeNode rootbinaryTreeNode) {
        if (rootbinaryTreeNode == null) {
            return 0;
        }
        int leftSubTreeHeight = getHeightRecursive(rootbinaryTreeNode.getLeftNode());
        int rightSubTreeHeight = getHeightRecursive(rootbinaryTreeNode.getRightNode());

        return (leftSubTreeHeight > rightSubTreeHeight) ? leftSubTreeHeight + 1 : rightSubTreeHeight + 1;
    }

    public int getHeightFromLevelOrder(BinaryTreeNode binaryTreeNode) {
        Queue<BinaryTreeNode> binaryTreeNodeQueue = new LinkedList<>();
        binaryTreeNodeQueue.add(binaryTreeNode);
        binaryTreeNodeQueue.add(null);
        int count = 1;
        while (!binaryTreeNodeQueue.isEmpty()) {
            BinaryTreeNode currentNode = binaryTreeNodeQueue.remove();
            if (currentNode != null) {
                if (currentNode.getLeftNode() == null && currentNode.getRightNode() == null) {
                    // return count from here for minimum height
                }
                if (currentNode.hasLeftNode()) {
                    binaryTreeNodeQueue.add(currentNode.getLeftNode());
                }
                if (currentNode.hasRightNode()) {
                    binaryTreeNodeQueue.add(currentNode.getRightNode());
                }
            } else {
                if (!binaryTreeNodeQueue.isEmpty()) {
                    count++;
                    binaryTreeNodeQueue.add(null);
                }
            }
        }
        return count;
    }

    public void deleteLevelOrder(BinaryTreeNode binaryTreeNode) {
        Queue<BinaryTreeNode> binaryTreeNodeQueue = new LinkedList<>();
        binaryTreeNodeQueue.add(binaryTreeNode);
        while (!binaryTreeNodeQueue.isEmpty()) {
            BinaryTreeNode currentNode = binaryTreeNodeQueue.remove();
        }
    }

    public void leavesNodeCount(BinaryTreeNode binaryTreeNode) {
        Queue<BinaryTreeNode> q1 = new LinkedList<>();
        int count = 0;
        q1.add(binaryTreeNode);
        while (!q1.isEmpty()) {
            BinaryTreeNode currentNode = q1.remove();
            if (currentNode.hasLeftNode()) {
                q1.add(currentNode.getLeftNode());
            }
            if (currentNode.hasRightNode()) {
                q1.add(currentNode.getRightNode());
            }
            if (currentNode.getRightNode() == null && currentNode.getLeftNode() == null) {
                count++;
            }
        }
        System.out.println("No of leaves node are - " + count);
    }

    public int findDaimeter(BinaryTreeNode rootbinaryTreeNode) {
        int left, right;
        if (rootbinaryTreeNode == null) {
            return 0;
        }
        left = findDaimeter(rootbinaryTreeNode.getLeftNode());
        right = findDaimeter(rootbinaryTreeNode.getRightNode());
        if (left + right > daimeter) {
            daimeter = left + right;
        }
        return Math.max(right, left) + 1;
    }

    public void findWidth(BinaryTreeNode binaryTreeNode) {
        Queue<BinaryTreeNode> q1 = new LinkedList<>();
        q1.add(binaryTreeNode);
        q1.add(null);
        int max = 0;
        int count = 0;
        while (!q1.isEmpty()) {
            BinaryTreeNode currentNode = q1.remove();
            if (currentNode != null) {
                if (currentNode.hasLeftNode()) {
                    q1.add(currentNode.getLeftNode());
                    count++;
                }
                if (currentNode.hasRightNode()) {
                    q1.add(currentNode.getRightNode());
                    count++;
                }
            } else {
                if (!q1.isEmpty()) {
                    q1.add(null);
                }
                if (count > max) {
                    max = count;
                }
                count = 0;
            }
        }
        System.out.println("width - " + max);
    }

    public void printPaths(BinaryTreeNode binaryTreeNode, ArrayList<Integer> integerList) {
        if (binaryTreeNode == null) {
            return;
        }
        integerList.add(binaryTreeNode.getData());
        if (binaryTreeNode.getLeftNode() == null && binaryTreeNode.getRightNode() == null) {
            System.out.println(integerList);
        } else {
            printPaths(binaryTreeNode.getLeftNode(), integerList);
            printPaths(binaryTreeNode.getRightNode(), integerList);
        }
    }

    public boolean findPathFromSum(BinaryTreeNode binaryTreeNode, int sum) {
        if (binaryTreeNode == null) {
            return false;
        } else if (binaryTreeNode.getData() == sum) {
            return true;
        } else {
            return findPathFromSum(binaryTreeNode.getLeftNode(), sum - binaryTreeNode.getData()) || findPathFromSum(binaryTreeNode.getRightNode(), sum - binaryTreeNode.getData());
        }
    }

    public BinaryTreeNode createMirrorTree(BinaryTreeNode binaryTreeNode) {
        BinaryTreeNode temp = null;
        if (binaryTreeNode != null) {
            createMirrorTree(binaryTreeNode.getLeftNode());
            createMirrorTree(binaryTreeNode.getRightNode());
            temp = binaryTreeNode.getLeftNode();
            binaryTreeNode.setLeftNode(binaryTreeNode.getRightNode());
            binaryTreeNode.setRightNode(temp);
        }
        return binaryTreeNode;
    }

    public boolean areMirrors(BinaryTreeNode binaryTreeNode1, BinaryTreeNode binaryTreeNode2) {
        if (binaryTreeNode1 == null && binaryTreeNode2 == null) {
            return true;
        }
        if (binaryTreeNode1 == null || binaryTreeNode2 == null) {
            return false;
        }
        if (binaryTreeNode1.getData() != binaryTreeNode2.getData()) {
            return false;
        } else {
            return areMirrors(binaryTreeNode1.getLeftNode(), binaryTreeNode2.getRightNode()) && areMirrors(binaryTreeNode1.getRightNode(), binaryTreeNode2.getLeftNode());
        }
    }

    public BinaryTreeNode deleteMax(BinaryTreeNode root) {
        if (root.getRightNode() == null) {
            return null;
        }
        return root.setRightNode(deleteMax(root.getRightNode()));
    }
}
