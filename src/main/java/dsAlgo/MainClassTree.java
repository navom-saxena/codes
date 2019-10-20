package dsAlgo;

import java.io.IOException;

public class MainClassTree {
    public static void main(String[] args) throws IOException {
//        BinaryTreeImplementation bst1 = new BinaryTreeImplementation(10);
//        bst1.insertNodeInBInaryTree(bst1.getRoot(),5);
//        bst1.insertNodeInBInaryTree(bst1.getRoot(), 15);
//        bst1.insertNodeInBInaryTree(bst1.getRoot(), 6);
//        bst1.insertNodeInBInaryTree(bst1.getRoot(), 12);
//        bst1.insertNodeInBInaryTree(bst1.getRoot(), 7);
//        bst1.insertNodeInBInaryTree(bst1.getRoot(), 4);
//        bst1.insertNodeInBInaryTree(bst1.getRoot(), 3);
//        bst1.insertNodeInBInaryTree(bst1.getRoot(), 2);
//        bst1.insertNodeInBInaryTree(bst1.getRoot(), 14);
//        bst1.insertNodeInBInaryTree(bst1.getRoot(), 17);
//        bst1.breathFirstTraversalOrInorder();
//        bst1.depthFirstTraversalOrInorder(bst1.getRoot());
//        bst1.iterativeDepthFirstTraversalOrInorder(bst1.getRoot());
//        bst1.iterativeDepthFirstTraversalOrPostorder(bst1.getRoot());
//        System.out.println(bst1.getHeightRecursive(bst1.getRoot()));
//        System.out.println(bst1.getHeightFromLevelOrder(bst1.getRoot()));
//        bst1.leavesNodeCount(bst1.getRoot());
//        bst1.findWidth(bst1.getRoot());
//        bst1.printPaths(bst1.getRoot(), new ArrayList<>());
//        System.out.println(bst1.findPathFromSum(bst1.getRoot(), 22));
//        bst1.createMirrorTree(bst1.getRoot());
//        bst1.breathFirstTraversalOrInorder();

        BinaryTreeImplementation bst2 = new BinaryTreeImplementation(4);
        bst2.insertNodeInBInaryTree(bst2.getRoot(), 5);
        bst2.insertNodeInBInaryTree(bst2.getRoot(), 15);
        bst2.insertNodeInBInaryTree(bst2.getRoot(), 0);
        bst2.insertNodeInBInaryTree(bst2.getRoot(), 1);
        bst2.insertNodeInBInaryTree(bst2.getRoot(), 7);
        bst2.insertNodeInBInaryTree(bst2.getRoot(), 17);
//        bst2.depthFirstTraversalOrPostorder(bst2.getRoot());
        bst2.breathFirstTraversalOrInorder();
        bst2.deleteMax(bst2.getRoot());
        bst2.breathFirstTraversalOrInorder();
    }

    public boolean checkTreesStructurely(BinaryTreeNode root1, BinaryTreeNode root2) {
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null || root2 == null) {
            return false;
        }
        return checkTreesStructurely(root1.getLeftNode(), root2.getRightNode()) &&
                checkTreesStructurely(root1.getRightNode(), root2.getLeftNode());
    }
}
