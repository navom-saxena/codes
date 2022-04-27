package dsalgo.leetcode.explore;

import dsalgo.leetcode.Models.*;

import java.util.*;
import java.util.Arrays;

public class Trees {

    public static void main(String[] args) {

    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/928/

    void preOrder(TreeNode node, List<Integer> res) {
        if (node == null) return;
        res.add(node.val);
        preOrder(node.left, res);
        preOrder(node.right, res);
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        Deque<TreeNode> stack = new ArrayDeque<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            res.add(node.val);
            if (node.right != null) stack.push(node.right);
            if (node.left != null) stack.push(node.left);
        }
        return res;
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/929/

    void inorder(TreeNode node, List<Integer> res) {
        if (node == null) return;
        inorder(node.left, res);
        res.add(node.val);
        inorder(node.right, res);
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode curr = root;
        while (!stack.isEmpty() || curr != null) {
            if (curr != null) {
                stack.push(curr);
                curr = curr.left;
            } else {
                TreeNode node = stack.pop();
                res.add(node.val);
                curr = node.right;
            }
        }
        //inorder(root, res);
        return res;
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/930/

    void postorder(TreeNode node, List<Integer> res) {
        if (node == null) return;
        postorder(node.left, res);
        postorder(node.right, res);
        res.add(node.val);
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        postorder(root, res);
        return res;
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/931/

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Deque<TreeNode> queue = new ArrayDeque<>();

        if (root == null) return res;
        queue.add(root);
        while (!queue.isEmpty()) {

            int n = queue.size();
            List<Integer> levelV = new ArrayList<>();

            for (int i = 0; i < n; i++) {
                TreeNode node = queue.remove();
                levelV.add(node.val);

                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }

            res.add(levelV);
        }

        return res;
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/535/

    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int l = maxDepth(root.left);
        int r = maxDepth(root.right);
        return Math.max(l,r) + 1;
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/536/

    boolean isSymmetricUtil(TreeNode n1, TreeNode n2) {
        if (n1 == null || n2 == null) return n1 == null && n2 == null;
        if (n1.val != n2.val) return false;
        return isSymmetricUtil(n1.left, n2.right) && isSymmetricUtil(n1.right, n2.left);
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isSymmetricUtil(root.left, root.right);
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/537/

    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;
        targetSum -= root.val;
        if (targetSum == 0 && root.left == null && root.right == null) return true;
        return hasPathSum(root.left, targetSum) || hasPathSum(root.right, targetSum);
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/538/

    int count = 0;

    public int countUnivalSubtrees(TreeNode root) {
        countUnivalSubtreesUtil(root);
        return count;
    }

    int countUnivalSubtreesUtil(TreeNode root) {
        if (root == null) return Integer.MIN_VALUE;
        int l = countUnivalSubtreesUtil(root.left);
        int r = countUnivalSubtreesUtil(root.right);
        if (l == Integer.MIN_VALUE) l = root.val;
        if (r == Integer.MIN_VALUE) r = root.val;
        if (l == root.val && r == root.val) {
            count++;
            return root.val;
        } else return Integer.MAX_VALUE;
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/133/conclusion/942/

    int postOrderIndex = 0;

    int linearSearchArr(int [] arr, int s, int e, int x) {
        for (int i = s; i <= e; i++) {
            if (arr[i] == x) return i;
        }
        return -1;
    }

    TreeNode buildTreeUtil(int [] inorder, int [] postorder, int s, int e) {
        if (s > e || postOrderIndex < 0) return null;
        int val = postorder[postOrderIndex];
        postOrderIndex--;
        TreeNode node = new TreeNode(val);
        int i = linearSearchArr(inorder, s, e, val);
        node.right = buildTreeUtil(inorder, postorder, i + 1, e);
        node.left = buildTreeUtil(inorder, postorder, s, i - 1);
        return node;
    }

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        postOrderIndex = postorder.length - 1;
        return buildTreeUtil(inorder, postorder, 0, inorder.length - 1);
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/133/conclusion/943/

    int preOrderIndex = 0;

    TreeNode buildTreeUtilPre(int [] inorder, int [] preorder, int s, int e) {
        if (s > e || preOrderIndex >= preorder.length) return null;
        int val = preorder[preOrderIndex];
        preOrderIndex++;
        TreeNode node = new TreeNode(val);
        int i = linearSearchArr(inorder, s, e, val);
        node.left = buildTreeUtilPre(inorder, preorder, s, i - 1);
        node.right = buildTreeUtilPre(inorder, preorder, i + 1, e);
        return node;
    }

    public TreeNode buildTreePre(int[] inorder, int[] preorder) {
        preOrderIndex = 0;
        return buildTreeUtil(inorder, preorder, 0, inorder.length - 1);
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/133/conclusion/994/

    static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }
    }

    public Node connect(Node root) {
        if (root == null) return null;
        if (root.left != null) root.left.next = root.right;
        if (root.next != null) {
            if (root.right != null) root.right.next = root.next.left;
        }
        connect(root.left);
        connect(root.right);
        return root;
    }

    public Node connect2(Node root) {
        if (root == null) return null;
        if (root.left != null) root.left.next = root.right;
        Node curr = root.next;
        Node c = root.right != null ? root.right : root.left;

        while (c != null && curr != null) {
            Node nextN = curr.left != null ? curr.left : curr.right;
            if (nextN != null) {
                c.next = nextN;
                break;
            }
            curr = curr.next;
        }

        connect2(root.right);
        connect2(root.left);
        return root;
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/133/conclusion/932/

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        if (root == p || root == q) return root;
        TreeNode l = lowestCommonAncestor(root.left, p, q);
        TreeNode r = lowestCommonAncestor(root.right, p, q);
        if (l != null && r != null) return root;
        return l != null ? l : r;
    }

//    https://leetcode.com/explore/learn/card/data-structure-tree/133/conclusion/995/

    public static class Codec {

        // Encodes a tree to a single string.

        void serializeUtils(TreeNode node, StringBuilder sb) {
            if (node == null) {
                sb.append("-");
                sb.append(".");
                return;
            }
            sb.append(node.val);
            sb.append(".");
            serializeUtils(node.left, sb);
            serializeUtils(node.right, sb);
        }


        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            serializeUtils(root, sb);
            return sb.toString();
        }

        TreeNode deserializeUtils(Deque<String> deque) {
            if (deque.isEmpty()) return null;
            String v = deque.pop();
            if (v.equals("-")) return null;
            TreeNode node = new TreeNode(Integer.parseInt(v));
            node.left = deserializeUtils(deque);
            node.right = deserializeUtils(deque);
            return node;
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            String [] values = data.split("\\.");
            Deque<String> deque = new ArrayDeque<>(Arrays.asList(values));
            return deserializeUtils(deque);
        }

    }

}