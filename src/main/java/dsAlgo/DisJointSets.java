package dsAlgo;

import java.util.HashMap;
import java.util.Map;

public class DisJointSets {

    public static Map<Integer, DisJointSetNode> hm = new HashMap<>();

    public static void main(String[] args) {
        DisJointSets djs = new DisJointSets();
        djs.makeSet(1);
        djs.makeSet(2);
        djs.makeSet(3);
        djs.makeSet(4);
        djs.makeSet(5);
        djs.makeSet(6);
        djs.makeSet(7);

        djs.union(1, 2);
        djs.union(2, 3);
        djs.union(4, 5);
        djs.union(6, 7);
        djs.union(5, 6);
        djs.union(3, 7);

        System.out.println(djs.findSet(1));
        System.out.println(djs.findSet(2));
        System.out.println(djs.findSet(3));
        System.out.println(djs.findSet(4));
        System.out.println(djs.findSet(5));
        System.out.println(djs.findSet(6));
        System.out.println(djs.findSet(7));

    }

    public void makeSet(int data) {
        DisJointSetNode node = new DisJointSetNode();
        node.data = data;
        node.parent = node;
        node.rank = 0;
        hm.put(data, node);
    }

    public DisJointSetNode findSet(DisJointSetNode disJointSetNode) {
        if (disJointSetNode.parent == disJointSetNode) {
            return disJointSetNode;
        }
        disJointSetNode.parent = findSet(disJointSetNode.parent);
        return disJointSetNode.parent;
    }

    public int findSet(int data) {
        return findSet(hm.get(data)).data;
    }

    public void union(int data1, int data2) {
        DisJointSetNode node1 = hm.get(data1);
        DisJointSetNode node2 = hm.get(data2);

        DisJointSetNode parent1 = findSet(node1);
        DisJointSetNode parent2 = findSet(node2);

        if (parent1.data == parent2.data) {
            return;
        }

        if (parent1.rank >= parent2.rank) {
            parent1.rank = (parent1.rank == parent2.rank) ? parent1.rank + 1 : parent1.rank;
            parent2.parent = parent1;
        } else {
            parent1.parent = parent2;
        }
    }

    class DisJointSetNode {
        int data;
        DisJointSetNode parent;
        int rank;
    }
}
