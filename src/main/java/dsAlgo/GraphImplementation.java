package dsAlgo;

import javafx.util.Pair;

import java.util.*;

public class GraphImplementation {

    private static boolean[][] visitedNodesAdjacencyMatrix;
    private static boolean[] visitedNodesAdjacencyList;

    private static void depthFirstTraversalAdjacencyMatrix(int i, int[][] adjacencyMatrix) {
        int[] nodeRow = adjacencyMatrix[i];
        for (int j = 0; j < nodeRow.length; j++) {
            if (nodeRow[j] == 1 && !visitedNodesAdjacencyMatrix[i][j]) {
                System.out.println("traversed node - " + adjacencyMatrix[i][j]);
                visitedNodesAdjacencyMatrix[i][j] = true;
                depthFirstTraversalAdjacencyMatrix(j, adjacencyMatrix);
            }
        }
    }

    private static void depthFirstTraversalAdjacencyList(int i, List<LinkedList<Integer>> adjacencyList) {
        LinkedList<Integer> edgesList = adjacencyList.get(i);
        for (int j = 0; j < edgesList.size(); j++) {
            if (!visitedNodesAdjacencyList[j]) {
                System.out.println("traversed node - " + edgesList.get(j));
                visitedNodesAdjacencyList[edgesList.get(j)] = true;
                depthFirstTraversalAdjacencyList(j, adjacencyList);
            }
        }
    }

    private static void breadthFirstTraversalAdjacencyMatrix(int[][] adjacencyMatrix) {
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < adjacencyMatrix[0].length; i++) {
            visitedNodesAdjacencyMatrix[0][i] = true;
            queue.add(i);
        }
        while (!queue.isEmpty()) {
            int node = queue.remove();
            System.out.println("traversed node - " + node);
            for (int j = 0; j < adjacencyMatrix[node].length; j++) {
                if (!visitedNodesAdjacencyMatrix[node][j]) {
                    visitedNodesAdjacencyMatrix[node][j] = true;
                    queue.add(j);
                }
            }
        }
    }

    private static void breadthFirstTraversalAdjacencyList(List<LinkedList<Integer>> adjacencyList) {
        Queue<Integer> queue = new LinkedList<>(adjacencyList.get(0));
        visitedNodesAdjacencyList[0] = true;
        while (!queue.isEmpty()) {
            int node = queue.remove();
            System.out.println("traversed node - " + node);
            visitedNodesAdjacencyList[node] = true;
            LinkedList<Integer> neighbourNodes = adjacencyList.get(node);
            for (Integer individualNeighbour : neighbourNodes) {
                if (!visitedNodesAdjacencyList[individualNeighbour]) {
                    queue.add(individualNeighbour);
                }
            }
        }
    }

    private static int[] shortestDistanceBFS(int startNode, List<LinkedList<Integer>> adjacencyList) {
        int[] distanceArray = new int[adjacencyList.size()];
        Queue<Integer> queue = new LinkedList<>();
        queue.add(startNode);
        distanceArray[startNode] = 0;
        visitedNodesAdjacencyList[startNode] = true;
        while (!queue.isEmpty()) {
            int node = queue.remove();
            System.out.println("traversed node - " + node);
            visitedNodesAdjacencyList[node] = true;
            LinkedList<Integer> neighbourNodes = adjacencyList.get(node);
            for (Integer individualNeighbour : neighbourNodes) {
                if (!visitedNodesAdjacencyList[individualNeighbour]) {
                    distanceArray[individualNeighbour] = distanceArray[node] + 1;
                    queue.add(individualNeighbour);
                }
            }
        }
        return distanceArray;
    }

    private static int getMinimumFromArray(int[] distanceArray) {
        int min = Integer.MAX_VALUE;
        for (int value : distanceArray) {
            if (value < min) {
                min = value;
            }
        }
        return min;
    }

    private static int getMinimumFromArray(int[] distanceArray, boolean[] visitedArray) {
        int min = Integer.MAX_VALUE;
        for (int value : distanceArray) {
            if (value < min && !visitedArray[value]) {
                min = value;
            }
        }
        return min;
    }

    private static Pair<Integer, Integer> getMinimumEdgeVerticesFromMatrix(int[][] adjacencyMatrix, boolean[] visitedArray) {
        int min = Integer.MAX_VALUE;
        int count = 0;
        int node = Integer.MAX_VALUE;
        int adjacentNode = Integer.MAX_VALUE;
        for (int[] adjacentNodesArray : adjacencyMatrix) {
            int adjacentNodeCount = 0;
            for (int value : adjacentNodesArray) {
                if (value < min && !visitedArray[count]) {
                    min = value;
                    node = count;
                    adjacentNode = adjacentNodeCount;

                }
                adjacentNodeCount++;
            }
            visitedArray[count] = true;
            count++;
        }
        return new Pair<>(node, adjacentNode);
    }

    public static int[] dijsktra(int startNode, int[][] adjacencyMatrix) {
        int[] distanceArray = new int[adjacencyMatrix.length];
        boolean[] visitedArray = new boolean[adjacencyMatrix.length];
        for (int i = 0; i < distanceArray.length; i++) {
            distanceArray[i] = Integer.MAX_VALUE;
            visitedArray[i] = false;
        }
        distanceArray[startNode] = 0;

        for (int vertex = 0; vertex < adjacencyMatrix.length; vertex++) {
            int minimumDistanceNode = getMinimumFromArray(distanceArray);
            visitedArray[minimumDistanceNode] = true;
            for (int adjacentNodes = 0; adjacentNodes < adjacencyMatrix.length; adjacentNodes++) {
                if (!visitedArray[vertex] && adjacencyMatrix[minimumDistanceNode][vertex] != 0
                        && distanceArray[adjacentNodes] != Integer.MAX_VALUE
                        && adjacencyMatrix[minimumDistanceNode][adjacentNodes]
                        + distanceArray[minimumDistanceNode] < distanceArray[vertex]) {
                    distanceArray[vertex] = adjacencyMatrix[minimumDistanceNode][adjacentNodes]
                            + distanceArray[minimumDistanceNode];
                }
            }
        }
        return distanceArray;
    }

    public static int[] bellmanFord(int startNode, int[][] adjacencyMatrix) {
        int[] distanceArray = new int[adjacencyMatrix.length];
        for (int i = 0; i < distanceArray.length; i++) {
            distanceArray[i] = Integer.MAX_VALUE;
        }
        distanceArray[startNode] = 0;
        for (int vertex = 0; vertex < adjacencyMatrix.length - 1; vertex++) {
            for (int adjacentNode = 0; adjacentNode < adjacencyMatrix.length; adjacentNode++) {
                if (adjacencyMatrix[vertex][adjacentNode] != 0
                        && distanceArray[adjacentNode] != Integer.MAX_VALUE
                        && adjacencyMatrix[vertex][adjacentNode]
                        + distanceArray[vertex] < distanceArray[adjacentNode]) {
                    distanceArray[adjacentNode] = adjacencyMatrix[vertex][adjacentNode]
                            + distanceArray[vertex];
                }
            }
        }
        return distanceArray;
    }

    private static int[] prims(int[][] adjacencyMatrix) {
        int[] mstArray = new int[adjacencyMatrix.length];
        int[] verticesEdgesKey = new int[adjacencyMatrix.length];
        boolean[] visitedArray = new boolean[adjacencyMatrix.length];
        for (int i = 0; i < mstArray.length; i++) {
            mstArray[i] = Integer.MAX_VALUE;
            verticesEdgesKey[i] = Integer.MAX_VALUE;
            visitedArray[i] = false;
        }
        mstArray[0] = 0;
        verticesEdgesKey[0] = 0;
        for (int vertex = 0; vertex < adjacencyMatrix.length; vertex++) {
            int minimumDistanceNode = getMinimumFromArray(verticesEdgesKey, visitedArray);
            visitedArray[minimumDistanceNode] = true;
            for (int adjacentNodes = 0; adjacentNodes < adjacencyMatrix.length; adjacentNodes++) {
                if (!visitedArray[vertex] && adjacencyMatrix[minimumDistanceNode][vertex] != 0
                        && adjacencyMatrix[minimumDistanceNode][vertex] < verticesEdgesKey[vertex]) {
                    mstArray[vertex] = minimumDistanceNode;
                    verticesEdgesKey[vertex] = adjacencyMatrix[minimumDistanceNode][vertex];
                }
            }
        }
        return mstArray;
    }

    private static int[] kruskals(int[][] adjacencyMatrix) {
        List<Integer> mstArray = new ArrayList<>();
        boolean[][] visitedEdge = new boolean[adjacencyMatrix.length][adjacencyMatrix.length];
        DisJointSets djs = new DisJointSets();
        for (int d = 0; d < adjacencyMatrix.length; d++) {
            djs.makeSet(d);
        }
        for (int v = 0; v < adjacencyMatrix.length; v++) {
            int minimumEdge = Integer.MAX_VALUE;
            for (int i = 0; i < adjacencyMatrix.length; i++) {
                for (int j = 0; j < adjacencyMatrix.length; j++) {
                    if (adjacencyMatrix[i][j] < minimumEdge && !visitedEdge[i][j]) {
                        visitedEdge[i][j] = true;
                        visitedEdge[j][i] = true;
                        int iThDs = djs.findSet(i);
                        int jThDs = djs.findSet(j);
                        if (iThDs != jThDs) {
                            djs.union(iThDs, jThDs);
                            mstArray.add(i);
                            mstArray.add(j);
                        }
                    }
                }
            }
        }
        return mstArray.stream().mapToInt(Integer::intValue).toArray();
    }

    public static void main(String[] args) {

        AdjacencyMatrixGraph adjacencyMatrixGraph = new AdjacencyMatrixGraph(5);
        visitedNodesAdjacencyMatrix = new boolean[adjacencyMatrixGraph.vertexCount][adjacencyMatrixGraph.vertexCount];
        depthFirstTraversalAdjacencyMatrix(0, adjacencyMatrixGraph.adjacencyMatrix);
        breadthFirstTraversalAdjacencyMatrix(adjacencyMatrixGraph.adjacencyMatrix);

        AdjacencyListGraph adjacencyListGraph = new AdjacencyListGraph(5);
        visitedNodesAdjacencyList = new boolean[adjacencyListGraph.vertexCount];
        depthFirstTraversalAdjacencyList(0, adjacencyListGraph.adjacencyList);
        breadthFirstTraversalAdjacencyList(adjacencyListGraph.adjacencyList);

        for (int[] ints : Collections.singletonList(shortestDistanceBFS(1, adjacencyListGraph.adjacencyList))) {
            System.out.println(Arrays.toString(ints));
        }

        Arrays.stream(prims(adjacencyMatrixGraph.adjacencyMatrix)).forEach(System.out::println);
        Arrays.stream(kruskals(adjacencyMatrixGraph.adjacencyMatrix)).forEach(System.out::println);
    }

    public static class AdjacencyMatrixGraph {
        int[][] adjacencyMatrix;
        int vertexCount;

        AdjacencyMatrixGraph(int vertexCount) {
            this.vertexCount = vertexCount;
            this.adjacencyMatrix = new int[vertexCount][vertexCount];
        }

        public void addEdge(int i, int j) {
            if (i >= 0 && i < this.vertexCount && j >= 0 && j < this.vertexCount) {
                this.adjacencyMatrix[i][j] = 1;
                this.adjacencyMatrix[j][i] = 1;
            }
        }

        public void deleteEdge(int i, int j) {
            if (i >= 0 && i < this.vertexCount && j >= 0 && j < this.vertexCount) {
                this.adjacencyMatrix[i][j] = 0;
                this.adjacencyMatrix[j][i] = 0;
            }
        }

        public int isEdge(int i, int j) {
            if (i >= 0 && i < this.vertexCount && j >= 0 && j < this.vertexCount) {
                return adjacencyMatrix[i][j];
            }
            return 0;
        }
    }

//    private int[] articulationPoints(int i, List<LinkedList<Integer>> adjacencyList) {
//        int [] distanceArray = new int[adjacencyList.size()];
//        int distanceCount = 0;
//        LinkedList<Integer> edgesList = adjacencyList.get(i);
//        for (int j = 0; j < edgesList.size(); j++) {
//            if (!visitedNodesAdjacencyList[j]) {
//                distanceCount++;
//                distanceArray[j] = distanceCount;
//                System.out.println("traversed node - " + edgesList.get(j));
//                visitedNodesAdjacencyList[edgesList.get(j)] = true;
//                depthFirstTraversalAdjacencyList(j, adjacencyList);
//            }
//        }
//    }

    public static class AdjacencyListGraph {
        List<LinkedList<Integer>> adjacencyList;
        int vertexCount;

        AdjacencyListGraph(int vertexCount) {
            this.vertexCount = vertexCount;
            this.adjacencyList = new ArrayList<>(vertexCount);
        }

        public void addEdge(int i, int j) {
            if (i >= 0 && i < this.vertexCount && j >= 0 && j < this.vertexCount) {
                LinkedList<Integer> edgesList = this.adjacencyList.get(i);
                edgesList.add(j);
                this.adjacencyList.add(i, edgesList);
                LinkedList<Integer> edgesListBi = this.adjacencyList.get(j);
                edgesListBi.add(i);
                this.adjacencyList.add(j, edgesListBi);
            }
        }
    }
}
