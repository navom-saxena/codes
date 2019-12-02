package dsalgo.hackerrank.datastructures;

import dsalgo.practice.HackerRankAux3;

import java.io.*;
import java.util.*;
import java.util.Arrays;

public class Queues {

    private static Set<Integer> primes;

    static {
        primes = HackerRankAux3.sieveOfEratosthenesHashSet(Double.valueOf(Math.pow(10, 6) + 1).intValue());
    }

    public static void main(String[] args) throws IOException {
//        twoStacksQueue();
//        half done minimum moves, to be done - logic to avoid loop in case of reaching same node
//        System.out.println(minimumMoves(new String[]{".X.", ".X.", "..."}, 0, 0, 0, 2));
        System.out.println(downToZero(12));
    }

    private static void twoStacksQueue() throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
        int firstInt = Integer.parseInt(br.readLine());
        Deque<Integer> pushStack = new ArrayDeque<>();
        Deque<Integer> popStack = new ArrayDeque<>();
        for (int z = 0; z < firstInt; z++) {
            String[] input = br.readLine().split(" ");
            String operation = input[0];
            switch (operation) {
                case "1": {
                    Integer pushValue = Integer.parseInt(input[1]);
                    pushStack.push(pushValue);
                    break;
                }
                case "2": {
                    if (popStack.isEmpty()) {
                        while (!pushStack.isEmpty()) {
                            popStack.push(pushStack.pop());
                        }
                    }
                    popStack.pop();
                    break;
                }
                case "3": {
                    if (popStack.isEmpty()) {
                        while (!pushStack.isEmpty()) {
                            popStack.push(pushStack.pop());
                        }
                    }
                    bw.write(popStack.peek() + "\n");
                    bw.flush();
                    break;
                }
            }
        }
        br.close();
        bw.close();
    }

    private static int minimumMoves(String[] grid, int startX, int startY, int goalX, int goalY) {
        String[][] returningGrid = new String[grid.length][];
        for (int i = 0; i < grid.length; i++) {
            String[] rowsArr = grid[i].split("");
            returningGrid[i] = rowsArr;
        }
        return minimumMoves(returningGrid, startX, startY, goalX, goalY);
    }

    private static int minimumMoves(String[][] grid, int startX, int startY, int goalX, int goalY) {
        class GridBox {
            int valueX;
            int valueY;
            int parentX;
            int parentY;
            int distanceFromStart;

            GridBox(int valueX, int valueY, int parentX, int parentY, int distanceFromStart) {
                this.valueX = valueX;
                this.valueY = valueY;
                this.parentX = parentX;
                this.parentY = parentY;
                this.distanceFromStart = distanceFromStart;
            }
        }
        int distance = Integer.MAX_VALUE;
        Deque<GridBox> queue = new ArrayDeque<>();
        GridBox start = new GridBox(startX, startY, Integer.MIN_VALUE, Integer.MIN_VALUE, 0);
        queue.addLast(start);
        while (!queue.isEmpty()) {
            GridBox currentBox = queue.removeFirst();
            if ((currentBox.valueX < grid[currentBox.valueX].length - 1
                    && currentBox.valueY < grid[currentBox.valueY].length - 1) &&
                    (currentBox.valueX > 0 && currentBox.valueY > 0)) {
                int nextValueX = currentBox.valueX + 1;
                int nextValueY = currentBox.valueY;
                if (nextValueX == goalX && nextValueY == goalY) {
                    if (currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY) {
                        distance = Math.min(distance, currentBox.distanceFromStart);
                    } else {
                        distance = Math.min(distance, currentBox.distanceFromStart + 1);
                    }
                } else if (!grid[nextValueX][nextValueY].equals("X")) {
                    if ((currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY)
                            || (currentBox.valueX == startX && currentBox.valueY == startY)) {
                        if (currentBox.distanceFromStart == 0) {
                            GridBox nextGridVertical = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                            queue.addLast(nextGridVertical);
                        } else {
                            GridBox nextGridVertical = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart);
                            queue.addLast(nextGridVertical);
                        }
                    } else {
                        GridBox nextGridVertical = new GridBox(nextValueX, nextValueY,
                                currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                        queue.addLast(nextGridVertical);
                    }
                }
                nextValueX = currentBox.valueX;
                nextValueY = currentBox.valueY + 1;
                if (nextValueX == goalX && nextValueY == goalY) {
                    if ((currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY)
                            || (currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)) {
                        distance = Math.min(distance, currentBox.distanceFromStart);
                    } else {
                        distance = Math.min(distance, currentBox.distanceFromStart + 1);
                    }
                } else if (!grid[nextValueX][nextValueY].equals("X")) {
                    if ((currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)
                            || (currentBox.valueX == startX && currentBox.valueY == startY)) {
                        if (currentBox.distanceFromStart == 0) {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                            queue.addLast(nextGridHorizontal);
                        } else {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart);
                            queue.addLast(nextGridHorizontal);
                        }
                    } else {
                        GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                        queue.addLast(nextGridHorizontal);
                    }
                }
                nextValueX = currentBox.valueX - 1 == currentBox.parentX ? currentBox.valueX : currentBox.valueX - 1;
                nextValueY = currentBox.valueY - 1 == currentBox.parentY ? currentBox.valueY : currentBox.valueY - 1;
                if (nextValueX == goalX && nextValueY == goalY) {
                    if ((currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY)
                            || (currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)) {
                        distance = Math.min(distance, currentBox.distanceFromStart);
                    } else {
                        distance = Math.min(distance, currentBox.distanceFromStart + 1);
                    }
                } else if (!grid[nextValueX][nextValueY].equals("X")) {
                    if ((currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)
                            || (currentBox.valueX == startX && currentBox.valueY == startY)) {
                        if (currentBox.distanceFromStart == 0) {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                            queue.addLast(nextGridHorizontal);
                        } else {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart);
                            queue.addLast(nextGridHorizontal);
                        }
                    } else {
                        GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                        queue.addLast(nextGridHorizontal);
                    }
                }
            } else if (currentBox.valueX < grid[currentBox.valueX].length - 1
                    && currentBox.valueY < grid[currentBox.valueY].length - 1) {
                int nextValueX = currentBox.valueX + 1;
                int nextValueY = currentBox.valueY;
                if (nextValueX == goalX && nextValueY == goalY) {
                    if (currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY) {
                        distance = Math.min(distance, currentBox.distanceFromStart);
                    } else {
                        distance = Math.min(distance, currentBox.distanceFromStart + 1);
                    }
                } else if (!grid[nextValueX][nextValueY].equals("X")) {
                    if ((currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY)
                            || (currentBox.valueX == startX && currentBox.valueY == startY)) {
                        if (currentBox.distanceFromStart == 0) {
                            GridBox nextGridVertical = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                            queue.addLast(nextGridVertical);
                        } else {
                            GridBox nextGridVertical = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart);
                            queue.addLast(nextGridVertical);
                        }
                    } else {
                        GridBox nextGridVertical = new GridBox(nextValueX, nextValueY,
                                currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                        queue.addLast(nextGridVertical);
                    }
                }
                nextValueX = currentBox.valueX;
                nextValueY = currentBox.valueY + 1;
                if (nextValueX == goalX && nextValueY == goalY) {
                    if ((currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY)
                            || (currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)) {
                        distance = Math.min(distance, currentBox.distanceFromStart);
                    } else {
                        distance = Math.min(distance, currentBox.distanceFromStart + 1);
                    }
                } else if (!grid[nextValueX][nextValueY].equals("X")) {
                    if ((currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)
                            || (currentBox.valueX == startX && currentBox.valueY == startY)) {
                        if (currentBox.distanceFromStart == 0) {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                            queue.addLast(nextGridHorizontal);
                        } else {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart);
                            queue.addLast(nextGridHorizontal);
                        }
                    } else {
                        GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                        queue.addLast(nextGridHorizontal);
                    }
                }
            } else if (currentBox.valueX < grid[currentBox.valueX].length - 1) {
                int nextValueX = currentBox.valueX + 1;
                int nextValueY = currentBox.valueY;
                if (nextValueX == goalX && nextValueY == goalY) {
                    if (currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY) {
                        distance = Math.min(distance, currentBox.distanceFromStart);
                    } else {
                        distance = Math.min(distance, currentBox.distanceFromStart + 1);
                    }
                } else if (!grid[nextValueX][nextValueY].equals("X")) {
                    if (currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY) {
                        if (currentBox.distanceFromStart == 0) {
                            GridBox nextGridVertical = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                            queue.addLast(nextGridVertical);
                        } else {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart);
                            queue.addLast(nextGridHorizontal);
                        }
                    } else {
                        GridBox nextGridVertical = new GridBox(nextValueX, nextValueY,
                                currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                        queue.addLast(nextGridVertical);
                    }
                }

            } else if (currentBox.valueY < grid[currentBox.valueY].length - 1) {
                int nextValueX = currentBox.valueX;
                int nextValueY = currentBox.valueY + 1;
                if (nextValueX == goalX && nextValueY == goalY) {
                    if ((currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY)
                            || (currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)) {
                        distance = Math.min(distance, currentBox.distanceFromStart);
                    } else {
                        distance = Math.min(distance, currentBox.distanceFromStart + 1);
                    }
                } else if (!grid[nextValueX][nextValueY].equals("X")) {
                    if ((currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)
                            || (currentBox.valueX == startX && currentBox.valueY == startY)) {
                        if (currentBox.distanceFromStart == 0) {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                            queue.addLast(nextGridHorizontal);
                        } else {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart);
                            queue.addLast(nextGridHorizontal);
                        }
                    } else {
                        GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                        queue.addLast(nextGridHorizontal);
                    }
                }
                nextValueX = currentBox.valueX - 1 == currentBox.parentX ? currentBox.valueX : currentBox.valueX - 1;
                nextValueY = currentBox.valueY - 1 == currentBox.parentY ? currentBox.valueY : currentBox.valueY - 1;
                if (nextValueX > 0 && nextValueY > 0) {
                    if (nextValueX == goalX && nextValueY == goalY) {
                        if ((currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY)
                                || (currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)) {
                            distance = Math.min(distance, currentBox.distanceFromStart);
                        } else {
                            distance = Math.min(distance, currentBox.distanceFromStart + 1);
                        }
                    } else if (!grid[nextValueX][nextValueY].equals("X")) {
                        if ((currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)
                                || (currentBox.valueX == startX && currentBox.valueY == startY)) {
                            if (currentBox.distanceFromStart == 0) {
                                GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                        currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                                queue.addLast(nextGridHorizontal);
                            } else {
                                GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                        currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart);
                                queue.addLast(nextGridHorizontal);
                            }
                        } else {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                            queue.addLast(nextGridHorizontal);
                        }
                    }
                }
            } else {
                int nextValueX = currentBox.valueX - 1 == currentBox.parentX ? currentBox.valueX : currentBox.valueX - 1;
                int nextValueY = currentBox.valueY - 1 == currentBox.parentY ? currentBox.valueY : currentBox.valueY - 1;
                if (nextValueX > 0 && nextValueY > 0) {
                    if (nextValueX == goalX && nextValueY == goalY) {
                        if ((currentBox.valueX == currentBox.parentX + 1 && currentBox.valueY == currentBox.parentY)) {
                            distance = Math.min(distance, currentBox.distanceFromStart);
                        } else {
                            distance = Math.min(distance, currentBox.distanceFromStart + 1);
                        }
                    } else if (!grid[nextValueX][nextValueY].equals("X")) {
                        if ((currentBox.valueY == currentBox.parentY + 1 && currentBox.valueX == currentBox.parentX)
                                || (currentBox.valueX == startX && currentBox.valueY == startY)) {
                            if (currentBox.distanceFromStart == 0) {
                                GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                        currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                                queue.addLast(nextGridHorizontal);
                            } else {
                                GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                        currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart);
                                queue.addLast(nextGridHorizontal);
                            }
                        } else {
                            GridBox nextGridHorizontal = new GridBox(nextValueX, nextValueY,
                                    currentBox.valueX, currentBox.valueY, currentBox.distanceFromStart + 1);
                            queue.addLast(nextGridHorizontal);
                        }
                    }
                }
            }
        }
        return distance;
    }

    private static int downToZero(int n) {
        int count = 0;
        while (n != 0) {
            if (n == 1) {
                count++;
                break;
            }
            if (primes.contains(n)) {
                n = n - 1;
            } else {
                int sqrt = Double.valueOf(Math.sqrt(n)).intValue();
                int lessThanSqrtValue = 1;
                for (int i = sqrt; i >= 2; i--) {
                    if (n % i == 0) {
                        lessThanSqrtValue = i;
                        break;
                    }
                }
                n = n / lessThanSqrtValue;
            }
            count++;
        }
        return count;
    }
}
