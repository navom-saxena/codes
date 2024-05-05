from __future__ import annotations

import math
import queue
import sys
from collections import deque
from functools import cmp_to_key
from typing import List, Set, Optional, Dict, Deque

from src.main.python.leetcode.Models import TreeNode, ListNode, Node


class CustomStack:

    def __init__(self, maxSize: int):
        self.custom_stack: List[int] = [0] * maxSize
        self.maxsize = maxSize
        self.i: int = -1

    def push(self, x: int) -> None:
        if self.i + 1 < self.maxsize:
            self.i += 1
            self.custom_stack[self.i] = x

    def pop(self) -> int:
        value: int = -1
        if self.i >= 0:
            value = self.custom_stack[self.i]
            self.i -= 1

        return value

    def increment(self, k: int, val: int) -> None:
        r: int = min(k, self.maxsize)
        for j in range(r):
            self.custom_stack[j] += val


# https://leetcode.com/problems/output-contest-matches/

def findContestMatch(self, n: int) -> str:
    rounds_count: int = int(math.log2(n))
    matches: List[str] = [str(x) for x in range(1, n + 1)]
    for _ in range(rounds_count):
        temp_list: List[str] = []
        i: int = 0
        j: int = len(matches) - 1
        while i < j:
            temp_list.append('(' + matches[i] + ',' + matches[j] + ')')
            i += 1
            j -= 1
        matches = temp_list

    return "".join(matches)


# https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree/

def decontaminateTree(node: Optional[TreeNode], val: int, global_set: Set[int]) -> None:
    if not node:
        return
    global_set.add(val)
    decontaminateTree(node=node.left, val=2 * val + 1, global_set=global_set)
    decontaminateTree(node=node.right, val=2 * val + 2, global_set=global_set)


class FindElements:

    def __init__(self, root: Optional[TreeNode]):
        self.global_set: Set[int] = set()
        decontaminateTree(node=root, val=0, global_set=self.global_set)

    def find(self, target: int) -> bool:
        return target in self.global_set


# https://leetcode.com/problems/remove-all-ones-with-row-and-column-flips/description/

def removeOnes(self, grid: List[List[int]]) -> bool:
    m: int = len(grid)
    n: int = len(grid[0])

    for j in range(n):
        if grid[0][j] == 1:
            for i in range(m):
                grid[i][j] = 1 if grid[i][j] == 0 else 0

    print(grid)

    for i in range(1, m):
        row_sum: int = 0
        for j in range(n):
            row_sum += grid[i][j]

        if not (row_sum == 0 or row_sum == n):
            return False

    return True


# https://leetcode.com/problems/delete-node-in-a-linked-list/

def deleteNode(self, node: ListNode):
    prev: ListNode = node
    curr: ListNode = node.next
    prev.val = curr.val
    prev.next = curr.next


# https://leetcode.com/problems/letter-tile-possibilities/

def swap(value: str, i: int, j: int) -> str:
    value_list: List[str] = list(value)
    temp: str = value_list[i]
    value_list[i] = value_list[j]
    value_list[j] = temp
    return "".join(value_list)


def numTilePossibilitiesUtils(tiles: str, visited: Set[str], i: int, running: str) -> None:
    if i == len(tiles):
        visited.add(running)
        return
    for j in range(i, len(tiles)):
        new_running: str = swap(running, i, j)
        if new_running not in visited:
            numTilePossibilitiesUtils(tiles=tiles, visited=visited, i=i + 1, running=new_running)


def createCombinations(tiles: str, i: int, combinations_set: Set[str], running: str) -> None:
    if i == len(tiles):
        combinations_set.add(running)
        return
    createCombinations(tiles=tiles, i=i + 1, combinations_set=combinations_set, running=running + tiles[i])
    createCombinations(tiles=tiles, i=i + 1, combinations_set=combinations_set, running=running)


def numTilePossibilities(self, tiles: str) -> int:
    visited: Set[str] = set()
    combinations_set: Set[str] = set()
    createCombinations(tiles=tiles, i=0, combinations_set=combinations_set, running="")
    for t in combinations_set:
        if t != "":
            numTilePossibilitiesUtils(tiles=tiles, visited=visited, i=0, running=tiles)
    return len(visited)


def numTilePossibilitiesOptimisedHelper(freq_dict: Dict[str, int]) -> int:
    res: int = 0

    for k, v in freq_dict.items():
        if v <= 0:
            continue
        freq_dict[k] = v - 1
        res += 1
        res += numTilePossibilitiesOptimisedHelper(freq_dict=freq_dict)
        freq_dict[k] = v

    return res


def numTilePossibilitiesOptimised(self, tiles: str) -> int:
    freq_dict: Dict[str, int] = {}
    for char in tiles:
        freq_dict[char] = freq_dict.get(char, 0) + 1

    return numTilePossibilitiesOptimisedHelper(freq_dict=freq_dict)


# https://leetcode.com/problems/permutations/

def permuteUtils(nums: List[int], i: int, res: List[List[int]]) -> None:
    if i == len(nums):
        res.append(list(nums))
        return
    for j in range(i, len(nums)):
        temp: int = nums[i]
        nums[i] = nums[j]
        nums[j] = temp
        permuteUtils(nums=nums, i=i + 1, res=res)
        temp: int = nums[i]
        nums[i] = nums[j]
        nums[j] = temp


def permute(self, nums: List[int]) -> List[List[int]]:
    res: List[List[int]] = []
    permuteUtils(nums=nums, i=0, res=res)
    return res


# https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/description/

def maxAncestorDiffUtils(node: Optional[TreeNode], min_range: int, max_range: int, res: List[int]) -> None:
    if not node:
        return
    val: int = node.val
    res[0] = max(abs(val - min_range), abs(max_range - val), res[0])
    min_range = min(min_range, val)
    max_range = max(max_range, val)
    maxAncestorDiffUtils(node=node.left, min_range=min_range, max_range=max_range, res=res)
    maxAncestorDiffUtils(node=node.right, min_range=min_range, max_range=max_range, res=res)


def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    res: List[int] = [- sys.maxsize + 1]
    maxAncestorDiffUtils(node=root, min_range=root.val, max_range=root.val, res=res)
    return res[0]


# https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/description/

def minAddToMakeValid(self, s: str) -> int:
    running_count: int = 0
    close_b_count: int = 0
    for b in s:
        running_count += 1 if b == '(' else -1

        if running_count < 0:
            close_b_count += 1
            running_count = 0

    return close_b_count + running_count


# https://leetcode.com/problems/matrix-block-sum/description/

def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
    m: int = len(mat)
    n: int = len(mat[0])
    prefix_mat: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix_mat[i][j] = mat[i - 1][j - 1] + prefix_mat[i - 1][j] + prefix_mat[i][j - 1] - prefix_mat[i - 1][
                j - 1]

    res: List[List[int]] = [[0] * n for _ in range(m)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            start_i: int = max(1, i - k)
            start_j: int = max(1, j - k)

            end_i: int = min(m, i + k)
            end_j: int = min(n, j + k)

            res[i - 1][j - 1] = prefix_mat[end_i][end_j] - prefix_mat[start_i - 1][end_j] - prefix_mat[end_i][
                start_j - 1] + prefix_mat[start_i - 1][start_j - 1]

    return res


# https://leetcode.com/problems/find-nearest-right-node-in-binary-tree/

def findNearestRightNode(self, root: TreeNode, u: TreeNode) -> Optional[TreeNode]:
    tree_q: Deque[Optional[TreeNode]] = deque()
    tree_q.append(root)
    tree_q.append(None)
    while tree_q:
        node: Optional[TreeNode] = tree_q.popleft()
        if node is None:
            tree_q.append(None)
            continue

        if node == u:
            return tree_q.popleft()
        if node.left:
            tree_q.append(node.left)
        if node.right:
            tree_q.append(node.right)

    return None


# https://leetcode.com/problems/smallest-number-in-infinite-set/

class SmallestInfiniteSet:

    def __init__(self):
        self.pq: queue.PriorityQueue = queue.PriorityQueue()
        self.already_popped: Set[int] = set()
        self.counter: int = 1

    def popSmallest(self) -> int:
        if not self.pq.empty() and self.counter > self.pq.queue[0]:
            v: int = self.pq.get()
            self.already_popped.add(v)
            return v
        else:
            v: int = self.counter
            self.already_popped.add(v)
            self.counter += 1
            return v

    def addBack(self, num: int) -> None:
        if num in self.already_popped:
            self.pq.put(num)
            self.already_popped.remove(num)


# https://leetcode.com/problems/subsets/

def subsetsUtils(nums: List[int], i: int, running: List[int], res: List[List[int]]) -> None:
    if i == len(nums):
        res.append(running)
        return
    subsetsUtils(nums=nums, i=i + 1, running=running, res=res)
    subsetsUtils(nums=nums, i=i + 1, running=running + [nums[i]], res=res)


def subsets(self, nums: List[int]) -> List[List[int]]:
    res: List[List[int]] = []
    subsetsUtils(nums=nums, i=0, running=[], res=res)
    return res


# https://leetcode.com/problems/battleships-in-a-board/description/


def countBattleships(self, board: List[List[str]]) -> int:
    m: int = len(board)
    n: int = len(board[0])

    res: int = 0
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'X':
                k: int = i + 1
                l_down: int = j + 1
                while k < m and board[k][j] == 'X':
                    board[k][j] = '.'
                    k += 1
                while l_down < n and board[i][l_down] == 'X':
                    board[i][l_down] = '.'
                    l_down += 1

                res += 1

    return res


# https://leetcode.com/problems/delete-leaves-with-a-given-value/

def removeLeafNodes(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
    if not root:
        return None
    left_node: Optional[TreeNode] = self.removeLeafNodes(root=root.left, target=target)
    right_node: Optional[TreeNode] = self.removeLeafNodes(root=root.right, target=target)
    if not left_node and not right_node and root.val == target:
        return None
    root.left = left_node
    root.right = right_node
    return root


# https://leetcode.com/problems/remove-all-occurrences-of-a-substring/description/

def removeOccurrences(s: str, part: str) -> str:
    rem_stack: Deque[str] = deque()
    for j, word in enumerate(s):
        rem_stack.append(word)

        i: int = len(part) - 1
        temp_stack: Deque[str] = deque()

        while len(rem_stack) > 0 and i >= 0 and rem_stack[-1] == part[i]:
            print(i)
            temp_stack.append(rem_stack.pop())
            i -= 1

        if i != -1:
            while len(temp_stack) > 0:
                rem_stack.append(temp_stack.pop())

    res: str = ""
    while rem_stack:
        res = rem_stack.pop() + res

    return res


# https://leetcode.com/problems/merge-in-between-linked-lists/

def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
    sentinel_node: ListNode = ListNode(-1)
    sentinel_node.next = list1

    start: Optional[ListNode] = None
    end: Optional[ListNode] = None

    curr: ListNode = sentinel_node
    i: int = -1
    while curr is not None:
        if curr.next is not None and i + 1 == a:
            start = curr
        elif i == b:
            end = curr.next

        i += 1
        curr = curr.next

    l2_end: Optional[ListNode] = list2
    while l2_end is not None and l2_end.next is not None:
        l2_end = l2_end.next

    if start is not None and end is not None:
        start.next = list2
        l2_end.next = end

    return sentinel_node.next


# https://leetcode.com/problems/diameter-of-n-ary-tree/

def diameterUtils(node: Node, max_diameter: List[int]) -> int:
    if not node:
        return 0
    h1: int = 0
    h2: int = 0
    for child in node.children:
        h: int = diameterUtils(node=child, max_diameter=max_diameter)
        if h >= h1:
            h2 = h1
            h1 = h
        elif h > h2:
            h2 = h

    subTree_h: int = h1 + h2
    max_diameter[0] = max(max_diameter[0], subTree_h)
    return h1 + 1


def diameter(self, root: Node) -> int:
    max_diameter: List[int] = [0]
    diameterUtils(node=root, max_diameter=max_diameter)
    return max_diameter[0]


# https://leetcode.com/problems/count-good-nodes-in-binary-tree/

def goodNodesUtil(self, node: TreeNode, max_v: int, count: List[int]) -> None:
    if not node:
        return
    if node.val >= max_v:
        count[0] += 1
    max_v = max(max_v, node.val)
    self.goodNodesUtil(node=node.left, max_v=max_v, count=count)
    self.goodNodesUtil(node=node.right, max_v=max_v, count=count)


def goodNodes(self, root: TreeNode) -> int:
    count: List[int] = [0]
    self.goodNodesUtil(node=root, max_v=- sys.maxsize + 1, count=count)
    return count[0]


# https://leetcode.com/problems/count-square-submatrices-with-all-ones/

def countSquares(self, matrix: List[List[int]]) -> int:
    m: int = len(matrix)
    n: int = len(matrix[0])
    count: int = 0
    prefix_mat: List[List[int]] = [[0] * n for _ in range(m)]
    for i in range(m):
        prefix_mat[i][0] = matrix[i][0]

    for j in range(n):
        prefix_mat[0][j] = matrix[0][j]

    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 1:
                prefix_mat[i][j] = min(prefix_mat[i - 1][j], prefix_mat[i][j - 1], prefix_mat[i - 1][j - 1]) + 1

    for i in range(m):
        for j in range(n):
            count += prefix_mat[i][j]

    return count


# https://leetcode.com/problems/find-all-duplicates-in-an-array/

def findDuplicates(self, nums: List[int]) -> List[int]:
    res: List[int] = []
    n: int = len(nums)

    for i in range(0, n):
        actual_v: int = abs(nums[i])
        index: int = actual_v % n
        if nums[index] < 0:
            res.append(actual_v)
        else:
            nums[index] = -nums[index]

    return res


# https://leetcode.com/problems/construct-quad-tree/

class QuadNode:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


def constructUtils(grid: List[List[int]], r: int, c: int, n: int) -> QuadNode:
    is_all_same: bool = True
    for i in range(n):
        for j in range(n):
            if grid[r][c] != grid[r + i][c + j]:
                is_all_same = False
                break

    if is_all_same:
        return QuadNode(val=grid[r][c], isLeaf=True, topLeft=None, topRight=None, bottomLeft=None, bottomRight=None)

    n //= 2

    top_left: QuadNode = constructUtils(grid=grid, r=r, c=c, n=n)
    top_right: QuadNode = constructUtils(grid=grid, r=r, c=c + n, n=n)
    bottom_left: QuadNode = constructUtils(grid=grid, r=r + n, c=c, n=n)
    bottom_right: QuadNode = constructUtils(grid=grid, r=r + n, c=c + n, n=n)

    return QuadNode(val=0, isLeaf=False, topLeft=top_left, topRight=top_right, bottomLeft=bottom_left,
                    bottomRight=bottom_right)


def construct(grid: List[List[int]]) -> QuadNode:
    return constructUtils(grid=grid, r=0, c=0, n=len(grid))


# https://leetcode.com/problems/iterator-for-combination/

def create_combinations(characters: str, i: int, combinationLength: int, running: str, res: List[str]):
    if len(running) == combinationLength:
        res.append(running)
        return

    if i == len(characters):
        return

    create_combinations(characters=characters, i=i + 1, combinationLength=combinationLength,
                        running=running + characters[i], res=res)
    create_combinations(characters=characters, i=i + 1, combinationLength=combinationLength,
                        running=running, res=res)


class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        self.comb_list: List[str] = []
        create_combinations(characters=characters, i=0, combinationLength=combinationLength, running="",
                            res=self.comb_list)
        self.sorted_comb_list: List[str] = sorted(self.comb_list)
        self.i: int = 0
        self.n: int = len(self.sorted_comb_list)
        print(self.sorted_comb_list)

    def next(self) -> str:
        v: str = self.sorted_comb_list[self.i]
        self.i += 1
        return v

    def hasNext(self) -> bool:
        return self.i < self.n


# https://leetcode.com/problems/binary-tree-pruning/

def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None

    left_node: Optional[TreeNode] = self.pruneTree(root=root.left)
    right_node: Optional[TreeNode] = self.pruneTree(root=root.right)

    root.left = left_node
    root.right = right_node

    if root.val == 1:
        return root

    if not left_node and not right_node:
        return None

    return root


# https://leetcode.com/problems/correct-a-binary-tree/

def correctBinaryTreeUtils(root: TreeNode, processed_nodes: Set[int]) -> Optional[TreeNode]:
    if not root:
        return None
    processed_nodes.add(root.val)
    if root.right and root.right.val in processed_nodes:
        return None
    root.right = correctBinaryTreeUtils(root=root.right, processed_nodes=processed_nodes)
    root.left = correctBinaryTreeUtils(root=root.left, processed_nodes=processed_nodes)
    return root


def correctBinaryTree(self, root: TreeNode) -> TreeNode:
    processed_nodes: Set[int] = set()
    return correctBinaryTreeUtils(root=root, processed_nodes=processed_nodes)


# https://leetcode.com/problems/generate-parentheses/description/

def generateParenthesisUtils(open_c: int, closed_c: int, n: int, running: List[str], res: List[str]) -> None:
    if open_c == n and closed_c == n:
        res.append("".join(running))
        return
    if closed_c > open_c or closed_c > n or open_c > n:
        return

    running.append("(")
    generateParenthesisUtils(open_c=open_c + 1, closed_c=closed_c, n=n, running=running, res=res)
    running.pop()

    running.append(")")
    generateParenthesisUtils(open_c=open_c, closed_c=closed_c + 1, n=n, running=running, res=res)
    running.pop()


def generateParenthesis(self, n: int) -> List[str]:
    res: List[str] = []
    generateParenthesisUtils(open_c=0, closed_c=0, n=n, running=[], res=res)
    return res


# https://leetcode.com/problems/queue-reconstruction-by-height/description/

def queueCompare(item1: List[int], item2: List[int]):
    if item1[0] < item2[0]:
        return 1
    elif item1[0] > item2[0]:
        return -1
    elif item1[1] < item2[1]:
        return -1
    else:
        return 1


def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
    sorted_people: List[List[int]] = sorted(people, key=cmp_to_key(queueCompare))
    for i in range(len(sorted_people)):
        curr: List[int] = sorted_people[i]
        t: int = curr[1]
        j: int = i
        while j > t:
            sorted_people[j] = sorted_people[j - 1]
            j -= 1
        sorted_people[t] = curr

    return sorted_people


# https://leetcode.com/problems/change-the-root-of-a-binary-tree/description/

def flipBinaryTreeUtils(self, node: Node, from_node: Optional[Node], root: Node) -> Node:
    parent: Node = node.parent
    left_node: Node = node.left
    right_node: Node = node.right

    node.parent = from_node

    if right_node == from_node:
        node.right = None
    if left_node == from_node:
        node.left = None

    if node == root:
        return node

    if node.left:
        node.right = left_node

    node.left = self.flipBinaryTreeUtils(node=parent, from_node=node, root=root)
    return node


def flipBinaryTree(self, root: Node, leaf: Node) -> Node:
    return self.flipBinaryTreeUtils(node=leaf, from_node=None, root=root)


# https://leetcode.com/problems/distribute-coins-in-binary-tree/

def distributeCoinsUtils(node: Optional[TreeNode], steps: List[int]) -> int:
    if not node:
        return 0
    left_extra: int = distributeCoinsUtils(node=node.left, steps=steps)
    right_extra: int = distributeCoinsUtils(node=node.right, steps=steps)
    steps[0] += abs(left_extra) + abs(right_extra)
    node_extra: int = left_extra + right_extra + node.val - 1
    return node_extra


def distributeCoins(self, root: Optional[TreeNode]) -> int:
    steps: List[int] = [0]
    distributeCoinsUtils(node=root, steps=steps)
    return steps[0]


# https://leetcode.com/problems/create-binary-tree-from-descriptions/

def createBinaryTree(self, descriptions: List[List[int]]) -> Optional[TreeNode]:
    root: Optional[TreeNode] = None
    nodes_mapping: Dict[int, TreeNode] = {}
    child_mapping: Set[int] = set()
    parent_mapping: List[int] = list()

    for description in descriptions:
        parent: int = description[0]
        child: int = description[1]
        is_left: bool = True if description[2] == 1 else False

        parent_node: Optional[TreeNode] = nodes_mapping.get(parent, TreeNode(val=parent))
        child_node: Optional[TreeNode] = nodes_mapping.get(child, TreeNode(val=child))

        if is_left:
            parent_node.left = child_node
        else:
            parent_node.right = child_node

        nodes_mapping[parent] = parent_node
        nodes_mapping[child] = child_node
        child_mapping.add(child)
        parent_mapping.append(parent)

    for parent in parent_mapping:
        if parent not in child_mapping:
            return nodes_mapping[parent]

    return None


# https://leetcode.com/problems/keys-and-rooms/description/

def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
    n: int = len(rooms)
    visited: Set[int] = set()
    room_queue: Deque[int] = deque()
    room_queue.extend(rooms[0])
    while room_queue:
        if len(visited) == n:
            return True
        key: int = room_queue.popleft()
        if key not in visited:
            visited.add(key)
            room_queue.extend(rooms[key])

    return len(visited) == n


# https://leetcode.com/problems/max-area-of-island/description/

def maxAreaOfIslandDfs(i: int, j: int, m: int, n: int, grid: List[List[int]], directions: List[List[int]]) -> int:
    if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == 0:
        return 0
    grid[i][j] = 0
    area: int = 1
    for d in directions:
        area += maxAreaOfIslandDfs(i=i + d[0], j=j + d[1], m=m, n=n, grid=grid, directions=directions)

    return area


def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    m: int = len(grid)
    n: int = len(grid[0])
    max_area: int = 0
    directions: List[List[int]] = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                area: int = maxAreaOfIslandDfs(i=i, j=j, m=m, n=n, grid=grid, directions=directions)
                max_area = max(max_area, area)

    return max_area


# https://leetcode.com/problems/build-an-array-with-stack-operations/description/

def buildArray(self, target: List[int], n: int) -> List[str]:
    idx: int = 0
    res: List[str] = []
    for i in range(1, n + 1):
        res.append("Push")

        if target[idx] == i:
            idx += 1
        else:
            res.append("Pop")

        if idx == len(target):
            return res

    return res


# https://leetcode.com/problems/minimum-number-of-operations-to-reinitialize-a-permutation/description/

def reinitializePermutation(self, n: int) -> int:
    operations_count: int = 0
    arr: List[int] = [i for i in range(n)]
    while True:
        perm: List[int] = []
        all_correct: int = True
        for i in range(n):
            if i % 2 == 0:
                perm[i] = arr[i // 2]
            else:
                perm[i] = arr[n // 2 + (i - 1) // 2]

            if perm[i] != i:
                all_correct = False

        arr = perm
        operations_count += 1
        if all_correct:
            break

    return operations_count


# https://leetcode.com/problems/range-addition/description/

def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
    arr: List[int] = [0] * length

    for update in updates:
        start: int = update[0]
        end: int = update[1]
        incr: int = update[2]
        arr[start] += incr
        if end + 1 < length:
            arr[end + 1] -= incr

    prefix_sum: int = 0
    for i, e in enumerate(arr):
        prefix_sum += e
        arr[i] = prefix_sum

    return arr


# https://leetcode.com/problems/interval-list-intersections/description/

def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    intersected_list: List[List[int]] = []
    m: int = len(firstList)
    n: int = len(secondList)
    i: int = 0
    j: int = 0

    while i < m and j < n:
        start_f: int = firstList[i][0]
        end_f: int = firstList[i][1]
        start_s: int = secondList[j][0]
        end_s: int = secondList[j][1]

        max_start: int = max(start_f, start_s)
        min_end: int = min(end_f, end_s)
        print(max_start, min_end)
        if max_start <= min_end:
            intersected_list.append([max_start, min_end])

        if end_f <= end_s:
            i += 1
        else:
            j += 1

    return intersected_list


# https://leetcode.com/problems/search-in-a-sorted-array-of-unknown-size/description/

class ArrayReader:
    @staticmethod
    def get(index: int) -> int:
        return 1


def search(self, reader: ArrayReader, target: int) -> int:
    low: int = 0
    high: int = 10001
    while low <= high:
        mid: int = low + (high - low) // 2
        ar_value: int = reader.get(index=mid)
        if ar_value == target:
            return mid
        elif ar_value < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1


# https://leetcode.com/problems/minimum-adjacent-swaps-to-reach-the-kth-smallest-number/description/

def swap_min_swap(self, nums: List[int], i: int, j: int) -> List[int]:
    temp: int = nums[i]
    nums[i] = nums[j]
    nums[j] = temp
    return nums


def getKSmallest(self, nums: str, k: int) -> List[int]:
    nums: List[int] = [int(x) for x in nums]
    n: int = len(nums)
    while k > 0:
        i: int = n - 2
        while nums[i] >= nums[i + 1]:
            i -= 1
        for j in range(n - 1, i, -1):
            if nums[j] > nums[i]:
                self.swap_min_swap(nums=nums, i=i, j=j)
                break

        sorted_rem: List[int] = nums[0: i + 1] + sorted(nums[i + 1:])
        nums = sorted_rem
        k -= 1

    return nums


def getSwaps(self, nums: str, new_nums: List[int]) -> int:
    nums: List[int] = [int(x) for x in nums]
    n: int = len(nums)
    swaps: int = 0
    i: int = 0
    while i < n:
        if nums[i] == new_nums[i]:
            i += 1
            continue
        else:
            j: int = i
            while j < n:
                if new_nums[j] != nums[i]:
                    j += 1
                else:
                    break
            while j != i:
                new_nums = self.swap_min_swap(new_nums, j, j - 1)
                j -= 1
                swaps += 1

    return swaps


def getMinSwaps(self, num: str, k: int) -> int:
    kth_smallest: List[int] = self.getKSmallest(nums=num, k=k)
    return self.getSwaps(nums=num, new_nums=kth_smallest)


# https://leetcode.com/problems/rotate-image/description/

def rotate(self, matrix: List[List[int]]) -> None:
    m: int = len(matrix)
    n: int = m - 1
    for i in range(0, m // 2):
        for j in range(0, math.ceil(m / 2)):
            temp: int = matrix[i][j]
            matrix[i][j] = matrix[n - j][i]
            matrix[n - j][i] = matrix[n - i][n - j]
            matrix[n - i][n - j] = matrix[j][n - i]
            matrix[j][n - i] = temp


# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/

def getRightIndex(val: int, preorder: List[int], postorder: List[int]) -> int:
    index: int = -1
    for i in range(len(postorder) - 1, 0, -1):
        if postorder[i] == val:
            index = i
            break

    right_index: int = index - 1
    right_val: int = -1
    if right_index >= 0:
        right_val = postorder[right_index]
    for i in range(0, len(preorder)):
        if preorder[i] == right_val:
            return i
    return -1


def constructFromPrePostUtil(preorder: List[int], postorder: List[int], low: int, high: int) -> Optional[TreeNode]:
    if low > high:
        return None

    val: int = preorder[low]
    node: Optional[TreeNode] = TreeNode(val=val)
    if low == high:
        return node

    right_index = getRightIndex(val=val, preorder=preorder, postorder=postorder)
    node.left = constructFromPrePostUtil(preorder=preorder, postorder=postorder, low=low + 1, high=right_index - 1)
    node.right = constructFromPrePostUtil(preorder=preorder, postorder=postorder, low=right_index, high=high)
    return node


def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    return constructFromPrePostUtil(preorder=preorder, postorder=postorder, low=0, high=len(preorder) - 1)
