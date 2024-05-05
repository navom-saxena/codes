from __future__ import annotations

import heapq
import random
import sys
from collections import deque, defaultdict
from functools import cmp_to_key
from typing import List, Set, Optional, Dict, Deque, Tuple

from sortedcontainers import SortedSet, SortedDict

from src.main.python.leetcode.Models import Node, TreeNode, Trie


# https://leetcode.com/problems/n-ary-tree-level-order-traversal/

def levelOrder(self, root: Node) -> List[List[int]]:
    res: List[List[int]] = []
    if root is None:
        return res
    level_deque: Deque[Node] = deque()
    level_deque.append(root)

    while len(level_deque) != 0:
        n: int = len(level_deque)
        level_nodes: List[int] = []
        for _ in range(n):
            node: Node = level_deque.popleft()
            if node is None:
                continue
            level_nodes.append(node.val)
            for child in node.children:
                level_deque.append(child)

        res.append(level_nodes)

    return res


# https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/

def kthSmallestUtils(node: Optional[TreeNode], counter: List[int], k: int, res: List[int]) -> None:
    if not node:
        return
    if res[0] >= 0:
        return

    kthSmallestUtils(node=node.left, counter=counter, k=k, res=res)
    counter[0] += 1
    if counter[0] == k:
        res[0] = node.val
        return

    kthSmallestUtils(node=node.right, counter=counter, k=k, res=res)


def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    counter: List[int] = [0]
    res: List[int] = [-1]
    kthSmallestUtils(node=root, counter=counter, k=k, res=res)
    return res[0]


# https://leetcode.com/problems/check-if-two-expression-trees-are-equivalent/description/

def checkEquivalenceUtils(node: Node, eval_mapping: Dict[str, int], signs: Set[str], numbers_used: Set[int]) -> int:
    if node.val not in signs:
        if node.val not in eval_mapping:
            rand_no: int = random.randint(a=1, b=5001)
            while rand_no in numbers_used:
                rand_no = random.randint(a=1, b=5001)
            eval_mapping[node.val] = rand_no
            numbers_used.add(rand_no)

        return eval_mapping[node.val]

    left_val: int = checkEquivalenceUtils(node=node.left, eval_mapping=eval_mapping, signs=signs,
                                          numbers_used=numbers_used)
    right_val: int = checkEquivalenceUtils(node=node.right, eval_mapping=eval_mapping, signs=signs,
                                           numbers_used=numbers_used)

    if node.val == '+':
        return left_val + right_val
    elif node.val == '-':
        return left_val - right_val
    elif node.val == '*':
        return left_val * right_val
    elif node.val == '/':
        return left_val // right_val

    return -1


def checkEquivalence(self, root1: Node, root2: Node) -> bool:
    eval_mapping: Dict[str, int] = {}
    signs: Set[str] = {'+', '-', '*', '/'}
    numbers_used: Set[int] = set()
    return checkEquivalenceUtils(node=root1, eval_mapping=eval_mapping, signs=signs, numbers_used=numbers_used) == \
        checkEquivalenceUtils(node=root2, eval_mapping=eval_mapping, signs=signs, numbers_used=numbers_used)


# https://leetcode.com/problems/egg-drop-with-2-eggs-and-n-floors/description/

def twoEggDropUtil(n: int, k: int, dp_eggs: List[List[int]]) -> int:
    if dp_eggs[n][k] != -1:
        return dp_eggs[n][k]
    if n <= 1:
        return 1
    if k == 1:
        return n

    dp_eggs[n][k] = sys.maxsize
    for i in range(1, n + 1):
        max_moves: int = max(twoEggDropUtil(n=i - 1, k=k - 1, dp_eggs=dp_eggs),
                             twoEggDropUtil(n=n - i, k=k, dp_eggs=dp_eggs)) + 1
        dp_eggs[n][k] = min(dp_eggs[n][k], max_moves)

    return dp_eggs[n][k]


def twoEggDrop(self, n: int) -> int:
    dp_eggs: List[List[int]] = [[-1] * 3 for _ in range(1001)]
    return twoEggDropUtil(n=n, k=2, dp_eggs=dp_eggs)


# https://leetcode.com/problems/guess-the-majority-in-a-hidden-array/description/

class ArrayReader(object):

    @staticmethod
    def query(a: int, b: int, c: int, d: int) -> int:
        return 4

    @staticmethod
    def length() -> int:
        return -1


def guessMajority(self, reader: ArrayReader) -> int:
    n: int = reader.length()
    q1: int = reader.query(0, 1, 2, 3)
    same: int = 1
    diff_value: int = 0
    diffIndex: int = -1

    for i in range(4, n):
        q: int = reader.query(0, 1, 2, i)
        if q1 == q:
            same += 1
        else:
            diff_value += 1
            if diffIndex == -1:
                diffIndex = i

    # compare 3 and 0
    q1: int = reader.query(1, 2, 3, 4)
    q: int = reader.query(0, 1, 2, 4)
    if q1 == q:
        same += 1
    else:
        diff_value += 1
        if diffIndex == -1:
            diffIndex = 0

    # compare 3 and 1
    q1: int = reader.query(0, 2, 3, 4)
    q: int = reader.query(0, 1, 2, 4)
    if q1 == q:
        same += 1
    else:
        diff_value += 1
        if diffIndex == -1:
            diffIndex = 1

    # compare 3 and 2
    q1: int = reader.query(0, 1, 3, 4)
    q: int = reader.query(0, 1, 2, 4)
    if q1 == q:
        same += 1
    else:
        diff_value += 1
        if diffIndex == -1:
            diffIndex = 2

    if same > diff_value:
        return 3
    elif same < diff_value:
        return diffIndex
    return -1


# https://leetcode.com/problems/where-will-the-ball-fall/

def isValid(self, i: int, j: int, m: int, n: int) -> bool:
    return 0 <= i < m and 0 <= j < n


def findBallUtil(self, r: int, c: int, m: int, n: int, grid: List[List[int]], dp_ball: List[List[int]]) -> int:
    if r == m:
        return c
    if not self.isValid(i=r, j=c, m=m, n=n):
        return -1
    if dp_ball[r][c] != -2:
        return dp_ball[r][c]

    if grid[r][c] == 1 and self.isValid(i=r, j=c + 1, m=m, n=n, ) and grid[r][c + 1] == 1:
        dp_ball[r][c] = self.findBallUtil(r=r + 1, c=c + 1, m=m, n=n, grid=grid, dp_ball=dp_ball)
        return dp_ball[r][c]

    elif grid[r][c] == -1 and self.isValid(i=r, j=c - 1, m=m, n=n, ) and grid[r][c - 1] == -1:
        dp_ball[r][c] = self.findBallUtil(r=r + 1, c=c - 1, m=m, n=n, grid=grid, dp_ball=dp_ball)
        return dp_ball[r][c]

    return -1


def findBall(self, grid: List[List[int]]) -> List[int]:
    m: int = len(grid)
    n: int = len(grid[0])
    dp_ball: List[List[int]] = [[-2] * n for _ in range(m)]
    res: List[int] = []
    for j in range(n):
        res.append(self.findBallUtil(r=0, c=j, m=m, n=n, grid=grid, dp_ball=dp_ball))

    return res


# https://leetcode.com/problems/sort-characters-by-frequency/

def frequencySort(self, s: str) -> str:
    freq_map: Dict[s, int] = {}
    for w in s:
        freq_map[w] = freq_map.get(w, 0) + 1

    sorted_dict: Dict[s, int] = dict(sorted(freq_map.items(), key=lambda x: x[1], reverse=True))
    res: str = ""
    for k, v in sorted_dict.items():
        for _ in range(v):
            res = res + k

    return res


# https://leetcode.com/problems/sort-integers-by-the-power-value/

def powerCompare(item1: Tuple[int, int], item2: Tuple[int, int]):
    if item1[1] < item2[1]:
        return -1
    elif item1[1] > item2[1]:
        return 1
    elif item1[0] < item2[0]:
        return -1
    else:
        return 1


def getKth(self, lo: int, hi: int, k: int) -> int:
    no_power_map: Dict[int, int] = {}
    for no in range(lo, hi + 1):
        n: int = no
        power: int = 0
        while n != 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = n * 3 + 1
            power += 1

        no_power_map[no] = power

    sorted_no: List[int] = [k[0] for k in sorted(no_power_map.items(), key=cmp_to_key(powerCompare))]
    return sorted_no[k - 1]


# https://leetcode.com/problems/binary-search-tree-iterator/description/

class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.res_list: List[int] = []
        self.inorderProcess(node=root, res=self.res_list)
        self.i: int = 0
        self.n: int = len(self.res_list)

    def inorderProcess(self, node: Optional[TreeNode], res: List[int]):
        if not node:
            return
        self.inorderProcess(node=node.left, res=res)
        res.append(node.val)
        self.inorderProcess(node=node.right, res=res)

    def next(self) -> int:
        if self.hasNext():
            v: int = self.res_list[self.i]
            self.i += 1
            return v
        return -1

    def hasNext(self) -> bool:
        return self.i < self.n


# https://leetcode.com/problems/find-distance-in-a-binary-tree/

def get_lca(self, root: Optional[TreeNode], p: int, q: int) -> Optional[TreeNode]:
    if not root:
        return None
    if root.val == p or root.val == q:
        return root
    left_node: Optional[TreeNode] = self.get_lca(root=root.left, p=p, q=q, )
    right_node: Optional[TreeNode] = self.get_lca(root=root.right, p=p, q=q, )

    if left_node is not None and right_node is not None:
        return root
    return left_node if left_node is not None else right_node


def diff(self, node: Optional[TreeNode], t: int, d: int) -> int:
    if not node:
        return 0
    if node.val == t:
        return d
    return max(self.diff(node=node.left, t=t, d=d + 1), self.diff(node=node.right, t=t, d=d + 1))


def findDistance(self, root: Optional[TreeNode], p: int, q: int) -> int:
    lca_node: Optional[TreeNode] = self.get_lca(root=root, p=p, q=q)

    left_diff: int = self.diff(node=lca_node, t=p, d=0)
    right_diff: int = self.diff(node=lca_node, t=q, d=0)
    return left_diff + right_diff


# https://leetcode.com/problems/validate-stack-sequences/description/

def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
    seqStack: Deque[int] = deque()
    i: int = 0
    j: int = 0
    m: int = len(pushed)
    n: int = len(popped)

    while i < m:
        seqStack.append(pushed[i])
        while j < n and len(seqStack) > 0 and seqStack[-1] == popped[j]:
            seqStack.pop()
            j += 1
        i += 1

    return i == m and j == n and len(seqStack) == 0


# https://leetcode.com/problems/stone-game/description/

def stoneGameUtils(piles: List[int], dp_arr: Dict[Tuple[int, int], int], i: int, j: int) -> int:
    if i > j:
        return 0
    if (i, j) in dp_arr:
        return dp_arr[(i, j)]

    is_even: bool = (j - i) % 2 == 0
    if not is_even:
        dp_arr[(i, j)] = max(piles[i] + stoneGameUtils(piles=piles, dp_arr=dp_arr, i=i + 1, j=j),
                             piles[j] + stoneGameUtils(piles=piles, dp_arr=dp_arr, i=i, j=j - 1))
    else:
        dp_arr[(i, j)] = min(- piles[i] + stoneGameUtils(piles=piles, dp_arr=dp_arr, i=i + 1, j=j),
                             - piles[j] + stoneGameUtils(piles=piles, dp_arr=dp_arr, i=i, j=j - 1))

    return dp_arr[(i, j)]


def stoneGame(self, piles: List[int]) -> bool:
    dp_arr: Dict[Tuple[int, int], int] = {}
    return stoneGameUtils(piles=piles, dp_arr=dp_arr, i=0, j=len(piles) - 1) > 0


# https://leetcode.com/problems/delete-nodes-and-return-forest/

def delNodesUtil(self, node: Optional[TreeNode], to_delete: Set[int], res: List[TreeNode]) -> Optional[TreeNode]:
    if not node:
        return None
    node.left = self.delNodesUtil(node=node.left, to_delete=to_delete, res=res)
    node.right = self.delNodesUtil(node=node.right, to_delete=to_delete, res=res)
    if node.val in to_delete:
        if node.left is not None:
            res.append(node.left)

        if node.right is not None:
            res.append(node.right)

        return None
    return node


def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
    to_del_set: Set[int] = set(to_delete)
    res: List[TreeNode] = []
    node: Optional[TreeNode] = self.delNodesUtil(node=root, to_delete=to_del_set, res=res)
    if node is not None:
        res.append(node)
    return res


# https://leetcode.com/problems/custom-sort-string/description/

def customSortString(self, order: str, s: str) -> str:
    mapping: Dict[str, int] = {}
    mapping_arr: Dict[int, Tuple[str, int]] = {}
    for i, o in enumerate(order):
        mapping[o] = i

    rem_list: List[str] = []
    for alphabet in s:
        if alphabet in mapping:
            idx: int = mapping.get(alphabet)
            t: Tuple[str, int] = mapping_arr.get(idx, (alphabet, 0))
            t_new: Tuple[str, int] = (t[0], t[1] + 1)
            mapping_arr[idx] = t_new
        else:
            rem_list.append(alphabet)

    res_list: List[str] = []
    for i in range(0, 26):
        if i in mapping_arr:
            t: Tuple[str, int] = mapping_arr.get(i)
            for j in range(t[1]):
                res_list.append(t[0])

    return "".join(res_list + rem_list)


# https://leetcode.com/problems/minimum-falling-path-sum/description/

def minFallingPathSumUtil(m: int, n: int, matrix: List[List[int]], dp_min_fall: List[List[int]]) -> None:
    for j in range(0, n):
        dp_min_fall[m - 1][j] = matrix[m - 1][j]

    for i in range(m - 2, -1, -1):
        for j in range(n - 1, -1, -1):
            for new_j in [-1, 0, 1]:
                new_value: int = (dp_min_fall[i + 1][j + new_j] + matrix[i][j]) if 0 <= j + new_j < n else sys.maxsize
                dp_min_fall[i][j] = min(dp_min_fall[i][j], new_value)


def minFallingPathSum(self, matrix: List[List[int]]) -> int:
    m: int = len(matrix)
    n: int = len(matrix[0])
    dp_min_fall: List[List[int]] = [[sys.maxsize] * n for _ in range(m)]
    minFallingPathSumUtil(m=m, n=n, matrix=matrix, dp_min_fall=dp_min_fall)
    res: int = sys.maxsize
    for j in range(0, n):
        res = min(res, dp_min_fall[0][j])

    return res


# https://leetcode.com/problems/find-positive-integer-solution-for-a-given-equation/

def f(x, y):
    return x + y


def findSolution(self, z: int) -> List[List[int]]:
    res: List[List[int]] = []
    x: int = 1
    y: int = 1000
    while x <= 1000 and y > 0:
        v: int = f(x, y)
        if v == z:
            res.append([x, y])
            x += 1
            y -= 1
        elif v < z:
            x += 1
        else:
            y -= 1

    return res


# https://leetcode.com/problems/regions-cut-by-slashes/

global regionsCount


def find(parents: List[int], x: int) -> int:
    if parents[x] == x:
        return x
    p: int = find(parents=parents, x=parents[x])
    parents[x] = p
    return p


def union(parents: List[int], rank: List[int], a: int, b: int) -> None:
    global regionsCount
    pA: int = find(parents=parents, x=a)
    pB: int = find(parents=parents, x=b)

    if pA != pB:
        if rank[pA] > rank[pB]:
            parents[pB] = pA
        elif rank[pA] < rank[pB]:
            parents[pA] = pB
        else:
            parents[pA] = pB
            rank[pB] += 1
    else:
        regionsCount += 1


def regionsBySlashes(self, grid: List[str]) -> int:
    m: int = len(grid)
    dots: int = m + 1

    parents: List[int] = [0] * (dots * dots)
    rank: List[int] = [0] * (dots * dots)
    for i in range(len(parents)):
        parents[i] = i
        rank[i] = 1

    for i in range(dots):
        for j in range(dots):
            if i == 0 or i == dots - 1 or j == 0 or j == dots - 1:
                cellNo: int = i * dots + j
                if cellNo == 0:
                    continue
                union(parents=parents, rank=rank, a=0, b=cellNo)

    for i in range(m):
        for j in range(m):
            if grid[i][j] == '/':
                cellNo1: int = i * dots + (j + 1)
                cellNo2: int = (i + 1) * dots + j
                union(parents=parents, rank=rank, a=cellNo1, b=cellNo2)
            elif grid[i][j] == '\\':
                cellNo1: int = i * dots + j
                cellNo2: int = (i + 1) * dots + (j + 1)
                union(parents=parents, rank=rank, a=cellNo1, b=cellNo2)

    return regionsCount


# https://leetcode.com/problems/combination-sum/description/

def combinationSumUtil(self, candidates: List[int], target: int, running_s: int, running: List[int], i: int,
                       res: List[List[int]]) -> None:
    if running_s > target or i >= len(candidates):
        return
    if running_s == target:
        res.append(list(running))
        return
    running.append(candidates[i])
    self.combinationSumUtil(candidates=candidates, target=target, running_s=running_s + candidates[i], running=running,
                            i=i, res=res)
    running.pop()
    self.combinationSumUtil(candidates=candidates, target=target, running_s=running_s, running=running,
                            i=i + 1, res=res)


def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    res: List[List[int]] = []
    self.combinationSumUtil(candidates=candidates, target=target, running_s=0, running=[], i=0, res=res)
    return res


# https://leetcode.com/problems/peak-index-in-a-mountain-array/description/

def peakIndexInMountainArray(self, arr: List[int]) -> int:
    low: int = 0
    high: int = len(arr) - 1
    while low <= high:
        mid: int = low + (high - low) // 2
        if arr[mid - 1] < arr[mid] > arr[mid + 1]:
            return mid
        elif arr[mid] < arr[mid + 1]:
            low = mid + 1
        else:
            high = mid - 1

    return -1


# https://leetcode.com/problems/kill-process/description/

def killProcessUtils(node: int, pc_m: Dict[int, List[int]], res: List[int]) -> None:
    res.append(node)
    for child in pc_m.get(node, []):
        killProcessUtils(node=child, pc_m=pc_m, res=res)


def killProcess(self, pid: List[int], ppid: List[int], kill: int) -> List[int]:
    n: int = len(pid)
    res: List[int] = []
    parent_child_mapping: Dict[int, List[int]] = {}
    for i in range(n):
        parent: int = ppid[i]
        parent_child_list: List[int] = parent_child_mapping.get(parent, [])
        parent_child_list.append(pid[i])
        parent_child_mapping[parent] = parent_child_list

    killProcessUtils(node=kill, pc_m=parent_child_mapping, res=res)
    return res


# https://leetcode.com/problems/extract-kth-character-from-the-rope-tree/

class RopeTreeNode(object):
    def __init__(self, length=0, val="", left=None, right=None):
        self.len = length
        self.val = val
        self.left = left
        self.right = right


def getKthCharacterUtils(node: Optional[RopeTreeNode]) -> str:
    if not node:
        return ""
    if node.len == 0:
        return node.val
    left_str: str = getKthCharacterUtils(node=node.left)
    right_str: str = getKthCharacterUtils(node=node.right)
    return left_str + right_str


def getKthCharacter(self, root: Optional[RopeTreeNode], k: int) -> str:
    str_value: str = getKthCharacterUtils(node=root)
    return str_value[k - 1]


# https://leetcode.com/problems/merge-strings-alternately/description/

def mergeAlternately(self, word1: str, word2: str) -> str:
    i: int = 0
    j: int = 0
    m: int = len(word1)
    n: int = len(word2)
    res: List[str] = []
    while i < m and j < n:
        res.append(word1[i])
        res.append(word2[j])
        i += 1
        j += 1

    while i < m:
        res.append(word1[i])
        i += 1

    while j < n:
        res.append(word2[j])
        j += 1

    return "".join(res)


# https://leetcode.com/problems/number-of-good-ways-to-split-a-string/

def numSplits(self, s: str) -> int:
    mapping: Dict[str, int] = {}
    running_set: Set[str] = set()
    for w in s:
        mapping[w] = mapping.get(w, 0) + 1

    res: int = 0
    for w in s:
        running_set.add(w)
        w_f: int = mapping.get(w, 0) - 1
        if w_f > 0:
            mapping[w] = w_f
        else:
            del mapping[w]

        if len(running_set) == len(mapping.keys()):
            res += 1

    return res


# https://leetcode.com/problems/minimum-cost-to-connect-sticks/

def connectSticks(self, sticks: List[int]) -> int:
    priority_queue: List[int] = []
    for stick in sticks:
        heapq.heappush(priority_queue, stick)

    res: int = 0
    while len(priority_queue) >= 2:
        smallest: int = heapq.heappop(priority_queue)
        smallest_s: int = heapq.heappop(priority_queue)
        sum_v: int = smallest + smallest_s
        heapq.heappush(priority_queue, sum_v)
        res += sum_v

    return res


# https://leetcode.com/problems/number-of-enclaves/description/

def numEnclavesUtils(grid: List[List[int]], i: int, j: int, m: int, n: int, directions: List[List[int]]) -> int:
    if i == -1 or i == m or j == -1 or j == n:
        return -1

    if grid[i][j] == 0:
        return 0

    grid[i][j] = 0
    v: int = 1
    out_flag: bool = False
    for direction in directions:
        new_i: int = i + direction[0]
        new_j: int = j + direction[1]
        v_d: int = numEnclavesUtils(grid=grid, i=new_i, j=new_j, m=m, n=n, directions=directions)
        if v_d == -1:
            out_flag = True
        v += v_d

    return -1 if out_flag else v


def numEnclaves(self, grid: List[List[int]]) -> int:
    res: int = 0
    m: int = len(grid)
    n: int = len(grid[0])
    directions: List[List[int]] = [[0, 1], [0, -1], [-1, 0], [1, 0]]
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                curr_res: int = numEnclavesUtils(grid=grid, i=i, j=j, m=m, n=n, directions=directions)
                res += 0 if curr_res == -1 else curr_res

    return res


# https://leetcode.com/problems/design-a-leaderboard/

def custom_key(element: Tuple[int, int]):
    return - (element[1])


class Leaderboard:

    def __init__(self):
        self.sorted_set: SortedSet[Tuple[int, int]] = SortedSet(key=custom_key)
        self.map: Dict[int, int] = defaultdict()

    def addScore(self, playerId: int, score: int) -> None:
        if playerId in self.map:
            oldScore: int = self.map[playerId]
            oldScoreTup: Tuple[int, int] = (playerId, oldScore)
            self.sorted_set.discard(oldScoreTup)

            newScore: int = oldScore + score
            newScoreTup: Tuple[int, int] = (playerId, newScore)
            self.sorted_set.add(newScoreTup)
            self.map[playerId] = newScore
        else:
            self.map[playerId] = score
            self.sorted_set.add((playerId, score))

    def top(self, K: int) -> int:
        sum_v: int = 0
        for tup in self.sorted_set:
            if K == 0:
                break
            sum_v += tup[1]
            K -= 1

        return sum_v

    def reset(self, playerId: int) -> None:
        self.sorted_set.discard((playerId, self.map.get(playerId, 0)))
        del self.map[playerId]


# https://leetcode.com/problems/design-hit-counter/

class HitCounter:

    def __init__(self):
        self.hitStore: Deque[int] = deque()

    def hit(self, timestamp: int) -> None:
        self.hitStore.append(timestamp)

    def getHits(self, timestamp: int) -> int:
        while len(self.hitStore) > 0 and self.hitStore[0] < timestamp - 300:
            self.hitStore.popleft()

        return len(self.hitStore)


# https://leetcode.com/problems/substrings-that-begin-and-end-with-the-same-letter/

def numberOfSubstrings(self, s: str) -> int:
    mapping: Dict[str, int] = {}
    for w in s:
        mapping[w] = mapping.get(w, 0) + 1

    res: int = 0
    for w in s:
        res += mapping[w]
        mapping[w] -= 1

    return res


# https://leetcode.com/problems/combination-sum-iii/description/

def combinationSum3Util(nums: List[int], i: int, k: int, n: int, s: int, running: List[int],
                        res: List[List[int]]) -> None:
    if s == n and len(running) == k:
        res.append(running.copy())
        return
    if s > n or len(running) > k:
        return
    if i >= len(nums):
        return
    running.append(nums[i])
    combinationSum3Util(nums=nums, i=i + 1, k=k, n=n, s=s + nums[i], running=running, res=res)
    running.pop()
    combinationSum3Util(nums=nums, i=i + 1, k=k, n=n, s=s, running=running, res=res)


def combinationSum3(self, k: int, n: int) -> List[List[int]]:
    res: List[List[int]] = []
    nums: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    combinationSum3Util(nums=nums, i=0, k=k, n=n, s=0, running=[], res=res)
    return res


# https://leetcode.com/problems/single-number-iii/description/

def singleNumber(self, nums: List[int]) -> List[int]:
    xor_no: int = 0
    for num in nums:
        xor_no ^= num

    rmb: int = 1
    while xor_no & rmb != rmb:
        rmb <<= 1

    x: int = 0
    y: int = 0

    for num in nums:
        if num & rmb:
            x ^= num
        else:
            y ^= num

    return [x, y]


# https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/description/

def get_days(weights: List[int], max_w: int) -> int:
    days: int = 0
    s: int = 0
    for w in weights:
        s += w
        if s > max_w:
            s = w
            days += 1

    return days


def shipWithinDays(self, weights: List[int], days: int) -> int:
    min_w: int = 0
    max_w: int = 0
    for w in weights:
        min_w = max(min_w, w)
        max_w += w

    optimal_w: int = max_w
    while min_w <= max_w:
        mid: int = min_w + (max_w - min_w) // 2
        no_days: int = self.get_days(weights=weights, max_w=mid)
        if no_days > days:
            min_w = mid + 1
        else:
            optimal_w = mid
            max_w = mid - 1

    return optimal_w


# https://leetcode.com/problems/find-duplicate-file-in-system/description/

def findDuplicate(self, paths: List[str]) -> List[List[str]]:
    mapping: Dict[str, List[str]] = dict()
    for path in paths:
        path_arr: List[str] = path.split(" ")
        root: str = path_arr[0]
        for i in range(1, len(path_arr)):
            file_arr: List[str] = path_arr[i].split("(")
            name: str = root + "/" + file_arr[0]
            content: str = file_arr[-1][0:-1]
            name_arr: List[str] = mapping.get(content, [])
            name_arr.append(name)
            mapping[content] = name_arr

    res: List[List[str]] = []
    for v in mapping.values():
        if len(v) > 1:
            res.append(v)

    return res


# https://leetcode.com/problems/longest-word-with-all-prefixes/description/

def addInTrie(root: Trie, word: str, i: int) -> None:
    if i >= len(word):
        root.is_word = True
        return

    c: str = word[i]
    child_node: Trie = root.children.get(c, Trie(val=c))
    root.children[c] = child_node
    addInTrie(root=child_node, word=word, i=i + 1)


def processTrie(root: Trie, sorted_map: SortedDict[int, List[str]], running: List[str]) -> None:
    if not root.is_word:
        return
    running.append(root.val)

    length: int = len(running)
    list_l: List[str] = sorted_map.get(length, [])
    list_l.append("".join(running))
    sorted_map[length] = list_l

    for child in root.children.values():
        processTrie(root=child, sorted_map=sorted_map, running=running)
    running.pop()


def longestWord(self, words: List[str]) -> str:
    sorted_map: SortedDict[int, List[str]] = SortedDict()
    root: Trie = Trie(val="")
    root.is_word = True
    for word in words:
        addInTrie(root=root, word=word, i=0, )

    processTrie(root=root, sorted_map=sorted_map, running=[])

    for k in reversed(sorted_map):
        v: List[str] = sorted_map[k]
        return sorted(v)[0]
    return ""


# https://leetcode.com/problems/combinations/description/

def combineUtils(i: int, n: int, k: int, running: List[int], res: List[List[int]]) -> None:
    if len(running) == k:
        res.append(list(running))
        return

    if i > n or len(running) > k:
        return

    running.append(i)
    combineUtils(i=i + 1, n=n, k=k, running=running, res=res)
    running.pop()
    combineUtils(i=i + 1, n=n, k=k, running=running, res=res)


def combine(self, n: int, k: int) -> List[List[int]]:
    res: List[List[int]] = []
    combineUtils(i=1, n=n, k=k, running=[], res=res)
    return res


# https://leetcode.com/problems/wiggle-sort/description/

def wiggleSort(self, nums: List[int]) -> None:
    n: int = len(nums)
    for i in range(1, n):
        if i % 2 != 0:
            first: int = nums[i - 1]
            second: int = nums[i]

            if first > second:
                nums[i - 1] = second
                nums[i] = first

            if i < n - 1 and nums[i + 1] > nums[i]:
                temp: int = nums[i]
                nums[i] = nums[i + 1]
                nums[i + 1] = temp


# https://leetcode.com/problems/palindromic-substrings/description/

def countSubstrings(self, s: str) -> int:
    n: int = len(s)
    res: int = 0
    for i in range(n):

        j: int = i
        k: int = i + 1
        while j >= 0 and k < n:
            if s[j] == s[k]:
                res += 1
                j -= 1
                k += 1
            else:
                break

        j = i
        k = i
        while j >= 0 and k < n:
            if s[j] == s[k]:
                res += 1
                j -= 1
                k += 1
            else:
                break

    return res


# https://leetcode.com/problems/game-of-life/description/

def gameOfLife(self, board: List[List[int]]) -> None:
    m: int = len(board)
    n: int = len(board[0])
    directions: List[List[int]] = [[0, 1], [1, 0], [0, -1], [-1, 0], [-1, -1], [1, 1], [-1, 1], [1, -1]]
    for i in range(m):
        for j in range(n):
            count: int = 0
            for direction in directions:
                new_i: int = i + direction[0]
                new_j: int = j + direction[1]
                if 0 <= new_i < m and 0 <= new_j < n:
                    if board[new_i][new_j] == 1 or board[new_i][new_j] == 10:
                        count += 1

            if board[i][j] == 1:
                if count < 2 or count > 3:
                    board[i][j] = 10
            else:
                if count == 3:
                    board[i][j] = -10

    for i in range(m):
        for j in range(n):
            if board[i][j] == 10:
                board[i][j] = 0
            elif board[i][j] == -10:
                board[i][j] = 1


# https://leetcode.com/problems/remove-all-ones-with-row-and-column-flips-ii/description/

def removeOnesUtils(grid: List[List[int]], m: int, n: int, row_col: Set[Tuple[str, int]], res: List[int],
                    flips: List[int]) -> None:
    found_one: bool = False

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and ('r', i) not in row_col and ('c', j) not in row_col:
                found_one = True
                row_col.add(('r', i))
                row_col.add(('c', j))
                flips[0] += 1
                removeOnesUtils(grid=grid, m=m, n=n, row_col=row_col, res=res, flips=flips)
                flips[0] -= 1
                row_col.remove(('r', i))
                row_col.remove(('c', j))

    if not found_one:
        res[0] = min(res[0], flips[0])


def removeOnes(self, grid: List[List[int]]) -> int:
    res: List[int] = [sys.maxsize]
    m: int = len(grid)
    n: int = len(grid[0])
    row_col: Set[Tuple[str, int]] = set()
    removeOnesUtils(grid=grid, m=m, n=n, row_col=row_col, res=res, flips=[0])
    return res[0]


# https://leetcode.com/problems/put-boxes-into-the-warehouse-i/

def maxBoxesInWarehouse(self, boxes: List[int], warehouse: List[int]) -> int:
    sorted_boxes: List[int] = sorted(boxes, reverse=True)
    warehouse_min: List[int] = []
    running_min: int = sys.maxsize

    m: int = len(boxes)
    n: int = len(warehouse)

    for i in range(n):
        running_min = min(running_min, warehouse[i])
        warehouse_min.append(running_min)

    i: int = m - 1
    count: int = 0
    for j in range(n - 1, -1, -1):
        if i >= 0 and sorted_boxes[i] <= warehouse_min[j]:
            i -= 1
            count += 1

    return count


# https://leetcode.com/problems/find-permutation/description/

def findPermutation(self, s: str) -> List[int]:
    n: int = len(s)
    res: List[int] = []
    perm_stack: Deque[int] = deque()

    for i in range(n):
        perm_stack.append(i + 1)
        if s[i] == 'I':
            while len(perm_stack) > 0:
                res.append(perm_stack.pop())

    perm_stack.append(n + 1)
    while len(perm_stack) > 0:
        res.append(perm_stack.pop())

    return res


# https://leetcode.com/problems/number-of-closed-islands/

def closedIslandUtil(grid: List[List[int]], i: int, j: int, m: int, n: int, directions: List[List[int]]) -> bool:
    if i < 0 or i == m or j < 0 or j == n:
        return False
    if grid[i][j] == 1:
        return True
    grid[i][j] = 1
    isClosed: bool = True
    for direction in directions:
        new_i: int = i + direction[0]
        new_j: int = j + direction[1]
        isClosed = closedIslandUtil(grid=grid, i=new_i, j=new_j, m=m, n=n, directions=directions) and isClosed

    return isClosed


def closedIsland(self, grid: List[List[int]]) -> int:
    count: int = 0
    directions: List[List[int]] = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    m: int = len(grid)
    n: int = len(grid[0])
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0 and closedIslandUtil(grid=grid, i=i, j=j, m=m, n=n, directions=directions):
                count += 1

    return count
