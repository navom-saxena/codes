from __future__ import annotations

import math
import sys
from collections import deque
from random import randrange
from typing import List, Set, Deque, Dict, Optional

from src.main.python.leetcode.Models import TreeNode, Node, LcaNode


# https://leetcode.com/problems/sqrtx

def mySqrt(self, x: int) -> int:
    if x == 1:
        return 1
    low: int = 0
    high: int = x // 2
    candidate: int = low
    while low <= high:
        mid: int = low + ((high - low) // 2)
        sq_no: int = mid * mid
        if sq_no == x:
            return mid
        elif sq_no > x:
            high = mid - 1
        else:
            candidate = mid
            low = mid + 1

    return candidate


# https://leetcode.com/problems/valid-boomerang/

def isBoomerang(self, points: List[List[int]]) -> bool:
    x1, y1 = points[0][0], points[0][1]
    x2, y2 = points[1][0], points[1][1]
    x3, y3 = points[2][0], points[2][1]
    # slope between 1 and 2
    slope12 = 0
    if x2 - x1 == 0:
        slope12 = 90
    else:
        slope12 = (y2 - y1) / (x2 - x1)

    # slope between 1 and 3
    slope13 = 0
    if x3 - x1 == 0:
        slope13 = 90
    else:
        slope13 = (y3 - y1) / (x3 - x1)

    # slope between 2 and 3
    slope23 = 0
    if x3 - x2 == 0:
        slope23 = 90
    else:
        slope23 = (y3 - y2) / (x3 - x2)

    if slope12 == slope13 or slope12 == slope23 or slope13 == slope23:
        return False
    return True


# https://leetcode.com/problems/check-if-n-and-its-double-exist/

def checkIfExist(self, arr: List[int]) -> bool:
    no_set: Set[int] = set()
    for num in arr:
        if (num % 2 == 0 and num // 2 in no_set) or num * 2 in no_set:
            return True
        no_set.add(num)

    return False


# https://leetcode.com/problems/excel-sheet-column-title/description/

def convertToTitle(self, columnNumber: int) -> str:
    excel_stack: Deque[str] = deque()
    while columnNumber:
        r: int = columnNumber % 26
        if r == 0:
            r = 26
        excel_stack.append(chr(r + 64))
        columnNumber -= r
        columnNumber //= 26

    res: str = ""
    while excel_stack:
        res += excel_stack.pop()

    return res


# https://leetcode.com/problems/valid-mountain-array/description/

def validMountainArray(self, arr: List[int]) -> bool:
    i: int = 1
    peak: bool = False
    while i < len(arr):
        if arr[i - 1] == arr[i]:
            return False
        if arr[i - 1] > arr[i]:
            if peak or i == 1:
                return False
            peak = True
            break
        i += 1

    while i < len(arr):
        if arr[i - 1] == arr[i]:
            return False
        if arr[i - 1] < arr[i]:
            return False
        i += 1

    return peak


# https://leetcode.com/problems/third-maximum-number/

def thirdMax(self, nums: List[int]) -> int:
    f_max: int = - sys.maxsize + 1
    s_max: int = - sys.maxsize + 1
    t_max: int = - sys.maxsize + 1
    for num in nums:
        if num > f_max:
            t_max = s_max
            s_max = f_max
            f_max = num
        elif f_max > num > s_max:
            t_max = s_max
            s_max = num
        elif s_max > num > t_max:
            t_max = num

    return f_max if t_max == - sys.maxsize + 1 else t_max


# https://leetcode.com/problems/missing-ranges/description/

def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
    res: List[List[int]] = []
    prev: int = lower - 1

    for i in range(0, len(nums) + 1):
        num = upper + 1 if i == len(nums) else nums[i]
        gap: int = num - prev
        if gap > 1:
            res.append([prev + 1, num - 1])
        prev = num

    return res


# https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/description/

def get_hcf(self, first: int, second: int) -> int:
    bigger: int = first if first >= second else second
    smaller: int = second if second <= first else first
    while smaller != 0:
        r: int = bigger % smaller
        if r == 0:
            return smaller
        bigger = smaller
        smaller = r

    return 1


def hasGroupsSizeX(self, deck: List[int]) -> bool:
    mapping: Dict[int, int] = {}
    for num in deck:
        mapping[num] = mapping.get(num, 0) + 1

    values_list: List[int] = list(mapping.values())
    if len(values_list) == 1:
        return values_list[0] > 1

    for i in range(0, len(values_list) - 1):
        if self.get_hcf(values_list[i], values_list[i + 1]) == 1:
            return False

    return True


# https://leetcode.com/problems/can-place-flowers/description/

def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    l: int = len(flowerbed)
    for i in range(l):
        if flowerbed[i] == 0:
            if (i == 0 or flowerbed[i - 1] == 0) and (i == l - 1 or flowerbed[i + 1] == 0):
                flowerbed[i] = 1
                n -= 1

    return n <= 0


# https://leetcode.com/problems/valid-word-abbreviation/description/

def validWordAbbreviation(self, word: str, abbr: str) -> bool:
    no: int = 0
    i: int = 0
    j: int = 0
    m: int = len(word)
    n: int = len(abbr)

    while i < m and j < n:
        c1: str = word[i]
        c2: str = abbr[j]

        if c1.isalpha() and c2.isalpha():
            if c1 != c2:
                return False
            i += 1
            j += 1
        else:
            if c2 == '0':
                return False

            no: int = 0
            while j < n and abbr[j].isnumeric():
                no = no * 10 + int(abbr[j])
                j += 1
            i += no

    return i == m and j == n


# https://leetcode.com/problems/long-pressed-name/description/

def isLongPressedName(self, name: str, typed: str) -> bool:
    i: int = 0
    j: int = 0
    m: int = len(name)
    n: int = len(typed)

    while i < m and j < n:
        c1: int = 1
        c2: int = 1
        p1: str = name[i]
        p2: str = typed[j]

        while i < m - 1 and name[i] == name[i + 1]:
            c1 += 1
            i += 1

        while j < n - 1 and typed[j] == typed[j + 1]:
            c2 += 1
            j += 1

        if c1 > c2 or p1 != p2:
            return False

        i += 1
        j += 1

    return i == m and j == n


# https://leetcode.com/problems/buddy-strings/description/

def buddyStrings(self, s: str, goal: str) -> bool:
    if len(s) != len(goal):
        return False

    if s == goal and len(s) != len(set(goal)):
        return True

    diffs: List[int] = []
    for i in range(len(s)):
        if s[i] != goal[i]:
            diffs.append(i)
            if len(diffs) > 2:
                return False

    return len(diffs) == 2 and s[diffs[0]] == goal[diffs[1]] and s[diffs[1]] == goal[diffs[0]]


# https://leetcode.com/problems/dot-product-of-two-sparse-vectors/

class SparseVector:
    def __init__(self, nums: List[int]):
        self.sv_dict: Dict[int, int] = {}
        for i, e in enumerate(nums):
            self.sv_dict[i] = e

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        dot_product: int = 0
        d1: Dict[int, int] = vec.sv_dict
        for k, v in d1.items():
            if k in self.sv_dict:
                dot_product += v * self.sv_dict[k]

        return dot_product


# https://leetcode.com/problems/minimize-product-sum-of-two-arrays/

def minProductSum(self, nums1: List[int], nums2: List[int]) -> int:
    nums1_sorted: List[int] = sorted(nums1)
    nums2_sorted: List[int] = sorted(nums2, reverse=True)

    min_product: int = 0
    for i in range(len(nums1)):
        min_product += nums1_sorted[i] * nums2_sorted[i]

    return min_product


# https://leetcode.com/problems/strictly-palindromic-number/

def get_base_str(n: int, base: int) -> str:
    base_stack: Deque[int] = deque()
    while n != 0:
        q: int = n // base
        r: int = n % base
        base_stack.append(r)
        n = q

    res: str = ""
    while base_stack:
        res += str(base_stack.pop())

    return res


def isPalindrome(number: str) -> bool:
    i: int = 0
    j: int = len(number) - 1
    while i < j:
        if number[i] != number[j]:
            return False
        i += 1
        j -= 1

    return True


def isStrictlyPalindromic(self, n: int) -> bool:
    for base in range(2, n - 1):
        base_str: str = get_base_str(n, base)
        if not isPalindrome(base_str):
            return False

    return True


# https://leetcode.com/problems/deepest-leaves-sum/

def deepestLeavesSumUtil(node: Optional[TreeNode], h: int, placeholder: List[int]) -> None:
    if not node:
        return
    if node.left is None and node.right is None:
        if placeholder[0] == h:
            placeholder[1] += node.val
        elif placeholder[0] < h:
            placeholder[0] = h
            placeholder[1] = node.val

        return

    deepestLeavesSumUtil(node=node.left, h=h + 1, placeholder=placeholder)
    deepestLeavesSumUtil(node=node.right, h=h + 1, placeholder=placeholder)


def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
    placeholder: List[int] = [0, 0]
    deepestLeavesSumUtil(node=root, h=0, placeholder=placeholder)
    return placeholder[1]


# https://leetcode.com/problems/queries-on-number-of-points-inside-a-circle/

def countPoints(self, points: List[List[int]], queries: List[List[int]]) -> List[int]:
    res: List[int] = [0] * len(queries)
    for i, query in enumerate(queries):
        x: int = query[0]
        y: int = query[1]
        r: int = query[2]
        for point in points:
            if math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2) <= r:
                res[i] += 1

    return res


# https://leetcode.com/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/

def minOperations(self, boxes: str) -> List[int]:
    left: int = 0
    right: int = 0
    steps: int = 0
    res: List[int] = [0] * len(boxes)
    for i in range(len(boxes)):
        if boxes[i] == '1':
            right += 1
            steps += i

    for i in range(0, len(boxes)):
        res[i] = steps
        if boxes[i] == '1':
            right -= 1
            left += 1
        steps += left
        steps -= right

    return res


# https://leetcode.com/problems/encode-and-decode-tinyurl/description/
class Codec:

    def __init__(self):
        self.char_arr: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.keys_dict: Dict[str, str] = {}

    def getKey(self) -> str:
        sb: str = ""
        for i in range(10):
            rand_int: int = randrange(62)
            sb += self.char_arr[rand_int]

        return sb

    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        key: str = self.getKey()
        while self.keys_dict.get(key, "") != "":
            key = self.getKey()

        self.keys_dict[key] = longUrl
        return key

    def decode(self, shortUrl: str) -> str:
        return self.keys_dict.get(shortUrl, "")


# https://leetcode.com/problems/partition-array-according-to-given-pivot/

def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
    k: int = 0
    same_count: int = 0
    larger: List[int] = []
    for i in range(len(nums)):
        if nums[i] < pivot:
            nums[k] = nums[i]
            k += 1
        elif nums[i] == pivot:
            same_count += 1
        else:
            larger.append(nums[i])

    while same_count:
        nums[k] = pivot
        same_count -= 1
        k += 1

    i: int = 0
    while k < len(nums):
        nums[k] = larger[i]
        i += 1
        k += 1

    return nums


# https://leetcode.com/problems/max-increase-to-keep-city-skyline/

def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
    row_max: List[int] = [0] * len(grid)
    col_max: List[int] = [0] * len(grid[0])

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            row_max[i] = max(row_max[i], grid[i][j])
            col_max[j] = max(col_max[j], grid[i][j])

    max_incr: int = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            max_incr += max(min(row_max[i], col_max[j]) - grid[i][j], 0)

    return max_incr


# https://leetcode.com/problems/clone-n-ary-tree/

def cloneTree(self, root: Node) -> Node:
    if not root:
        return root
    new_root: Node = Node(val=root.val)
    for child in root.children:
        new_root.children.append(self.cloneTree(root=child))

    return new_root


# https://leetcode.com/problems/minimize-maximum-pair-sum-in-array/

def minPairSum(self, nums: List[int]) -> int:
    sorted_nums: List[int] = sorted(nums)
    min_pair_sum: int = sys.maxsize
    i: int = 0
    j: int = len(nums) - 1
    max_sum: int = 0
    while i < j:
        sum_v: int = sorted_nums[i] + sorted_nums[j]
        max_sum = max(max_sum, sum_v)
        i += 1
        j -= 1

    return max_sum


# https://leetcode.com/problems/find-leaves-of-binary-tree/

def findLeavesUtil(root: Optional[TreeNode], depth_map: Dict[int, List[int]]) -> int:
    if not root:
        return -1
    left_depth: int = findLeavesUtil(root=root.left, depth_map=depth_map)
    right_depth: int = findLeavesUtil(root=root.right, depth_map=depth_map)
    final_depth: int = max(left_depth, right_depth) + 1
    if final_depth >= 0:
        depth_list: List[int] = depth_map.get(final_depth, [])
        depth_list.append(root.val)
        depth_map[final_depth] = depth_list
    return final_depth


def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
    depth_map: Dict[int, List[int]] = {}
    findLeavesUtil(root=root, depth_map=depth_map)
    res: List[List[int]] = [depth_map[k] for k in sorted(depth_map)]
    return res


# https://leetcode.com/problems/arithmetic-subarrays/

def checkArithmeticSubarrays(self, nums: List[int], l1: List[int], r: List[int]) -> List[bool]:
    res: List[bool] = [False] * len(l1)
    for i in range(len(l1)):
        left: int = l1[i]
        right: int = r[i]
        subArr: List[int] = nums[left:right + 1]
        sorted_subArr: List[int] = sorted(subArr)
        if len(sorted_subArr) < 2:
            continue
        d: int = sorted_subArr[1] - sorted_subArr[0]
        ap: bool = True
        for j in range(1, len(sorted_subArr)):
            if sorted_subArr[j] - sorted_subArr[j - 1] != d:
                ap = False
                break
        if ap:
            res[i] = True

    return res


# https://leetcode.com/problems/nested-list-weight-sum/

class NestedInteger:
    def __init__(self, value=None):
        """
       If value is not specified, initializes an empty list.
       Otherwise, initializes a single integer equal to value.
       """

    def isInteger(self):
        """
       @return True if this NestedInteger holds a single integer, rather than a nested list.
       rtype bool
       """

    def add(self, elem):
        """
       Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
       rtype void
       """

    def setInteger(self, value):
        """
       Set this NestedInteger to hold a single integer equal to value.
       rtype void
       """

    def getInteger(self):
        """
       @return the single integer that this NestedInteger holds, if it holds a single integer
       Return None if this NestedInteger holds a nested list
       :rtype int
       """

    def getList(self):
        """
       @return the nested list that this NestedInteger holds, if it holds a nested list
       Return None if this NestedInteger holds a single integer
       :rtype List[NestedInteger]
       """


def dfs_nested(nested_list, depth):
    total = 0
    for nested in nested_list:
        if nested.isInteger():
            total += nested.getInteger() * depth
        else:
            total += dfs_nested(nested.getList(), depth + 1)
    return total


def depthSum(self, nestedList: List[NestedInteger]) -> int:
    return dfs_nested(nested_list=nestedList, depth=1)


# https://leetcode.com/problems/all-paths-from-source-to-target/

def allPathsSourceTargetUtils(graph: List[List[int]], curr: int, running_docs: List[int], res: List[List[int]],
                              destination: int) -> None:
    if curr == destination:
        res.append(list(running_docs))
        return

    for neighbour in graph[curr]:
        running_docs.append(neighbour)
        allPathsSourceTargetUtils(graph=graph, curr=neighbour, running_docs=running_docs, res=res,
                                  destination=destination)
        running_docs.pop()


def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
    res: List[List[int]] = []
    allPathsSourceTargetUtils(graph=graph, curr=0, running_docs=[0], res=res, destination=len(graph) - 1)
    return res


# https://leetcode.com/problems/all-possible-full-binary-trees/description/

def getAllPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
    if n == 1:
        return [TreeNode()]
    res: List[Optional[TreeNode]] = []
    for i in range(1, n, 2):
        left_nodes: List[Optional[TreeNode]] = self.getAllPossibleFBT(i)
        right_nodes: List[Optional[TreeNode]] = self.getAllPossibleFBT(n - i - 1)
        for left_node in left_nodes:
            for right_node in right_nodes:
                node: Optional[TreeNode] = TreeNode()
                node.left = left_node
                node.right = right_node
                res.append(node)

    return res


def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
    return self.getAllPossibleFBT(n=n)


# https://leetcode.com/problems/design-an-expression-tree-with-evaluate-function/

class PostfixNode:

    def __init__(self, holder: str, left=None, right=None):
        self.holder: str = holder
        self.left: PostfixNode = left
        self.right: PostfixNode = right

    def evaluate(self) -> int:
        holder: str = self.holder
        left: PostfixNode = self.left
        right: PostfixNode = self.right
        if holder.isnumeric():
            return int(holder)
        if holder == '+':
            return left.evaluate() + right.evaluate()
        elif holder == '-':
            return left.evaluate() - right.evaluate()
        elif holder == '*':
            return left.evaluate() * right.evaluate()
        else:
            return left.evaluate() // right.evaluate()


def buildTreeUtils(postfix: List[str], idx: List[int]) -> Optional[PostfixNode]:
    if idx[0] < 0:
        return None
    holder: str = postfix[idx[0]]
    idx[0] -= 1
    node: PostfixNode = PostfixNode(holder=holder)
    if not holder.isnumeric():
        node: PostfixNode = PostfixNode(holder=holder)
        node.right = buildTreeUtils(postfix=postfix, idx=idx)
        node.left = buildTreeUtils(postfix=postfix, idx=idx)
    return node


def buildTree(self, postfix: List[str]) -> PostfixNode:
    idx: List[int] = [len(postfix) - 1]
    return buildTreeUtils(postfix=postfix, idx=idx)


# https://leetcode.com/problems/split-bst/
def splitBST(self, root: Optional[TreeNode], target: int) -> List[Optional[TreeNode]]:
    if not root:
        return [None, None]
    if root.val <= target:
        right_subTree_res: List[Optional[TreeNode]] = self.splitBST(root=root.right, target=target)
        root.right = right_subTree_res[0]
        right_subTree_res[0] = root
        return right_subTree_res
    else:
        left_subTree_res: List[Optional[TreeNode]] = self.splitBST(root=root.left, target=target)
        root.left = left_subTree_res[1]
        left_subTree_res[1] = root
        return left_subTree_res


# https://leetcode.com/problems/all-elements-in-two-binary-search-trees/description/

def preOrder_Traversal(node: Optional[TreeNode], v: List[int]) -> None:
    if not node:
        return
    preOrder_Traversal(node=node.left, v=v)
    v.append(node.val)
    preOrder_Traversal(node=node.right, v=v)


def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
    tree1_values: List[int] = []
    tree2_values: List[int] = []
    preOrder_Traversal(node=root1, v=tree1_values)
    preOrder_Traversal(node=root2, v=tree2_values)
    i: int = 0
    j: int = 0
    m: int = len(tree1_values)
    n: int = len(tree2_values)
    res: List[int] = []
    while i < m and j < n:
        if tree1_values[i] <= tree2_values[j]:
            res.append(tree1_values[i])
            i += 1
        else:
            res.append(tree2_values[j])
            j += 1

    while i < m:
        res.append(tree1_values[i])
        i += 1

    while j < n:
        res.append(tree2_values[j])
        j += 1

    return res


# https://leetcode.com/problems/partition-labels/

def partitionLabels(self, s: str) -> List[int]:
    char_mapping: Dict[str, List[int]] = {}
    for i, char in enumerate(s):
        first_last: List[int] = char_mapping.get(char, [])
        if not first_last:
            first_last = [i, i]
        else:
            first_last[1] = i
        char_mapping[char] = first_last

    sorted_fl: List[List[int]] = sorted(char_mapping.values(), key=lambda x: x[0])
    res: List[int] = []
    start: int = 0
    last: int = 0
    for char_l in sorted_fl:
        if char_l[0] > last:
            res.append(last - start + 1)
            start = char_l[0]

        last = max(last, char_l[1])

    res.append(last - start + 1)
    return res


# https://leetcode.com/problems/count-number-of-distinct-integers-after-reverse-operations/description/

def countDistinctIntegers(self, nums: List[int]) -> int:
    num_set: Set[int] = set()
    for num in nums:
        num_set.add(num)
        rev: int = 0
        while num:
            r: int = num % 10
            rev = rev * 10 + r
            num //= 10
        num_set.add(rev)

    return len(num_set)


# https://leetcode.com/problems/find-root-of-n-ary-tree/description/

def findRootUtils(self, node: Node, children_set: Set[Node]) -> None:
    if not node:
        return
    for child in node.children:
        if child not in children_set:
            children_set.add(child)
            self.findRootUtils(node=child, children_set=children_set)


def findRoot(self, tree: List[Node]) -> Node:
    children_set: Set[Node] = set()
    for node in tree:
        if node not in children_set:
            self.findRootUtils(node=node, children_set=children_set)
    for node in tree:
        if node not in children_set:
            return node
    return Node(-1)


# https://leetcode.com/problems/longest-common-subsequence-between-sorted-arrays/description/

def longestCommonSubsequence(self, arrays: List[List[int]]) -> List[int]:
    no_count: Dict[int, int] = {}
    for array in arrays:
        for num in array:
            no_count[num] = no_count.get(num, 0) + 1

    res: List[int] = []
    for k, v in no_count.items():
        if v == len(arrays):
            res.append(k)

    return sorted(res)


# https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/description/

def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
    to_edges: List[int] = [0] * n
    for edge in edges:
        from_node: int = edge[0]
        to_node: int = edge[1]
        to_edges[to_node] += 1

    res: List[int] = []
    for i in range(n):
        if not to_edges[i]:
            res.append(i)

    return res


# https://leetcode.com/problems/watering-plants/

def wateringPlants(self, plants: List[int], capacity: int) -> int:
    steps: int = 0
    running_capacity: int = capacity
    for i, water in enumerate(plants):
        rem: int = running_capacity - water
        if rem < 0:
            running_capacity = capacity - water
            steps += i + i + 1
        else:
            running_capacity = rem
            steps += 1

    return steps


# https://leetcode.com/problems/design-most-recently-used-queue/description/


class MruNode:

    def __init__(self, val: int):
        self.val = val
        self.next: Optional[MruNode] = None


class MRUQueue:

    def __init__(self, n: int):
        self.sentinel_node = MruNode(0)
        curr: MruNode = self.sentinel_node
        self.last: MruNode = self.sentinel_node
        for i in range(1, n + 1):
            curr.next = MruNode(i)
            curr = curr.next
            self.last = curr

    def getQueue(self):
        test = self.sentinel_node
        lis = []
        while test:
            lis.append(test.val)
            test = test.next
        return lis

    def fetch(self, k: int) -> int:
        curr: MruNode = self.sentinel_node
        prev: MruNode = self.sentinel_node
        for i in range(k):
            prev = curr
            curr = curr.next
        value: int = curr.val
        if curr != self.last:
            next_node = curr.next
            prev.next = next_node
            curr.next = None
            self.last.next = curr
            self.last = curr

        return value


# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/

def lowestCommonAncestor(self, p: LcaNode, q: LcaNode) -> LcaNode:
    actual_p: LcaNode = p
    actual_q: LcaNode = q
    while p != q:
        p = p.parent
        q = q.parent
        if p is None:
            p = actual_q
        if q is None:
            q = actual_p

    return p


# https://leetcode.com/problems/count-sorted-vowel-strings/description/

def countVowelStrings(self, n: int) -> int:
    counts: List[int] = [1] * 5
    for _ in range(n):
        prefix_sum: int = 0
        for i in range(5):
            prefix_sum += counts[i]
            counts[i] = prefix_sum

    return counts[4]


# https://leetcode.com/problems/find-and-replace-pattern/

def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
    pattern_order: List[str] = []
    for p in pattern:
        pattern_order.append(p)

    res: List[str] = []
    len_p: int = len(pattern)

    for word in words:

        mapping: Dict[str, str] = {}
        rev_mapping: Dict[str, str] = {}
        is_same: bool = True

        for i in range(len_p):
            w: str = word[i]
            p: str = pattern[i]
            if mapping.get(w, p) != p or rev_mapping.get(p, w) != w:
                is_same = False
                break
            mapping[w] = p
            rev_mapping[p] = w

        if not is_same:
            continue

        if is_same:
            res.append(word)

    return res


# https://leetcode.com/problems/find-valid-matrix-given-row-and-column-sums/

def get_min_idx(arr: List[int], contains_set: Set[int]) -> int:
    min_value: int = sys.maxsize
    min_idx: int = -1
    for i, a in enumerate(arr):
        if a < min_value and i not in contains_set:
            min_value = a
            min_idx = i

    return min_idx


def restoreMatrix(self, rowSum: List[int], colSum: List[int]) -> List[List[int]]:
    m: int = len(rowSum)
    n: int = len(colSum)

    row_set: Set[int] = set()
    col_set: Set[int] = set()
    matrix: List[List[int]] = []

    for r in range(m):
        matrix.append([0] * n)

    count: int = 0
    while len(row_set) != m and len(col_set) != n:

        rI: int = get_min_idx(rowSum, row_set)
        cI: int = get_min_idx(colSum, col_set)

        if rowSum[rI] < colSum[cI]:
            matrix[rI][cI] = rowSum[rI]
            colSum[cI] -= rowSum[rI]
            row_set.add(rI)
        else:
            matrix[rI][cI] = colSum[cI]
            rowSum[rI] -= colSum[cI]
            col_set.add(cI)

    return matrix


# https://leetcode.com/problems/find-the-winner-of-the-circular-game/

class DDLNode:

    def __init__(self, val):
        self.val: int | str = val
        self.next: Optional[DDLNode] = None
        self.prev: Optional[DDLNode] = None


def findTheWinner(self, n: int, k: int) -> int:
    start: Optional[DDLNode] = DDLNode(1)
    curr: Optional[DDLNode] = start
    for i in range(2, n + 1):
        node: Optional[DDLNode] = DDLNode(i)
        curr.next = node
        node.prev = curr
        curr = curr.next

    curr.next = start
    start.prev = curr

    curr = start

    while n > 1:

        for i in range(k - 1):
            curr = curr.next

        prev_node: Optional[DDLNode] = curr.prev
        next_node: Optional[DDLNode] = curr.next
        prev_node.next = next_node
        next_node.prev = prev_node
        curr = next_node
        n -= 1

    return curr.val


# https://leetcode.com/problems/design-browser-history/

class BrowserHistory:

    def __init__(self, homepage: str):
        self.start: Optional[DDLNode] = DDLNode(val=homepage)
        self.curr = self.start

    def visit(self, url: str) -> None:
        node: Optional[DDLNode] = DDLNode(val=url)
        self.curr.next = node
        node.prev = self.curr
        self.curr = self.curr.next

    def back(self, steps: int) -> str:
        while self.curr.prev and steps:
            self.curr = self.curr.prev
            steps -= 1

        return self.curr.val

    def forward(self, steps: int) -> str:
        while self.curr.next and steps:
            self.curr = self.curr.next
            steps -= 1

        return self.curr.val
