import functools
import heapq
from typing import List, Mapping, Optional, Set
from collections import deque

from src.main.python.leetcode.Models import *


# https://leetcode.com/problems/find-numbers-with-even-number-of-digits/

def findNumbers(self, nums: List[int]) -> int:
    count: int = 0
    for num in nums:
        length: int = 0
        if num == 0:
            continue

        while num > 0:
            length += 1
            num //= 10

        if length % 2 == 0:
            count += 1

    return count


# https://leetcode.com/problems/find-greatest-common-divisor-of-array/

def gcd(first: int, second: int) -> int:
    mod: int = first % second
    if mod == 0:
        return second

    div: int = first // second
    return gcd(first=max(second, div), second=min(second, div))


def findGCD(self, nums: List[int]) -> int:
    max_no: int = max(nums)
    min_no: int = min(nums)

    return gcd(max_no, min_no)


# https://leetcode.com/problems/find-target-indices-after-sorting-array/

def targetIndices(self, nums: List[int], target: int) -> List[int]:
    nums.sort()
    res: List[int] = []
    for i, a in enumerate(nums):
        if a == target:
            res.append(i)

    return res


# https://leetcode.com/problems/di-string-match/

def diStringMatch(self, s: str) -> List[int]:
    a: int = 0
    b: int = len(s)
    res: List[int] = []
    if len(s) == 0:
        return res

    for i in range(len(s)):
        if s[i] == 'I':
            res.append(a)
            a += 1
        else:
            res.append(b)
            b -= 1

    res.append(b)
    return res


# https://leetcode.com/problems/maximum-number-of-pairs-in-array/

def numberOfPairs(self, nums: List[int]) -> List[int]:
    mapping: Mapping[int, int] = {}
    for num in nums:
        c: int = mapping.get(num, 0)
        c += 1
        mapping[num] = c

    leftover_count: int = 0
    dup_count: int = 0
    for k, v in mapping.items():
        dup_count += v // 2
        leftover_count += v % 2

    return [dup_count, leftover_count]


# https://leetcode.com/problems/defanging-an-ip-address/

# noinspection SpellCheckingInspection
def defangIPaddr(self, address: str) -> str:
    res: List[str] = []
    for alphabet in address:
        if alphabet == '.':
            res.append('[')
            res.append(alphabet)
            res.append(']')
        else:
            res.append(alphabet)

    return "".join(res)


# https://leetcode.com/problems/jewels-and-stones/

def numJewelsInStones(self, jewels: str, stones: str) -> int:
    jewels_set: Set[str] = set()
    for jewel in jewels:
        jewels_set.add(jewel)

    count: int = 0
    for stone in stones:
        if stone in jewels_set:
            count += 1

    return count


# https://leetcode.com/problems/design-parking-system/

class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.slots: Mapping[int, int] = {1: big, 2: medium, 3: small}

    def addCar(self, carType: int) -> bool:
        count: int = self.slots[carType]
        if count == 0:
            return False
        self.slots[carType] -= 1
        return True


# https://leetcode.com/problems/subtract-the-product-and-sum-of-digits-of-an-integer/

def subtractProductAndSum(self, n: int) -> int:
    if n == 0:
        return 0
    product: int = 1
    total_sum: int = 0
    while n > 0:
        last_digit: int = n % 10
        product *= last_digit
        total_sum += last_digit
        n /= 10

    return product - total_sum


# https://leetcode.com/problems/range-sum-of-bst/

def rangeSumBST(root: Optional[TreeNode], low: int, high: int) -> int:
    subtree_sum: int = 0

    if not root:
        return subtree_sum
    elif low <= root.val <= high:
        subtree_sum += root.val + rangeSumBST(root.left, low, high) + rangeSumBST(root.right, low, high)
    elif low <= root.val:
        subtree_sum += rangeSumBST(root.left, low, high)
    else:
        subtree_sum += rangeSumBST(root.right, low, high)

    return subtree_sum


# https://leetcode.com/problems/sorting-the-sentence/

def comparator(a: str, b: str):
    a_value: int = int(a[-1])
    b_value: int = int(b[-1])
    if a_value < b_value:
        print(a_value, b_value)
        return 1
    else:
        print(a_value, b_value)
        return -1


def sortSentence(self, s: str) -> str:
    l: List[str] = s.split(" ")
    sorted_l: List[str] = sorted(l, key=functools.cmp_to_key(comparator))
    return " ".join([x[0: -1] for x in sorted_l])


# https://leetcode.com/problems/count-asterisks/

def countAsterisks(self, s: str) -> int:
    count: int = 0
    asterisk_count: int = 0
    for alphabet in s:
        if alphabet == '|':
            asterisk_count += 1
        elif alphabet == '*' and asterisk_count % 2 == 0:
            count += 1

    return count


# https://leetcode.com/problems/reverse-words-in-a-string-iii/

def reverseWords(self, s: str) -> str:
    l: List[str] = s.split(" ")
    for i, word in enumerate(l):
        l[i] = str(reversed(word))

    return " ".join(l)


# https://leetcode.com/problems/rings-and-rods/

def countPoints(self, rings: str) -> int:
    rods: List[Set[str]] = [set() for _ in range(10)]

    for i in range(0, len(rings) - 1, 2):
        color: str = rings[i]
        rod: int = int(rings[i + 1])

        rods[rod].add(color)

    count: int = 0
    for s in rods:
        if 'R' in s and 'G' in s and 'B' in s:
            count += 1

    return count


# https://leetcode.com/problems/remove-outermost-parentheses/

def removeOuterParentheses(self, s: str) -> str:
    n: int = len(s)
    indexes: List[int] = [-1] * n
    unbalanced_value: int = 0

    for i, p in enumerate(s):
        if p == '(':
            unbalanced_value += 1
        else:
            unbalanced_value -= 1

        if unbalanced_value == 1 and (i == 0 or indexes[i - 1] == 0):
            indexes[i] = 1
        elif unbalanced_value == 0:
            indexes[i] = 0

    return "".join([x for i, x in enumerate(s) if not (indexes[i] == 0 or indexes[i] == 1)])


# https://leetcode.com/problems/replace-all-digits-with-characters/

def shift(c: str, x: int) -> str:
    return chr(ord(c) + x)


def replaceDigits(self, s: str) -> str:
    l: List[str] = []

    for i, c in enumerate(s):
        if c.isdigit():
            new_c: str = shift(s[i - 1], int(c))
            l.append(new_c)
        else:
            l.append(c)

    return "".join(l)


# https://leetcode.com/problems/evaluate-boolean-binary-tree/

def evaluateTree(root: Optional[TreeNode]) -> bool:
    if not root:
        return False

    if root.left is None and root.right is None:
        return True if root.val == 1 else False

    return evaluateTree(root=root.left) or evaluateTree(root=root.right) \
        if root.val == 2 else evaluateTree(root=root.left) and evaluateTree(root=root.right)


# https://leetcode.com/problems/merge-two-binary-trees/

def mergeTrees(root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
    if root1 is None and root2 is None:
        return None

    if root1 is None or root2 is None:
        return root1 if root1 else root2

    sum_val: int = root1.val + root2.val

    left_node: Optional[TreeNode] = mergeTrees(root1=root1.left, root2=root2.left)
    right_node: Optional[TreeNode] = mergeTrees(root1=root1.right, root2=root2.right)

    new_node: Optional[TreeNode] = TreeNode(sum_val, left=left_node, right=right_node)
    return new_node


# https://leetcode.com/problems/search-in-a-binary-search-tree/

def searchBST(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    if not root:
        return None

    if root.val == val:
        return root

    elif root.val > val:
        return searchBST(root=root.left, val=val)

    else:
        return searchBST(root=root.right, val=val)


# https://leetcode.com/problems/destination-city/

# noinspection SpellCheckingInspection
def destCity(self, paths: List[List[str]]) -> str:
    incoming_map: Mapping[str, int] = {}
    outgoing_map: Mapping[str, int] = {}

    for path in paths:
        incoming: str = path[0]
        outgoing: str = path[1]
        incoming_map[incoming] = incoming_map.get(incoming, 0) + 1
        outgoing_map[outgoing] = outgoing_map.get(outgoing, 0) + 1

    for k, v in outgoing_map.items():
        if v == 1 and incoming_map.get(k, 0) == 0:
            return k

    return ""


# https://leetcode.com/problems/reverse-string/

def reverseString(self, s: List[str]) -> None:
    i: int = 0
    j: int = len(s) - 1
    while i < j:
        temp: str = s[i]
        s[i] = s[j]
        s[j] = temp
        i += 1
        j -= 1


# https://leetcode.com/problems/n-ary-tree-preorder-traversal/

def preorder_utils(node: Node, preorder_list: List[int]) -> None:
    if not node:
        return
    preorder_list.append(node.val)
    for child in node.children:
        preorder_utils(node=child, preorder_list=preorder_list)


def preorder(self, root: Node) -> List[int]:
    preorder_list: List[int] = []
    preorder_utils(node=root, preorder_list=preorder_list)
    return preorder_list


# https://leetcode.com/problems/first-letter-to-appear-twice/

def repeatedCharacter(self, s: str) -> str:
    count_arr: List[int] = [0] * 26

    for c in s:
        if count_arr[ord(c) - 97] == 1:
            return c
        count_arr[ord(c) - 97] += 1

    return ""


# https://leetcode.com/problems/logger-rate-limiter/

class Logger:
    def __init__(self):
        self.logger_timestamp_map: Mapping[str, int] = {}

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        should_print: bool = False
        if self.logger_timestamp_map.get(message, 0) <= timestamp:
            should_print = True
        self.logger_timestamp_map[message] = timestamp + 10

        return should_print


# https://leetcode.com/problems/sort-array-by-parity/

def sortArrayByParity(self, nums: List[int]) -> List[int]:
    i: int = 0
    j: int = len(nums) - 1

    while i < j:

        if nums[i] % 2 != 0 and nums[j] % 2 == 0:
            temp: int = nums[i]
            nums[i] = nums[j]
            nums[j] = temp

            i += 1
            j -= 1

        elif nums[i] % 2 == 0 and nums[j] % 2 != 0:
            i += 1
            j -= 1

        elif nums[i] % 2 != 0:
            j -= 1

        elif nums[j] % 2 == 0:
            i += 1

    return nums


# https://leetcode.com/problems/counting-bits/

def countBits(self, n: int) -> List[int]:
    bit_list: List[int] = [0] * (n + 1)
    for i, e in enumerate(bit_list):
        if i % 2 == 0:
            bit_list[i] = bit_list[i // 2]
        else:
            bit_list[i] = bit_list[i // 2] + 1

    return bit_list


# https://leetcode.com/problems/robot-return-to-origin/

def judgeCircle(self, moves: str) -> bool:
    vertical_pos: int = 0
    horizontal_pos: int = 0

    for move in moves:
        if move == 'U':
            vertical_pos += 1
        elif move == 'D':
            vertical_pos -= 1
        elif move == 'L':
            horizontal_pos += 1
        else:
            horizontal_pos -= 1

    return vertical_pos == 0 and horizontal_pos == 0


# https://leetcode.com/problems/merge-similar-items/

def mergeSimilarItems(self, items1: List[List[int]], items2: List[List[int]]) -> List[List[int]]:
    mapping: Mapping[int, int] = {}

    for item in items1:
        v = item[0]
        w = item[1]
        mapping[v] = mapping.get(v, 0) + w

    for item in items2:
        v = item[0]
        w = item[1]
        mapping[v] = mapping.get(v, 0) + w

    l: List[List[int]] = []
    for k, v in sorted(mapping.items()):
        l.append([k, v])

    return l


# https://leetcode.com/problems/height-checker/

def heightChecker(self, heights: List[int]) -> int:
    sorted_h: List[int] = sorted(heights)
    count: int = 0

    for i, e in enumerate(heights):
        if sorted_h[i] != e:
            count += 1

    return count


# https://leetcode.com/problems/hamming-distance/

def hammingDistance(self, x: int, y: int) -> int:
    v: int = x ^ y
    count: int = 0

    while v != 0:
        count = v % 2
        v //= 2

    return count


# https://leetcode.com/problems/maximum-units-on-a-truck/

def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
    box_type_l: List[tuple] = [(box[1], box[0]) for box in boxTypes]
    heapq._heapify_max(box_type_l)
    boxes: int = 0
    total_types: int = 0

    while boxes < truckSize:
        v: tuple = heapq.heappop(box_type_l)
        box_types: int = v[0]
        box: int = v[1]

        if boxes + box <= truckSize:
            total_types += box_types * box
            boxes += box
        else:
            total_types += box_types * (truckSize - boxes)
            boxes += truckSize - boxes

    return total_types


# https://leetcode.com/problems/middle-of-the-linked-list/

def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
    f: ListNode = head
    s: ListNode = head

    while f is not None and f.next is not None:
        f = f.next.next
        s = s.next

    return s


# https://leetcode.com/problems/unique-number-of-occurrences/

def uniqueOccurrences(self, arr: List[int]) -> bool:
    mapping: Mapping[int, int] = {}
    for a in arr:
        mapping[a] = mapping.get(a, 0) + 1

    distinct_v: set = set(mapping.values())
    return len(mapping.values()) == len(distinct_v)


# https://leetcode.com/problems/check-if-number-has-equal-digit-count-and-digit-value/

def digitCount(self, num: str) -> bool:
    digit_map: Mapping[int, int] = {}
    for no in num:
        no_int: int = int(no)
        digit_map[no_int] = digit_map.get(no_int, 0) + 1

    for i, no in enumerate(num):
        if digit_map.get(i, 0) != int(no):
            return False

    return True


# https://leetcode.com/problems/invert-binary-tree/

def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None

    left_node: Optional[TreeNode] = invertTree(root=root.left)
    right_node: Optional[TreeNode] = invertTree(root=root.right)

    root.left = right_node
    root.right = left_node

    return root


# https://leetcode.com/problems/count-prefixes-of-a-given-string/description/

def countPrefixes(self, words: List[str], s: str) -> int:
    n: int = len(s)
    count: int = 0

    for w in words:
        is_prefix: bool = True
        len_w: int = len(w)

        if len_w > n:
            continue

        for i in range(min(len_w, n)):
            if s[i] != w[i]:
                is_prefix = False
                break

        if is_prefix:
            count += 1

    return count


# https://leetcode.com/problems/binary-tree-inorder-traversal/

def inorderTraversalUtils(root: Optional[TreeNode], inorder_l: List[int]) -> None:
    if not root:
        return

    inorderTraversalUtils(root=root.left, inorder_l=inorder_l)
    inorder_l.append(root.val)
    inorderTraversalUtils(root=root.right, inorder_l=inorder_l)


def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    inorder_l: List[int] = []
    inorderTraversalUtils(root, inorder_l)

    return inorder_l


# https://leetcode.com/problems/maximum-depth-of-binary-tree/

def maxDepth(root: Optional[TreeNode]) -> int:
    if not root:
        return 0

    left_depth: int = maxDepth(root=root.left)
    right_depth: int = maxDepth(root=root.right)

    return max(left_depth, right_depth) + 1


# https://leetcode.com/problems/reverse-linked-list/

def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
        return None

    if head.next is None:
        return head

    reversed_list: Optional[ListNode] = reverseList(head=head.next)
    temp: Optional[ListNode] = head.next
    head.next = None
    temp.next = head

    return reversed_list


# https://leetcode.com/problems/squares-of-a-sorted-array/

def sortedSquares(self, nums: List[int]) -> List[int]:
    n: int = len(nums)
    i: int = 0
    while i < n and nums[i] < 0:
        i += 1

    i -= 1
    j: int = i + 1

    squared: List[int] = []
    while i >= 0 or j < n:

        if i >= 0 and j < n:

            a: int = nums[i] * nums[i]
            b: int = nums[j] * nums[j]

            if a < b:
                squared.append(a)
                i -= 1
            else:
                squared.append(b)
                j += 1

        elif i >= 0:
            a: int = nums[i] * nums[i]
            squared.append(a)
            i -= 1

        else:
            b: int = nums[j] * nums[j]
            squared.append(b)
            j += 1

    return squared


# https://leetcode.com/problems/smallest-index-with-equal-value/

def smallestEqual(self, nums: List[int]) -> int:
    for i, e in enumerate(nums):
        if i % 10 == e:
            return i

    return -1


# https://leetcode.com/problems/intersection-of-two-arrays/

def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    set_nums: Set = set(nums1)
    res: List[int] = []
    for num in nums2:
        if num in set_nums:
            res.append(num)
            set_nums.remove(num)

    return res


# https://www.geeksforgeeks.org/stack-in-python/

def removeDuplicates(self, s: str) -> str:
    stack: deque = deque()
    for word in s:
        if stack and stack[-1] == word:
            stack.pop()
        else:
            stack.append(word)

    return "".join(stack)


# https://leetcode.com/problems/single-number/

def singleNumber(self, nums: List[int]) -> int:
    return functools.reduce(lambda x, y: x ^ y, nums)


# https://leetcode.com/problems/baseball-game/

def calPoints(self, operations: List[str]) -> int:
    stack: deque = deque()

    for operation in operations:

        if operation.lstrip("-").isdigit():
            ops: int = int(operation)
            stack.append(ops)

        elif len(stack) >= 2 and operation == '+':
            s: int = stack[-1] + stack[-2]
            stack.append(s)

        elif len(stack) >= 1 and operation == 'D':
            d = 2 * stack[-1]
            stack.append(d)

        elif len(stack) >= 1:
            stack.pop()

    return sum(stack)


# https://leetcode.com/problems/sum-of-root-to-leaf-binary-numbers/

def bin_to_dec(running_no: List[int]) -> int:
    no: int = 0
    mul: int = 1
    for v in running_no[::-1]:
        if v == 1:
            no += mul
        mul *= 2

    return no


def sumRootToLeafUtil(root: TreeNode, sum_l: List[int], running_no: List[int]) -> None:
    if not root:
        return

    running_no.append(root.val)
    if root.left is None and root.right is None:
        no: int = bin_to_dec(running_no)
        sum_l[0] += no

    sumRootToLeafUtil(root=root.left, sum_l=sum_l, running_no=running_no)
    sumRootToLeafUtil(root=root.right, sum_l=sum_l, running_no=running_no)

    del running_no[-1]


def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
    sum_l: List[int] = []
    running_no: List[int] = []
    sumRootToLeafUtil(root=root, sum_l=sum_l, running_no=running_no)
    return sum_l[0]
