import functools
import heapq
import math
import sys
from heapq import heapify, heappush, heappop
from typing import List, Mapping, Optional, Set, Deque, Tuple
from collections import deque

from src.main.python.leetcode.Models import *


# https://leetcode.com/problems/check-if-the-sentence-is-pangram/

def checkIfPangram(self, sentence: str) -> bool:
    count_arr: List[int] = [0] * 26
    for word in sentence:
        count_arr[ord(word) - 97] += 1

    for f in count_arr:
        if f == 0:
            return False

    return True


# https://leetcode.com/problems/delete-columns-to-make-sorted/

def minDeletionSize(self, strs: List[str]) -> int:
    if len(strs) == 0:
        return 0

    count: int = 0
    n: int = len(strs[0])
    for i in range(n):
        c: int = 0
        for letters in strs:
            if ord(letters[i]) < c:
                count += 1
                break

            c = ord(letters[i])

    return count


# https://leetcode.com/problems/find-the-difference-of-two-arrays/

def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
    nums1_set: Set = set(nums1)
    nums2_set: Set = set(nums2)

    nums1_sub: List[int] = []
    nums2_sub: List[int] = []

    for no in nums1_set:
        if no not in nums2_set:
            nums1_sub.append(no)

    for no in nums2_set:
        if no not in nums1_set:
            nums2_sub.append(no)

    return [nums1_sub, nums2_sub]


# https://leetcode.com/problems/univalued-binary-tree/

def isUnivalTreeUtil(root: TreeNode, v: int) -> bool:
    if not root:
        return True

    if root.val != v:
        return False

    return isUnivalTreeUtil(root=root.left, v=v) and isUnivalTreeUtil(root=root.right, v=v)


def isUnivalTree(self, root: Optional[TreeNode]) -> bool:
    if not root:
        return True

    v: int = root.val
    return isUnivalTreeUtil(root=root, v=v)


# https://leetcode.com/problems/minimum-absolute-difference/

def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
    arr.sort()

    min_diff: float = math.inf
    for i in range(len(arr) - 1):

        diff: int = arr[i + 1] - arr[i]
        if diff < min_diff:
            min_diff = diff

    res: List[List[int]] = []
    for i in range(len(arr) - 1):
        diff: int = arr[i + 1] - arr[i]
        if diff == min_diff:
            res.append([arr[i], arr[i + 1]])

    return res


# https://leetcode.com/problems/pascals-triangle/

def generate(self, numRows: int) -> List[List[int]]:
    res: List[List[int]] = [[1]]
    if numRows == 1:
        return res
    res.append([1, 1])

    for i in range(2, numRows):
        ith_list: List[int] = [1]
        prev: List[int] = res[i - 1]
        for j in range(len(prev) - 1):
            v: int = prev[j] + prev[j + 1]
            ith_list.append(v)
        ith_list.append(1)
        res.append(ith_list)

    return res


# https://leetcode.com/problems/island-perimeter/

def calculate_perimeter(grid: List[List[int]], directions: List[List[int]], i: int, j: int) -> int:
    if i < 0 or i == len(grid) or j < 0 or j == len(grid[i]) or grid[i][j] == 0:
        return 1

    if grid[i][j] == -1:
        return 0

    grid[i][j] = -1
    perimeter: int = 0
    for direction in directions:
        perimeter += calculate_perimeter(grid=grid, directions=directions, i=i + direction[0], j=j + direction[1])

    return perimeter


def islandPerimeter(self, grid: List[List[int]]) -> int:
    perimeter: int = 0
    directions: List[List[int]] = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                perimeter += calculate_perimeter(grid, directions, i, j)

    return perimeter


# https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/

def sortedArrayToBSTUtils(nums: List[int], low: int, high: int) -> Optional[TreeNode]:
    if low > high:
        return None
    mid: int = low + (high - low) // 2
    node: Optional[TreeNode] = TreeNode(val=nums[mid])
    node.left = sortedArrayToBSTUtils(nums=nums, low=low, high=mid - 1)
    node.right = sortedArrayToBSTUtils(nums=nums, low=mid + 1, high=high)
    return node


def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
    return sortedArrayToBSTUtils(nums, 0, len(nums) - 1)


# https://leetcode.com/problems/fibonacci-number/

def fib(self, n: int) -> int:
    a: int = 0
    b: int = 1

    if n == 0:
        return 0
    elif n == 1:
        return 1

    i = 2
    while i <= n:
        c: int = a + b
        a = b
        b = c

    return b


# https://leetcode.com/problems/fizz-buzz/

def fizzBuzz(self, n: int) -> List[str]:
    res: List[str] = []

    for i in range(n):
        if (i + 1) % 3 == 0 and (i + 1) % 5 == 0:
            res.append("FizzBuzz")
        elif (i + 1) % 3 == 0:
            res.append("Fizz")
        elif (i + 1) % 5 == 0:
            res.append("Buzz")
        else:
            res.append("{}".format(i + 1))

    return res


# https://leetcode.com/problems/sort-array-by-increasing-frequency/

def compare(item1: List[int], item2: List[int]) -> int:
    if item1[1] > item2[1]:
        return -1
    elif item1[1] < item2[1]:
        return 1
    else:
        return item2[0] - item1[0]


def frequencySort(self, nums: List[int]) -> List[int]:
    mapping: Mapping[int, int] = {}

    for num in nums:
        mapping[num] = mapping.get(num, 0) + 1

    # noinspection PyTypeChecker
    sorted_mapping: Optional[List[Tuple[int, int]]] = sorted(mapping.items(), key=functools.cmp_to_key(compare))
    res: List[int] = []
    for k, v in sorted_mapping:
        for i in range(v):
            res.append(k)

    return res


# https://leetcode.com/problems/toeplitz-matrix/

def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
    m: int = len(matrix)
    n: int = len((matrix[0]))

    for i in range(m):
        v: int = matrix[i][0]
        j: int = 1
        while i + j < m and j < n:
            if matrix[i + j][j] != v:
                return False
            j += 1

    if n == 1:
        return True

    for j in range(1, n):
        v: int = matrix[0][j]
        i: int = 1
        while i < m and i + j < n:
            if matrix[i][i + j] != v:
                return False
            i += 1

    return True


# https://leetcode.com/problems/relative-sort-array/description/

def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
    order: List[int] = [0] * 1001
    arr2_set: Set[int] = set(arr2)
    not_in_arr2: List[int] = [0] * 1001

    for v in arr1:
        if v in arr2_set:
            order[v] += 1
        else:
            not_in_arr2[v] += 1

    res: List[int] = []
    for v in arr2:
        freq: int = order[v]
        for _ in range(freq):
            res.append(v)

    for i, v in enumerate(not_in_arr2):
        if v > 0:
            for _ in range(v):
                res.append(i)

    return res


# https://leetcode.com/problems/find-common-characters/description/

def commonChars(self, words: List[str]) -> List[str]:
    count_arr: List[int] = [sys.maxsize] * 26

    for word in words:
        single_word_count_arr: List[int] = [0] * 26
        for w in word:
            single_word_count_arr[ord(w) - 97] += 1

        for i in range(26):
            count_arr[i] = min(count_arr[i], single_word_count_arr[i])

    res: List[str] = []
    for i in range(26):
        for _ in range(count_arr[i]):
            res.append(chr(i + 97))

    return res


# https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/

def countCharacters(self, words: List[str], chars: str) -> int:
    chars_dict: Mapping[str, int] = {}

    for c in chars:
        chars_dict[c] = chars_dict.get(c, 0) + 1

    total_len: int = 0
    for w in words:
        words_dict: Mapping[str, int] = {}
        for c in w:
            words_dict[c] = words_dict.get(c, 0) + 1

        can_build: bool = True
        for k, v in words_dict.items():
            if not (k in chars_dict and v <= chars_dict.get(k, 0)):
                can_build = False
                break

        if can_build:
            total_len += len(w)

    return total_len


# https://leetcode.com/problems/leaf-similar-trees/

def leafSimilarUtils(root: Optional[TreeNode], leaves_list: List[int]) -> None:
    if not root:
        return

    if root.left is None and root.right is None:
        leaves_list.append(root.val)

    leafSimilarUtils(root=root.left, leaves_list=leaves_list)
    leafSimilarUtils(root=root.right, leaves_list=leaves_list)


def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
    root1_leaves_list: List[int] = []
    root2_leaves_list: List[int] = []
    leafSimilarUtils(root=root1, leaves_list=root1_leaves_list)
    leafSimilarUtils(root=root2, leaves_list=root2_leaves_list)

    if len(root1_leaves_list) != len(root2_leaves_list):
        return False

    for i in range(len(root2_leaves_list)):
        if root2_leaves_list[i] != root2_leaves_list[i]:
            return False

    return True


# https://leetcode.com/problems/number-complement/

def findComplement(self, num: int) -> int:
    no_comp_bin: List[int] = []
    while num != 0:
        rem: int = num % 2
        no_comp_bin.append(1 if rem is 0 else 0)
        num //= 2

    comp: int = 0
    for i, e in enumerate(no_comp_bin):
        comp += e * int(math.pow(2, i))

    return comp


# https://leetcode.com/problems/divisor-game/

def divisorGame(self, n: int) -> bool:
    return n % 2 == 0


# https://leetcode.com/problems/unique-email-addresses/

def numUniqueEmails(self, emails: List[str]) -> int:
    distinct_emails: Set[str] = set()
    for email in emails:
        split_arr: List[str] = email.split("@")
        email_id: str = split_arr[0].replace(".", "")
        email_id: str = email_id.split("+")[0]
        new_email: str = email_id + "@" + split_arr[1]
        distinct_emails.add(new_email)

    return len(distinct_emails)


# https://leetcode.com/problems/uncommon-words-from-two-sentences/

def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
    s1_split: List[str] = s1.split(" ")
    s2_split: List[str] = s2.split(" ")

    s1_dict: Mapping[str, int] = {}
    s2_dict: Mapping[str, int] = {}

    for v in s1_split:
        s1_dict[v] = s1_dict.get(v, 0) + 1

    for v in s2_split:
        s2_dict[v] = s2_dict.get(v, 0) + 1

    final_v: List[str] = []

    for k, v in s1_dict.items():
        if v == 1 and k not in s2_dict:
            final_v.append(k)

    for k, v in s2_dict.items():
        if v == 1 and k not in s1_dict:
            final_v.append(k)

    return final_v


# https://leetcode.com/problems/palindrome-permutation/

def canPermutePalindrome(self, s: str) -> bool:
    mapping: Mapping[str, int] = {}
    for v in s:
        mapping[v] = mapping.get(v, 0) + 1

    one_or_less: bool = True
    for k, v in mapping.items():
        if v % 2 != 0:
            if one_or_less:
                one_or_less = False
            else:
                return False

    return True


# https://leetcode.com/problems/similar-rgb-color/

def similarRGB_pair(c: str) -> str:
    min_diff: float = math.inf
    ans: int = -1

    for i in range(16):
        diff: float = (int(c, 16) - i * 17) ** 2
        if diff < min_diff:
            min_diff = diff
            ans = i

    return hex(ans)[-1] * 2


def similarRGB(self, color: str) -> str:
    ans: str = "#"
    for i in range(1, 6, 2):
        ans += similarRGB_pair(color[i:i + 2])

    return ans


# https://leetcode.com/problems/binary-tree-preorder-traversal/

def preorder_traversal_utils(root: Optional[TreeNode], preorder_traversal: List[int]) -> None:
    if not root:
        return
    preorder_traversal.append(root.val)
    preorder_traversal_utils(root=root.left, preorder_traversal=preorder_traversal)
    preorder_traversal_utils(root=root.right, preorder_traversal=preorder_traversal)


def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    preorder_traversal: List[int] = []
    preorder_traversal_utils(root=root, preorder_traversal=preorder_traversal)
    return preorder_traversal


# https://leetcode.com/problems/design-hashset/

class LLNode:

    def __init__(self, val=None):
        self.val: int = val
        self.next: Optional[LLNode] = None


class MyHashSet:

    def __init__(self):
        self.n: int = 1000
        self.base_arr: List[LLNode] = []
        for i in range(self.n):
            self.base_arr.append(LLNode(val=math.inf))

    def add(self, key: int) -> None:
        if self.contains(key=key):
            return
        mod: int = key % self.n
        ll_node: LLNode = self.base_arr[mod]
        next_node: LLNode = ll_node.next
        new_node: LLNode = LLNode(val=key)
        ll_node.next = new_node
        new_node.next = next_node

    def remove(self, key: int) -> None:
        mod: int = key % self.n
        ll_node: LLNode = self.base_arr[mod]
        while ll_node and ll_node.next is not None:
            if ll_node.next.val == key:
                ll_node.next = ll_node.next.next

            ll_node = ll_node.next

    def contains(self, key: int) -> bool:
        mod: int = key % self.n
        ll_node: LLNode = self.base_arr[mod]
        while ll_node is not None:
            if ll_node.val == key:
                return True

            ll_node = ll_node.next

        return False


# https://leetcode.com/problems/number-of-1-bits/

def hammingWeight(self, n: int) -> int:
    no: int = 0
    while n != 0:
        no += n % 2
        n //= 2

    return no


# https://leetcode.com/problems/count-binary-substrings/

def countBinarySubstrings(self, s: str) -> int:
    prev: int = 0
    curr: int = 1
    count: int = 0
    for i in range(1, len(s)):
        if s[i - 1] != s[i]:
            count += min(prev, curr)
            prev = curr
            curr = 1
        else:
            curr += 1

    count += min(prev, curr)

    return count


# https://leetcode.com/problems/special-positions-in-a-binary-matrix/

def numSpecial(self, mat: List[List[int]]) -> int:
    m: int = len(mat)
    n: int = len(mat[0])
    row_map: Mapping[int, List[int]] = {}
    col_map: Mapping[int, List[int]] = {}

    for i in range(m):
        for j in range(n):
            if mat[i][j] == 1:
                row_map[i] = row_map.get(i, [])
                row_map[i].append(j)

                col_map[j] = col_map.get(j, [])
                col_map[j].append(i)

    count: int = 0
    for i, col_values in row_map.items():
        if len(col_values) == 1 and len(col_map.get(col_values[0])) == 1:
            count += 1

    return count


# https://leetcode.com/problems/minimum-number-of-operations-to-convert-time/

def convertTime(self, current: str, correct: str) -> int:
    current_arr: List[str] = current.split(":")
    correct_arr: List[str] = correct.split(":")

    current_hour: int = int(current_arr[0])
    current_min: int = int(current_arr[1])
    correct_hour: int = int(correct_arr[0])
    correct_min: int = int(correct_arr[1])

    hour_diff: int = correct_hour - current_hour if correct_hour >= current_hour \
        else 24 - current_hour + correct_hour
    min_diff: int = correct_min - current_min if correct_min >= current_min \
        else 60 - current_min + correct_min
    hour_diff = hour_diff - 1 if current_min > correct_min else hour_diff

    total_min: int = 60 * hour_diff + min_diff
    count: int = 0
    div_arr: List[int] = [60, 15, 5, 1]
    i: int = 0
    while total_min > 0 and i < len(div_arr):
        if total_min >= div_arr[i]:
            count += total_min // div_arr[i]
            total_min %= div_arr[i]
        else:
            i += 1

    return count


# https://leetcode.com/problems/mean-of-array-after-removing-some-elements/description/

def trimMean(self, arr: List[int]) -> float:
    n: int = len(arr)
    t: int = int(n * 0.05)
    min_heap: heapq = []
    max_heap: heapq = []
    heapify(min_heap)
    heapify(max_heap)

    s: int = 0
    for no in arr:
        s += no
        heappush(min_heap, no)
        heappush(max_heap, - no)
        if len(min_heap) > t:
            heappop(min_heap)
        if len(max_heap) > t:
            heappop(max_heap)

    min_elements_sum: int = 0
    while len(max_heap) > 0:
        min_elements_sum += - heappop(max_heap)

    max_elements_sum: int = 0
    while len(min_heap) > 0:
        max_elements_sum += heappop(min_heap)

    mean: float = (s - min_elements_sum - max_elements_sum) / (n - (2 * t))
    return mean


# https://leetcode.com/problems/design-hashmap/

class LLNodeM:

    def __init__(self, key: float, value: float):
        self.next: Optional[LLNodeM] = None
        self.key: float = key
        self.value: float = value


class MyHashMap:

    def __init__(self):
        self.n: int = 1000
        self.arr: List[LLNodeM] = []
        for _ in range(self.n):
            self.arr.append(LLNodeM(key=math.inf, value=math.inf))

    def put(self, key: int, value: int) -> None:
        mod: int = key % self.n
        sentinel_node: LLNodeM = self.arr[mod]
        head: LLNodeM = sentinel_node
        while sentinel_node is not None:
            if sentinel_node.key == key:
                sentinel_node.value = value
                return
            sentinel_node = sentinel_node.next

        new_node: LLNodeM = LLNodeM(key=key, value=value)
        next_node: LLNodeM = head.next
        head.next = new_node
        new_node.next = next_node

    def get(self, key: int) -> int:
        mod: int = key % self.n
        sentinel_node: LLNodeM = self.arr[mod]
        while sentinel_node is not None:
            if sentinel_node.key == key:
                return int(sentinel_node.value)
            sentinel_node = sentinel_node.next

        return -1

    def remove(self, key: int) -> None:
        mod: int = key % self.n
        sentinel_node: LLNodeM = self.arr[mod]
        while sentinel_node is not None and sentinel_node.next is not None:
            if sentinel_node.next.key == key:
                sentinel_node.next = sentinel_node.next.next
                return

            sentinel_node = sentinel_node.next


# https://leetcode.com/problems/last-stone-weight/description/

def lastStoneWeight(self, stones: List[int]) -> int:
    max_heap: heapq = []

    for stone in stones:
        heappush(max_heap, -stone)

    while len(max_heap) > 1:
        first_max: int = - heappop(max_heap)
        second_max: int = - heappop(max_heap)
        diff: int = first_max - second_max
        if diff > 0:
            heappush(max_heap, - diff)

    return 0 if len(max_heap) == 0 else - heappop(max_heap)


# https://leetcode.com/problems/make-the-string-great/

def makeGood(self, s: str) -> str:
    stack: Deque[str] = deque()
    for w in s:
        is_w_capital: bool = w.isupper()
        if len(stack) >= 1:
            is_last_capital: bool = stack[-1].isupper()
            if stack[-1].lower() == w.lower() and is_w_capital != is_last_capital:
                stack.pop()
            else:
                stack.append(w)
        else:
            stack.append(w)

    return "".join(stack)


# https://leetcode.com/problems/occurrences-after-bigram/

def findOccurrences(self, text: str, first: str, second: str) -> List[str]:
    arr: List[str] = text.split(" ")
    res: List[str] = []
    for i in range(len(arr) - 2):
        if arr[i] == first and arr[i + 1] == second:
            res.append(arr[i + 2])

    return res


# https://leetcode.com/problems/add-digits/

def addDigits(self, num: int) -> int:
    res: int = 0
    while num > 0:
        res += num % 10
        num //= 10
        if num == 0:
            if res <= 9:
                break
            num = res
            res = 0

    return res


# if (num == 0) return 0;
# if (num % 9 == 0) return 9;
# return num % 9;


# https://leetcode.com/problems/find-all-k-distant-indices-in-an-array/description/

def findKDistantIndices(self, nums: List[int], key: int, k: int) -> List[int]:
    key_indexes: List[int] = []
    for i, e in enumerate(nums):
        if e == key:
            key_indexes.append(i)

    i: int = 0
    res: List[int] = []
    for j in range(len(nums)):
        if i < len(key_indexes) and abs(key_indexes[i] - j) <= k:
            res.append(j)

        if i < len(key_indexes) - 1 and j - key_indexes[i] == k:
            i += 1

    return res


# https://leetcode.com/problems/majority-element/description/

def majorityElement(self, nums: List[int]) -> int:
    count: int = 1
    maj_elem: int = nums[0]
    for i in range(1, len(nums)):
        if count == 0:
            maj_elem = nums[i]

        if nums[i] == maj_elem:
            count += 1
        else:
            count -= 1

    return maj_elem


# https://leetcode.com/problems/largest-subarray-length-k/description/

def largestSubarray(self, nums: List[int], k: int) -> List[int]:
    max_elem: int = 0
    max_elem_index: int = 0
    for i in range(0, len(nums) - k + 1):
        if nums[i] > max_elem:
            max_elem = nums[i]
            max_elem_index = i

    return nums[max_elem_index: max_elem_index + k]


# https://leetcode.com/problems/valid-anagram/

def isAnagram(self, s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    count_arr_s: List[int] = [0] * 26
    count_arr_t: List[int] = [0] * 26
    for i in range(len(s)):
        count_arr_s[ord(s[i]) - 97] += 1
        count_arr_t[ord(t[i]) - 97] += 1

    for i in range(0, 26):
        if count_arr_s[i] != count_arr_t[i]:
            return False

    return True


# https://leetcode.com/problems/flip-game/

def generatePossibleNextMoves(self, currentState: str) -> List[str]:
    res: List[str] = []
    for i in range(0, len(currentState) - 1):
        if currentState[i] == '+' and currentState[i + 1] == '+':
            new_state: str = currentState[:i] + "--" + currentState[i + 2:]
            res.append(new_state)

    return res


# https://leetcode.com/problems/missing-number/

def missingNumber(self, nums: List[int]) -> int:
    n: int = len(nums)
    xor_n: int = n
    for i in range(n):
        xor_n ^= i ^ nums[i]

    return xor_n


# https://leetcode.com/problems/implement-queue-using-stacks/

class MyQueue:

    def __init__(self):
        self.stack_1: Deque = deque()
        self.stack_2: Deque = deque()

    def push(self, x: int) -> None:
        self.stack_1.append(x)

    def pop(self) -> int:
        if len(self.stack_2) != 0:
            return self.stack_2.pop()
        else:
            while len(self.stack_1) > 0:
                self.stack_2.append(self.stack_1.pop())
        return self.stack_2.pop() if len(self.stack_2) != 0 else -1

    def peek(self) -> int:
        if len(self.stack_2) != 0:
            return self.stack_2[-1]
        else:
            while len(self.stack_1) > 0:
                self.stack_2.append(self.stack_1.pop())
        return self.stack_2[-1] if len(self.stack_2) != 0 else -1

    def empty(self) -> bool:
        return len(self.stack_1) == 0 and len(self.stack_2) == 0


# https://leetcode.com/problems/merge-two-sorted-lists/

def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    sentinel_node: ListNode = ListNode(-1)
    curr_node: ListNode = sentinel_node

    while list1 is not None or list2 is not None:
        if list1 is not None and list2 is not None:
            if list1.val <= list2.val:
                next_node = list1.next
                list1.next = None
                curr_node.next = list1
                list1 = next_node
            else:
                next_node = list2.next
                list2.next = None
                curr_node.next = list2
                list2 = next_node
            curr_node = curr_node.next
        elif list1 is not None:
            curr_node.next = list1
            break
        elif list2 is not None:
            curr_node.next = list2
            break

    return sentinel_node.next


# https://leetcode.com/problems/min-cost-climbing-stairs/

def minCostClimbingStairs(self, cost: List[int]) -> int:
    n: int = len(cost)
    dp: List[int] = [100000] * (n + 1)
    length: int = len(dp)
    dp[length - 1] = 0
    dp[length - 2] = cost[n - 1]
    for i in range(length - 3, -1, -1):
        dp[i] = min(dp[i + 1], dp[i + 2]) + cost[i]

    return min(dp[0], dp[1])


# https://leetcode.com/problems/excel-sheet-column-number/

def titleToNumber(self, columnTitle: str) -> int:
    res: int = 0
    n = len(columnTitle)
    for i in range(n - 1, -1, -1):
        res += int(math.pow(26, n - 1 - i) * (ord(columnTitle[i]) - 64))

    return res


# https://leetcode.com/problems/reverse-only-letters/

def reverseOnlyLetters(self, s: str) -> str:
    i: int = 0
    j: int = len(s) - 1
    s: List[str] = list(s)
    while i < j:
        if not s[i].isalpha():
            i += 1
        elif not s[j].isalpha():
            j -= 1
        else:
            temp: str = s[i]
            s[i] = s[j]
            s[j] = temp
            i += 1
            j -= 1

    return "".join(s)


# https://leetcode.com/problems/move-zeroes/

def moveZeroes(self, nums: List[int]) -> None:
    n: int = len(nums)
    zero_counter: int = 0
    for i in range(n):
        if nums[i] != 0:
            nums[zero_counter] = nums[i]
            zero_counter += 1

    for i in range(zero_counter, n):
        nums[i] = 0


# https://leetcode.com/problems/flood-fill/

def floodFillUtils(image: List[List[int]], i: int, j: int, old_color: int, color: int, directions: List[List[int]]) \
        -> None:
    if i < 0 or i >= len(image) or j < 0 or j >= len(image[i]) or image[i][j] == color or image[i][j] != old_color:
        return
    image[i][j] = color
    for direction in directions:
        new_i: int = i + direction[0]
        new_j: int = j + direction[1]
        floodFillUtils(image=image, i=new_i, j=new_j, old_color=old_color, color=color, directions=directions)


def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
    directions: List[List[int]] = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    floodFillUtils(image, sr, sc, image[sr][sc], color, directions)
    return image


# https://leetcode.com/problems/contains-duplicate/description/

def containsDuplicate(self, nums: List[int]) -> bool:
    numbers_set: Set[int] = set()
    for num in nums:
        if num in numbers_set:
            return True
        numbers_set.add(num)

    return False


# https://leetcode.com/problems/consecutive-characters/

def maxPower(self, s: str) -> int:
    n: int = len(s)
    if n == 1:
        return 1
    max_power: int = 0
    running_power: int = 1
    for i in range(1, n):
        if s[i] == s[i - 1]:
            running_power += 1
        else:
            running_power = 1
        max_power = max(max_power, running_power)

    return max_power


# https://leetcode.com/problems/binary-tree-paths/description/

def binaryTreePathsUtils(root: Optional[TreeNode], res: List[str], curr: List[int]) -> None:
    if not root:
        return
    curr.append(root.val)
    if root.left is None and root.right is None:
        res.append("->".join([str(x) for x in curr]))
        curr.pop()
        return
    binaryTreePathsUtils(root=root.left, res=res, curr=curr)
    binaryTreePathsUtils(root=root.right, res=res, curr=curr)
    curr.pop()


def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
    res: List[str] = []
    binaryTreePathsUtils(root=root, res=res, curr=[])
    return res


# https://leetcode.com/problems/two-sum-iv-input-is-a-bst/

def findTargetUtils(root: Optional[TreeNode], mapping: Set[int], k: int) -> bool:
    if not root:
        return False
    if k - root.val in mapping:
        return True
    mapping.add(root.val)
    left_value: bool = findTargetUtils(root=root.left, mapping=mapping, k=k)
    if left_value:
        return True
    return findTargetUtils(root=root.right, mapping=mapping, k=k)


def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
    mapping: Set[int] = set()
    return findTargetUtils(root=root, mapping=mapping, k=k)


# https://leetcode.com/problems/special-array-with-x-elements-greater-than-or-equal-x/description/

def specialArray(self, nums: List[int]) -> int:
    count_arr: List[int] = [0] * 1001
    for num in nums:
        count_arr[num] += 1

    remaining_elements: int = len(nums)
    for i in range(1001):
        if i == remaining_elements:
            return i
        remaining_elements -= count_arr[i]

    return -1


# https://leetcode.com/problems/most-frequent-number-following-key-in-an-array/

def mostFrequent(self, nums: List[int], key: int) -> int:
    n: int = len(nums)
    mapping: Mapping[int, int] = {}
    for i in range(1, n):
        if nums[i - 1] == key:
            target: int = nums[i]
            mapping[target] = mapping.get(target, 0) + 1

    sorted_mapping: List[Tuple[int, int]] = sorted(mapping.items(), key=lambda x: x[1], reverse=True)
    return sorted_mapping[0][0]


# https://leetcode.com/problems/longest-uncommon-subsequence-i/

def findLUSlength(self, a: str, b: str) -> int:
    if a == b:
        return -1
    return max(len(a), len(b))


# https://leetcode.com/problems/largest-triangle-area/

def calculate_area(a: List[int], b: List[int], c: List[int]) -> float:
    return (a[0] * (b[1] - c[1]) + (b[0] * (c[1] - a[1])) + (c[0] * (a[1] - b[1]))) / 2.0


def largestTriangleArea(self, points: List[List[int]]) -> float:
    max_area: float = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                area: float = abs(calculate_area(points[i], points[j], points[k]))
                if area > max_area:
                    max_area = area

    return max_area
