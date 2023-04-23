import heapq
import sys
from collections import deque
from heapq import heappush, heappop
from typing import List, Mapping, Set, Optional, Deque, Tuple

from src.main.python.leetcode.Models import TreeNode, ListNode


# https://leetcode.com/problems/find-the-difference/

def findTheDifference(s: str, t: str) -> str:
    s_map: Mapping[str, int] = {}
    for a in s:
        s_map[a] = s_map.get(a, 0) + 1

    t_map: Mapping[str, int] = {}
    for a in t:
        t_map[a] = t_map.get(a, 0) + 1

    for k, v in t_map.items():
        if s_map.get(k, 0) < v:
            return k

    return ""


# https://leetcode.com/problems/relative-ranks/

def findRelativeRanks(score: List[int]) -> List[str]:
    rank_map: Mapping[int, str] = {
        1: "Gold Medal",
        2: "Silver Medal",
        3: "Bronze Medal",
    }
    sorted_score = sorted(score, reverse=True)
    sorted_map: Mapping[int, int] = {}
    for i, e in enumerate(sorted_score):
        sorted_map[e] = i + 1

    res: List[str] = []
    for s in score:
        rank: int = sorted_map.get(s, 0)
        final_rank: str = rank_map.get(rank, str(rank))
        res.append(final_rank)

    return res


# https://leetcode.com/problems/maximum-population-year/

def maximumPopulation(logs: List[List[int]]) -> int:
    mapping: Mapping[int, int] = {}
    for log in logs:
        start_year: int = log[0]
        end_year: int = log[1]
        mapping[start_year] = mapping.get(start_year, 0) + 1
        mapping[end_year] = mapping.get(end_year, 0) - 1

    sorted_dict: Mapping[int, int] = {k: mapping[k] for k in sorted(mapping)}

    running_count: int = 0
    max_count: int = 0
    first_year: int = 0
    for k, v in sorted_dict.items():
        running_count += v
        if running_count > max_count:
            max_count = running_count
            first_year = k

    return first_year


# https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/

def findDisappearedNumbers(nums: List[int]) -> List[int]:
    n: int = len(nums)
    for e in nums:
        if e > 0:
            new_index = 0 if e == n else e
            nums[new_index] *= -1

    res: List[int] = []
    for i, e in enumerate(nums):
        if e > 0:
            no: int = n if i == 0 else i
            res.append(no)

    return res


# https://leetcode.com/problems/element-appearing-more-than-25-in-sorted-array/

def binary_search_first(arr: List[int], x: int, low: int, high: int) -> int:
    first_occ: int = low
    while low <= high:
        mid: int = low + ((high - low) // 2)
        if arr[mid] == x:
            first_occ = mid
            high = mid - 1
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1

    return first_occ


def binary_search_last(arr: List[int], x: int, low: int, high: int) -> int:
    last_occ: int = low
    while low <= high:
        mid: int = low + ((high - low) // 2)
        if arr[mid] == x:
            last_occ = mid
            low = mid + 1
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1

    return last_occ


def check_length_at_check(self, arr: List[int], check: int) -> int:
    n: int = len(arr)
    num: int = arr[check]
    first_occurrence: int = self.binary_search_first(arr, num, 0, check)
    last_occurrence: int = self.binary_search_last(arr, num, check, n - 1)
    return last_occurrence - first_occurrence + 1


def findSpecialInteger(self, arr: List[int]) -> int:
    n: int = len(arr)
    first: int = n // 4
    half: int = n // 2
    third: int = n * 3 // 4

    checks: List[int] = [first, half, third]
    for check in checks:
        size: int = self.check_length_at_check(arr=arr, check=check)
        if size > n // 4:
            return arr[check]

    return -1


# https://leetcode.com/problems/first-unique-character-in-a-string/

def firstUniqChar(s: str) -> int:
    count_arr: List[int] = [0] * 26
    for character in s:
        char_int: int = ord(character) - 91
        count_arr[char_int] += 1

    for i, character in enumerate(s):
        char_int: int = ord(character) - 91
        if count_arr[char_int] == 1:
            return i

    return -1


# https://leetcode.com/problems/rank-transform-of-an-array/

def arrayRankTransform(arr: List[int]) -> List[int]:
    mapping: Set[int] = set()
    for num in arr:
        mapping.add(num)

    sorted_mapping: List[int] = sorted(mapping)
    dict_values: Mapping[int, int] = {}

    i: int = 1
    for no in sorted_mapping:
        dict_values[no] = i
        i += 1

    res: List[int] = []
    for no in arr:
        res.append(dict_values[no])

    return res


# https://leetcode.com/problems/roman-to-integer/

def romanToInt(s: str) -> int:
    n: int = len(s)
    mapping: Mapping[str, int] = {
        "I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000,
        "IV": 4, "IX": 9, "XL": 40, "XC": 90, "CD": 400, "CM": 900
    }

    res: int = 0
    i: int = 0
    while i < n:
        if i < n - 1 and s[i: i + 1] in mapping:
            res += mapping.get(s[i: i + 1], 0)
            i += 1
        elif s[i] in mapping:
            res += mapping.get(s[i], 0)

        i += 1

    return res


# https://leetcode.com/problems/largest-substring-between-two-equal-characters/

def maxLengthBetweenEqualCharacters(s: str) -> int:
    indexes: Mapping[str, int] = {}
    n: int = len(s)
    res: int = -1
    for i in range(n):
        if s[i] in indexes:
            res = max(res, i - indexes[s[i]])
        else:
            indexes[s[i]] = i

    return res


# https://leetcode.com/problems/convert-1d-array-into-2d-array/

def construct2DArray(original: List[int], m: int, n: int) -> List[List[int]]:
    length: int = len(original)
    if length != m * n:
        return []

    k: int = 0
    res: List[List[int]] = []
    for i in range(m):
        col: List[int] = []
        for j in range(n):
            col.append(original[k])
            k += 1
        res.append(col)

    return res


# https://leetcode.com/problems/shortest-completing-word/

def shortestCompletingWord(licensePlate: str, words: List[str]) -> str:
    license_mapping: Mapping[str, int] = {}
    for w in licensePlate:
        if w.isalpha():
            w = w.lower()
            license_mapping[w] = license_mapping.get(w, 0) + 1

    min_len: int = 1001
    min_length_word: str = ""
    for word in words:
        words_map: Mapping[str, int] = {}
        for w in words:
            w = w.lower()
            words_map[w] = words_map.get(w, 0) + 1

        not_found_flag: bool = False
        for k, v in license_mapping.items():
            if k not in words_map or words_map.get(k, 0) < v:
                not_found_flag = True
                break

        if not not_found_flag:
            if len(word) < min_len:
                min_len = len(word)
                min_length_word = word

    return min_length_word


# https://leetcode.com/problems/minimum-distance-between-bst-nodes/


def minDiffInBST_util(self, root: Optional[TreeNode], p_m: List[int]) -> None:
    if not root:
        return
    self.minDiffInBST_util(root=root.left, p_m=p_m)
    if p_m[0] == -1:
        p_m[0] = root.val
    else:
        val: int = root.val
        prev: int = p_m[0]
        abs_min: int = p_m[1]
        curr_min: int = val - prev
        abs_min = min(abs_min, curr_min)
        p_m[0] = val
        p_m[1] = abs_min
    self.minDiffInBST_util(root=root.right, p_m=p_m)


def minDiffInBST(self, root: Optional[TreeNode]) -> int:
    prev_and_min: List[int] = [-1, 100001]
    self.minDiffInBST_util(root=root, p_m=prev_and_min)
    return prev_and_min[1]


# https://leetcode.com/problems/monotonic-array/

def isMonotonic(nums: List[int]) -> bool:
    inc: bool = True
    dec: bool = True

    for i in range(1, len(nums)):
        inc = inc and nums[i] >= nums[i - 1]
        dec = dec and nums[i] <= nums[i - 1]

    return inc or dec


# https://leetcode.com/problems/same-tree/

def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if not p or not q:
        return not p and not q

    if p.val != q.val:
        return False

    # noinspection PyArgumentList
    return isSameTree(q.left) and isSameTree(q.right)


# https://leetcode.com/problems/ransom-note/

def canConstruct(ransomNote: str, magazine: str) -> bool:
    mag_map: Mapping[str, int] = {}
    mapp: Mapping[str, int] = {}

    for rs in ransomNote:
        mag_map[rs] = mag_map.get(rs, 0) + 1

    for m in magazine:
        mapp[m] = mapp.get(m, 0) + 1

    for k, v in mag_map:
        if k not in mapp or v > mapp.get(k, 0):
            return False

    return True


class MyStack:

    def __init__(self):
        self.q1: Deque[int] = deque()
        self.q2: Deque[int] = deque()

    def push(self, x: int) -> None:
        self.q1.append(x)

    def pop(self) -> int:
        n: int = len(self.q1)
        for _ in range(n - 1):
            self.q2.append(self.q1.popleft())

        t: int = self.q1.pop()
        while self.q2:
            self.q1.append(self.q2.popleft())
        return t

    def top(self) -> int:
        n: int = len(self.q1)
        for _ in range(n - 1):
            self.q2.append(self.q1.popleft())

        t: int = self.q1.popleft()
        self.q2.append(t)
        while self.q2:
            self.q1.append(self.q2.popleft())
        return t

    def empty(self) -> bool:
        return not self.q1 and not self.q2


# https://leetcode.com/problems/maximum-score-after-splitting-a-string/

def maxScore(s: str) -> int:
    n: int = len(s)
    zeros_arr: List[int] = [0] * n
    ones_arr: List[int] = [0] * n

    running_count: int = 0
    for i in range(0, n):
        running_count += 1 if s[i] == '0' else 0
        zeros_arr[i] = running_count

    running_count = 0
    for i in range(n - 1, -1, -1):
        ones_arr[i] = running_count
        running_count += 1 if s[i] == '1' else 0

    max_score: int = 0
    for i in range(0, n - 1):
        score: int = zeros_arr[i] + ones_arr[i]
        max_score = max(max_score, score)

    return max_score


# https://leetcode.com/problems/meeting-rooms/

def canAttendMeetings(intervals: List[List[int]]) -> bool:
    interval_map: Mapping[int, int] = {}
    for interval in intervals:
        start: int = interval[0]
        end: int = interval[1]
        interval_map[start] = interval_map.get(start, 0) + 1
        interval_map[end] = interval_map.get(end, 0) - 1

    sorted_interval_map: Mapping[int, int] = {k: interval_map[k] for k in sorted(interval_map)}
    overlap_time: int = 0
    for k, v in sorted_interval_map.items():
        overlap_time += v
        if overlap_time > 1:
            return False

    return True


# https://leetcode.com/problems/detect-capital/

def detectCapitalUse(word: str) -> bool:
    n: int = len(word)
    is_first_capital: bool = word[0].isupper()
    rest_capital: int = 0
    for i in range(1, n):
        if word[i].isupper():
            rest_capital += 1

    if rest_capital == 0:
        return True
    if is_first_capital and (rest_capital == 0 or rest_capital == n - 1):
        return True
    return False


# https://leetcode.com/problems/max-consecutive-ones/

def findMaxConsecutiveOnes(nums: List[int]) -> int:
    max_ones: int = 0
    running_ones: int = 0
    for num in nums:
        if num == 1:
            running_ones += 1
        else:
            running_ones = 0
        max_ones = max(max_ones, running_ones)

    return max_ones


# https://leetcode.com/problems/diameter-of-binary-tree/

def diameterOfBinaryTreeUtils(root: Optional[TreeNode], max_diameter: List[int]) -> int:
    if not root:
        return 0
    left_node: int = diameterOfBinaryTreeUtils(root=root.left, max_diameter=max_diameter)
    right_node: int = diameterOfBinaryTreeUtils(root=root.right, max_diameter=max_diameter)

    max_diameter[0] = max(max_diameter[0], left_node + right_node + 1)
    return max(left_node, right_node) + 1


def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    max_diameter: List[int] = [0]
    diameterOfBinaryTreeUtils(root=root, max_diameter=max_diameter)
    return max_diameter[0] - 1


# https://leetcode.com/problems/kth-missing-positive-number/

def findKthPositive(arr: List[int], k: int) -> int:
    low: int = 0
    high: int = len(arr) - 1
    while low <= high:
        mid: int = low + (high - low) // 2
        pos: int = arr[mid] - 1 - mid
        if pos < k:
            low = mid + 1
        else:
            high = mid - 1

    return arr[high] + k - (arr[high] - high - 1) if high >= 0 else k


# https://leetcode.com/problems/nim-game/description/

def canWinNim(n: int) -> bool:
    return n % 4 != 0


# https://leetcode.com/problems/path-crossing/

def isPathCrossing(path: str) -> bool:
    before_paths: Set[Tuple] = set()
    initial: Tuple = (0, 0)
    before_paths.add(initial)
    for path in path:
        if path == 'N':
            initial = (initial[0] + 1, initial[1])
        elif path == 'S':
            initial = (initial[0] - 1, initial[1])
        elif path == 'W':
            initial = (initial[0], initial[1] + 1)
        elif path == 'E':
            initial = (initial[0], initial[1] - 1)

        if initial in before_paths:
            return True
        before_paths.add(initial)

    return False


# https://leetcode.com/problems/intersection-of-two-arrays-ii/

def intersect(nums1: List[int], nums2: List[int]) -> List[int]:
    count_arr: List[int] = [0] * 1001
    for num in nums1:
        count_arr[num] += 1

    res_arr: List[int] = []
    for num in nums2:
        if count_arr[num] > 0:
            res_arr.append(num)
            count_arr[num] -= 1

    return res_arr


# https://leetcode.com/problems/binary-search/

def search(nums: List[int], target: int) -> int:
    low: int = 0
    high: int = len(nums) - 1
    while low <= high:
        mid: int = low + (high - low) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            high = mid + 1
        else:
            low = mid - 1

    return low


# https://leetcode.com/problems/rotate-string/description/

def rotateString(s: str, goal: str) -> bool:
    if len(goal) != len(s):
        return False
    new_s: str = s + s
    return goal in new_s


# https://leetcode.com/problems/kth-largest-element-in-a-stream/

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.min_heap: heapq = []
        self.k = k

        for num in nums:
            heappush(self.min_heap, num)
            if len(self.min_heap) > k:
                heappop(self.min_heap)

    def add(self, val: int) -> int:
        heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heappop(self.min_heap)

        return self.min_heap[0]


# https://leetcode.com/problems/reformat-the-string/

def reformat(s: str) -> str:
    int_values: List[str] = []
    str_values: List[str] = []
    for value in s:
        if value.isdigit():
            int_values.append(value)
        else:
            str_values.append(value)

    m: int = len(int_values)
    n: int = len(str_values)

    if abs(m - n) > 1:
        return ""

    flag: bool = m - n > 0
    res: List[str] = []
    while int_values or str_values:
        res.append(int_values.pop()) if flag else res.append(str_values.pop())
        flag = not flag

    return "".join(res)


# https://leetcode.com/problems/happy-number/

def isHappy(n: int) -> bool:
    repeated: Set[int] = set()
    while n != 1:
        sum_values: int = 0
        while n != 0:
            last_digit: int = n % 10
            sum_values += last_digit ** 2
            n //= 10

        if sum_values in repeated:
            return False

        repeated.add(sum_values)
        n = sum_values

    return True


# https://leetcode.com/problems/closest-binary-search-tree-value/description/

def closestValueUtil(root: Optional[TreeNode], target: float, min_value: List[int]) -> None:
    if not root:
        return
    diff: int = root.val - target
    if abs(diff) < min_value[0]:
        min_value[0] = abs(diff)
        min_value[1] = root.val

    if diff == 0:
        return
    elif diff > 0:
        closestValueUtil(root=root.left, target=target, min_value=min_value)
    else:
        closestValueUtil(root=root.right, target=target, min_value=min_value)


def closestValue(root: Optional[TreeNode], target: float) -> int:
    min_value: List[int] = [sys.maxsize, -1]
    closestValueUtil(root=root, target=target, min_value=min_value)
    return min_value[0]


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

def maxProfit(prices: List[int]) -> int:
    min_price: int = sys.maxsize
    max_profit: int = 0
    for p in prices:
        if p < min_price:
            min_price = p
        elif p - min_price > max_profit:
            max_profit = p - min_price

    return max_profit


# https://leetcode.com/problems/intersection-of-two-linked-lists/description/

def get_length(node: Optional[ListNode]) -> int:
    length: int = 0
    while node is not None:
        length += 1
        node = node.next

    return length


def getIntersectionNode(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    a_len: int = get_length(headA)
    b_len: int = get_length(headB)
    d: int = a_len - b_len

    currA: ListNode = headA
    currB: ListNode = headB

    while d > 0 and currA:
        currA = currA.next
        d -= 1

    while d < 0 and currB:
        currB = currB.next
        d += 1

    while currA is not None and currB is not None:
        if currA == currB:
            return currA
        currA = currA.next
        currB = currB.next

    return None


# https://leetcode.com/problems/longest-palindrome/description/

def longestPalindrome(s: str) -> int:
    mapping: Mapping[str, int] = {}
    for word in s:
        mapping[word] = mapping.get(word, 0) + 1

    count: int = 0
    odd_count: int = 0
    for k, v in mapping.items():
        if v % 2 == 0:
            count += v
        else:
            if odd_count == 0:
                count += 1
                odd_count += 1
            count += v - 1

    return count


# https://leetcode.com/problems/distance-between-bus-stops/

def distanceBetweenBusStops(distance: List[int], start: int, destination: int) -> int:
    n: int = len(distance)
    s: int = start if start <= destination else destination
    e: int = destination if destination > start else start
    cyclic_distance: List[int] = distance + distance
    f_distance: int = sum(cyclic_distance[s: e])
    b_distance: int = sum(cyclic_distance[e: s + n])
    return min(f_distance, b_distance)


# https://leetcode.com/problems/find-winner-on-a-tic-tac-toe-game/description/

def tictactoe(moves: List[List[int]]) -> str:
    n: int = 3
    rows: List[int] = [0] * n
    cols: List[int] = [0] * n
    d1: List[int] = [0] * n
    d2: List[int] = [0] * n

    moves_count: int = 0
    for i, move in enumerate(moves):
        moves_count += 1
        r: int = move[0]
        c: int = move[1]

        val: int = 1 if i % 2 == 0 else -1
        rows[r] += val
        cols[c] += val
        if r == c:
            d1[r] += val
        if r + c == 2:
            d2[r] += val

    s_d1: int = sum(d1)
    s_d2: int = sum(d2)

    if 3 in rows or 3 in cols or s_d1 == 3 or s_d2 == 3:
        return "A"
    if -3 in rows or -3 in cols or s_d1 == -3 or s_d2 == -3:
        return "B"
    if moves_count == n * n:
        return "Draw"
    return "Pending"


# https://leetcode.com/problems/reverse-bits/

def reverseBits(n: int) -> int:
    bin_no: List[int] = []
    while n != 0:
        bin_no.append(n % 2)
        n /= 2
    bin_length: int = len(bin_no)
    for _ in range(bin_length, 32):
        bin_no.append(0)

    reverse_no: int = 0
    for i in range(len(bin_no), -1, -1):
        reverse_no += (2 ** (31 - i)) * bin_no[i]

    return reverse_no


# https://leetcode.com/problems/palindrome-number/

def isPalindrome(x: int) -> bool:
    if x < 0:
        return False
    v: int = 0
    placeholder: int = x
    while x > 0:
        r: int = x % 10
        v = v * 10 + r
        x //= 10

    return placeholder == v


# https://leetcode.com/problems/remove-element/

def removeElement(nums: List[int], val: int) -> int:
    pointer: int = 0
    for i in range(len(nums)):
        if nums[i] != val:
            nums[pointer] = nums[i]
            pointer += 1

    return pointer


# https://leetcode.com/problems/add-strings/

def addStrings(num1: str, num2: str) -> str:
    m: int = len(num1) - 1
    n: int = len(num2) - 1
    added_value: Deque[int] = deque()
    carry_over: int = 0
    while m >= 0 and n >= 0:
        no: int = int(num1[m]) + int(num2[n]) + carry_over
        added_value.appendleft(no % 10)
        carry_over = no // 10
        m -= 1
        n -= 1

    while m >= 0:
        no: int = int(num1[m]) + carry_over
        added_value.appendleft(no % 10)
        carry_over = no // 10
        m -= 1

    while n >= 0:
        no: int = int(num2[n]) + carry_over
        added_value.appendleft(no % 10)
        carry_over = no // 10
        n -= 1

    if carry_over > 0:
        added_value.appendleft(carry_over)

    return "".join([str(d) for d in added_value])


# https://leetcode.com/problems/merge-two-2d-arrays-by-summing-values/

def mergeArrays(nums1: List[List[int]], nums2: List[List[int]]) -> List[List[int]]:
    mapping: Mapping[int, int] = {}
    for num_values in nums1 + nums2:
        num_id: int = num_values[0]
        val: int = num_values[1]
        mapping[num_id] = mapping.get(num_id, 0) + val

    return [[k, mapping[k]] for k in sorted(mapping)]


# https://leetcode.com/problems/add-binary/

def addBinary(a: str, b: str) -> str:
    carry_over: int = 0
    i: int = len(a) - 1
    j: int = len(b) - 1
    queue: Deque[int] = deque()
    while i > -1 and j > -1:
        int_i: int = int(a[i])
        int_j: int = int(b[j])
        new_no: int = int_i + int_j + carry_over
        r: int = new_no % 2
        carry_over = new_no // 2
        queue.appendleft(r)
        i -= 1
        j -= 1

    while i > -1:
        int_i: int = int(a[i])
        new_no: int = int_i + carry_over
        r: int = new_no % 2
        carry_over = new_no // 2
        queue.appendleft(r)
        i -= 1

    while j > -1:
        int_j: int = int(b[j])
        new_no: int = int_j + carry_over
        r: int = new_no % 2
        carry_over = new_no // 2
        queue.appendleft(r)
        j -= 1

    if carry_over:
        queue.appendleft(carry_over)

    return "".join([str(v) for v in queue])


# https://leetcode.com/problems/climbing-stairs/

def climbStairs(n: int) -> int:
    if n == 0:
        return 0
    elif n == 1:
        return 1
    steps: List[int] = [0] * (n + 1)
    steps[1] = 1
    steps[2] = 2
    for i in range(3, n + 1):
        steps[i] = steps[i - 1] + steps[i - 2]

    return steps[-1]


# https://leetcode.com/problems/binary-watch/

def generateCombinationsForHour(nums: List[int], i: int, k: int, sum_v: int, values: List[Tuple[str, int]]):
    if sum_v >= 12:
        return
    if i >= len(nums):
        values.append((str(sum_v), k))
        return
    generateCombinationsForHour(nums, i + 1, k=k + 1, sum_v=sum_v + nums[i], values=values)
    generateCombinationsForHour(nums, i + 1, k=k, sum_v=sum_v, values=values)


def generateCombinationsForMin(nums: List[int], i: int, k: int, sum_v: int, values: List[Tuple[str, int]]):
    if sum_v >= 60:
        return
    if i >= len(nums):
        minute_str: str = str(sum_v) if sum_v > 9 else "0" + str(sum_v)
        values.append((minute_str, k))
        return
    generateCombinationsForMin(nums, i + 1, k=k + 1, sum_v=sum_v + nums[i], values=values)
    generateCombinationsForMin(nums, i + 1, k=k, sum_v=sum_v, values=values)


def readBinaryWatch(turnedOn: int) -> List[str]:
    if turnedOn > 8:
        return []
    hours: List[int] = [1, 2, 4, 8]
    minutes: List[int] = [1, 2, 4, 8, 16, 32]
    hours_combinations: List[Tuple[str, int]] = []
    min_combinations: List[Tuple[str, int]] = []
    generateCombinationsForHour(nums=hours, i=0, k=0, sum_v=0, values=hours_combinations)
    generateCombinationsForMin(nums=minutes, i=0, k=0, sum_v=0, values=min_combinations)

    result: List[str] = []
    for hour in hours_combinations:
        for minute in min_combinations:
            if hour[1] + minutes[1] == turnedOn:
                result.append(hour[0] + ":" + minute[0])

    return result


# https://leetcode.com/problems/find-if-path-exists-in-graph/

def bfs_valid_path(adjacency_map: Mapping[int, List[int]], source: int, destination: int) -> bool:
    visited_nodes: Set[int] = set()
    bfs_queue: Deque[int] = deque()
    bfs_queue.append(source)
    while bfs_queue:
        cur_node: int = bfs_queue.popleft()
        if cur_node == destination:
            return True
        if cur_node in visited_nodes:
            continue
        visited_nodes.add(cur_node)
        neighbours: List[int] = adjacency_map.get(cur_node, [])
        for neighbour in neighbours:
            bfs_queue.append(neighbour)

    return False


def validPath(edges: List[List[int]], source: int, destination: int) -> bool:
    adjacency_map: Mapping[int, List[int]] = {}
    for edge in edges:
        from_node: int = edge[0]
        to_edge: int = edge[1]

        from_node_edges = adjacency_map.get(from_node, [])
        from_node_edges.append(to_edge)
        adjacency_map[from_node] = from_node_edges

        to_node_edges = adjacency_map.get(to_edge, [])
        to_node_edges.append(from_node)
        adjacency_map[to_edge] = to_node_edges

    return bfs_valid_path(adjacency_map=adjacency_map, source=source, destination=destination)


# https://leetcode.com/problems/guess-number-higher-or-lower/

def guess(num: int) -> int:
    return num


def guessNumber(n: int) -> int:
    low: int = 1
    high: int = n
    while low <= high:
        mid: int = low + (high - low) // 2
        idx: int = guess(mid)
        if idx == 0:
            return mid
        elif idx == 1:
            low = mid + 1
        else:
            high = mid - 1

    return -1


# https://leetcode.com/problems/positions-of-large-groups/

def largeGroupPositions(s: str) -> List[List[int]]:
    n: int = len(s)
    result: List[List[int]] = []
    curr_value: str = s[0]
    counter: int = 1
    for i in range(1, n):
        if s[i] == curr_value:
            counter += 1
        else:
            if counter >= 3:
                result.append([i - counter, i - 1])
            counter = 1
            curr_value = s[i]

    if counter >= 3:
        result.append([n - counter, n - 1])

    return result


# https://leetcode.com/problems/remove-duplicates-from-sorted-array/

def removeDuplicates(nums: List[int]) -> int:
    k: int = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[k] = nums[i]
            k += 1

    return k


# https://leetcode.com/problems/duplicate-zeros/

def duplicateZeros(arr: List[int]) -> None:
    n: int = len(arr)
    shifts: int = 0
    for num in arr:
        if num == 0:
            shifts += 1

    for i in range(n - 1, -1, -1):
        if arr[i] == 0:
            shifts -= 1
            if i + shifts + 1 < n:
                arr[i + shifts + 1] = 0
        if i + shifts < n:
            arr[i + shifts] = arr[i]


# https://leetcode.com/problems/remove-duplicates-from-sorted-list/

def deleteDuplicates(head: Optional[ListNode]) -> Optional[ListNode]:
    sentinel_node: Optional[ListNode] = ListNode(-100)

    sentinel_node.next = head
    prev: Optional[ListNode] = sentinel_node
    curr: Optional[ListNode] = sentinel_node.next

    while curr is not None:
        if curr.val != prev.val:
            prev.next = curr
            prev = prev.next

        curr = curr.next

    prev.next = None
    return sentinel_node.next


# https://leetcode.com/problems/reverse-string-ii/

def reverseStr(s: str, k: int) -> str:
    main_queue: Deque[str] = deque()
    rev_queue: Deque[str] = deque()
    prev_mod: int = 0

    for i in range(len(s)):
        rem_value: int = i // k

        if prev_mod != rem_value:
            prev_mod = rem_value
            while rev_queue:
                main_queue.append(rev_queue.pop())

        if rem_value % 2 == 0:
            rev_queue.append(s[i])
        else:
            main_queue.append(s[i])

    while rev_queue:
        main_queue.append(rev_queue.pop())
    return "".join(main_queue)
