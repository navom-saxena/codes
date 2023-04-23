import math
import sys
from collections import deque
from datetime import datetime
from typing import List, Optional, Set, Mapping, Deque

from src.main.python.leetcode.Models import ListNode, TreeNode


# https://leetcode.com/problems/palindrome-linked-list/
def get_length(node: Optional[ListNode]) -> int:
    length: int = 0
    while node is not None:
        length += 1
        node = node.next

    return length


def reverse_linkedList(node: Optional[ListNode]) -> Optional[ListNode]:
    if not node or node.next is None:
        return node
    reversed_node: Optional[ListNode] = reverse_linkedList(node=node.next)
    if node.next is not None:
        next_node: Optional[ListNode] = node.next
        node.next = None
        next_node.next = node
    return reversed_node


def isPalindrome(head: Optional[ListNode]) -> bool:
    fastPointer: Optional[ListNode] = head
    slowPointer: Optional[ListNode] = head
    prevNode: Optional[ListNode] = None

    while fastPointer is not None and fastPointer.next is not None:
        prevNode = slowPointer
        fastPointer = fastPointer.next.next
        slowPointer = slowPointer.next

    if fastPointer is None:
        prevNode.next = None
        head2: Optional[ListNode] = slowPointer
    else:
        head2: Optional[ListNode] = slowPointer.next
        slowPointer.next = None

    reversed_head: Optional[ListNode] = reverse_linkedList(head2)
    while head is not None and reversed_head is not None:
        if head.val != reversed_head.val:
            return False
        head = head.next
        reversed_head = reversed_head.next

    return True


# https://leetcode.com/problems/check-if-array-is-sorted-and-rotated/

def check(nums: List[int]) -> bool:
    if nums[0] < nums[len(nums) - 1]:
        for i in range(0, len(nums) - 1):
            if nums[i] > nums[i + 1]:
                return False
    else:
        flag: bool = False
        for i in range(0, len(nums) - 1):
            if nums[i] > nums[i + 1] and not flag:
                flag = True
            elif nums[i] > nums[i + 1] and flag:
                return False

    return True


# https://leetcode.com/problems/reverse-vowels-of-a-string/description/

def reverseVowels(s: str) -> str:
    low: int = 0
    high: int = len(s) - 1
    vowels: Set[str] = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
    s_list: List[str] = list(s)
    print(s_list)
    while low < high:
        if s_list[low] in vowels and s_list[high] in vowels:
            temp: str = s_list[low]
            s_list[low] = s_list[high]
            s_list[high] = temp
            low += 1
            high -= 1
        elif s_list[low] in vowels:
            high -= 1
        elif s_list[high] in vowels:
            low += 1
        else:
            low += 1
            high -= 1

    return "".join(s_list)


# https://leetcode.com/problems/two-sum/

def twoSum(nums: List[int], target: int) -> List[int]:
    num_map: Mapping[int, int] = {}
    for i, num in enumerate(nums):
        if target - num in num_map:
            return [i, num_map[target - num]]
        num_map[num] = i

    return []


# https://leetcode.com/problems/find-the-town-judge/

def findJudge(n: int, trust: List[List[int]]) -> int:
    if n == 1 and not trust:
        return 1
    trust_map: Mapping[int, List[Set[int]]] = {}
    for t in trust:
        p1: int = t[0]
        p2: int = t[1]

        p1_list: List[Set[int]] = trust_map.get(p1, [set(), set()])
        p1_trusts: Set[int] = p1_list[0]
        p1_trustee: Set[int] = p1_list[1]
        p1_trusts.add(p2)
        trust_map[p1] = [p1_trusts, p1_trustee]

        p2_list: List[Set[int]] = trust_map.get(p2, [set(), set()])
        p2_trusts: Set[int] = p2_list[0]
        p2_trustee: Set[int] = p2_list[1]
        p2_trustee.add(p1)
        trust_map[p2] = [p2_trusts, p2_trustee]

    for p, p_list in trust_map.items():
        trusts: Set[int] = p_list[0]
        trusted_by: Set[int] = p_list[1]
        if len(trusts) == 0 and len(trusted_by) == n - 1:
            return p

    return -1


# https://leetcode.com/problems/determine-if-two-events-have-conflict/description/

def haveConflict(event1: List[str], event2: List[str]) -> bool:
    e1s: int = int(event1[0][:2]) * 60 + int(event1[0][3:])
    e1e: int = int(event1[1][:2]) * 60 + int(event1[1][3:])
    e2s: int = int(event2[0][:2]) * 60 + int(event2[0][3:])
    e2e: int = int(event2[1][:2]) * 60 + int(event2[1][3:])

    if e1s <= e2s <= e1e:
        return True
    elif e1s <= e2e <= e1e:
        return True
    elif e2s <= e1s <= e2e:
        return True
    else:
        return False


# https://leetcode.com/problems/count-odd-numbers-in-an-interval-range/

def countOdds(low: int, high: int) -> int:
    total_no: int = high - low + 1
    half: int = total_no // 2
    if (low % 2 == 0 and high % 2 == 0) or low % 2 == 0 or high % 2 == 0:
        return half
    return half + 1


# https://leetcode.com/problems/find-mode-in-binary-search-tree/

def findModeUtil(root: Optional[TreeNode], mapping: Mapping[int, int]) -> None:
    if not root:
        return
    mapping[root.val] = mapping.get(root.val, 0) + 1
    findModeUtil(root=root.left, mapping=mapping)
    findModeUtil(root=root.right, mapping=mapping)


def findMode(root: Optional[TreeNode]) -> List[int]:
    mapping: Mapping[int, int] = {}

    findModeUtil(root=root, mapping=mapping)

    max_f: int = 0
    for k, v in mapping.items():
        max_f = max(max_f, v)

    res: List[int] = []
    for k, v in mapping.items():
        if v == max_f:
            res.append(k)
    return res


# https://leetcode.com/problems/longest-continuous-increasing-subsequence/

def findLengthOfLCIS(nums: List[int]) -> int:
    max_count: int = 0
    running_count: int = 1
    for i in range(1, len(nums)):
        if nums[i - 1] < nums[i]:
            running_count += 1
        else:
            running_count = 1

        max_count = max(max_count, running_count)

    return running_count


# https://leetcode.com/problems/balanced-binary-tree/

def check_is_balanced(root: Optional[TreeNode], res: List[bool]) -> int:
    if not root:
        return 0
    if not res[0]:
        return -1
    left_height: int = check_is_balanced(root=root.left, res=res)
    right_height: int = check_is_balanced(root=root.right, res=res)
    if abs(left_height - right_height) > 1:
        res[0] = False
    return max(left_height, right_height) + 1


def isBalanced(root: Optional[TreeNode]) -> bool:
    res: List[bool] = [True]
    check_is_balanced(root=root, res=res)
    return res[0]


# https://leetcode.com/problems/base-7/

def convertToBase7(num: int) -> str:
    if num == 0:
        return "0"

    q1: Deque[int] = deque()
    is_negative: bool = False
    if num < 0:
        is_negative = True
        num *= -1

    while num != 0:
        rem: int = num % 7
        q1.append(rem)
        num //= 7

    stack_list: List[int] = []
    while len(q1):
        stack_list.append(q1.pop())

    sign: str = "-" if is_negative else ""
    return sign + "".join(str(x) for x in stack_list)


# https://leetcode.com/problems/path-sum/

def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
    if not root:
        return False
    targetSum -= root.val
    if root.left is None and root.right is None:
        return targetSum == 0
    return self.hasPathSum(root=root.left, targetSum=targetSum) or self.hasPathSum(root=root.right,
                                                                                   targetSum=targetSum)


# https://leetcode.com/problems/student-attendance-record-i/

def checkRecord(s: str) -> bool:
    late_count: int = 0
    absent_count: int = 0
    for attendance in s:
        if attendance == 'L':
            late_count += 1
            if late_count >= 3:
                return False
        else:
            late_count = 0
            if attendance == 'A':
                absent_count += 1
                if absent_count >= 2:
                    return False

    return True


# https://leetcode.com/problems/number-of-days-between-two-dates/

def daysBetweenDates(date1: str, date2: str) -> int:
    x = datetime.strptime(date1, '%Y-%m-%d')
    y = datetime.strptime(date2, '%Y-%m-%d')
    return abs((y - x).days)


# https://leetcode.com/problems/confusing-number/

def confusingNumber(n: int) -> bool:
    invalid_no: Set[int] = {2, 3, 4, 5, 7}
    valid_dict: Mapping[int, int] = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6, }
    n_copy: int = n
    new_no: int = 0
    while n_copy != 0:
        rem: int = n_copy % 10
        if rem in invalid_no:
            return False
        new_no = new_no * 10 + valid_dict[rem]
        n_copy //= 10

    return new_no != n


# https://leetcode.com/problems/backspace-string-compare/

def backspaceCompare(s: str, t: str) -> bool:
    m: int = len(s) - 1
    n: int = len(t) - 1
    m_hash_count: int = 0
    n_hash_count: int = 0

    while m > -1 or n > -1:
        if m > -1 and s[m] == '#':
            m_hash_count += 1
            m -= 1
            continue
        if m_hash_count:
            m_hash_count -= 1
            m -= 1
            continue
        if n > -1 and t[n] == '#':
            n_hash_count += 1
            n -= 1
            continue
        if n_hash_count:
            n_hash_count -= 1
            n -= 1
            continue
        if m > -1 and n > -1 and s[m] != t[n]:
            return False
        if m > -1 and n > -1 and s[m] == t[n]:
            m -= 1
            n -= 1
        else:
            break

    return m < 0 and n < 0


# https://leetcode.com/problems/strobogrammatic-number/

def isStrobogrammatic(num: str) -> bool:
    valid_dict: Mapping[int, int] = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6, }
    i: int = 0
    j: int = len(num) - 1
    while i <= j:
        no: int = int(num[i])
        no_j: int = int(num[j])
        reverse_no: int = valid_dict.get(no, -1)

        if reverse_no != no_j:
            return False
        if i == j and (no == '6' or no == '9'):
            return False

        i += 1
        j -= 1

    return True


# https://leetcode.com/problems/linked-list-cycle/

def hasCycle(head: Optional[ListNode]) -> bool:
    if head is None:
        return False
    slow_pointer: Optional[ListNode] = head
    fast_pointer: Optional[ListNode] = head.next

    while fast_pointer is not None and fast_pointer.next is not None:
        if fast_pointer == slow_pointer:
            return True

        fast_pointer = fast_pointer.next.next
        slow_pointer = slow_pointer.next

    return False


# https://leetcode.com/problems/summary-ranges/

def summaryRanges(nums: List[int]) -> List[str]:
    n: int = len(nums)
    res: List[str] = []
    if n == 0:
        return res

    initial_no: int = nums[0]
    curr_no: int = nums[0]
    for i in range(1, n):
        if nums[i] - curr_no > 1:
            if initial_no != curr_no:
                res.append(f"{initial_no}->{curr_no}")
            else:
                res.append(f"{initial_no}")
            initial_no = nums[i]
        curr_no = nums[i]

    if initial_no != curr_no:
        res.append(f"{initial_no}->{curr_no}")
    else:
        res.append(f"{initial_no}")

    return res


# https://leetcode.com/problems/largest-number-at-least-twice-of-others/

def dominantIndex(nums: List[int]) -> int:
    max_value: int = 0
    max_index: int = -1
    for i, num in enumerate(nums):
        if max_value < num:
            max_value = num
            max_index = i

    for num in nums:
        if num != max_value and num > max_value / 2:
            return -1

    return max_index


# https://leetcode.com/problems/convert-a-number-to-hexadecimal/description/

def toHex(num: int) -> str:
    if num == 0:
        return "0"
    if num < 0:
        num += (1 << 32)
    hex_stack: Deque[str] = deque()
    mapping: Mapping[int, str] = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    while num != 0:
        rem: int = num % 16
        hex_stack.append(mapping.get(rem, str(rem)))
        num //= 16

    res: str = ""
    while hex_stack:
        res += hex_stack.pop()

    return res


# https://leetcode.com/problems/merge-sorted-array/

def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    main_idx: int = m + n - 1
    m -= 1
    n -= 1
    while m > -1 and n > -1:
        if nums1[m] >= nums2[n]:
            nums1[main_idx] = nums1[m]
            m -= 1
        else:
            nums1[main_idx] = nums2[n]
            n -= 1
        main_idx -= 1

    while n > -1:
        nums1[main_idx] = nums2[n]
        main_idx -= 1
        n -= 1


# https://leetcode.com/problems/subtree-of-another-tree/

def check_subtree(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    if not root or not subRoot:
        return root is None and subRoot is None
    return root.val == subRoot.val and check_subtree(root=root.left, subRoot=subRoot.left) and check_subtree(
        root=root.right, subRoot=subRoot.right)


def isSubtree(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    if not root or not subRoot:
        return root is None and subRoot is None
    if check_subtree(root=root, subRoot=subRoot):
        return True
    return isSubtree(root=root.left, subRoot=subRoot) or isSubtree(root=root.right, subRoot=subRoot)


# https://leetcode.com/problems/remove-linked-list-elements/

def removeElements(head: Optional[ListNode], val: int) -> Optional[ListNode]:
    sentinel_node: Optional[ListNode] = ListNode(val=-1)
    sentinel_node.next = head
    prev: Optional[ListNode] = sentinel_node
    curr: Optional[ListNode] = head
    while curr:
        if curr.val != val:
            prev.next = curr
            prev = curr

        curr = curr.next

    prev.next = None
    return sentinel_node.next


# https://leetcode.com/problems/power-of-two/description/

def isPowerOfTwo(n: int) -> bool:
    if n == 0 or n < 0:
        return False

    # i: int = 1
    # while i < n:
    #     i *= 2
    #
    # return i == n

    return math.log2(n) % 1 == 0


# https://leetcode.com/problems/maximum-product-of-three-numbers/

def maximumProduct(nums: List[int]) -> int:
    max_f: int = -sys.maxsize - 1
    max_s: int = -sys.maxsize - 1
    max_t: int = -sys.maxsize - 1
    min_f: int = sys.maxsize
    min_s: int = sys.maxsize
    for num in nums:
        if num >= max_f:
            max_t = max_s
            max_s = max_f
            max_f = num
        elif max_s <= num < max_f:
            max_t = max_s
            max_s = num
        elif max_t <= num < max_s:
            max_t = num

        if num <= min_f:
            min_s = min_f
            min_f = num
        elif min_s >= num > min_f:
            min_s = num

    return max(max_f * max_s * max_t, min_f * min_s * max_f)


# https://leetcode.com/problems/power-of-three/

def isPowerOfThree(n: int) -> bool:
    if n == 0 or n < 0:
        return False

    i: int = 1
    while i < n:
        i *= 3

    return i == n


# https://leetcode.com/problems/find-closest-number-to-zero/

def findClosestNumber(nums: List[int]) -> int:
    closest_no: int = sys.maxsize
    for num in nums:
        abs_num: int = abs(num)
        abs_closest: int = abs(closest_no)

        if abs_num <= abs_closest:
            if abs_num == abs_closest:
                closest_no = max(closest_no, num)
            else:
                closest_no = num

    return closest_no


# https://leetcode.com/problems/valid-palindrome/

def isPalindrome1(s: str) -> bool:
    i: int = 0
    j: int = len(s) - 1
    while i < j:
        if not s[i].isalnum() or s[i] == ' ':
            i += 1
            continue
        if not s[j].isalnum() or s[j] == ' ':
            j -= 1
            continue
        if s[i].lower() == s[j].lower():
            i += 1
            j -= 1

    return True


# https://leetcode.com/problems/sentence-similarity/description/

def areSentencesSimilar(sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
    if len(sentence1) != len(sentence2):
        return False
    similar_dict: Mapping[str, Set[str]] = {}
    for similarPair in similarPairs:
        first: str = similarPair[0]
        second: str = similarPair[1]
        first_set: Set[str] = similar_dict.get(first, set())
        first_set.add(second)
        second_set: Set[str] = similar_dict.get(second, set())
        second_set.add(first)
        similar_dict[first] = first_set
        similar_dict[second] = second_set

    for i in range(len(sentence1)):
        first: str = sentence1[i]
        second: str = sentence2[i]

        if first == second or second in similar_dict.get(first, set()) or first in similar_dict.get(second, set()):
            continue
        else:
            return False

    return True


# https://leetcode.com/problems/rectangle-overlap/description/

def isRectangleOverlap(rec1: List[int], rec2: List[int]) -> bool:
    return rec1[0] < rec2[2] and rec1[1] < rec2[3] and rec2[0] < rec1[2] and rec2[1] < rec1[3]


# https://leetcode.com/problems/plus-one/description/

def plusOne(self, digits: List[int]) -> List[int]:
    carryover: int = 1
    for i in range(len(digits) - 1, -1, -1):
        new_val: int = (digits[i] + carryover)
        digits[i] = new_val % 10
        carryover = new_val // 10

    if carryover > 0:
        digits = [1] + digits

    return digits


# https://leetcode.com/problems/repeated-substring-pattern/description/

def repeatedSubstringPattern(self, s: str) -> bool:
    for i in range(1, (len(s) // 2) + 1):
        check_str: str = s[0:i]
        check_flag: bool = True
        for j in range(0, len(s), i):
            if s[j: j + i] != check_str:
                check_flag = False
                break
        if check_flag:
            return True

    return False


# https://leetcode.com/problems/is-subsequence/

def isSubsequence(self, s: str, t: str) -> bool:
    s_pointer: int = 0
    for i in range(0, len(t)):
        if s_pointer < len(s) and t[i] == s[s_pointer]:
            s_pointer += 1

    return s_pointer == len(s)


# https://leetcode.com/problems/maximum-average-subarray-i/

def findMaxAverage(self, nums: List[int], k: int) -> float:
    i: int = 0
    running_sum: int = 0

    while i < k and i < len(nums):
        running_sum += nums[i]
        i += 1

    max_avg: float = running_sum / k
    while i < len(nums):
        running_sum = running_sum + nums[i] - nums[i - k]
        max_avg = max(max_avg, running_sum / k)
        i += 1

    return max_avg


# https://leetcode.com/problems/search-insert-position/description/

def searchInsert(self, nums: List[int], target: int) -> int:
    low: int = 0
    high: int = len(nums) - 1
    while low <= high:
        mid: int = low + (high - low) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return low


# https://leetcode.com/problems/rearrange-spaces-between-words/description/

def reorderSpaces(self, text: str) -> str:
    words: List[str] = []
    spaces: int = 0
    first_idx: int = 0
    is_letter: bool = False
    for i in range(len(text)):
        if text[i].isalpha() and not is_letter:
            is_letter = True
            first_idx = i
        elif text[i] == ' ':
            if is_letter:
                words.append(text[first_idx: i])
                is_letter = False
            spaces += 1

    if is_letter:
        words.append(text[first_idx: len(text)])

    if not spaces or len(words) == 1:
        return words[0] + (" " * spaces)

    spaces_needed: int = spaces // (len(words) - 1)
    rem: int = spaces % (len(words) - 1)
    res: str = ""
    for i in range(len(words) - 1):
        res = res + words[i] + (" " * spaces_needed)

    res += words[len(words) - 1]
    res += " " * rem

    return res


# https://leetcode.com/problems/first-bad-version/

def isBadVersion() -> bool:
    return True


def firstBadVersion(self, n: int) -> int:
    low: int = 0
    high: int = n
    first_bad: int = low
    while low <= high:
        mid: int = low + (high - low) // 2
        if isBadVersion():
            first_bad = mid
            high = mid - 1
        else:
            low = mid + 1

    return first_bad


# https://leetcode.com/problems/valid-perfect-square/description/

def isPerfectSquare(self, num: int) -> bool:
    low: int = 0
    high: int = num // 2 + 1
    while low <= high:
        mid: int = low + (high - low) // 2
        sq: int = mid * mid
        if sq == num:
            return True
        elif sq < num:
            low = mid + 1
        else:
            high = mid - 1

    return False


# https://leetcode.com/problems/license-key-formatting/description/

def licenseKeyFormatting(self, s: str, k: int) -> str:
    license_stack: Deque[str] = deque()
    for value in s:
        if value.isalnum():
            license_stack.append(value)

    groups_count: int = len(license_stack) // k
    rem: int = len(license_stack) % k

    result: str = ""
    while rem and license_stack:
        result += license_stack.popleft().upper()
        rem -= 1

    while groups_count:
        if result:
            result += "-"
        group_val: int = k
        while group_val and license_stack:
            result += license_stack.popleft().upper()
            group_val -= 1
        groups_count -= 1

    return result


# https://leetcode.com/problems/number-of-valid-clock-times/

def countTime(self, time: str) -> int:
    possibilities: int = 1

    for i in [0, 1, 3, 4]:
        if time[i] == '?':
            if i == 0:
                if time[i + 1] in ['0', '1', '2', '3', ]:
                    possibilities *= 3
                elif time[i + 1] in ['4', '5', '6', '7', '8', '9', ]:
                    possibilities *= 2
                else:
                    possibilities *= 24
            elif i == 1:
                if time[i - 1] in ['0', '1', ]:
                    possibilities *= 10
                elif time[i - 1] in ['2', ]:
                    possibilities *= 4
            elif i == 3:
                possibilities *= 6
            else:
                possibilities *= 10

    return possibilities


# https://leetcode.com/problems/length-of-last-word/

def lengthOfLastWord(self, s: str) -> int:
    length: int = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i].isalpha():
            length += 1
        elif not s[i].isalpha() and length:
            break

    return length


# https://leetcode.com/problems/isomorphic-strings/

def isIsomorphic(self, s: str, t: str) -> bool:
    mapping_s_t: Mapping[str, str] = {}
    mapping_t_s: Mapping[str, str] = {}
    for i in range(0, len(s)):
        if s[i] not in mapping_s_t and t[i] not in mapping_t_s:
            mapping_s_t[s[i]] = t[i]
            mapping_t_s[t[i]] = s[i]

        if mapping_s_t.get(s[i], '') != t[i] or mapping_t_s.get(t[i], '') != s[i]:
            return False

    return True


# https://leetcode.com/problems/contains-duplicate-ii/

def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    mapping: Mapping[int, List[int]] = {}
    for i, e in enumerate(nums):
        idx_list: List[int] = mapping.get(e, [])
        if idx_list and i - idx_list[-1] <= k:
            return True

        idx_list.append(i)
        mapping[e] = idx_list

    return False


# https://leetcode.com/problems/latest-time-by-replacing-hidden-digits/submissions/936492588/

def maximumTime(self, time: str) -> str:
    time: List[str] = list(time)

    for i in [0, 1, 3, 4]:
        if time[i] == '?':
            if i == 0:
                if time[i + 1] in ['0', '1', '2', '3', ]:
                    time[i] = '2'
                elif time[i + 1] in ['4', '5', '6', '7', '8', '9', ]:
                    time[i] = '1'
                else:
                    time[i] = '2'
                    time[i + 1] = '3'
            elif i == 1:
                if time[i - 1] in ['0', '1', ]:
                    time[i] = '9'
                elif time[i - 1] in ['2', ]:
                    time[i] = '3'
            elif i == 3:
                time[i] = '5'
            elif i == 4:
                time[i] = '9'

    return "".join(time)


# https://leetcode.com/problems/read-n-characters-given-read4/

def read4(buf4):
    return 5


def read(self, buf: List[str], n):
    idx: int = 0
    while idx < n:
        buf4: List[str] = [' '] * 4
        num_read: int = read4(buf4)
        if not num_read:
            break

        for i in range(0, num_read):
            if idx == n:
                return n
            buf[idx] = buf4[i]
            idx += 1

    return idx


# https://leetcode.com/problems/longest-common-prefix/description/

def longestCommonPrefix(self, strs: List[str]) -> str:
    result: str = strs[0]

    for i in range(1, len(strs)):
        word: str = strs[i]
        if len(word) < len(result):
            result = result[: len(word)]
        for j in range(len(word)):
            if j < len(result) and result[j] != word[j]:
                result = result[: j]

    return result


# https://leetcode.com/problems/valid-parentheses/

def isValid(self, s: str) -> bool:
    opening_set: Set[str] = {'(', '{', '['}
    p_stack: Deque[str] = deque()
    for parenthesis in s:
        if parenthesis in opening_set:
            p_stack.append(parenthesis)
        elif p_stack:
            popped: str = p_stack.pop()
            if (popped == '(' and parenthesis == ')') or \
                    (popped == '{' and parenthesis == '}') or \
                    (popped == '[' and parenthesis == ']'):
                continue
            else:
                return False
        else:
            return False

    return not p_stack


# https://leetcode.com/problems/valid-palindrome-ii/

def validPalindromeUtils(s: str, i: int, j: int, isFlag: bool) -> bool:
    if i > j:
        return True
    if s[i] == s[j]:
        return validPalindromeUtils(s=s, i=i + 1, j=j - 1, isFlag=isFlag)
    if not isFlag:
        return validPalindromeUtils(s=s, i=i + 1, j=j, isFlag=True) or validPalindromeUtils(s=s, i=i, j=j - 1,
                                                                                            isFlag=True)

    return False


def validPalindrome(self, s: str) -> bool:
    return validPalindromeUtils(s=s, i=0, j=0, isFlag=False)


# https://leetcode.com/problems/valid-word-square/description/

def validWordSquare(self, words: List[str]) -> bool:
    for i in range(len(words)):
        for j in range(len(words[i])):
            if j >= len(words) or i >= len(words[j]) or words[i][j] != words[j][i]:
                return False

    return True


# https://leetcode.com/problems/design-compressed-string-iterator/description/

class StringIterator:

    def __init__(self, compressedString: str):
        self.i: int = 0
        self.compressedString: str = compressedString
        self.last_letter: str = " "
        self.count: int = 0

    def next(self) -> str:
        if self.count:
            self.count -= 1
            return self.last_letter
        elif self.i < len(self.compressedString):
            self.last_letter = self.compressedString[self.i]
            self.i += 1
            while self.i < len(self.compressedString) and self.compressedString[self.i].isnumeric():
                self.count = self.count * 10 + int(self.compressedString[self.i])
                self.i += 1
            self.count -= 1
            return self.last_letter

        else:
            return " "

    def hasNext(self) -> bool:
        return self.i < len(self.compressedString) or self.count


# https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/

def strStr(self, haystack: str, needle: str) -> int:
    i: int = 0
    j: int = 0
    while i < len(haystack):
        if j < len(needle) and needle[j] == haystack[i]:
            i += 1
            j += 1
        else:
            i = i - j + 1
            j = 0
        if j == len(needle):
            return i - j

    return -1


# https://leetcode.com/problems/perfect-number/

def checkPerfectNumber(self, num: int) -> bool:
    if num == 1:
        return False
    divisor_sum: int = 0
    sqrt_no: int = int(math.sqrt(num))
    print(sqrt_no)
    for i in range(1, sqrt_no + 1):
        other_no: int = num // i
        if i * other_no == num:
            divisor_sum = divisor_sum + i + other_no

    return divisor_sum == 2 * num
