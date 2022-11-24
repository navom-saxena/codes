import functools
from typing import List, Mapping, Optional, Set

from python.leetcode.Models import TreeNode


# https://leetcode.com/problems/find-numbers-with-even-number-of-digits/

def findNumbers(self, nums: List[int]) -> int:
    count: int = 0
    for num in nums:
        length: int = 0
        if num == 0:
            continue

        while num > 0:
            length += 1
            num = num // 10

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
        self.slots[carType] = self.slots[carType] - 1
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

def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
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
        l[i] = reversed(word)

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









