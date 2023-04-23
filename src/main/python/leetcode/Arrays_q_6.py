import sys
from collections import deque
from typing import List, Set, Deque, Dict, Tuple


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
