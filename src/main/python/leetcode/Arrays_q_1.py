import sys
from collections import deque
from typing import List, Set, Mapping
import collections
import math


# https://leetcode.com/problems/build-array-from-permutation/

def buildArray(nums: List[int]) -> List[int]:
    n: int = len(nums)
    for i in range(n):
        no: int = nums[i]
        old_no: int = nums[no] % n
        new_no: int = old_no * n
        nums[i] = (nums[i] + new_no)

    for i in range(n):
        new_no: int = nums[i] // n
        nums[i] = new_no

    return nums


# https://leetcode.com/problems/concatenation-of-array/

def getConcatenation(nums: List[int]) -> List[int]:
    n: int = len(nums)
    ans: List[int] = [0] * 2 * n
    print(ans)
    for i, no in enumerate(nums):
        ans[i] = no
        ans[i + n] = no

    return ans


# https://leetcode.com/problems/running-sum-of-1d-array/

def runningSum(nums: List[int]) -> List[int]:
    cumulative_sum: int = 0
    n: int = len(nums)
    for i in range(n):
        cumulative_sum += nums[i]
        nums[i] = cumulative_sum

    return nums


# https://leetcode.com/problems/final-value-of-variable-after-performing-operations/

def finalValueAfterOperations(operations: List[str]) -> int:
    final_value: int = 0
    for operation in operations:
        if operation[0] == '+' or operation[-1] == '+':
            final_value += 1
        else:
            final_value -= 1

    return final_value


# https://leetcode.com/problems/richest-customer-wealth/

def maximumWealth(accounts: List[List[int]]) -> int:
    max_wealth: int = 0
    for i in accounts:
        wealth: int = 0
        for j in i:
            wealth += j
        if wealth > max_wealth:
            max_wealth = wealth
    return max_wealth


# https://leetcode.com/problems/shuffle-the-array/

def shuffle(nums: List[int], n: int) -> List[int]:
    n: int = len(nums)
    ans: List[int] = []
    j: int = n // 2
    for i in range(n // 2):
        ans.append(nums[i])
        ans.append(nums[j])
        j += 1

    return ans


# https://leetcode.com/problems/maximum-number-of-words-found-in-sentences/

def mostWordsFound(self, sentences: List[str]) -> int:
    max_length: int = 0
    for sentence in sentences:
        words: list[str] = sentence.split(" ")
        max_length = max(max_length, len(words))

    return max_length


# https://leetcode.com/problems/number-of-good-pairs/

def numIdenticalPairs(nums: List[int]) -> int:
    len(nums)
    count_arr: List[int] = [0] * 101
    for e in nums:
        count_arr[e] += 1

    pairs: int = 0
    for counts in count_arr:
        if counts > 2:
            pairs += math.comb(counts, 2)

    return pairs


# https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/

def kidsWithCandies(candies: List[int], extraCandies: int) -> List[bool]:
    ans: List[bool] = []
    max_c: int = 0
    for candy in candies:
        if max_c < candy:
            max_c = candy

    for candy in candies:
        if candy + extraCandies >= max_c:
            ans.append(True)
        else:
            ans.append(False)
    return ans


# https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/

def smallerNumbersThanCurrent(nums: List[int]) -> List[int]:
    count_arr: List[int] = [0] * 101
    for num in nums:
        count_arr[num] += 1

    for i, num in enumerate(count_arr):
        if i > 0:
            count_arr[i] += count_arr[i - 1]

    for i, num in enumerate(nums):
        if num > 0:
            nums[i] = count_arr[num - 1]
        else:
            nums[i] = 0

    return nums


# https://leetcode.com/problems/decompress-run-length-encoded-list/

def decompressRLElist(nums: List[int]) -> List[int]:
    n: int = len(nums)
    res = []
    for i in range(0, n - 1, 2):
        f = nums[i]
        v = nums[i + 1]
        for j in range(f):
            res.append(v)
    return res


# https://leetcode.com/problems/decode-xored-array/

def decode(encoded: List[int], first: int) -> List[int]:
    arr: List[int] = [first]
    for num in encoded:
        xor = arr[-1] ^ num
        arr.append(xor)

    return arr


# https://leetcode.com/problems/create-target-array-in-the-given-order/

def createTargetArray(nums: List[int], index: List[int]) -> List[int]:
    n: int = len(index)
    arr = []
    for i in range(n):
        arr.insert(index[i], nums[i])

    return arr


# https://leetcode.com/problems/shuffle-string/

def restoreString(s: str, indices: List[int]) -> str:
    n: int = len(indices)
    res: List[int] = [0] * n
    for i in range(n):
        res[indices[i]] = s[i]
    return "".join(res)


# https://leetcode.com/problems/design-an-ordered-stream/

class OrderedStream:

    def __init__(self, n: int):
        self.n: int = n
        self.stream: List[str] = [""] * n
        self.ptr = 0

    def insert(self, idKey: int, value: str) -> List[str]:
        self.stream[idKey - 1] = value
        res = []
        while self.ptr < self.n and self.stream[self.ptr] != '':
            res.append(self.stream[self.ptr])
            self.ptr += 1

        return res


# https://leetcode.com/problems/count-items-matching-a-rule/

def countMatches(items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
    count: int = 0
    if ruleKey == "type":
        index = 0
    elif ruleKey == "color":
        index = 1
    else:
        index = 2
    for item in items:
        if item[index] == ruleValue:
            count += 1

    return count


# https://leetcode.com/problems/number-of-arithmetic-triplets/

def arithmeticTriplets(nums: List[int], diff: int) -> int:
    value_set: Set[int] = set()
    for v in nums:
        value_set.add(v)

    count: int = 0
    for i, v in enumerate(nums):
        if v + diff in value_set and v + (2 * diff) in value_set:
            count += 1

    return count


# https://leetcode.com/problems/largest-local-values-in-a-matrix/

def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
    m: int = len(grid) - 2
    new_grid: List[List[int]] = [[0 for _ in range(m)] for _ in range(m)]
    for i in range(0, m):
        for j in range(0, m):
            new_grid[i][j] = max(grid[i][j], grid[i][j + 1], grid[i][j + 2],
                                 grid[i + 1][j], grid[i + 1][j + 1], grid[i + 1][j + 2],
                                 grid[i + 2][j], grid[i + 2][j + 1], grid[i + 2][j + 2])

    return new_grid


# https://leetcode.com/problems/sum-of-all-odd-length-subarrays/

def sumOddLengthSubarrays(self, arr: List[int]) -> int:
    result: int = 0
    n: int = len(arr)
    for i, no in enumerate(arr):
        start: int = n - i
        end: int = i + 1
        total: int = start * end
        odd: int = total // 2
        if total % 2 != 0:
            odd += 1
        result += odd * no

    return result


# https://leetcode.com/problems/find-anagram-mappings/

def anagramMappings(self, nums1: List[int], nums2: List[int]) -> List[int]:
    n: int = len(nums1)
    d2 = {}
    for i in range(n):
        d2[nums2[i]] = i

    res: List[int] = []
    for i in range(n):
        res.append(d2[nums1[i]])

    return res


# https://leetcode.com/problems/unique-morse-code-words/

def uniqueMorseRepresentations(self, words: List[str]) -> int:
    m_codes: List[str] = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..",
                          ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.",
                          "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
    unique_v: Set[str] = set()
    for word in words:
        m_code: str = ""
        for alphabet in word:
            m_code = m_code + m_codes[ord(alphabet) - 97]
        unique_v.add(m_code)

    return len(unique_v)


# https://leetcode.com/problems/count-number-of-pairs-with-absolute-difference-k/

def countKDifference(self, nums: List[int], k: int) -> int:
    d: Mapping[int, int] = {}
    for num in nums:
        d[num] = d.get(num, 0) + 1

    k_count: int = 0
    for num in nums:
        f: int = d.get(num + k, 0)
        k_count += f

    return k_count


# https://leetcode.com/problems/minimum-number-of-moves-to-seat-everyone/

def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
    sorted_seats: List[int] = sorted(seats)
    sorted_students: List[int] = sorted(students)
    return sum(abs(x - y) for x, y in zip(sorted_seats, sorted_students))


# https://leetcode.com/problems/check-if-two-string-arrays-are-equivalent/

def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
    return "".join(word1) == "".join(word2)


# https://leetcode.com/problems/sort-the-people/

def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
    d: Mapping[int, str] = {}
    for i, height in enumerate(heights):
        d[height] = names[i]

    od = collections.OrderedDict(sorted(d.items(), reverse=True))
    return list(od.values())


# https://leetcode.com/problems/count-the-number-of-consistent-strings/

def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
    allowed_set: Set[str] = set(allowed)
    count: int = 0
    for word in words:
        not_found: bool = False
        for alphabet in word:
            if alphabet not in allowed_set:
                not_found = True
                break

        if not not_found:
            count += 1

    return count


# https://leetcode.com/problems/truncate-sentence/

def truncateSentence(self, s: str, k: int) -> str:
    spaces: int = 0
    for i, w in enumerate(s):
        if w == ' ':
            spaces += 1
            if spaces == k:
                return s[0: i]

    return s


# https://leetcode.com/problems/maximum-product-difference-between-two-pairs/

def maxProductDifference(self, nums: List[int]) -> int:
    max_no: int = - sys.maxsize
    max_second: int = - sys.maxsize
    min_no: int = sys.maxsize
    min_second = sys.maxsize
    for no in nums:
        if no > max_no:
            max_second = max_no
            max_no = no
        elif no > max_second:
            max_second = no

        if no < min_no:
            min_second = min_no
            min_no = no
        elif no < min_second:
            min_second = no

    return max_no * max_second - (min_no * min_second)


# https://leetcode.com/problems/count-good-triplets/description/

def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
    out = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if abs(arr[i] - arr[j]) <= a:
                for k in range(j + 1, len(arr)):
                    if abs(arr[j] - arr[k]) <= b:
                        if abs(arr[i] - arr[k]) <= c:
                            out += 1

    return out


# https://leetcode.com/problems/flipping-an-image/

def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
    for row in image:
        n: int = len(row)
        for i in range(n // 2):
            row[i] = row[n - i]

    for row in image:
        n: int = len(row)
        for i in range(n):
            if row[i] == 1:
                row[i] = 0
            else:
                row[i] = 1

    return image


# https://leetcode.com/problems/count-equal-and-divisible-pairs-in-an-array/

def countPairs(self, nums: List[int], k: int) -> int:
    d: Mapping[int, List[int]] = {}
    c: int = 0

    for i, num in enumerate(nums):
        if d.get(num) is not None:
            l: List[int] = d.get(num)
            l.append(i)
        else:
            l: List[int] = [i]

        d[num] = l

    for v in d.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if (v[i] * v[j]) % k == 0:
                    c += 1

    return c


# https://leetcode.com/problems/intersection-of-three-sorted-arrays/description/

def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
    res: List[int] = []
    i: int = 0
    j: int = 0
    k: int = 0

    while i < len(arr1) and j < len(arr2) and k < len(arr3):
        a: int = arr1[i]
        b: int = arr2[j]
        c: int = arr3[k]

        if a == b == c:
            res.append(a)
            i += 1
            j += 1
            k += 1
        elif min(a, b, c) == a:
            i += 1
        elif min(a, b, c) == b:
            j += 1
        else:
            k += 1

    return res


# https://leetcode.com/problems/matrix-diagonal-sum/

def diagonalSum(self, mat: List[List[int]]) -> int:
    n: int = len(mat)
    s: int = 0
    for i in range(n):
        s += mat[i][i]

    for i in range(n):
        j: int = n - 1 - i
        if i != j:
            s += mat[i][j]

    return s


# https://leetcode.com/problems/maximum-product-of-two-elements-in-an-array/

def maxProduct(self, nums: List[int]) -> int:
    first_max: int = 0
    second_max: int = 0

    for num in nums:
        if num > first_max:
            second_max = first_max
            first_max = num
        elif num > second_max:
            second_max = num

    return (first_max - 1) * (second_max - 1)


# https://leetcode.com/problems/minimum-time-visiting-all-points/description/

def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
    n: int = len(points)
    time: int = 0
    for i in range(n - 1):
        first: List[int] = points[i]
        second: List[int] = points[i + 1]

        max_time: int = max(abs(second[1] - first[1]), abs(second[0] - first[0]))
        time += max_time

    return time


# https://leetcode.com/problems/sum-of-all-subset-xor-totals/description/

def subsetXORSumUtil(nums: List[int], i: int, running_xor: int, total: int) -> None:
    if i == len(nums):
        total[0] += running_xor
        return
    else:
        subsetXORSumUtil(nums=nums, i=i + 1, running_xor=running_xor ^ nums[i], total=total)
        subsetXORSumUtil(nums=nums, i=i + 1, running_xor=running_xor, total=total)


def subsetXORSum(self, nums: List[int]) -> int:
    total: int = 0
    subsetXORSumUtil(nums, 0, 0, total)
    return total


# https://leetcode.com/problems/find-the-highest-altitude/

def largestAltitude(self, gain: List[int]) -> int:
    max_alt: int = 0
    initial: int = 0

    for points in gain:
        initial += points
        max_alt = max(max_alt, initial)

    return max_alt


# https://leetcode.com/problems/number-of-rectangles-that-can-form-the-largest-square/

def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
    count: int = 0
    max_len: int = 0

    for rectangle in rectangles:
        min_len_sq: int = min(rectangle[0], rectangle[1])
        if min_len_sq > max_len:
            max_len = min_len_sq
            count = 1
        elif min_len_sq == max_len:
            count += 1

    return count


# https://leetcode.com/problems/cells-with-odd-values-in-a-matrix/description/

def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
    odd_rows: List[bool] = [False] * m
    odd_columns: List[bool] = [False] * n
    for index in indices:
        odd_rows[index[0]] ^= True
        odd_columns[index[1]] ^= True

    r = 0
    c = 0
    for b in odd_rows:
        if b:
            r += 1
    for b in odd_columns:
        if b:
            c += 1

    return (n * r) + (m * c) - 2 * (r * c)


# https://leetcode.com/problems/find-first-palindromic-string-in-the-array/


def firstPalindrome(self, words: List[str]) -> str:
    for word in words:
        i: int = 0
        j: int = len(word) - 1
        is_palindrome: bool = True

        while i < j:
            if word[i] != word[j]:
                is_palindrome = False
                break
            i += 1
            j -= 1

        if is_palindrome:
            return word

    return ""


# https://leetcode.com/problems/minimum-operations-to-make-the-array-increasing/

def minOperations(self, nums: List[int]) -> int:
    total_increment: int = 0
    last_incr_value: int = nums[0]

    for i in range(1, len(nums)):
        diff: int = nums[i] - last_incr_value
        if diff == 0:
            last_incr_value = nums[i] + 1
            total_increment += 1
        elif diff < 0:
            last_incr_value = nums[i] + (diff * -1) + 1
            total_increment += (diff * -1) + 1
        else:
            last_incr_value = nums[i]

    return total_increment


# https://leetcode.com/problems/find-n-unique-integers-sum-up-to-zero/

def sumZero(self, n: int) -> List[int]:
    res: List[int] = []
    if n % 2 != 0:
        res.append(0)

    r: int = (n // 2) + 1

    for i in range(1, r):
        res.append(i)
        res.append(-i)

    return res


# https://leetcode.com/problems/counting-words-with-a-given-prefix/

def prefixCount(self, words: List[str], pref: str) -> int:
    count: int = 0
    for word in words:
        if len(pref) > len(word):
            continue

        is_prefix: bool = True
        for i in range(len(pref)):
            if pref[i] != word[i]:
                is_prefix = False
                break

        if is_prefix:
            count += 1

    return count


# https://leetcode.com/problems/moving-average-from-data-stream/

class MovingAverage:

    def __init__(self, size: int):
        self.n: int = size
        self.linked_list: deque = deque()
        self.sum: int = 0

    def next(self, val: int) -> float:
        self.sum += val
        self.linked_list.append(val)
        if len(self.linked_list) > self.n:
            popped: int = self.linked_list.popleft()
            self.sum -= popped

        return self.sum / len(self.linked_list)


# https://leetcode.com/problems/single-row-keyboard/

def calculateTime(self, keyboard: str, word: str) -> int:

    alphabets: Mapping[str, int] = {}
    for i, e in enumerate(keyboard):
        alphabets[e] = i

    initial: int = 0
    total: int = 0

    for k in word:
        total += abs(alphabets[k] - initial)
        initial = alphabets[k]

    return total





