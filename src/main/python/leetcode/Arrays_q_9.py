import sys
from typing import List, Set, Dict, Tuple
import heapq


# https://leetcode.com/problems/find-words-containing-character/

def findWordsContaining(self, words: List[str], x: str) -> List[int]:
    res: List[int] = []
    for i, word in enumerate(words):
        for character in word:
            if character == x:
                res.append(i)
                break

    return res


# https://leetcode.com/problems/widest-vertical-area-between-two-points-containing-no-points/description/

def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
    n: int = len(points)
    arr: List[int] = [0] * n
    res: int = -1

    for point in points:
        x: int = point[0]
        y: int = point[1]
        arr.append(x)

    sorted_arr: List[int] = sorted(arr)
    prev: int = sorted_arr[0]
    for i in range(1, n):
        curr: int = sorted_arr[i]
        res = max(res, curr - prev)
        prev = curr

    return res


# https://leetcode.com/problems/number-of-employees-who-met-the-target/description/

def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
    res: int = 0
    for hour in hours:
        if hour >= target:
            res += 1

    return res


# https://leetcode.com/problems/count-pairs-whose-sum-is-less-than-target/description/

def countPairs(self, nums: List[int], target: int) -> int:
    res: int = 0
    i: int = 0
    j: int = len(nums) - 1
    while i < j:
        if nums[i] + nums[j] < target:
            res += j - i
            i += 1
        else:
            j -= 1

    return res


# https://leetcode.com/problems/sum-of-values-at-indices-with-k-set-bits/description/

def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
    res: int = 0
    n: int = len(nums)
    mem_arr: List[int] = [0] * (n + 1)
    print(n, len(mem_arr))
    for i in range(0, n):
        mem_arr[i] = mem_arr[i // 2] + i % 2
        if mem_arr[i] == k:
            res += nums[i]

    return res


# https://leetcode.com/problems/left-and-right-sum-differences/description/

def leftRightDifference(self, nums: List[int]) -> List[int]:
    n: int = len(nums)
    left_sum: List[int] = [0] * n
    answer: List[int] = [0] * n

    prefix_sum: int = 0
    for i in range(n):
        left_sum[i] = prefix_sum
        prefix_sum += nums[i]

    suffix_sum: int = 0
    for i in range(n - 1, -1):
        answer[i] = abs(left_sum[i] - suffix_sum)
        suffix_sum += nums[i]

    return answer


# https://leetcode.com/problems/minimum-number-game/description/

def numberGame(self, nums: List[int]) -> List[int]:
    res: List[int] = []
    min_heap: List[int] = []
    for num in nums:
        heapq.heappush(min_heap, num)

    while len(min_heap) > 0:
        min_element: int = heapq.heappop(min_heap)
        second_min_element: int = heapq.heappop(min_heap)
        res.append(second_min_element)
        res.append(min_element)

    return res


# https://leetcode.com/problems/minimum-operations-to-exceed-threshold-value-i/

def minOperations(self, nums: List[int], k: int) -> int:
    nums: List[int] = sorted(nums)
    n: int = len(nums)
    low: int = 0
    high: int = n - 1
    while low <= high:
        mid: int = low + (high - low) // 2
        mid_value: int = nums[mid]
        if mid_value >= k:
            high = mid - 1
        else:
            low = mid + 1

    return high + 1


# https://leetcode.com/problems/difference-between-element-sum-and-digit-sum-of-an-array/description/

def differenceOfSum(self, nums: List[int]) -> int:
    element_sum: int = 0
    digit_sum: int = 0
    for num in nums:
        element_sum += num
        while num > 0:
            digit_sum += num % 10
            num //= 10

    return abs(element_sum - digit_sum)


# https://leetcode.com/problems/check-if-a-string-is-an-acronym-of-words/description/

def isAcronym(self, words: List[str], s: str) -> bool:
    n: int = len(s)
    if n != len(words):
        return False
    for i, word in enumerate(words):
        if s[i] != word[0]:
            return False

    return True


# https://leetcode.com/problems/find-common-elements-between-two-arrays/description/

def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
    nums1_set: Set[int] = set(nums1)
    nums2_set: Set[int] = set(nums2)
    res1: int = 0
    res2: int = 0
    for num in nums1:
        if num in nums2_set:
            res1 += 1

    for num in nums2:
        if num in nums1_set:
            res2 += 1

    return [res1, res2]


# https://leetcode.com/problems/subarrays-distinct-element-sum-of-squares-i/description

def sumCounts(self, nums: List[int]) -> int:
    n: int = len(nums)
    res: int = 0

    for i in range(1, n + 1):
        window_dict: Dict[int, int] = dict()
        for j in range(0, i):
            window_dict[nums[j]] = window_dict.get(nums[j], 0) + 1
        res += len(window_dict) ** 2
        for j in range(i, n):
            window_dict[nums[j]] = window_dict.get(nums[j], 0) + 1
            remove_key: int = nums[j - i]
            f: int = window_dict[remove_key]
            if f == 1:
                window_dict.pop(remove_key)
            else:
                window_dict[remove_key] = f - 1
            res += len(window_dict) ** 2

    return res


# https://leetcode.com/problems/maximum-sum-with-exactly-k-elements/description/

def maximizeSum(self, nums: List[int], k: int) -> int:
    max_no: int = 0
    for num in nums:
        if num > max_no:
            max_no = num

    res: int = (max_no * k) + (((k - 1) * k) // 2)
    return res


# https://leetcode.com/problems/find-the-integer-added-to-array-i/

def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
    n: int = len(nums1)
    sum1: int = 0
    sum2: int = 0
    for num in nums1:
        sum1 += num
    for num in nums2:
        sum2 += num

    return (sum2 - sum1) // n


# https://leetcode.com/problems/find-maximum-number-of-string-pairs/

def maximumNumberOfStringPairs(self, words: List[str]) -> int:
    word_dict: Dict[str, int] = dict()
    for word in words:
        sorted_word: str = word if ord(word[0]) < ord(word[1]) else word[1] + word[0]
        word_dict[sorted_word] = word_dict.get(sorted_word, 0) + 1

    res: int = 0
    for k, v in word_dict.items():
        res += v // 2

    return res


# https://leetcode.com/problems/sum-of-squares-of-special-elements/description/

def sumOfSquares(self, nums: List[int]) -> int:
    n: int = len(nums)
    res: int = 0
    for i, num in enumerate(nums):
        if n % (i + 1) == 0:
            res += num ** 2

    return res


# https://leetcode.com/problems/count-tested-devices-after-test-operations/description/

def countTestedDevices(self, batteryPercentages: List[int]) -> int:
    decay: int = 0
    for i, num in enumerate(batteryPercentages):
        percent: int = num - decay
        if percent > 0:
            decay += 1

    return decay


# https://leetcode.com/problems/count-elements-with-maximum-frequency/description/

def maxFrequencyElements(self, nums: List[int]) -> int:
    res: int = 0
    freq_map: Dict[int, int] = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1

    max_f: int = 0
    for k, v in freq_map.items():
        if v > max_f:
            res = v
            max_f = v
        elif v == max_f:
            res += v

    return res


# https://leetcode.com/problems/separate-the-digits-in-an-array/description/

def separateDigits(self, nums: List[int]) -> List[int]:
    res: List[int] = []
    for num in nums:
        found_flag: bool = False
        for i in range(5, -1, -1):
            denominator: int = 10 ** i
            q: int = num // denominator
            if q != 0 or found_flag:
                found_flag = True
                res.append(q)
            num %= denominator

    return res


# https://leetcode.com/problems/sort-integers-by-the-number-of-1-bits/description/

def get_bits(num: int) -> int:
    num_bits: int = 0
    while num > 0:
        num_bits += num & 1
        num >>= 1

    return num_bits


def sortByBits(self, arr: List[int]) -> List[int]:
    bits_count_map: Dict[int, List[int]] = dict()
    for num in arr:
        num_bits: int = get_bits(num)
        bit_nums: List[int] = bits_count_map.get(num_bits, [])
        bit_nums.append(num)
        bits_count_map[num_bits] = bit_nums

    res: List[int] = []
    for k in sorted(bits_count_map.keys()):
        bit_nums: List[int] = bits_count_map[k]
        for num in sorted(bit_nums):
            res.append(num)

    return res


# https://leetcode.com/problems/delete-greatest-value-in-each-row/description/

def deleteGreatestValue(self, grid: List[List[int]]) -> int:
    res: int = 0
    m: int = len(grid)
    n: int = len(grid[0])
    for i, row in enumerate(grid):
        row = sorted(row, reverse=True)
        grid[i] = row

    for j in range(0, n):
        max_at_c: int = 0
        for i in range(0, m):
            max_at_c = max(max_at_c, grid[i][j])
        res += max_at_c

    return res


# https://leetcode.com/problems/find-the-distinct-difference-array/description/

def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
    n: int = len(nums)
    res: List[int] = [0] * n

    running_set: Set[int] = set()
    for i in range(n - 1, -1, -1):
        res[i] = len(running_set)
        running_set.add(nums[i])

    running_set = set()
    for i in range(0, n):
        running_set.add(nums[i])
        res[i] = len(running_set) - res[i]

    return res


# https://leetcode.com/problems/n-repeated-element-in-size-2n-array/description/

def repeatedNTimes(self, nums: List[int]) -> int:
    seen: Set[int] = set()
    for num in nums:
        if num in seen:
            return num
        seen.add(num)

    return -1


# https://leetcode.com/problems/number-of-senior-citizens/description/

def countSeniors(self, details: List[str]) -> int:
    res: int = 0
    for detail in details:
        if int(detail[11:13]) > 60:
            res += 1

    return res


# https://leetcode.com/problems/neither-minimum-nor-maximum/description/

def findNonMinOrMax(self, nums: List[int]) -> int:
    min_no: int = sys.maxsize
    max_no: int = - sys.maxsize

    for num in nums:
        if num < min_no:
            min_no = num

        if num > max_no:
            max_no = num

        if max_no > num > min_no:
            return num

    for num in nums:
        if max_no > num > min_no:
            return num

    return -1


# https://leetcode.com/problems/maximum-strong-pair-xor-i/description/

def maximumStrongPairXor(self, nums: List[int]) -> int:
    n: int = len(nums)
    max_xor: int = 0
    nums = sorted(nums)
    for i in range(n):
        low: int = i + 1
        high: int = len(nums) - 1
        while low <= high:
            mid: int = low + (high - low) // 2
            if nums[mid] <= 2 * nums[i]:
                low = mid + 1
            else:
                high = mid - 1

        for j in range(i + 1, high + 1):
            max_xor = max(max_xor, nums[i] ^ nums[j])

    return max_xor


# https://leetcode.com/problems/find-the-number-of-good-pairs-i/description/

def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
    res: int = 0
    for num2 in nums2:
        for num1 in nums1:
            if num1 % (num2 * k) == 0:
                res += 1

    return res


# https://leetcode.com/problems/find-the-xor-of-numbers-which-appear-twice/description/

def duplicateNumbersXOR(self, nums: List[int]) -> int:
    count_dict: Dict[int, int] = dict()

    for num in nums:
        count_dict[num] = count_dict.get(num, 0) + 1

    res: int = 0
    for k, v in count_dict.items():
        if v == 2:
            res ^= k

    return res


# https://leetcode.com/problems/special-array-i/description/

def isArraySpecial(self, nums: List[int]) -> bool:
    n: int = len(nums)
    for i in range(n - 1):
        if not ((nums[i] % 2 == 0 and nums[i + 1] % 2 != 0) or (nums[i] % 2 != 0 and nums[i + 1] % 2 == 0)):
            return False

    return True


# https://leetcode.com/problems/find-the-peaks/description/

def findPeaks(self, mountain: List[int]) -> List[int]:
    n: int = len(mountain)
    res: List[int] = list()
    for i in range(1, n - 1):
        if mountain[i - 1] < mountain[i] > mountain[i + 1]:
            res.append(i)

    return res


# https://leetcode.com/problems/row-with-maximum-ones/description/

def rowAndMaximumOnes(self, mat: List[List[int]]) -> List[int]:
    res: List[int] = [-1, -1]
    for i, row in enumerate(mat):
        ones_count: int = 0
        for col_value in row:
            if col_value == 1:
                ones_count += 1
        if res[1] < ones_count:
            res[0] = i
            res[1] = ones_count

    return res


# https://leetcode.com/problems/find-missing-and-repeated-values/description/

def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
    n: int = len(grid)
    sum_v: int = (n ** 2) * ((n ** 2) + 1) // 2
    sum_grid: int = 0
    a: int = -1
    repeated: Set[int] = set()
    for rows in grid:
        for value in rows:
            if value in repeated:
                a = value
            repeated.add(value)
            sum_grid += value

    b: int = abs(sum_v - sum_grid + a)
    return [a, b]


# https://leetcode.com/problems/largest-positive-integer-that-exists-with-its-negative/description/

def findMaxK(self, nums: List[int]) -> int:
    num_set: Set[int] = set()
    max_no: int = -1
    for num in nums:
        if (num * -1) in num_set and abs(num) > max_no:
            max_no = abs(num)
        num_set.add(num)

    return max_no


# https://leetcode.com/problems/two-out-of-three/description/

def twoOutOfThree(self, nums1: List[int], nums2: List[int], nums3: List[int]) -> List[int]:
    num_dict: Dict[int, Set[int]] = dict()
    res: List[int] = list()

    for num in nums1:
        num_set = num_dict.get(num, set())
        num_set.add(1)
        num_dict[num] = num_set

    for num in nums2:
        num_set = num_dict.get(num, set())
        num_set.add(2)
        num_dict[num] = num_set

    for num in nums3:
        num_set = num_dict.get(num, set())
        num_set.add(3)
        num_dict[num] = num_set

    for num, num_set in num_dict.items():
        if len(num_set) >= 2:
            res.append(num)

    return res




