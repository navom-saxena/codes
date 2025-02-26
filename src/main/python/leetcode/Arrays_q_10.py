import heapq
import math
from typing import List, Set, Dict, Tuple


# https://leetcode.com/problems/assign-cookies/description/?envType=problem-list-v2&envId=array

def findContentChildren(self, g: List[int], s: List[int]) -> int:
    g: List[int] = sorted(g, reverse=True)
    s: List[int] = sorted(s, reverse=True)

    j: int = 0
    for i in range(len(g)):
        if j < len(s) and g[i] <= s[j]:
            j += 1

    return j


# https://leetcode.com/problems/distribute-candies/description/?envType=problem-list-v2&envId=array

def distributeCandies(self, candyType: List[int]) -> int:
    candy_set: Set[int] = set()
    for candy in candyType:
        candy_set.add(candy)

    return min(len(candyType) // 2, len(candy_set))


# https://leetcode.com/problems/longest-harmonious-subsequence/description/?envType=problem-list-v2&envId=array

def findLHS(self, nums: List[int]) -> int:
    num_elements: Dict[int, int] = dict()
    max_len: int = 0
    for num in nums:
        num_elements[num] = num_elements.get(num, 0) + 1

    for k, v in num_elements.items():
        if k + 1 in num_elements:
            max_len = max(max_len, num_elements[k] + num_elements[k + 1])
    return max_len


# https://leetcode.com/problems/range-addition-ii/?envType=problem-list-v2&envId=array

def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
    min_intersection_r: int = m
    min_intersection_c: int = n

    for op in ops:
        min_intersection_r = min(op[0], min_intersection_r)
        min_intersection_c = min(op[1], min_intersection_c)

    return min_intersection_r * min_intersection_c


# https://leetcode.com/problems/minimum-index-sum-of-two-lists/description/?envType=problem-list-v2&envId=array

def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
    l1_dict: Dict[str, int] = dict()
    l2_dict: Dict[str, int] = dict()

    for i in range(len(list1)):
        word: str = list1[i]
        if word not in l1_dict:
            l1_dict[word] = i

    for i in range(len(list2)):
        word: str = list2[i]
        if word not in l2_dict:
            l2_dict[word] = i

    result: Dict[int, List[str]] = dict()
    for k, v in l1_dict.items():
        if k in l2_dict:
            index_value: int = v + l2_dict[k]
            result_list: List[str] = result.get(index_value, [])
            result_list.append(k)
            result[index_value] = result_list

    min_key: int = min(result.keys())
    return result[min_key]


# https://leetcode.com/problems/number-of-lines-to-write-string/?envType=problem-list-v2&envId=array

def numberOfLines(self, widths: List[int], s: str) -> List[int]:
    row_length: int = 0
    lines: int = 0

    row_reset_counter: int = 0
    i: int = 0
    for i in range(len(s)):
        ith_ord: int = ord(s[i]) - 97
        if row_length + widths[ith_ord] <= 100:
            row_length += widths[ith_ord]
        else:
            lines += 1
            row_length = widths[ith_ord]

    if row_length > 1:
        lines += 1

    result: List[int] = [lines, row_length]
    return result


# https://leetcode.com/problems/lemonade-change/description/?envType=problem-list-v2&envId=array

def lemonadeChange(self, bills: List[int]) -> bool:
    running_change: List[int] = [0] * 5
    for bill in bills:
        to_return: int = bill - 5

        i: int = 2
        while to_return > 0 and i >= 0:
            if running_change[i] != 0 and to_return - (i * 5) >= 0:
                to_return -= i * 5
                running_change[i] -= 1
            else:
                i -= 1

        if to_return > 0:
            return False

        running_change[bill // 5] += 1

    return True


# https://leetcode.com/problems/surface-area-of-3d-shapes/description/?envType=problem-list-v2&envId=array

def surfaceArea(self, grid: List[List[int]]) -> int:
    surface_area: int = 0
    n: int = len(grid)

    for i in range(n):
        for j in range(n):
            curr_size: int = grid[i][j]
            up: int = 0 if i - 1 < 0 else grid[i - 1][j]
            down: int = 0 if i + 1 >= n else grid[i + 1][j]
            left: int = 0 if j - 1 < 0 else grid[i][j - 1]
            right: int = 0 if j + 1 >= n else grid[i][j + 1]
            height: int = 0 if curr_size == 0 else 1
            base: int = 0 if curr_size == 0 else 1

            surface_area += (height + base + max(curr_size - up, 0) + max(curr_size - down, 0) + max(curr_size - left,
                                                                                                     0) + max(
                curr_size - right, 0))

    return surface_area


# https://leetcode.com/problems/largest-perimeter-triangle/description/?envType=problem-list-v2&envId=array

def largestPerimeter(self, nums: List[int]) -> int:
    nums.sort()

    for i in range(len(nums) - 1, -1, -1):
        if (i - 2 >= 0 and nums[i - 2] + nums[i - 1] >= nums[i] and nums[i - 2] + nums[i] >= nums[i - 1] and nums[
            i - 1] + nums[i] >= nums[i - 2]):
            return nums[i] + nums[i - 1] + nums[i - 2]

    return 0


# https://leetcode.com/problems/find-resultant-array-after-removing-anagrams/description/?envType=problem-list-v2&envId=array

def get_anagram(word: str) -> str:
    return ''.join(sorted(word))


def removeAnagrams(self, words: List[str]) -> List[str]:
    words_len: int = len(words)
    result: List[str] = [words[0]]

    for i in range(1, words_len):
        word: str = words[i]
        prev_word: str = words[i - 1]
        if get_anagram(word) == get_anagram(prev_word):
            continue
        else:
            result.append(word)

    return result


# https://leetcode.com/problems/maximize-sum-of-array-after-k-negations/description/?envType=problem-list-v2&envId=array

def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
    nums.sort()

    largest_sum: int = 0
    min_absolute: int = 10 ** 5
    for num in nums:
        if k > 0 >= num:
            k -= 1
            num = -num
        largest_sum += num
        min_absolute = min(min_absolute, num)

    k = - 1
    if k > 0 and k % 2 != 0:
        largest_sum -= min_absolute

    return largest_sum


# https://leetcode.com/problems/calculate-amount-paid-in-taxes/?envType=problem-list-v2&envId=array

def calculateTax(self, brackets: List[List[int]], income: int) -> float:
    total_tax: int = 0
    already_taxed_income: int = 0
    previous_bracket: int = 0

    for bracket in brackets:
        upperbound: int = bracket[0]
        percent: int = bracket[1]
        actual_bracket: int = upperbound - previous_bracket

        bracket_income: int = min(income - already_taxed_income, actual_bracket)
        total_tax += bracket_income * percent / 100

        already_taxed_income += bracket_income
        previous_bracket = upperbound

    return total_tax


# https://leetcode.com/problems/string-matching-in-an-array/?envType=problem-list-v2&envId=array

def get_lps_array(pattern: str) -> List[int]:
    lps: List[int] = [0] * len(pattern)
    current_index: int = 1
    length: int = 0

    while current_index < len(pattern):
        if pattern[current_index] == pattern[length]:
            length += 1
            lps[current_index] = length
            current_index += 1
        else:
            if length > 0:
                lps[current_index] = length - 1
            else:
                current_index += 1

    return lps


def kmp_search(text: str, pattern: str, lps: List[int]) -> bool:
    text_index: int = 0
    pattern_index: int = 0

    while text_index < len(text):
        if text[text_index] == pattern[pattern_index]:
            text_index += 1
            pattern_index += 1
            if text_index == len(text):
                return True
        else:
            if pattern_index > 0:
                pattern_index = lps[pattern_index - 1]
            else:
                text_index += 1

    return False


def stringMatching(self, words: List[str]) -> List[str]:
    matched_words: List[str] = list()
    n: int = len(words)

    for i in range(n):

        lps: List[int] = get_lps_array(words[i])
        for j in range(n):
            if i == j:
                continue
            else:
                if kmp_search(text=words[j], pattern=words[i], lps=lps):
                    matched_words.append(words[i])
                    break

    return matched_words


# https://leetcode.com/problems/find-nearest-point-that-has-the-same-x-or-y-coordinate/description/?envType=problem-list-v2&envId=array

def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
    min_manhattan_distance: int = 10 ** 5
    min_index: int = len(points)

    for i in range(len(points)):
        point: List[int] = points[i]
        if not (point[0] == x or point[1] == y):
            continue
        manhattan_distance: int = abs(x - point[0]) + abs(y - point[1])
        if manhattan_distance < min_manhattan_distance:
            min_manhattan_distance = manhattan_distance
            min_index = i

    return -1 if min_index == len(points) else min_index


# https://leetcode.com/problems/maximum-ascending-subarray-sum/description/?envType=problem-list-v2&envId=array

def maxAscendingSum(self, nums: List[int]) -> int:
    max_ascending_sum: int = 0
    running_sum: int = 0
    prev: int = 0

    for num in nums:
        if num >= prev:
            running_sum += num
            prev = num
        else:
            running_sum = num

        max_ascending_sum = max(max_ascending_sum, running_sum)

    return max_ascending_sum


# https://leetcode.com/problems/sign-of-the-product-of-an-array/description/?envType=problem-list-v2&envId=array

def arraySign(self, nums: List[int]) -> int:
    product: int = 1
    for num in nums:
        product *= num

    if product > 0:
        return 1
    elif product < 0:
        return -1
    else:
        return 0


# https://leetcode.com/problems/minimum-distance-to-the-target-element/description/?envType=problem-list-v2&envId=array

def getMinDistance(self, nums: List[int], target: int, start: int) -> int:
    i: int = start
    j: int = start - 1
    n: int = len(nums)
    while i < n or j >= 0:
        if i < n and nums[i] == target:
            return abs(i - start)
        elif i < n:
            i += 1
        if j >= 0 and nums[j] == target:
            return abs(j - start)
        elif j >= 0:
            j -= 1

    return -1


# https://leetcode.com/problems/check-if-all-the-integers-in-a-range-are-covered/description/?envType=problem-list-v2&envId=array

def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
    range_arr: List[int] = [0] * 51

    for range_v in ranges:
        left_r: int = range_v[0]
        right_r: int = range_v[1]
        range_arr[left_r] += 1
        range_arr[right_r] -= 1

    running_sum: int = 0
    for i in range(len(range_arr)):
        running_sum += range_arr[i]
        if left <= i <= right and running_sum <= 0:
            return False

    return True


# https://leetcode.com/problems/determine-whether-matrix-can-be-obtained-by-rotation/?envType=problem-list-v2&envId=array

def rotate_mat(mat: List[List[int]]) -> List[List[int]]:
    m: int = len(mat)
    n: int = len(mat[0])
    new_mat: List[List[int]] = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            new_mat[j][n - i - 1] = mat[i][j]
    return new_mat


def is_same(mat: List[List[int]], target: List[List[int]]) -> bool:
    if len(mat) != len(target) or len(mat[0]) != len(target[0]):
        return False

    m: int = len(mat)
    n: int = len(mat[0])

    for i in range(m):
        for j in range(n):
            if mat[i][j] != target[i][j]:
                return False

    return True


def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
    for i in range(4):
        if is_same(mat, target):
            return True
        mat = rotate_mat(mat)

    return False


# https://leetcode.com/problems/remove-one-element-to-make-the-array-strictly-increasing/description/?envType=problem-list-v2&envId=array

def canBeIncreasing(self, nums: List[int]) -> bool:
    c: int = 0
    prev: int = nums[0]
    for i in range(1, len(nums)):
        if prev < nums[i]:
            prev = nums[i]
            continue
        else:
            c += 1
            if i > 1 and nums[i] <= nums[i - 2]:
                prev = nums[i - 1]
            else:
                prev = nums[i]

    return c < 2


# https://leetcode.com/problems/check-if-string-is-a-prefix-of-array/description/?envType=problem-list-v2&envId=array

def isPrefixString(self, s: str, words: List[str]) -> bool:
    s_index: int = 0
    for word in words:
        for i in range(len(word)):
            if s_index < len(s) and s[s_index] != word[i]:
                return False
            if s_index == len(s) and 0 < i < len(word):
                return False

            s_index += 1

    return s_index >= len(s)


# https://leetcode.com/problems/number-of-strings-that-appear-as-substrings-in-word/?envType=problem-list-v2&envId=array

def numOfStrings(self, patterns: List[str], word: str) -> int:
    count: int = 0
    for p in patterns:
        if p in word:
            count += 1

    return count


# https://leetcode.com/problems/find-the-middle-index-in-array/description/?envType=problem-list-v2&envId=array

def findMiddleIndex(self, nums: List[int]) -> int:
    n: int = len(nums)
    post_sum_arr: List[int] = [0] * n
    running_sum: int = 0
    for i in range(n - 1, -1, -1):
        post_sum_arr[i] = running_sum
        running_sum += nums[i]

    running_sum = 0
    for i in range(n):
        if running_sum == post_sum_arr[i]:
            return i
        running_sum += nums[i]

    return -1


# https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/description/?envType=problem-list-v2&envId=array

def minimumDifference(self, nums: List[int], k: int) -> int:
    nums.sort()
    min_diff: int = 2 ** 100
    for i in range(k - 1, len(nums)):
        min_no: int = nums[i - (k - 1)]
        max_no: int = nums[i]
        min_diff = min(min_diff, max_no - min_no)

    return min_diff


# https://leetcode.com/problems/count-special-quadruplets/description/?envType=problem-list-v2&envId=array

def countQuadruplets(self, nums: List[int]) -> int:
    n: int = len(nums)
    freq_map: Dict[int, int] = dict()

    result: int = 0
    for i in range(n - 1, 1, -1):
        for j in range(i - 1, 0, -1):
            for k in range(j - 1, -1, -1):
                curr_sum: int = nums[i] + nums[j] + nums[k]
                result += freq_map.get(curr_sum, 0)

        freq_map[nums[i]] = freq_map.get(nums[i], 0) + 1

    return result


# https://leetcode.com/problems/maximum-difference-between-increasing-elements/description/?envType=problem-list-v2&envId=array

def maximumDifference(self, nums: List[int]) -> int:
    n: int = len(nums)

    result: int = -1
    min_element: int = nums[0]

    for i in range(1, n):
        if nums[i] > min_element:
            result = max(result, nums[i] - min_element)
        else:
            min_element = nums[i]

    return result


# https://leetcode.com/problems/two-furthest-houses-with-different-colors/?envType=problem-list-v2&envId=array

def maxDistance(self, colors: List[int]) -> int:
    n: int = len(colors)
    i: int = 0
    j: int = n - 1

    while i < n:
        if colors[i] == colors[n - 1]:
            i += 1
        else:
            return n - 1 - i
        if colors[j] == colors[0]:
            j -= 1
        else:
            return j

    return -1


# https://leetcode.com/problems/find-subsequence-of-length-k-with-the-largest-sum/description/?envType=problem-list-v2&envId=array

def maxSubsequence(self, nums: List[int], k: int) -> List[int]:
    n: int = len(nums)
    min_heap = []

    for i in range(n):
        heap_element: Tuple[int, int] = (i, nums[i])
        heapq.heappush(min_heap, (heap_element[1], heap_element))
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    indices: List[int] = list()
    while len(min_heap) > 0:
        key, element = heapq.heappop(min_heap)
        indices.append(element[0])

    indices.sort()
    result: List[int] = list()
    for index in indices:
        result.append(nums[index])

    return result


# https://leetcode.com/problems/finding-3-digit-even-numbers/description/?envType=problem-list-v2&envId=array

def findEvenNumbers(self, digits: List[int]) -> List[int]:
    counter_map: Dict[int, int] = dict()
    for d in digits:
        counter_map[d] = counter_map.get(d, 0) + 1

    result: List[int] = list()

    for i in range(100, 999, 2):
        num: int = i
        ones: int = num % 10
        num //= 10
        tens: int = num % 10
        num //= 10
        hundredth: int = num % 10

        print(i, hundredth, tens, ones)

        required_counter: Dict[int, int] = dict()
        required_counter[hundredth] = required_counter.get(hundredth, 0) + 1
        required_counter[tens] = required_counter.get(tens, 0) + 1
        required_counter[ones] = required_counter.get(ones, 0) + 1

        print(counter_map, required_counter)

        if required_counter[hundredth] <= counter_map.get(hundredth, 0) and required_counter[tens] <= counter_map.get(
                tens, 0) and required_counter[ones] <= counter_map.get(ones, 0):
            result.append(i)

    return result


# https://leetcode.com/problems/minimum-cost-of-buying-candies-with-discount/description/?envType=problem-list-v2&envId=array

def minimumCost(self, cost: List[int]) -> int:
    n: int = len(cost)
    cost.sort(reverse=True)

    total_cost: int = 0
    for i in range(0, n, 3):
        first: int = cost[i]
        second: int = cost[i + 1] if i < n - 1 else 0
        total_cost += first + second

    return total_cost


# https://leetcode.com/problems/check-if-every-row-and-column-contains-all-numbers/description/?envType=problem-list-v2&envId=array

def checkValid(self, matrix: List[List[int]]) -> bool:
    n: int = len(matrix)

    for i in range(n):
        row_set: Set[int] = set()
        col_set: Set[int] = set()
        for j in range(n):
            value_row: int = matrix[i][j]
            value_col: int = matrix[j][i]
            if (not 1 <= value_row <= n) or (not 1 <= value_col <= n):
                return False
            if value_row in row_set:
                return False
            row_set.add(value_row)

            if value_col in col_set:
                return False
            col_set.add(value_col)

    return True


# https://leetcode.com/problems/count-elements-with-strictly-smaller-and-greater-elements/?envType=problem-list-v2&envId=array

def countElements(self, nums: List[int]) -> int:
    n: int = len(nums)
    if n <= 2:
        return 0

    nums.sort()
    result: int = 0

    for i in range(2, n - 1):
        if nums[0] < nums[i] < nums[n - 1]:
            result += 1

    return result


# https://leetcode.com/problems/sort-even-and-odd-indices-independently/description/?envType=problem-list-v2&envId=array

def sortEvenOdd(self, nums: List[int]) -> List[int]:
    n: int = len(nums)
    # odd_indices_values: List[int] = list()
    # even_indices_value: List[int] = list()
    #
    # for i in range(n):
    #     if i % 2 == 0:
    #         even_indices_value.append(nums[i])
    #     else:
    #         odd_indices_values.append(nums[i])
    #
    # odd_indices_values.sort(reverse=True)
    # even_indices_value.sort()
    #
    # j: int = 0
    # k: int = 0
    # for i in range(n):
    #     if i % 2 == 0:
    #         nums[i] = even_indices_value[j]
    #         j += 1
    #     else:
    #         nums[i] = odd_indices_values[k]
    #         k += 1
    #
    # return nums

    # O(1) space
    for i in range(0, n, 2):
        for j in range(0, n - 2 - i, 2):
            if nums[j] > nums[j + 2]:
                temp: int = nums[j]
                nums[j] = nums[j + 2]
                nums[j + 2] = temp

    for i in range(1, n, 2):
        for j in range(1, n - 1 - i, 2):
            if nums[j] < nums[j + 2]:
                temp: int = nums[j]
                nums[j] = nums[j + 2]
                nums[j + 2] = temp

    return nums


# https://leetcode.com/problems/count-hills-and-valleys-in-an-array/?envType=problem-list-v2&envId=array

def countHillValley(self, nums: List[int]) -> int:
    n: int = len(nums)

    count: int = 0
    prev_no: int = nums[0]
    for i in range(1, n - 1):
        curr_no: int = nums[i]
        next_no: int = nums[i + 1]
        if (curr_no > prev_no and curr_no > next_no) or (curr_no < prev_no and curr_no < next_no):
            count += 1
        if curr_no != next_no:
            prev_no = curr_no

    return count


# https://leetcode.com/problems/intersection-of-multiple-arrays/description/?envType=problem-list-v2&envId=array

def intersection(self, nums: List[List[int]]) -> List[int]:
    n: int = 1001
    intersection_arr: List[int] = [0] * n
    m: int = len(nums)

    for num_arr in nums:
        for num in num_arr:
            intersection_arr[num] += 1

    result: List[int] = list()
    for i in range(n):
        if intersection_arr[i] == m:
            result.append(i)

    return result


# https://leetcode.com/problems/min-max-game/description/?envType=problem-list-v2&envId=array

def minMaxGame(self, nums: List[int]) -> int:
    n: int = len(nums)
    while n > 1:
        newNums: List[int] = [0] * (n // 2)
        for i in range(n // 2):
            if i % 2 == 0:
                newNums[i] = min(nums[2 * i], nums[2 * i + 1])
            else:
                newNums[i] = max(nums[2 * i], nums[2 * i + 1])

        nums = newNums
        n = len(nums)

    return nums[0]


# https://leetcode.com/problems/check-if-matrix-is-x-matrix/description/?envType=problem-list-v2&envId=array

def checkXMatrix(self, grid: List[List[int]]) -> bool:
    n: int = len(grid)
    d1_count: int = 0
    d2_count: int = 0

    for i in range(n):
        for j in range(n):
            d1_condition: bool = i == j and grid[i][j] != 0
            d2_condition: bool = i + j == n and grid[i][j] != 0
            if d1_condition:
                d1_count += 1
            if d2_condition:
                d2_count += 1
            if not (d1_condition or d2_condition) and grid[i][j] == 0:
                return False

    return d1_count == n and d2_count == n


# https://leetcode.com/problems/minimum-amount-of-time-to-fill-cups/description/?envType=problem-list-v2&envId=array

def fillCups(self, amount: List[int]) -> int:
    heap = []
    for num in amount:
        if num > 0:
            heapq.heappush(heap, - num)

    seconds: int = 0
    while len(heap) > 1:
        max1: int = - heapq.heappop(heap)
        max2: int = - heapq.heappop(heap)
        max1 -= 1
        max2 -= 1
        if max1 > 0:
            heapq.heappush(heap, - max1)
        if max2 > 0:
            heapq.heappush(heap, - max2)

        seconds += 1

    while len(heap) > 0:
        max1: int = - heapq.heappop(heap)
        max1 -= 1
        if max1 > 0:
            heapq.heappush(heap, - max1)

        seconds += 1

    return seconds


# http://leetcode.com/problems/best-poker-hand/description/?envType=problem-list-v2&envId=array

def bestHand(self, ranks: List[int], suits: List[str]) -> str:
    if len(set(suits)) == 1:
        return "Flush"
    rank_dict: Dict[int, int] = dict()
    for rank in ranks:
        rank_dict[rank] = rank_dict.get(rank, 0) + 1

    rank_set: Set[int] = set(rank_dict.values())
    if 3 in rank_set or 4 in rank_set or 5 in rank_set:
        return "Three of a Kind"
    elif 2 in rank_set:
        return "Pair"
    else:
        return "High Card"


# https://leetcode.com/problems/minimum-hours-of-training-to-win-a-competition/?envType=problem-list-v2&envId=array

def minNumberOfHours(self, initialEnergy: int, initialExperience: int, energy: List[int], experience: List[int]) -> int:
    hours: int = 0
    n: int = len(energy)

    running_energy: int = initialEnergy
    running_experience: int = initialExperience

    for i in range(n):
        net_energy: int = running_energy - energy[i]
        if net_energy <= 0:
            running_energy += (net_energy * -1) + 1
            hours += (net_energy * -1) + 1
        running_energy -= energy[i]

        net_experience: int = running_experience - experience[i]
        if net_experience <= 0:
            running_experience += (net_experience * -1) + 1
            hours += (net_experience * -1) + 1
        running_experience += experience[i]

    return hours


# https://leetcode.com/problems/find-subarrays-with-equal-sum/description/?envType=problem-list-v2&envId=array

def findSubarrays(self, nums: List[int]) -> bool:
    subarrays: Set[int] = set()
    n: int = len(nums)

    for i in range(0, n - 1):
        subarr_sum: int = nums[i] + nums[i + 1]
        if subarr_sum in subarrays:
            return True
        subarrays.add(subarr_sum)

    return False


# https://leetcode.com/problems/most-frequent-even-element/?envType=problem-list-v2&envId=array

def mostFrequentEven(self, nums: List[int]) -> int:
    freq_dict: Dict[int, int] = dict()
    for num in nums:
        if num % 2 == 0:
            freq_dict[num] = freq_dict.get(num, 0) + 1

    max_freq: int = 0
    for k, v in freq_dict.items():
        if v > max_freq:
            max_freq = v

    if max_freq == 0:
        return -1
    max_freq_no: int = 10 ** 5
    for k, v in freq_dict.items():
        if v == max_freq:
            max_freq_no = min(max_freq_no, k)

    return max_freq_no


# https://leetcode.com/problems/the-employee-that-worked-on-the-longest-task/description/?envType=problem-list-v2&envId=array

def hardestWorker(self, n: int, logs: List[List[int]]) -> int:
    employees: List[int] = [0] * n
    running_time: int = 0

    for log in logs:
        employee: int = log[0]
        time_taken: int = log[1] - running_time
        employees[employee] += time_taken
        running_time = log[1]

    max_time: int = 0
    for i in range(n):
        if employees[i] > max_time:
            max_time = employees[i]

    for i in range(n):
        if employees[i] == max_time:
            return i

    return -1


# https://leetcode.com/problems/average-value-of-even-numbers-that-are-divisible-by-three/description/?envType=problem-list-v2&envId=array

def averageValue(self, nums: List[int]) -> int:
    divisible_by_three_sum: int = 0
    n: int = 0

    for num in nums:
        if (num + divisible_by_three_sum) % 6 == 0:
            divisible_by_three_sum += num
            n += 1

    return 0 if n == 0 else divisible_by_three_sum // n


# https://leetcode.com/problems/odd-string-difference/?envType=problem-list-v2&envId=array

def oddString(self, words: List[str]) -> str:
    n: int = len(words[0])
    difference_dict: Dict[str, List[str]] = dict()

    for word in words:
        diff_product: List[str] = list()
        for i in range(1, n):
            diff_product.append(str(ord(word[i]) - ord(word[i - 1])))

        print(diff_product)
        key: str = "-".join(diff_product)
        diff_list: List[str] = difference_dict.get(key, [])
        diff_list.append(word)
        difference_dict[key] = diff_list

    for k, v in difference_dict.items():
        if len(v) == 1:
            return v[0]

    return ""


# https://leetcode.com/problems/apply-operations-to-an-array/?envType=problem-list-v2&envId=array

def applyOperations(self, nums: List[int]) -> List[int]:
    n: int = len(nums)
    num_zeros: int = 0

    for i in range(0, n - 1):
        if nums[i] == nums[i + 1]:
            nums[i] *= 2
            nums[i + 1] = 0
            num_zeros += 1

    shift_index: int = 0
    for i in range(n):
        if nums[i] != 0:
            nums[shift_index] = nums[i]
            shift_index += 1

    while shift_index < n:
        nums[shift_index] = 0
        shift_index += 1

    return nums


# https://leetcode.com/problems/number-of-distinct-averages/description/?envType=problem-list-v2&envId=array

def distinctAverages(self, nums: List[int]) -> int:
    nums.sort()
    n: int = len(nums)
    i: int = 0
    j: int = n - 1
    avg_set: Set[float] = set()
    while i < j:
        avg: float = (nums[i] + nums[j]) / 2
        avg_set.add(avg)
        i += 1
        j -= 1

    return len(avg_set)


# https://leetcode.com/problems/shortest-distance-to-target-string-in-a-circular-array/?envType=problem-list-v2&envId=array

def closetTarget(self, words: List[str], target: str, startIndex: int) -> int:
    n: int = len(words)

    right_move: int = -1
    left_move: int = -1

    for i in range(0, n):
        circular_index: int = (i + startIndex) % n
        if words[circular_index] == target:
            right_move = i
            break

    for i in range(0, -n, -1):
        circular_index: int = (i + startIndex + n) % n
        if words[circular_index] == target:
            left_move = -1 * i
            break

    return min(right_move, left_move)


# https://leetcode.com/problems/maximum-enemy-forts-that-can-be-captured/description/?envType=problem-list-v2&envId=array

def captureForts(self, forts: List[int]) -> int:
    n: int = len(forts)
    max_zeros: int = 0

    running_count: int = 0
    previous_non_zero: int = -1
    for i in range(n):
        if forts[i] != 0:
            if previous_non_zero >= 0 and forts[previous_non_zero] * forts[i] == -1:
                max_zeros = max(max_zeros, i - previous_non_zero - 1)
            previous_non_zero = i

    return max_zeros


# https://leetcode.com/problems/minimum-common-value/description/?envType=problem-list-v2&envId=array

def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
    i: int = 0
    j: int = 0

    m: int = len(nums1)
    n: int = len(nums2)

    while i < m and j < n:
        if nums1[i] == nums2[j]:
            return nums1[i]
        elif nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1

    return -1


# https://leetcode.com/problems/form-smallest-number-from-two-digit-arrays/?envType=problem-list-v2&envId=array

def minNumber(self, nums1: List[int], nums2: List[int]) -> int:
    digits_arr: List[int] = [0] * 10
    for num in nums1:
        digits_arr[num] += 1
    for num in nums2:
        digits_arr[num] += 1

    for i in range(10):
        if digits_arr[i] > 1:
            return i

    min_1: int = min(nums1)
    min_2: int = min(nums2)
    return 10 * (min(min_1, min_2)) + max(min_1, min_2)


# https://leetcode.com/problems/find-the-width-of-columns-of-a-grid/description/?envType=problem-list-v2&envId=array

def get_int_len(a: int) -> int:
    int_len: int = 0 if a > 0 else 1
    a = abs(a)

    while a > 0:
        a //= 10
        int_len += 1

    return int_len


def findColumnWidth(self, grid: List[List[int]]) -> List[int]:
    m: int = len(grid)
    n: int = len(grid[0])
    ans: List[int] = [0] * n

    for j in range(n):
        for i in range(m):
            int_len: int = get_int_len(grid[i][j])
            ans[j] = max(ans[j], int_len)

    return ans


# https://leetcode.com/problems/count-distinct-numbers-on-board/description/?envType=problem-list-v2&envId=array

def distinctIntegers(self, n: int) -> int:
    return n - 1 if n > 1 else 1


# https://leetcode.com/problems/determine-the-winner-of-a-bowling-game/description/?envType=problem-list-v2&envId=array

def isWinner(self, player1: List[int], player2: List[int]) -> int:
    n: int = len(player1)
    p1_score: int = 0
    p2_score: int = 0

    for i in range(n):
        if (i > 0 and player1[i - 1] == 10) or (i > 1 and player1[i - 2]) == 10:
            p1_score += player1[i] * 2
        else:
            p1_score += player1[i]

    for i in range(n):
        if (i > 0 and player2[i - 1] == 10) or (i > 1 and player2[i - 2]) == 10:
            p2_score += player2[i] * 2
        else:
            p2_score += player2[i]

    if p1_score > p2_score:
        return 1
    elif p1_score < p2_score:
        return 2
    else:
        return 0


# https://leetcode.com/problems/take-gifts-from-the-richest-pile/description/?envType=problem-list-v2&envId=array

def pickGifts(self, gifts: List[int], k: int) -> int:
    max_heap = []

    for gift in gifts:
        heapq.heappush(max_heap, - gift)

    while k > 0:
        max_gift: int = - heapq.heappop(max_heap)
        remaining_gift: int = - math.floor(math.sqrt(max_gift))
        heapq.heappush(max_heap, remaining_gift)
        k -= 1

    total_gift: int = 0
    for gift in max_heap:
        total_gift += abs(gift)

    return total_gift
