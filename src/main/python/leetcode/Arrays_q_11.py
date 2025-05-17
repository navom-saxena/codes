import heapq
import math
import sys
from typing import List, Set, Dict, Tuple


# https://leetcode.com/problems/find-the-maximum-divisibility-score/description/?envType=problem-list-v2&envId=array

def maxDivScore(self, nums: List[int], divisors: List[int]) -> int:
    nums: List[int] = sorted(nums)
    divisors: List[int] = sorted(divisors)

    n_nums: int = len(nums)
    n_divisors: int = len(divisors)
    result: int = -1
    max_count: int = -1

    for i in range(n_divisors - 1, -1, -1):
        divisor: int = divisors[i]
        count: int = 0
        for j in range(n_nums - 1, -1, -1):
            num: int = nums[j]
            if num < divisor:
                break
            if num % divisor == 0:
                count += 1
        if count >= max_count:
            max_count = count
            result = divisor

    return result


# https://leetcode.com/problems/find-the-array-concatenation-value/description/?envType=problem-list-v2&envId=array

def findTheArrayConcVal(self, nums: List[int]) -> int:
    n: int = len(nums)
    i: int = 0
    j: int = n - 1
    result: int = 0
    while i <= j:
        if i == j:
            result += nums[i]
        else:
            result += int(str(nums[i]) + str(nums[j]))
        i += 1
        j -= 1

    return result


# https://leetcode.com/problems/prime-in-diagonal/?envType=problem-list-v2&envId=array

def sieveOfEratosthenes(n: int) -> Set[int]:
    n += 1
    primes: Set[int] = set()
    isPrime: List[bool] = [True] * n

    isPrime[0] = isPrime[1] = False
    for p in range(2, int(math.sqrt(n)) + 1):
        if isPrime[p]:
            for p_mul in range(p * p, n, p):
                isPrime[p_mul] = False

    for p in range(n):
        if isPrime[p]:
            primes.add(p)

    return primes


def diagonalPrime(self, nums: List[List[int]]) -> int:
    n: int = len(nums)
    max_no: int = -1
    diagonals: Set[int] = set()

    for i in range(n):
        for j in range(n):
            if i == j or i + j == n:
                num: int = nums[i][j]
                max_no = max(max_no, num)
                diagonals.add(num)

    primes: Set[int] = sieveOfEratosthenes(max_no)
    primesAvailable: Set[int] = primes.intersection(diagonals)

    max_no = -1
    for p in primesAvailable:
        max_no = max(max_no, p)

    return max_no


# https://leetcode.com/problems/buy-two-chocolates/description/?envType=problem-list-v2&envId=array

def buyChoco(self, prices: List[int], money: int) -> int:
    min_price: int = 101
    min_price_2: int = 101

    for price in prices:
        if price <= min_price:
            min_price_2 = min_price
            min_price = price
        elif price < min_price_2:
            min_price_2 = price

    if min_price == 101 or min_price_2 == 101:
        return money

    remaining: int = money - min_price - min_price_2
    if remaining >= 0:
        return remaining
    else:
        return money


# https://leetcode.com/problems/semi-ordered-permutation/description/?envType=problem-list-v2&envId=array

def semiOrderedPermutation(self, nums: List[int]) -> int:
    n: int = len(nums)
    one_index: int = -1
    n_index: int = -1
    one_found_after: bool = False

    for i in range(n):
        if nums[i] == 1:
            one_index = i
        elif nums[i] == n:
            if one_index == -1:
                one_found_after = True
            n_index = i

    swaps: int = one_index + (n - 1 - n_index) - 1 if one_found_after else one_index + (n - 1 - n_index)
    return swaps


# https://leetcode.com/problems/find-the-losers-of-the-circular-game/description/?envType=problem-list-v2&envId=array

def circularGameLosers(self, n: int, k: int) -> List[int]:
    nums: List[int] = [0] * n

    curr: int = 0
    turns: int = 1
    while nums[curr] == 0:
        nums[curr] += 1
        curr = (curr + (k * turns)) % n
        turns += 1

    remaining: List[int] = list()
    for i in range(n):
        if i > 0 and nums[i] == 0:
            remaining.append(i + 1)

    return remaining


# https://leetcode.com/problems/number-of-beautiful-pairs/description/?envType=problem-list-v2&envId=array

def findFirstDigit(num: int) -> int:
    prev: int = 0
    while num > 0:
        prev = num % 10
        num //= 10

    return prev


def findGcd(num1: int, num2: int) -> int:
    bigger: int = num1 if num1 >= num2 else num2
    smaller: int = num2 if num1 > num2 else num1
    print(bigger, smaller)
    if bigger == 1 or smaller == 1:
        return 1

    remainder: int = bigger % smaller
    print(bigger, smaller, remainder)
    if remainder == 0:
        return smaller
    else:
        return findGcd(smaller, remainder)


def countBeautifulPairs(self, nums: List[int]) -> int:
    n: int = len(nums)
    pairs_count: int = 0
    firstDigitMap: Dict[int, int] = dict()

    for i in range(n):
        firstDigit: int = findFirstDigit(nums[i])
        lastDigit: int = nums[i] % 10

        for firstNoFromMap, freq in firstDigitMap.items():
            if findGcd(firstNoFromMap, lastDigit) == 1:
                pairs_count += freq

        firstDigitMap[firstDigit] = firstDigitMap.get(firstDigit, 0) + 1

    return pairs_count


# https://leetcode.com/problems/longest-even-odd-subarray-with-threshold/description/?envType=problem-list-v2&envId=array

def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
    n: int = len(nums)
    maxLength: int = 0
    runningLen: int = 0

    for i in range(n):
        if nums[i] > threshold:
            runningLen = 0
        elif runningLen == 0 and nums[i] % 2 == 0:
            runningLen = 1
        elif runningLen >= 1 and nums[i - 1] % 2 != nums[i] % 2:
            runningLen += 1
        else:
            runningLen = 1 if nums[i] % 2 == 0 else 0
        maxLength = max(maxLength, runningLen)

    return maxLength


# https://leetcode.com/problems/longest-alternating-subarray/description/?envType=problem-list-v2&envId=array

def alternatingSubarray(self, nums: List[int]) -> int:
    n: int = len(nums)
    maxLength: int = -1
    runningLen: int = -1

    for i in range(1, n):
        if runningLen > 0 and nums[i] == nums[i - 2]:
            runningLen += 1
        else:
            runningLen = 2 if nums[i] - nums[i - 1] == 1 else -1
        maxLength = max(maxLength, runningLen)

    return maxLength


# https://leetcode.com/problems/minimum-operations-to-collect-elements/description/?envType=problem-list-v2&envId=array

def minOperations(self, nums: List[int], k: int) -> int:
    n: int = len(nums)
    numSet: Set[int] = set()
    count: int = 0

    for i in range(n - 1, -1, -1):
        if nums[i] <= k:
            numSet.add(nums[i])
        if len(numSet) == k:
            return count
        count += 1

    return -1


# https://leetcode.com/problems/max-pair-sum-in-an-array/description/?envType=problem-list-v2&envId=array

def findMaxDigit(num: int) -> int:
    prev: int = 0
    maxDigit: int = 0
    while num > 0:
        prev = num % 10
        maxDigit = max(maxDigit, prev)
        num //= 10

    return maxDigit


def maxSum(self, nums: List[int]) -> int:
    maxDigitsMap: Dict[int, List[int]] = dict()

    for num in nums:
        maxDigit: int = findMaxDigit(num)
        maxDigitList: List[int] = maxDigitsMap.get(maxDigit, [])
        maxDigitList.append(num)
        maxDigitsMap[maxDigit] = maxDigitList

    maxSumValue: int = -1
    for maxDigit, maxDigitList in maxDigitsMap.items():
        maxDigitList.sort(reverse=True)
        if len(maxDigitList) >= 2:
            maxSumValue = max(maxSumValue, maxDigitList[0] + maxDigitList[1])

    return maxSumValue


# https://leetcode.com/problems/minimum-right-shifts-to-sort-the-array/description/?envType=problem-list-v2&envId=array

def minimumRightShifts(self, nums: List[int]) -> int:
    n: int = len(nums)
    swapIndex: int = -1

    for i in range(1, n):
        if nums[i - 1] > nums[i]:
            if swapIndex != -1:
                return -1
            swapIndex = i

    if swapIndex == -1 and nums[0] <= nums[n - 1]:
        return 0
    elif swapIndex != -1 and nums[0] > nums[n - 1]:
        return -1
    return n - swapIndex


# https://leetcode.com/problems/check-if-array-is-good/?envType=problem-list-v2&envId=array

def isGood(self, nums: List[int]) -> bool:
    n: int = len(nums)
    freqArr: List[int] = [0] * n

    for i in range(n):
        num: int = nums[i]
        if 0 >= num or num > n - 1:
            return False
        freqArr[num] += 1

    for i in range(n):
        if i != n - 1 and freqArr[i] > 1:
            return False
        elif i == n - 1 and freqArr[i] != 2:
            return False

    return True


# https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-i/description/?envType=problem-list-v2&envId=array

def getLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
    n: int = len(groups)
    result: List[str] = [words[0]]

    prev: int = groups[0]
    for i in range(1, n):
        if groups[i] != prev:
            prev = groups[i]
            result.append(words[i])

    return result


# https://leetcode.com/problems/maximum-value-of-an-ordered-triplet-i/description/?envType=problem-list-v2&envId=array

def maximumTripletValue(self, nums: List[int]) -> int:
    result: int = 0
    n: int = len(nums)
    maxDiff: int = 0
    maxPrefix: int = 0
    for i in range(n):
        result = max(result, maxDiff * nums[i])
        maxPrefix = max(maxPrefix, nums[i])
        maxDiff = max(maxDiff, maxPrefix - nums[i])

    return result


# https://leetcode.com/problems/last-visited-integers/description/?envType=problem-list-v2&envId=array

def lastVisitedIntegers(self, nums: List[int]) -> List[int]:
    n: int = len(nums)
    seen: List[int] = list()
    ans: List[int] = list()

    k: int = 0
    for i in range(n):
        value: int = nums[i]
        if value != -1:
            seen.append(nums[i])
            k = 0
        else:
            k += 1
            if k > len(seen):
                ans.append(value)
            else:
                ans.append(seen[len(seen) - k])

    return ans


# https://leetcode.com/problems/find-indices-with-index-and-value-difference-i/description/?envType=problem-list-v2&envId=array

def findIndices(self, nums: List[int], indexDifference: int, valueDifference: int) -> List[int]:
    n: int = len(nums)
    result: List[int] = [-1, -1]
    prefixMinIndex: int = 0
    prefixMaxIndex: int = 0
    prefixMin: int = 10000
    prefixMax: int = 0

    for i in range(indexDifference, n):
        if prefixMin > nums[i - indexDifference]:
            prefixMin = nums[i - indexDifference]
            prefixMinIndex = i - indexDifference
        if prefixMax < nums[i - indexDifference]:
            prefixMax = nums[i - indexDifference]
            prefixMaxIndex = i - indexDifference

        if abs(nums[i] - prefixMax) >= valueDifference:
            result[0] = prefixMaxIndex
            result[1] = i
        if abs(nums[i] - prefixMin) >= valueDifference:
            result[0] = prefixMinIndex
            result[1] = i

    return result


# https://leetcode.com/problems/matrix-similarity-after-cyclic-shifts/description/?envType=problem-list-v2&envId=array

def areSimilar(self, mat: List[List[int]], k: int) -> bool:
    m: int = len(mat)
    n: int = len(mat[0])
    k %= n

    for i in range(m):
        for j in range(n):
            if mat[i][(j + k) % n] == mat[i][j]:
                continue
            else:
                return False

    return True


# https://leetcode.com/problems/smallest-missing-integer-greater-than-sequential-prefix-sum/description/?envType=problem-list-v2&envId=array

def missingInteger(self, nums: List[int]) -> int:
    n: int = len(nums)
    prefixSum: int = nums[0]
    for i in range(1, n):
        if nums[i] == nums[i - 1] + 1:
            prefixSum += nums[i]
        else:
            break

    numsSet: Set[int] = set(nums)

    while prefixSum in numsSet:
        prefixSum += 1

    return prefixSum


# https://leetcode.com/problems/maximum-area-of-longest-diagonal-rectangle/description/?envType=problem-list-v2&envId=array

def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
    maxDiagonal: float = 0.0
    maxArea: int = 0

    for dimension in dimensions:
        recLength: int = dimension[0]
        recWidth: int = dimension[1]

        diagonal: float = math.sqrt((recLength ** 2) + (recWidth ** 2))
        area: int = recLength * recWidth
        if diagonal > maxDiagonal:
            maxArea = area
            maxDiagonal = diagonal
        elif diagonal == maxDiagonal and area > maxArea:
            maxArea = area

    return maxArea


# https://leetcode.com/problems/count-the-number-of-incremovable-subarrays-i/description/?envType=problem-list-v2&envId=array

def incremovableSubarrayCount(self, nums: List[int]) -> int:
    n: int = len(nums)
    if n == 1:
        return 1
    if n == 2:
        return 3

    result: int = 1
    left: int = 0
    while left < n - 1 and nums[left] < nums[left + 1]:
        left += 1

    result += left + 1

    right: int = n - 1
    while right > 0 and nums[right] > nums[right - 1]:
        right -= 1

    result += (n - right)

    if left >= right:
        return n * (n + 1) // 2

    l: int = 0
    r = right
    while l <= left and r < n:
        if nums[l] < nums[r]:
            result += (n - r)
            l += 1
        else:
            r += 1

    return result


# https://leetcode.com/problems/divide-an-array-into-subarrays-with-minimum-cost-i/?envType=problem-list-v2&envId=array

def minimumCost(self, nums: List[int]) -> int:
    n: int = len(nums)
    minimumNo: int = sys.maxsize - 1
    secondMinimumNo = sys.maxsize - 1

    for i in range(1, n):
        num = nums[i]
        if num < minimumNo:
            secondMinimumNo = minimumNo
            minimumNo = num
        elif minimumNo <= num < secondMinimumNo:
            secondMinimumNo = num

    return nums[0] + minimumNo + secondMinimumNo


# https://leetcode.com/problems/count-prefix-and-suffix-pairs-i/description/?envType=problem-list-v2&envId=array

def isPrefixAndSuffix(str1: str, str2: str) -> bool:
    m: int = len(str2)
    n: int = len(str1)

    if m < n:
        return False

    i: int = 0
    j: int = 0

    while i < n and j < m:
        if str1[i] != str2[j]:
            return False
        i += 1
        j += 1

    i = n - 1
    j = m - 1

    while i >= 0 and j >= 0:
        if str1[i] != str2[j]:
            return False
        i -= 1
        j -= 1

    return True


def countPrefixSuffixPairs(self, words: List[str]) -> int:
    n: int = len(words)
    result: int = 0

    for i in range(n):
        str1: str = words[i]
        for j in range(i + 1, n):
            str2: str = words[j]
            if isPrefixAndSuffix(str1, str2):
                result += 1

    return result


# https://leetcode.com/problems/modify-the-matrix/description/?envType=problem-list-v2&envId=array

def modifiedMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
    m: int = len(matrix)
    n: int = len(matrix[0])

    columnMaxValue: List[int] = [-1] * n

    for i in range(m):
        for j in range(n):
            columnMaxValue[j] = max(columnMaxValue[j], matrix[i][j])

    newMat: List[List[int]] = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            newMat[i][j] = matrix[i][j]
            if newMat[i][j] == -1:
                newMat[i][j] = columnMaxValue[j]

    return newMat


# https://leetcode.com/problems/split-the-array/?envType=problem-list-v2&envId=array

def isPossibleToSplit(self, nums: List[int]) -> bool:
    freqMap: Dict[int, int] = dict()

    for num in nums:
        freqMap[num] = freqMap.get(num, 0) + 1
        if freqMap[num] > 2:
            return False

    return True


# https://leetcode.com/problems/maximum-number-of-operations-with-the-same-score-i/description/?envType=problem-list-v2&envId=array

def maxOperations(self, nums: List[int]) -> int:
    n: int = len(nums)
    initialScore: int = nums[0] + nums[1]
    count: int = 1

    for i in range(2, n - 1, 2):
        if nums[i] + nums[i + 1] != initialScore:
            return count
        count += 1

    return count


# https://leetcode.com/problems/type-of-triangle/description/?envType=problem-list-v2&envId=array

def triangleType(self, nums: List[int]) -> str:
    if nums[0] + nums[1] <= nums[2] or nums[0] + nums[2] <= nums[1] or nums[1] + nums[2] <= nums[0]:
        return "none"

    numsSet = set(nums)
    n: int = len(numsSet)
    if n == 1:
        return "equilateral"
    elif n == 2:
        return "isosceles"
    else:
        return "scalene"


# https://leetcode.com/problems/apple-redistribution-into-boxes/?envType=problem-list-v2&envId=array

def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
    capacity.sort(reverse=True)
    n: int = len(capacity)
    boxesCount: int = 0
    allApples: int = 0

    for a in apple:
        allApples += a

    for i in range(n):
        allApples -= capacity[i]
        if allApples <= 0:
            return i + 1
    return n


# https://leetcode.com/problems/check-if-grid-satisfies-conditions/description/?envType=problem-list-v2&envId=array

def satisfiesConditions(self, grid: List[List[int]]) -> bool:
    m: int = len(grid)
    n: int = len(grid[0])

    for i in range(m):
        for j in range(n):
            if i < m - 1 and grid[i][j] != grid[i + 1][j]:
                return False

            if j < n - 1 and grid[i][j] == grid[i][j + 1]:
                return False

    return True


# https://leetcode.com/problems/shortest-subarray-with-or-at-least-k-i/description/?envType=problem-list-v2&envId=array

def updateOr(orArr: List[int], num: int) -> None:
    i: int = 0
    while num > 0:
        if num & 1 == 1:
            orArr[i] += 1
        i += 1
        num >>= 1


def removeFromOr(orArr: List[int], num: int) -> None:
    i: int = 0
    while num > 0:
        if num & 1 == 1:
            orArr[i] -= 1
        i += 1
        num >>= 1


def getNumberFromBits(orArr: List[int]) -> int:
    num: int = 0
    for i in range(32):
        if orArr[i] >= 1:
            num += 2 ** i

    return num


def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
    n: int = len(nums)
    minimumLength: int = sys.maxsize
    left: int = 0
    runningOr: List[int] = [0] * 32

    for i in range(n):
        updateOr(runningOr, num=nums[i])
        while getNumberFromBits(runningOr) >= k and left <= i:
            minimumLength = min(minimumLength, i - left + 1)
            removeFromOr(orArr=runningOr, num=nums[left])
            left += 1

    return -1 if minimumLength == sys.maxsize else minimumLength


# https://leetcode.com/problems/longest-strictly-increasing-or-strictly-decreasing-subarray/description/?envType=problem-list-v2&envId=array

def longestMonotonicSubarray(self, nums: List[int]) -> int:
    n: int = len(nums)
    longestIncreasingLen: int = 0
    longestDecreasingLen: int = 0
    runningIncreasingLen: int = 0
    runningDecreasingLen: int = 0

    for i in range(1, n):
        if nums[i - 1] < nums[i]:
            runningIncreasingLen += 1
            longestIncreasingLen = max(longestIncreasingLen, runningIncreasingLen)
        else:
            runningIncreasingLen = 1

        if nums[i - 1] > nums[i]:
            runningDecreasingLen += 1
            longestDecreasingLen = max(longestDecreasingLen, runningDecreasingLen)
        else:
            runningDecreasingLen = 1

    return max(longestIncreasingLen, longestDecreasingLen)


# https://leetcode.com/problems/alternating-groups-i/description/?envType=problem-list-v2&envId=array

def numberOfAlternatingGroups(self, colors: List[int]) -> int:
    n: int = len(colors)
    colors.extend(colors)

    result: int = 0

    for i in range(n):
        if (colors[i] == 0 and colors[i + 1] == 1 and colors[i + 2] == 0) or (
                colors[i] == 1 and colors[i + 1] == 0 and colors[i + 2] == 1):
            result += 1

    return result


# https://leetcode.com/problems/maximum-height-of-a-triangle/description/?envType=problem-list-v2&envId=array

def calculateHeight(red: int, blue: int, isRed: bool, depth: int) -> int:
    if isRed:
        red -= depth
        if red >= 0:
            return calculateHeight(red, blue, isRed=False, depth=depth + 1)
        else:
            return depth - 1
    else:
        blue -= depth
        if blue >= 0:
            return calculateHeight(red, blue, isRed=True, depth=depth + 1)
        else:
            return depth - 1


def maxHeightOfTriangle(self, red: int, blue: int) -> int:
    return max(calculateHeight(red, blue, isRed=True, depth=1), calculateHeight(red, blue, isRed=False, depth=1))


# https://leetcode.com/problems/make-a-square-with-the-same-color/description/?envType=problem-list-v2&envId=array

def canMakeSquare(self, grid: List[List[str]]) -> bool:
    for i in range(2):
        for j in range(2):
            b: int = 0
            w: int = 0
            directions: List[List[int]] = [[0, 0], [0, 1], [1, 0], [1, 1]]
            for direction in directions:
                newI: int = i + direction[0]
                newJ: int = j + direction[1]
                if grid[newI][newJ] == 'B':
                    b += 1
                else:
                    w += 1
            if w in [0, 1] or b in [0, 1]:
                return True

    return False


# https://leetcode.com/problems/find-if-digit-game-can-be-won/description/?envType=problem-list-v2&envId=array

def canAliceWin(self, nums: List[int]) -> bool:
    n: int = len(nums)
    singleDigitSum: int = 0
    doubleDigitSum: int = 0

    for i in range(n):
        if nums[i] <= 9:
            singleDigitSum += nums[i]
        else:
            doubleDigitSum += nums[i]

    return singleDigitSum != doubleDigitSum


# https://leetcode.com/problems/design-neighbor-sum-service/description/?envType=problem-list-v2&envId=array

class NeighborSum:

    def __init__(self, grid: List[List[int]]):
        self.adjacentSumMap: Dict[int, int] = dict()
        self.diagonalSumMap: Dict[int, int] = dict()
        n: int = len(grid)
        for i in range(n):
            for j in range(n):
                sumValue: int = 0
                diagonalValue: int = 0
                if i > 0:
                    sumValue += grid[i - 1][j]
                if i < n - 1:
                    sumValue += grid[i + 1][j]
                if j > 0:
                    sumValue += grid[i][j - 1]
                if j < n - 1:
                    sumValue += grid[i][j + 1]
                if i > 0 and j > 0:
                    diagonalValue += grid[i - 1][j - 1]
                if i > 0 and j < n - 1:
                    diagonalValue += grid[i - 1][j + 1]
                if i < n - 1 and j > 0:
                    diagonalValue += grid[i + 1][j - 1]
                if i < n - 1 and j < n - 1:
                    diagonalValue += grid[i + 1][j + 1]

                self.adjacentSumMap[grid[i][j]] = sumValue
                self.diagonalSumMap[grid[i][j]] = diagonalValue

    def adjacentSum(self, value: int) -> int:
        return self.adjacentSumMap[value]

    def diagonalSum(self, value: int) -> int:
        return self.diagonalSumMap[value]


# https://leetcode.com/problems/find-the-number-of-winning-players/?envType=problem-list-v2&envId=array

def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
    playerDict: Dict[int, List[int]] = dict()
    playerSet: Set[int] = set()

    for p in pick:
        player: int = p[0]
        color: int = p[1]
        playerColor: List[int] = playerDict.get(player, [0] * 11)
        playerColor[color] += 1
        playerDict[player] = playerColor
        playerSet.add(player)

    winsCount: int = 0
    for p in playerSet:
        playerColor = playerDict[p]
        for colorF in playerColor:
            if colorF > p:
                winsCount += 1
                break

    return winsCount


# https://leetcode.com/problems/snake-in-matrix/description/?envType=problem-list-v2&envId=array

def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
    i: int = 0
    j: int = 0

    for command in commands:
        if command == 'UP':
            i -= 1
        elif command == 'DOWN':
            i += 1
        elif command == 'LEFT':
            j -= 1
        elif command == 'RIGHT':
            j += 1

    return i * n + j


# https://leetcode.com/problems/the-two-sneaky-numbers-of-digitville/description/?envType=problem-list-v2&envId=array

def getSneakyNumbers(self, nums: List[int]) -> List[int]:
    n: int = len(nums)
    nums.sort()

    duplicateNums: List[int] = list()
    runningValue: int = 0
    for i in range(n):
        if nums[i] != runningValue:
            duplicateNums.append(nums[i])
        else:
            runningValue += 1

    return duplicateNums


# https://leetcode.com/problems/final-array-state-after-k-multiplication-operations-i/description/?envType=problem-list-v2&envId=array

def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
    n: int = len(nums)
    heap = []
    for i in range(n):
        heap.append((nums[i], i))

    heapq.heapify(heap)

    while k > 0:
        minElement: Tuple[int, int] = heapq.heappop(heap)
        minElementNew = (minElement[0] * multiplier, minElement[1])
        heapq.heappush(heap, minElementNew)
        k -= 1

    result: List[int] = [0] * n
    for t in heap:
        index: int = t[1]
        value: int = t[0]
        result[index] = value

    return result


# https://leetcode.com/problems/find-indices-of-stable-mountains/description/?envType=problem-list-v2&envId=array

def stableMountains(self, height: List[int], threshold: int) -> List[int]:
    n: int = len(height)
    result: List[int] = list()

    for i in range(1, n):
        if height[i - 1] > threshold:
            result.append(i)

    return result
