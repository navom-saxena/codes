import heapq
import math
import sys
from typing import List, Set, Dict, Tuple, re
from Arrays_q_11 import sieveOfEratosthenes


# https://leetcode.com/problems/smallest-index-with-digit-sum-equal-to-index/description/?envType=problem-list-v2&envId=array

def smallestIndex(self, nums: List[int]) -> int:
    n: int = len(nums)
    for i in range(n):
        num: int = nums[i]
        sumDigits: int = 0
        while num > 0:
            remainder = num % 10
            sumDigits += remainder
            num //= 10

        if sumDigits == i:
            return i
    return -1


# https://leetcode.com/problems/check-if-any-element-has-prime-frequency/?envType=problem-list-v2&envId=array

def checkPrimeFrequency(self, nums: List[int]) -> bool:
    n: int = len(nums)
    primes: Set[int] = sieveOfEratosthenes(101)
    freqMap: Dict[int, int] = dict()

    for i in range(n):
        num: int = nums[i]
        freqMap[num] = freqMap.get(num, 0) + 1

    for value in freqMap.values():
        if value in primes:
            return True
    return False


# https://leetcode.com/problems/coupon-code-validator/?envType=problem-list-v2&envId=array

def validateCoupons(self, code: List[str], businessLine: List[str], isActive: List[bool]) -> List[str]:
    n: int = len(code)
    interimResult: List[str] = list()

    for i in range(n):
        if bool(re.fullmatch(r"^[a-zA-Z0-9_]+$", code[i])) and isActive[i] and businessLine[i] in ["electronics",
                                                                                                   "grocery",
                                                                                                   "pharmacy",
                                                                                                   "restaurant"]:
            interimResult.append(f"{businessLine[i]}-{code[i]}")

    interimResult.sort()
    result: List[str] = list()
    for word in interimResult:
        codeWord: str = word.split("-")[1]
        result.append(codeWord)

    return result
