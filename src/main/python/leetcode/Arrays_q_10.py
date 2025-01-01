import queue
import sys
from typing import List, Set, Dict, Tuple
import heapq

# https://leetcode.com/problems/assign-cookies/description/?envType=problem-list-v2&envId=array

def findContentChildren(self, g: List[int], s: List[int]) -> int:
    g: List[int] = sorted(g, reverse=True)
    s: List[int] = sorted(s, reverse=True)

    j: int = 0
    for i in range(len(g)):
        if j < len(s) and g[i] <= s[j]:
            j += 1

    return j
