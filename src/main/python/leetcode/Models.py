from typing import Dict


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=None, children=None, parent=None):
        self.val = val
        self.children = children
        self.parent = parent
        self.left = None
        self.right = None


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class LcaNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


class Trie:
    def __init__(self, val: str):
        self.val: str = val
        self.children: Dict[str, Trie] = dict()
        self.is_word: bool = False
