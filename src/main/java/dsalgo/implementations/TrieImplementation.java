package dsalgo.implementations;

import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

public class TrieImplementation {
    TrieNode root;

    TrieImplementation() {
        this.root = new TrieNode();
    }

    public void displayTrieTreeLevelOrder() {
        Queue<TrieNode> queue = new LinkedList<>();
        TrieNode currentNode = this.root;
        queue.add(currentNode);
        while (!queue.isEmpty()) {
            TrieNode dequeuedNode = queue.remove();
            if (dequeuedNode != null) {
                Map<Character, TrieNode> charChildMapping = dequeuedNode.getAllChildMapping();
                System.out.println(charChildMapping.keySet());
                charChildMapping.forEach((key, value) -> {
//                    System.out.print(key + " ");
                    queue.add(value);
                });
            }
//            System.out.println();
        }
    }

    public void displayTrieTreeWordWise(String word) {
        TrieNode currentNode = this.root;
        int wordLength = word.length();
        for (int i = 0; i < wordLength; i++) {
            char character = word.charAt(i);
            TrieNode childCharacterNode = currentNode.getCharacterNode(character);
            if (childCharacterNode != null) {
                System.out.print(character + " ");
                currentNode = childCharacterNode;
            } else {
                System.out.println();
                System.out.println("character " + character + " not in trie");
                break;
            }
        }
        System.out.println();
    }

    public void insertWordInTrie(String word) {
        int length = word.length();
        TrieNode currentNode = this.root;
        for (int i = 0; i < length; i++) {
            char character = word.charAt(i);
            if (currentNode.doesTrieHasCharChild(character)) {
                currentNode = currentNode.getCharacterNode(character);
            } else {
                TrieNode newNode = new TrieNode();
                currentNode.insertChar(character, newNode);
                currentNode = newNode;
            }
        }
        currentNode.setIsWord(true);
    }

    public boolean searchWordInTrie(String word) {
        int length = word.length();
        TrieNode currentNode = this.root;
        for (int i = 0; i < length; i++) {
            char character = word.charAt(i);
            TrieNode childCharacterNode = currentNode.getCharacterNode(character);
            if (childCharacterNode != null) {
                System.out.print(character + " ");
                currentNode = childCharacterNode;
            } else {
                System.out.println();
                System.out.println("character " + character + " not in trie");
                return false;
            }
        }
        System.out.println();
        return currentNode.getIsWord();
    }

    public void deleteWordInTrie(String word) {
        int length = word.length();
        TrieNode currentNode = this.root;
        for (int i = 0; i < length; i++) {
            char character = word.charAt(i);
            TrieNode childCharacterNode = currentNode.getCharacterNode(character);
            if (childCharacterNode != null) {
                System.out.print(character + " ");
                currentNode = childCharacterNode;
            } else {
                System.out.println("character " + character + " not in trie");
                break;
            }
            currentNode.setIsWord(false);
        }
    }

    public void deleteTrieNodeRecursively(TrieNode current, String word, int index) {
        if (index == word.length()) {
            if (current.getAllChildMapping().keySet().size() == 0) {
                System.out.println(current.getAllChildMapping() + " -- " + current);
                current = null;
                System.out.println(current + " --- ");
                return;
            } else {
                return;
            }
        }
        char character = word.charAt(index);
        TrieNode node = current.getCharacterNode(character);
        deleteTrieNodeRecursively(node, word, index + 1);
        Map<Character, TrieNode> mapping = current.getAllChildMapping();
        System.out.println(mapping + " <- mapping");
        if (current.getCharacterNode(character) == null) {
            mapping.remove(character);
        }
        if (mapping.keySet().size() == 0) {
            current = null;
        }
    }
}
