package dsalgo.implementations;

import java.util.HashMap;
import java.util.Map;

class TrieNode {
    private Map<Character, TrieNode> hm;
    private boolean isWord;

    TrieNode() {
        this.hm = new HashMap<>();
        this.isWord = false;
    }

    void insertChar(char character, TrieNode childNode) {
        this.hm.put(character, childNode);
    }

    boolean doesTrieHasCharChild(char character) {
        return this.hm.containsKey(character);
    }

    TrieNode getCharacterNode(char character) {
        return this.hm.get(character);
    }

    Map<Character, TrieNode> getAllChildMapping() {
        return this.hm;
    }

    boolean getIsWord() {
        return this.isWord;
    }

    void setIsWord(boolean flag) {
        this.isWord = flag;
    }
}
