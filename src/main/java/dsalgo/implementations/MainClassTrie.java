package dsalgo.implementations;

public class MainClassTrie {
    public static void main(String[] args) {
        TrieImplementation trieTreeObj = new TrieImplementation();
        trieTreeObj.insertWordInTrie("sma");
//        trieTreeObj.insertWordInTrie("smart");
        trieTreeObj.displayTrieTreeLevelOrder();
        trieTreeObj.deleteTrieNodeRecursively(trieTreeObj.root, "sma", 0);
        trieTreeObj.displayTrieTreeLevelOrder();
    }
}
