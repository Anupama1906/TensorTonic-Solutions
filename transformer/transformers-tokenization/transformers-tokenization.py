import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        self.vocab_size += 4

        word_list=set()

        for text in texts:
            words = text.lower().split(" ")
            for word in words:
               word_list.add(word)
        word_list = sorted(list(word_list))

        for word in word_list:
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size+=1  
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        id_list = []
        words = text.lower().split()
        for word in words:
            if word not in self.word_to_id.keys():
                id_list.append(self.word_to_id[self.unk_token])
                continue
            id_list.append(self.word_to_id[word])
        return id_list
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        res = []
        for id in ids:
            if id in self.id_to_word:
                res.append(self.id_to_word[id])
            else:
                res.append(self.unk_token)
        res_str = " ".join(res)
        return res_str
