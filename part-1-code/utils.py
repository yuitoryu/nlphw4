import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    words = word_tokenize(text)

    # Probability of applying each transformation
    synonym_prob = 0.3      # 10% chance to replace with a synonym
    typo_prob = 0.15        # 5% chance to introduce a typo

    # Define nearby keys for simple typo simulation
    keyboard_neighbors = {
        'a': ['s', 'q', 'z'], 'b': ['v', 'n'], 'c': ['x', 'v'], 'd': ['s', 'f', 'e'],
        'e': ['w', 'r', 'd'], 'f': ['d', 'g', 'r'], 'g': ['f', 'h', 't'], 'h': ['g', 'j', 'y'],
        'i': ['u', 'o', 'k'], 'j': ['h', 'k', 'u'], 'k': ['j', 'l', 'i'], 'l': ['k', 'o'],
        'm': ['n', 'j'], 'n': ['b', 'm'], 'o': ['i', 'p', 'l'], 'p': ['o'],
        'q': ['w', 'a'], 'r': ['e', 't', 'f'], 's': ['a', 'd', 'w', 'x'],
        't': ['r', 'y', 'g'], 'u': ['y', 'i', 'j'], 'v': ['c', 'b'], 'w': ['q', 'e', 's'],
        'x': ['z', 'c', 's'], 'y': ['t', 'u', 'h'], 'z': ['x', 's']
    }

    new_words = []
    for w in words:
        # Try synonym replacement
        if random.random() < synonym_prob:
            synsets = wordnet.synsets(w)
            if synsets:
                lemmas = synsets[0].lemmas()
                if lemmas:
                    synonym = lemmas[0].name().replace("_", " ")
                    if synonym.lower() != w.lower():
                        w = synonym

        # Try introducing a typo
        elif random.random() < typo_prob and len(w) > 1:
            idx = random.randint(0, len(w) - 1)
            if w[idx].lower() in keyboard_neighbors:
                neighbors = keyboard_neighbors[w[idx].lower()]
                new_char = random.choice(neighbors)
                w = w[:idx] + new_char + w[idx + 1:]

        new_words.append(w)

    new_text = TreebankWordDetokenizer().detokenize(new_words)
    example["text"] = new_text

    ##### YOUR CODE ENDS HERE ######

    return example
