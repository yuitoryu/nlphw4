# data_statistics.py
import os
from transformers import T5TokenizerFast
from collections import Counter
import numpy as np

def analyze_data():
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    splits = ['train', 'dev']
    stats = {}
    
    for split in splits:
        # Read natural language data
        with open(f'data/{split}.nl', 'r', encoding='utf-8') as f:
            nl_lines = [line.strip() for line in f.readlines()]
        
        # Read SQL data
        with open(f'data/{split}.sql', 'r', encoding='utf-8') as f:
            sql_lines = [line.strip() for line in f.readlines()]
        
        # Compute statistics
        nl_lengths = []
        sql_lengths = []
        nl_vocab = set()
        sql_vocab = set()
        
        for nl, sql in zip(nl_lines, sql_lines):
            # Natural language statistics
            nl_tokens = tokenizer.tokenize(nl)
            nl_lengths.append(len(nl_tokens))
            nl_vocab.update(nl_tokens)
            
            # SQL statistics
            sql_tokens = tokenizer.tokenize(sql)
            sql_lengths.append(len(sql_tokens))
            sql_vocab.update(sql_tokens)
        
        stats[split] = {
            'num_examples': len(nl_lines),
            'mean_nl_length': np.mean(nl_lengths),
            'mean_sql_length': np.mean(sql_lengths),
            'nl_vocab_size': len(nl_vocab),
            'sql_vocab_size': len(sql_vocab),
            'max_nl_length': max(nl_lengths),
            'max_sql_length': max(sql_lengths)
        }
    
    return stats

if __name__ == "__main__":
    stats = analyze_data()
    for split, data in stats.items():
        print(f"\n{split.upper()} SET STATISTICS:")
        print(f"Number of examples: {data['num_examples']}")
        print(f"Mean NL length: {data['mean_nl_length']:.2f}")
        print(f"Mean SQL length: {data['mean_sql_length']:.2f}")
        print(f"NL vocabulary size: {data['nl_vocab_size']}")
        print(f"SQL vocabulary size: {data['sql_vocab_size']}")
        print(f"Max NL length: {data['max_nl_length']}")
        print(f"Max SQL length: {data['max_sql_length']}")
