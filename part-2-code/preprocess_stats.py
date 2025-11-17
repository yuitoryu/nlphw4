import os
import numpy as np
from collections import Counter
from transformers import T5TokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
from load_data import T5Dataset  # Reuse the dataset implementation we defined earlier

def analyze_preprocessing():
    """Analyze dataset statistics before and after preprocessing"""
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    print("=" * 60)
    print("DATA PREPROCESSING ANALYSIS")
    print("=" * 60)
    
    splits = ['train', 'dev', 'test']
    results = {}
    
    for split in splits:
        print(f"\nAnalyzing {split} set...")
        
        # Instantiate the dataset (which triggers preprocessing)
        dataset = T5Dataset('data', split)
        
        # Raw data statistics
        nl_path = f'data/{split}.nl'
        with open(nl_path, 'r', encoding='utf-8') as f:
            raw_nl_lines = [line.strip() for line in f.readlines()]
        
        # Processed data statistics
        processed_nl_texts = dataset.encoder_inputs
        
        # Compute statistics over raw NL examples
        raw_nl_lengths = []
        raw_nl_tokens_all = []
        for line in raw_nl_lines:
            tokens = tokenizer.tokenize(line)
            raw_nl_lengths.append(len(tokens))
            raw_nl_tokens_all.extend(tokens)
        
        # Compute statistics over processed NL examples
        processed_nl_lengths = []
        processed_nl_tokens_all = []
        for text in processed_nl_texts:
            tokens = tokenizer.tokenize(text)
            processed_nl_lengths.append(len(tokens))
            processed_nl_tokens_all.extend(tokens)
        
        # Collect SQL statistics when available
        if split != 'test':
            sql_path = f'data/{split}.sql'
            with open(sql_path, 'r', encoding='utf-8') as f:
                raw_sql_lines = [line.strip() for line in f.readlines()]
            
            raw_sql_lengths = []
            raw_sql_tokens_all = []
            for line in raw_sql_lines:
                tokens = tokenizer.tokenize(line)
                raw_sql_lengths.append(len(tokens))
                raw_sql_tokens_all.extend(tokens)
            
            # SQL after preprocessing (identical because SQL is untouched)
            processed_sql_texts = dataset.decoder_targets
            processed_sql_lengths = []
            processed_sql_tokens_all = []
            for text in processed_sql_texts:
                tokens = tokenizer.tokenize(text)
                processed_sql_lengths.append(len(tokens))
                processed_sql_tokens_all.extend(tokens)
        else:
            raw_sql_lines = []
            raw_sql_lengths = []
            raw_sql_tokens_all = []
            processed_sql_lengths = []
            processed_sql_tokens_all = []
        
        results[split] = {
            'raw': {
                'num_examples': len(raw_nl_lines),
                'nl_mean_length': np.mean(raw_nl_lengths),
                'nl_max_length': max(raw_nl_lengths) if raw_nl_lengths else 0,
                'nl_min_length': min(raw_nl_lengths) if raw_nl_lengths else 0,
                'nl_vocab_size': len(set(raw_nl_tokens_all)),
                'sql_mean_length': np.mean(raw_sql_lengths) if raw_sql_lengths else 0,
                'sql_max_length': max(raw_sql_lengths) if raw_sql_lengths else 0,
                'sql_min_length': min(raw_sql_lengths) if raw_sql_lengths else 0,
                'sql_vocab_size': len(set(raw_sql_tokens_all)) if raw_sql_tokens_all else 0,
            },
            'processed': {
                'num_examples': len(processed_nl_texts),
                'nl_mean_length': np.mean(processed_nl_lengths),
                'nl_max_length': max(processed_nl_lengths) if processed_nl_lengths else 0,
                'nl_min_length': min(processed_nl_lengths) if processed_nl_lengths else 0,
                'nl_vocab_size': len(set(processed_nl_tokens_all)),
                'sql_mean_length': np.mean(processed_sql_lengths) if processed_sql_lengths else 0,
                'sql_max_length': max(processed_sql_lengths) if processed_sql_lengths else 0,
                'sql_min_length': min(processed_sql_lengths) if processed_sql_lengths else 0,
                'sql_vocab_size': len(set(processed_sql_tokens_all)) if processed_sql_tokens_all else 0,
            }
        }
        
        # Print detailed statistics
        print(f"\n{split.upper()} SET - RAW DATA:")
        print(f"  Examples: {results[split]['raw']['num_examples']}")
        print(f"  NL - Mean length: {results[split]['raw']['nl_mean_length']:.2f} tokens")
        print(f"  NL - Max length: {results[split]['raw']['nl_max_length']} tokens")
        print(f"  NL - Vocabulary: {results[split]['raw']['nl_vocab_size']} unique tokens")
        
        if split != 'test':
            print(f"  SQL - Mean length: {results[split]['raw']['sql_mean_length']:.2f} tokens")
            print(f"  SQL - Max length: {results[split]['raw']['sql_max_length']} tokens")
            print(f"  SQL - Vocabulary: {results[split]['raw']['sql_vocab_size']} unique tokens")
        
        print(f"\n{split.upper()} SET - PROCESSED DATA:")
        print(f"  Examples: {results[split]['processed']['num_examples']}")
        print(f"  NL - Mean length: {results[split]['processed']['nl_mean_length']:.2f} tokens")
        print(f"  NL - Max length: {results[split]['processed']['nl_max_length']} tokens")
        print(f"  NL - Vocabulary: {results[split]['processed']['nl_vocab_size']} unique tokens")
        
        if split != 'test':
            print(f"  SQL - Mean length: {results[split]['processed']['sql_mean_length']:.2f} tokens")
            print(f"  SQL - Max length: {results[split]['processed']['sql_max_length']} tokens")
            print(f"  SQL - Vocabulary: {results[split]['processed']['sql_vocab_size']} unique tokens")
        
        # Show the effect of preprocessing
        if split != 'test':
            nl_length_change = results[split]['processed']['nl_mean_length'] - results[split]['raw']['nl_mean_length']
            print(f"  NL length change due to prefix: +{nl_length_change:.2f} tokens")
    
    return results

def print_latex_tables(results):
    """Generate LaTeX tables"""
    print("\n" + "=" * 60)
    print("LATEX TABLES FOR REPORT")
    print("=" * 60)
    
    # Table 1: Raw data statistics
    print("\n% Table 1: Data statistics before any pre-processing")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l c c c}")
    print("\\hline")
    print("\\textbf{Statistics Name} & \\textbf{Train} & \\textbf{Dev} & \\textbf{Test} \\\\")
    print("\\hline")
    
    stats_names = [
        ("Number of examples", 'num_examples'),
        ("Mean NL length", 'nl_mean_length'),
        ("Mean SQL length", 'sql_mean_length'),
        ("Max NL length", 'nl_max_length'),
        ("Max SQL length", 'sql_max_length'),
        ("NL vocabulary size", 'nl_vocab_size'),
        ("SQL vocabulary size", 'sql_vocab_size')
    ]
    
    for stat_name, stat_key in stats_names:
        if stat_key == 'sql_mean_length' or stat_key == 'sql_max_length' or stat_key == 'sql_vocab_size':
            # SQL stats don't exist for test set
            train_val = f"{results['train']['raw'][stat_key]:.2f}" if 'mean' in stat_key else f"{results['train']['raw'][stat_key]}"
            dev_val = f"{results['dev']['raw'][stat_key]:.2f}" if 'mean' in stat_key else f"{results['dev']['raw'][stat_key]}"
            test_val = "N/A"
        else:
            train_val = f"{results['train']['raw'][stat_key]:.2f}" if 'mean' in stat_key else f"{results['train']['raw'][stat_key]}"
            dev_val = f"{results['dev']['raw'][stat_key]:.2f}" if 'mean' in stat_key else f"{results['dev']['raw'][stat_key]}"
            test_val = f"{results['test']['raw'][stat_key]:.2f}" if 'mean' in stat_key else f"{results['test']['raw'][stat_key]}"
        
        print(f"{stat_name} & {train_val} & {dev_val} & {test_val} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Data statistics before any pre-processing}")
    print("\\label{tab:raw_stats}")
    print("\\end{table}")
    
    # Table 2: Processed data statistics
    print("\n% Table 2: Data statistics after pre-processing")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l c c c}")
    print("\\hline")
    print("\\textbf{Statistics Name} & \\textbf{Train} & \\textbf{Dev} & \\textbf{Test} \\\\")
    print("\\hline")
    
    for stat_name, stat_key in stats_names:
        if stat_key == 'sql_mean_length' or stat_key == 'sql_max_length' or stat_key == 'sql_vocab_size':
            # SQL stats don't exist for test set
            train_val = f"{results['train']['processed'][stat_key]:.2f}" if 'mean' in stat_key else f"{results['train']['processed'][stat_key]}"
            dev_val = f"{results['dev']['processed'][stat_key]:.2f}" if 'mean' in stat_key else f"{results['dev']['processed'][stat_key]}"
            test_val = "N/A"
        else:
            train_val = f"{results['train']['processed'][stat_key]:.2f}" if 'mean' in stat_key else f"{results['train']['processed'][stat_key]}"
            dev_val = f"{results['dev']['processed'][stat_key]:.2f}" if 'mean' in stat_key else f"{results['dev']['processed'][stat_key]}"
            test_val = f"{results['test']['processed'][stat_key]:.2f}" if 'mean' in stat_key else f"{results['test']['processed'][stat_key]}"
        
        print(f"{stat_name} & {train_val} & {dev_val} & {test_val} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Data statistics after pre-processing}")
    print("\\label{tab:processed_stats}")
    print("\\end{table}")

def analyze_token_distribution():
    """Analyze the token distribution"""
    print("\n" + "=" * 60)
    print("TOKEN DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Analyze token frequencies in the training split
    with open('data/train.nl', 'r', encoding='utf-8') as f:
        nl_lines = [line.strip() for line in f.readlines()]
    
    with open('data/train.sql', 'r', encoding='utf-8') as f:
        sql_lines = [line.strip() for line in f.readlines()]
    
    all_nl_tokens = []
    all_sql_tokens = []
    
    for nl, sql in zip(nl_lines, sql_lines):
        nl_tokens = tokenizer.tokenize(nl)
        sql_tokens = tokenizer.tokenize(sql)
        all_nl_tokens.extend(nl_tokens)
        all_sql_tokens.extend(sql_tokens)
    
    # Compute the most frequent tokens
    nl_token_freq = Counter(all_nl_tokens)
    sql_token_freq = Counter(all_sql_tokens)
    
    print(f"\nTotal unique NL tokens: {len(nl_token_freq)}")
    print(f"Total unique SQL tokens: {len(sql_token_freq)}")
    
    print("\nTop 10 most common NL tokens:")
    for token, count in nl_token_freq.most_common(10):
        print(f"  {token}: {count} occurrences")
    
    print("\nTop 10 most common SQL tokens:")
    for token, count in sql_token_freq.most_common(10):
        print(f"  {token}: {count} occurrences")
    
    # Analyze out-of-vocabulary coverage
    print(f"\nTokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Coverage of NL tokens: {len(nl_token_freq) / tokenizer.vocab_size * 100:.2f}%")
    print(f"Coverage of SQL tokens: {len(sql_token_freq) / tokenizer.vocab_size * 100:.2f}%")

def check_data_quality():
    """Check dataset quality"""
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)
    
    # Verify matching line counts across files
    for split in ['train', 'dev']:
        nl_file = f'data/{split}.nl'
        sql_file = f'data/{split}.sql'
        
        with open(nl_file, 'r', encoding='utf-8') as f:
            nl_lines = f.readlines()
        
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_lines = f.readlines()
        
        print(f"\n{split.upper()} set:")
        print(f"  NL lines: {len(nl_lines)}")
        print(f"  SQL lines: {len(sql_lines)}")
        print(f"  Match: {len(nl_lines) == len(sql_lines)}")
        
        if len(nl_lines) != len(sql_lines):
            print(f"  WARNING: Line count mismatch!")
    
    # Inspect the test split
    test_nl_file = 'data/test.nl'
    with open(test_nl_file, 'r', encoding='utf-8') as f:
        test_lines = f.readlines()
    print(f"\nTEST set: {len(test_lines)} NL examples")

if __name__ == "__main__":
    # Run the full analysis suite
    check_data_quality()
    results = analyze_preprocessing()
    analyze_token_distribution()
    # print_latex_tables(results)
    
    # print("\n" + "=" * 60)
    # print("PREPROCESSING ANALYSIS COMPLETE")
    # print("=" * 60)
    # print("\nNext steps:")
    # print("1. Copy the LaTeX tables to your report")
    # print("2. Use the statistics for Q4 in your assignment")
    # print("3. Begin training with: python train_t5.py --finetune --experiment_name baseline")
