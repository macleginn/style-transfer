import os
import sys
from itertools import zip_longest
from unicodedata import category
from random import shuffle
from math import ceil
import argparse
import requests
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import datasets

from tqdm.auto import tqdm


class ClassificationHead(nn.Module):
    def __init__(self, model_dim=768, innder_dim=4096, n_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(model_dim, innder_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(innder_dim, n_classes)
        )
        
    def forward(self, x):
        return self.model(x)


# Global variables for POS tagging state machines
FIRST_SUBWORD_MARKERS = {'Ġ', '▁'}
STATE_TRANSITION_TABLE = {
    ('seen_punct',    'punct'):    'seen_punct',
    ('seen_first',    'punct'):    'seen_punct',
    ('seen_nonfirst', 'punct'):    'seen_punct',
    ('seen_punct',    'first'):    'seen_first',
    ('seen_first',    'first'):    'seen_first',
    ('seen_nonfirst', 'first'):    'seen_first',
    ('seen_punct',    'nonfirst'): 'seen_first',        # NB
    ('seen_first',    'nonfirst'): 'seen_nonfirst',
    ('seen_nonfirst', 'nonfirst'): 'seen_nonfirst'
}
ACTION_SELECTION_TABLE = {
    # PUNCT is identified using lookup
    ('seen_punct',    'punct'):    'output',               
    ('seen_first',    'punct'):    'output',
    ('seen_nonfirst', 'punct'):    'output',
    ('seen_punct',    'first'):    'predict',
    ('seen_first',    'first'):    'predict',
    ('seen_nonfirst', 'first'):    'predict',
    ('seen_punct',    'nonfirst'): 'predict',           # NB
    ('seen_first',    'nonfirst'): 'skip',
    ('seen_nonfirst', 'nonfirst'): 'skip'
}

codepoints = range(sys.maxunicode + 1)
punctuation = {
    c for i in codepoints
    if (
        category(c := chr(i)).startswith("P")
        # Some punctuation signs are classified as maths signs in Unicode
        or category(c := chr(i)).startswith("S")  
    )
}


def tokenize_and_classify(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    actions = []
    state = 'seen_punct'
    for token in tokens:
        if token in punctuation or token[1:] in punctuation:
            token_type = 'punct'
        elif token[0] in FIRST_SUBWORD_MARKERS:
            token_type = 'first'
        else:
            token_type = 'nonfirst'
        actions.append(ACTION_SELECTION_TABLE[(state, token_type)])
        state = STATE_TRANSITION_TABLE[(state, token_type)]
    return tokens, actions
    

def get_nonfirst_subword_prefix(
        tokeniser,
        test_word="honorificabilitudinitatibus",
        step=1
        ):
    if step > 5:
        raise ValueError("The tokeniser is returning a single token after 5 test-word doublings.")
    tokenised = tokeniser.tokenize(test_word)
    if len(tokenised) == 1:
        return get_nonfirst_subword_prefix(
            tokeniser,
            test_word=test_word * 2,
        )
    return tokenised[0], tokenised[1]
    

def parse_pos_data(url):
    response = requests.get(url)
    examples = response.text.strip().split('\n\n')
    return [
        {
            'tokens': [token for token, _ in (line.split() for line in example.splitlines() if line.strip())],
            'tags': [tag for _, tag in (line.split() for line in example.splitlines() if line.strip())]
        }
        for example in examples
    ]


import string_constants

def detokenize_list(tokens):
    """
    Converts a list of tokens into a single string with specific punctuation handling.
    - Punctuation signs are not preceded by a white space.
    - Exception: quote marks can be preceded by a whitespace.
    - Quote marks (and opening brackets/symbols) should not be followed by a whitespace.
    - For double quotes ("): if not inside a double quote pair, add a space before an opening "; if inside, no space before a closing ".
    """
    if not tokens:
        return ""

    result_buffer = [tokens[0]]
    in_double_quote = False
    # Initialize in_double_quote state based on the first token if it's a double quote
    if tokens[0] == '"':
        in_double_quote = True

    for i in range(1, len(tokens)):
        current_token = tokens[i]
        prev_token = tokens[i-1]
        needs_space = True  # Default: add a space

        if current_token == '"':
            if not in_double_quote:  # Current token is an OPENING "
                # Add space before this opening quote, unless prev_token is an opening symbol like '('.
                # This check ensures (" doesn't become ( ".
                # Assumes _OPENING_SYMBOLS_NO_POST_SPACE contains '(', '[', etc. but not necessarily '"' itself for this specific check.
                if prev_token in string_constants._OPENING_SYMBOLS_NO_POST_SPACE:
                    needs_space = False
                # else: needs_space remains True (e.g. for "word ...")
            else:  # Current token is a CLOSING "
                needs_space = False # No space before a closing quote (e.g. "... word")
            in_double_quote = not in_double_quote # Toggle the state
        elif prev_token == '"' and in_double_quote:
            # If we are inside a double quote pair, no space before the current token
            needs_space = False
        else:
            # Logic for tokens that are not symmetrical double quotes.
            if (
                current_token in string_constants._PUNCT_NO_PRE_SPACE
                or prev_token in string_constants._OPENING_SYMBOLS_NO_POST_SPACE
            ):
                needs_space = False

            # A bunch of ad hoc rules for apostrophes. To handle them
            # in a more general way, we would need to know the detailed context
            # 1. No space before if we are at the end of the input
            if current_token == "'" and i == len(tokens) - 1:
                    needs_space = False
            # 2. No space before the word following the apostrophe at the beginning of the input
            elif i == 1 and prev_token == "'":
                needs_space = False
        

        # Handle hyphens for compound words like "wheel-chair"
        # This logic is applied after the main spacing rules and can override needs_space.
        if current_token == '-' and (prev_token[-1].isalnum() if prev_token else False):
            needs_space = False
        if prev_token == '-' and (current_token[0].isalnum() if current_token else False):
            needs_space = False
            
        if needs_space:
            result_buffer.append(" ")
        result_buffer.append(current_token)
        
    return ''.join(result_buffer)


def load_data():
    DEV_URL = "https://eurphon.info/static/xglue-pos/en.dev"
    TEST_URL =  "https://eurphon.info/static/xglue-pos/en.test"
    TRAIN_URL = "https://eurphon.info/static/xglue-pos/en.train"
    if not os.path.exists('pos_data/train.json'):
        train_data = parse_pos_data(TRAIN_URL)
        dev_data = parse_pos_data(DEV_URL)
        test_data = parse_pos_data(TEST_URL)

        datasets.Dataset.from_list(train_data).to_json('pos_data/train.json')
        datasets.Dataset.from_list(dev_data).to_json('pos_data/dev.json')
        datasets.Dataset.from_list(test_data).to_json('pos_data/test.json')
    else:
        train_df = pd.read_json('pos_data/train.json', lines=True)
        dev_df = pd.read_json('pos_data/dev.json', lines=True)
        test_df = pd.read_json('pos_data/test.json', lines=True)
        
        train_data = datasets.Dataset.from_pandas(train_df)
        dev_data = datasets.Dataset.from_pandas(dev_df)
        test_data = datasets.Dataset.from_pandas(test_df)
    return train_data, dev_data, test_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a POS tagger on XGLUE dataset for English that uses next-token representations from a language model"
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="ibm-granite/granite-3.3-2b-instruct",
        help="Hugging Face tag of the pre-trained model to use",
    )
    parser.add_argument(
        "--model-dimension",
        type=int,
        default=2048,
        help="Dimension of the model (default: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer (default: 5e-5)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of epochs to train (default: 3)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization (default: 512)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pos_tagger_model",
        help="Directory to save the trained model (default: pos_tagger_model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def compress_tokens_and_actions(tokens, actions):
    """
    Compress the lists of tokens and actions by appending 'skip' actions to the preceding action
    and appending respective non-first-subword tokens to the preceding token.
    """
    compressed_tokens = []
    compressed_actions = []
    for token, action in zip(tokens, actions):
        if action == 'skip':
            compressed_tokens[-1] += '-'
            compressed_tokens[-1] += token

            compressed_actions[-1] += ' '
            compressed_actions[-1] += action
        else:
            compressed_tokens.append(token)
            compressed_actions.append(action)
    return compressed_tokens, compressed_actions
    

def main():
    train_data, dev_data, test_data = load_data()
    args = parse_args()

    # Check that the "predict" actions produced by tokenize_and_classify
    # align with the POS tags in the training data
    for example in tqdm(train_data):
        detokenized_sentence = detokenize_list(example['tokens'])
        tokens, actions = compress_tokens_and_actions(
            *tokenize_and_classify(
                detokenized_sentence,
                AutoTokenizer.from_pretrained(args.model_tag))
        ) 
        if len(actions) != len(example['tags']):
            print(f"Mismatch in example: {detokenized_sentence}")
            print(f"Have {len(example['tags'])} tags, but got {len(actions)} actions.")
            print(
                pd.DataFrame.from_records(
                    zip_longest(example['tokens'], example['tags'], tokens, actions),
                    columns=['Token', 'Tag', 'Tokenized', 'Action']
                ))
            break


if __name__ == "__main__":
    main()
