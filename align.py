#!/usr/bin/env python3
'''
Performs word alignment using machine learning.

Rather than the "Pharaoh" output format used by GIZA++ and other older word
alignment tools, which maps tokens on one side to tokens on the other, the
output of this program is a repeating array of four integers: the range of
characters in the original "from" text (as inclusive start & exclusive end
indexes) followed by the aligned range in the original "to" text. The inputs
therefore do not need to be tokenized beforehand, and no knowledge of the
tokenization is needed to make use of the alignment result.
'''
import argparse
import sys
from spacy.lang.en import English
from spacy.lang.ja import Japanese
from transformers import pipeline
from simplify import simplify

# BERT-based model pretrained on the Kyoto Free Translation Task (KFTT) dataset.
MODEL = 'qiyuw/WSPAlign-ft-kftt'

# This model is trained to find the part of `context` that corresponds to the
# part of `question` that's wrapped in this marker.
MARKER = ' ¶ '

# Predictions with scores below the threshold will be discarded. A threshold of
# 0.1 filters out the AI's "wild guesses," while 0.9 and above will return only
# what the AI seems to consider "high confidence" results.
DEFAULT_THRESHOLD = 0.1

# These tokenizers are used to mark words (often morphemes) in the 'from' text
# for alignment; the ML pipeline gets fed untokenized strings which get broken
# down differently and mapped to vocab ids by the model's BertTokenizer.
TOKENIZERS = {
  'en': English().tokenizer,
  'ja': Japanese().tokenizer
}

def get_token_ranges(language: str, text: str) -> list[tuple[int, int]]:
  '''Tokenizes the text and returns an array of (start, end) for each token.'''
  tokenizer = TOKENIZERS[language]
  return [(t.idx, t.idx + len(t)) for t in tokenizer(text)]

def find_token_indexes(token_ranges: list[tuple[int, int]], start: int, end: int) -> list[int]:
  '''Finds the token ranges that intersect the given range.'''
  indexes: list[int] = []
  for i, (token_start, token_end) in enumerate(token_ranges):
    if end > token_start and start < token_end:
      indexes += [i]
  return indexes

def wrap_token(from_text: str, start: int, end: int, start_marker: str = MARKER, end_marker: str = MARKER) -> str:
  '''Wraps the part of the text to be aligned.'''
  return f'{from_text[:start]}{start_marker}{from_text[start:end]}{end_marker}{from_text[end:]}'

def print_alignment(from_text: str, from_start: int, from_end: int, to_text: str, to_start: int, to_end: int, score: float|None = None, is_above_threshold: bool = True):
  '''Shows a visual of the alignment result for a token on stderr.'''
  if not sys.stderr.isatty():
    return
  color = '\033[32m' if is_above_threshold else '\033[31m'
  print(wrap_token(from_text, from_start, from_end, color, '\033[m'), file=sys.stderr)
  print(wrap_token(to_text, to_start, to_end, color, '\033[m'), file=sys.stderr, end='\n\n' if score is None else '')
  if score is not None:
    print(f' \033[1;30m(score = {score:.10f})\033[m', file=sys.stderr, end='\n\n')

def align_forward(
  from_token_ranges: list[tuple[int, int]],
  to_token_ranges: list[tuple[int, int]],
  from_text: str,
  to_text: str,
  threshold: float = DEFAULT_THRESHOLD) -> list[tuple[int, int]]:
  '''
  Runs the ML model and returns a list of token pairs mapping indexes of tokens
  in `from_token_ranges` to those of `to_token_ranges`.
  '''
  result: list[tuple[int, int]] = []
  pipe = pipeline('question-answering', model=MODEL)

  for from_token, (from_start, from_end) in enumerate(from_token_ranges):
    prediction = pipe(
      question=wrap_token(from_text, from_start, from_end),
      context=to_text)

    is_above_threshold = prediction['score'] >= threshold
    print_alignment(from_text, from_start, from_end, to_text, prediction['start'], prediction['end'], prediction['score'], is_above_threshold)

    if is_above_threshold:
      to_tokens = find_token_indexes(to_token_ranges, prediction['start'], prediction['end'])
      result += [(from_token, to_token) for to_token in to_tokens]

  return result

def align_reverse(
  from_token_ranges: list[tuple[int, int]],
  to_token_ranges: list[tuple[int, int]],
  from_text: str,
  to_text: str,
  threshold: float = DEFAULT_THRESHOLD) -> list[tuple[int, int]]:
  '''
  Calls align_forward with the from and to swapped, then swaps the results back.
  '''
  result = align_forward(to_token_ranges, from_token_ranges, to_text, from_text, threshold)
  return [(to_token, from_token) for (from_token, to_token) in result]

def token_pairs_to_ranges(
  from_token_ranges: list[tuple[int, int]],
  to_token_ranges: list[tuple[int, int]],
  token_pairs: list[tuple[int, int]]) -> list[int]:
  '''
  Converts a list of token index pairs (`from_token`, `to_token`) to a flat
  array of `from_start`, `from_end`, `to_start`, and `to_end`.
  '''
  result: list[int] = []

  for (from_token, to_token) in token_pairs:
    (from_start, from_end) = from_token_ranges[from_token]
    (to_start, to_end) = to_token_ranges[to_token]
    result += [from_start, from_end, to_start, to_end]

  assert len(result) % 4 == 0
  return result

def align(
  from_language: str,
  from_text: str,
  to_language: str,
  to_text: str,
  threshold: float = DEFAULT_THRESHOLD,
  symmetric: bool = False,
  simplify_result: bool = True) -> list[int]:
  '''
  Returns an flat array of `from_start`, `from_end`, `to_start`, and `to_end`,
  repeated for every token in `from_text` that aligns to a part of `to_text`,
  and vice versa if `symmetric` is true.
  '''
  from_token_ranges = get_token_ranges(from_language, from_text)
  to_token_ranges = get_token_ranges(to_language, to_text)

  token_pairs = align_forward(from_token_ranges, to_token_ranges, from_text, to_text, threshold)
  if symmetric:
    token_pairs += align_reverse(from_token_ranges, to_token_ranges, from_text, to_text, threshold)

  result = token_pairs_to_ranges(from_token_ranges, to_token_ranges, token_pairs)
  return simplify(result, from_text, to_text) if simplify_result else result

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--from-language', type=str, required=True, choices=TOKENIZERS.keys())
  parser.add_argument('--from-text', type=str, required=True)
  parser.add_argument('--to-language', type=str, required=True, choices=TOKENIZERS.keys())
  parser.add_argument('--to-text', type=str, required=True)
  parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
  parser.add_argument('--symmetric', action='store_true', default=False)
  parser.add_argument('--no-simplify', action='store_true', default=False)
  args = parser.parse_args()

  result = align(
    args.from_language,
    args.from_text,
    args.to_language,
    args.to_text,
    args.threshold,
    args.symmetric,
    not args.no_simplify)

  print(','.join(str(i) for i in result))
