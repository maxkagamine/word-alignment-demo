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

# BERT-based model pretrained on the Kyoto Free Translation Task (KFTT) dataset.
MODEL = 'qiyuw/WSPAlign-ft-kftt'

# This model is trained to find the part of `context` that corresponds to the
# part of `question` that's wrapped in this marker.
MARKER = ' ¶ '

# Predictions with scores below the threshold will be discarded. A threshold of
# 0.1 filters out the AI's "wild guesses," while 0.9 and above will return only
# what the AI seems to consider "high confidence" results.
DEFAULT_THRESHOLD = 0.1

# These tokenizers are used solely to mark words (often morphemes) in the 'from'
# text for alignment; the ML pipeline gets fed untokenized strings which get
# broken down differently and mapped to vocab ids by the model's BertTokenizer.
TOKENIZERS = {
  'en': English().tokenizer,
  'ja': Japanese().tokenizer
}

def get_token_ranges(language: str, text: str) -> list[tuple[int, int]]:
  '''Tokenizes the text and returns an array of (start, end) for each token.'''
  tokenizer = TOKENIZERS[language]
  return [(t.idx, t.idx + len(t)) for t in tokenizer(text)]

def wrap_token(from_text: str, start: int, end: int, start_marker: str = MARKER, end_marker: str = MARKER) -> str:
  '''Wraps the part of the text to be aligned.'''
  return f'{from_text[:start]}{start_marker}{from_text[start:end]}{end_marker}{from_text[end:]}'

def print_alignment(from_text: str, from_start: int, from_end: int, to_text: str|None, to_start: int|None, to_end: int|None):
  '''Shows a visual of the alignment result for a token on stderr.'''
  if not sys.stderr.isatty():
    return
  color = '\033[32m' if to_text is not None else '\033[31m'
  print(wrap_token(from_text, from_start, from_end, color, '\033[m'), file=sys.stderr, end='\n' if to_text is not None else '\n\n')
  if to_text is not None:
    print(wrap_token(to_text, to_start, to_end, color, '\033[m'), file=sys.stderr, end='\n\n')

def align_forward(from_language: str, from_text: str, to_text: str, threshold: float = DEFAULT_THRESHOLD) -> list[int]:
  '''
  Returns an flat array of `from_start`, `from_end`, `to_start`, and `to_end`,
  repeated for every token in `from_text` that aligns to a part of `to_text`.
  '''
  result: list[int] = []
  pipe = pipeline('question-answering', model=MODEL)

  for (from_start, from_end) in get_token_ranges(from_language, from_text):
    prediction = pipe(
      question=wrap_token(from_text, from_start, from_end),
      context=to_text)

    if prediction['score'] < threshold:
      print_alignment(from_text, from_start, from_end, None, None, None)
      continue

    print_alignment(from_text, from_start, from_end, to_text, prediction['start'], prediction['end'])
    result += [from_start, from_end, prediction['start'], prediction['end']]

  assert len(result) % 4 == 0
  return result

def align_reverse(from_text: str, to_language: str, to_text: str, threshold: float = DEFAULT_THRESHOLD) -> list[int]:
  '''
  Calls align_forward with the from and to swapped, then swaps the results back.
  '''
  result = align_forward(to_language, to_text, from_text, threshold)
  reversed_result: list[int] = []
  for i in range(0, len(result), 4):
    reversed_result += [result[i + 2], result[i + 3], result[i], result[i + 1]]
  return reversed_result

def dedupe(result: list[int]) -> list[int]:
  '''Filters out duplicate alignment results.'''
  deduped = []
  for i in range(0, len(result), 4):
    is_dupe = False
    for j in range(0, len(deduped), 4):
      if result[i:i+4] == deduped[j:j+4]:
        is_dupe = True
    if not is_dupe:
      deduped += result[i:i+4]
  return deduped

def align(
  from_language: str,
  from_text: str,
  to_language: str,
  to_text: str,
  threshold: float = DEFAULT_THRESHOLD,
  symmetric: bool = False) -> list[int]:
  '''
  Returns an flat array of `from_start`, `from_end`, `to_start`, and `to_end`,
  repeated for every token in `from_text` that aligns to a part of `to_text`,
  and vice versa if `symmetric` is true.
  '''
  result = align_forward(from_language, from_text, to_text, threshold)
  if symmetric:
    result += align_reverse(from_text, to_language, to_text, threshold)
    result = dedupe(result)
  return result

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--from-language', type=str, required=True, choices=TOKENIZERS.keys())
  parser.add_argument('--from-text', type=str, required=True)
  parser.add_argument('--to-language', type=str, required=True, choices=TOKENIZERS.keys())
  parser.add_argument('--to-text', type=str, required=True)
  parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
  parser.add_argument('--symmetric', action='store_true', default=False)
  args = parser.parse_args()

  result = align(
    args.from_language,
    args.from_text,
    args.to_language,
    args.to_text,
    args.threshold,
    args.symmetric)

  print(','.join(str(i) for i in result))
