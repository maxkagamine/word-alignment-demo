#!/usr/bin/env python3
'''
Runs awesome-align with the same tokenization and output format as align.py
'''
import argparse
from align import get_token_ranges, print_alignment, TOKENIZERS
from subprocess import call

INPUT_FILE = 'awesome-align-input.txt'
OUTPUT_FILE = 'awesome-align-output.txt'

def build_tokenized_string(text: str, token_ranges: list[tuple[int, int]]):
  return ' '.join(text[t[0]:t[1]] for t in token_ranges)

def build_input(from_text: str, from_token_ranges: list[tuple[int, int]], to_text: str, to_token_ranges: list[tuple[int, int]]):
  from_tokenized = build_tokenized_string(from_text, from_token_ranges)
  to_tokenized = build_tokenized_string(to_text, to_token_ranges)
  return f'{from_tokenized} ||| {to_tokenized}'

def run_awesome():
  ret = call([
    'python3', '-m', 'awesome_align.run_align',
    '--output_file', '../' + OUTPUT_FILE,
    '--model_name_or_path', 'bert-base-multilingual-cased',
    '--data_file', '../' + INPUT_FILE
  ], cwd='./awesome-align')
  if ret != 0:
    raise ValueError('awesome-align failed')

def parse_output(output: str, from_text: str, from_token_ranges: list[tuple[int, int]], to_text: str, to_token_ranges: list[tuple[int, int]]):
  result: list[int] = []
  token_mappings = [(int(pair[0]), int(pair[1])) for pair in [pair.split('-') for pair in output.strip().split(' ')]]
  token_mappings.sort(key=lambda pair: pair[0])
  for (from_token, to_token) in token_mappings:
    [from_start, from_end] = from_token_ranges[from_token]
    [to_start, to_end] = to_token_ranges[to_token]

    print_alignment(from_text, from_start, from_end, to_text, to_start, to_end)
    result += [from_start, from_end, to_start, to_end]

  assert len(result) % 4 == 0
  return result

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--from-language', type=str, required=True, choices=TOKENIZERS.keys())
  parser.add_argument('--from-text', type=str, required=True)
  parser.add_argument('--to-language', type=str, required=True, choices=TOKENIZERS.keys())
  parser.add_argument('--to-text', type=str, required=True)
  args = parser.parse_args()

  from_token_ranges = get_token_ranges(args.from_language, args.from_text)
  to_token_ranges = get_token_ranges(args.to_language, args.to_text)

  input_str = build_input(args.from_text, from_token_ranges, args.to_text, to_token_ranges)
  with open(INPUT_FILE, 'w') as file:
    print(input_str, file=file)

  run_awesome()

  result: list[int] = []
  with open(OUTPUT_FILE, 'r') as file:
    result = parse_output(file.readline(), args.from_text, from_token_ranges, args.to_text, to_token_ranges)

  print(','.join(str(i) for i in result))

