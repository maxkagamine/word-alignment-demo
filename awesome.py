#!/usr/bin/env python3
'''
Runs awesome-align with the same tokenization and output format as align.py
'''
import argparse
import os
from align import get_token_ranges, print_alignment, TOKENIZERS
from simplify import simplify
from subprocess import call
from tempfile import NamedTemporaryFile

def build_tokenized_string(text: str, token_ranges: list[tuple[int, int]]):
  return ' '.join(text[t[0]:t[1]] for t in token_ranges)

def build_input(from_text: str, from_token_ranges: list[tuple[int, int]], to_text: str, to_token_ranges: list[tuple[int, int]]):
  from_tokenized = build_tokenized_string(from_text, from_token_ranges)
  to_tokenized = build_tokenized_string(to_text, to_token_ranges)
  return f'{from_tokenized} ||| {to_tokenized}'

def run_awesome(model: str, input_file_path: str, output_file_path: str):
  if '/' in model and not model.startswith('/'):
    model = '../' + model
  ret = call([
    'python3', '-m', 'awesome_align.run_align',
    '--output_file', os.path.join('..', output_file_path),
    '--model_name_or_path', model,
    '--data_file', os.path.join('..', input_file_path),
    '--cache_dir', '../models/bert'
  ], cwd='./awesome-align')
  if ret != 0:
    raise ValueError('awesome-align failed')

def parse_output(
    output: str,
    from_text: str,
    from_token_ranges: list[tuple[int, int]],
    to_text: str,
    to_token_ranges: list[tuple[int, int]],
    simplify_result: bool = True):
  result: list[int] = []
  token_mappings = [(int(pair[0]), int(pair[1])) for pair in [pair.split('-') for pair in output.strip().split(' ')]]
  token_mappings.sort(key=lambda pair: pair[0])
  for (from_token, to_token) in token_mappings:
    [from_start, from_end] = from_token_ranges[from_token]
    [to_start, to_end] = to_token_ranges[to_token]

    print_alignment(from_text, from_start, from_end, to_text, to_start, to_end)
    result += [from_start, from_end, to_start, to_end]

  assert len(result) % 4 == 0
  return simplify(result, from_text, to_text) if simplify_result else result

def align(
    from_language: str,
    from_text: str,
    to_language: str,
    to_text: str,
    model: str = 'bert-base-multilingual-cased',
    simplify_result: bool = True):
  with NamedTemporaryFile(mode='+w', prefix='awesome-', suffix='.tmp', dir='.') as input_file, \
       NamedTemporaryFile(mode='+w', prefix='awesome-', suffix='.tmp', dir='.') as output_file:

    from_token_ranges = get_token_ranges(from_language, from_text)
    to_token_ranges = get_token_ranges(to_language, to_text)

    input_str = build_input(from_text, from_token_ranges, to_text, to_token_ranges)
    print(input_str, file=input_file, flush=True)

    run_awesome(model, input_file.name, output_file.name)

    return parse_output(
      output_file.readline(),
      from_text,
      from_token_ranges,
      to_text,
      to_token_ranges,
      simplify_result)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--from-language', type=str, required=True, choices=TOKENIZERS.keys())
  parser.add_argument('--from-text', type=str, required=True)
  parser.add_argument('--to-language', type=str, required=True, choices=TOKENIZERS.keys())
  parser.add_argument('--to-text', type=str, required=True)
  parser.add_argument('--model', type=str, default='bert-base-multilingual-cased')
  parser.add_argument('--no-simplify', action='store_true', default=False)
  args = parser.parse_args()

  result = align(
    args.from_language,
    args.from_text,
    args.to_language,
    args.to_text,
    args.model,
    not args.no_simplify)

  print(','.join(str(i) for i in result))

