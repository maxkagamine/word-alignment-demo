'''
An algorithm for merging word alignments, such that for any selected span of the
"from" text, the resulting aligned span(s) in the "to" text remains the same,
and vice versa.

Two alignments are eligible to be merged if:

- One is entirely contained within the other on both sides; or
- Both point to the same span on one side and are overlapping, adjacent, or
  separated by only whitespace on the other.

This doesn't produce the optimal result in all cases (it fails the "abc-abc"
example in simplify_slow.py, for instance), but it's good enough for real-world
sentence pairs and executes in milliseconds rather than minutes.
'''
import re

def simplify(alignments: list[int], from_text: str, to_text: str) -> list[int]:
  # Group into tuples, remove duplicates from symmetrizing, and sort
  result = list(set(group_alignments(alignments)))
  result.sort()

  removed = set()
  modified_list = True

  # _debug(result, from_text, to_text)

  while modified_list:
    modified_list = False

    for i, current in enumerate(result):
      if current in removed:
        continue

      for other in result[i + 1:]:
        if other in removed:
          continue

        merged = merge_alignments(current, other, from_text, to_text)
        if merged is None:
          continue

        result[i] = current = merged
        removed.add(other)
        modified_list = True

        # _debug([x for x in result if not x in removed], from_text, to_text)

    result = [x for x in result if not x in removed]
    removed.clear()

  return ungroup_alignments(result)

def merge_alignments(
  left: tuple[int, int, int, int],
  right: tuple[int, int, int, int],
  from_text: str,
  to_text: str) -> tuple[int, int, int, int] | None:

  # Check if left is entirely contained within right
  if left[0] >= right[0] and left[1] <= right[1] and \
     left[2] >= right[2] and left[3] <= right[3]:
    return right

  # Check if right is entirely contained within left
  if right[0] >= left[0] and right[1] <= left[1] and \
     right[2] >= left[2] and right[3] <= left[3]:
    return left

  # Check if 'from' is the same and 'to' is overlapping/adjacent
  if left[0] == right[0] and left[1] == right[1] and \
     is_overlapping_or_adjacent(left[2], left[3], right[2], right[3], to_text):
    return (left[0], left[1], min(left[2], right[2]), max(left[3], right[3]))

  # Check if 'to' is the same and 'from' is overlapping/adjacent
  if left[2] == right[2] and left[3] == right[3] and \
     is_overlapping_or_adjacent(left[0], left[1], right[0], right[1], from_text):
    return (min(left[0], right[0]), max(left[1], right[1]), left[2], left[3])

  return None

def is_overlapping_or_adjacent(start1: int, end1: int, start2: int, end2: int, text: str) -> bool:
  '''
  Returns true if [start1,end1) overlaps with, touches, or is separated only by
  whitespace from [start2,end2).
  '''
  return (end1 < start2 and not re.search(r'\S', text[end1:start2])) or \
         (start1 <= end2 and end1 >= start2) or \
         (start1 > end2 and not re.search(r'\S', text[end2:start1]))

def group_alignments(alignments: list[int]) -> list[tuple[int, int, int, int]]:
  return [tuple(alignments[i*4:(i+1)*4]) for i in range(len(alignments)//4)]

def ungroup_alignments(alignments: list[tuple[int, int, int, int]]) -> list[int]:
  return [x for group in alignments for x in group]

def _debug(alignments: list[tuple[int, int, int, int]], from_text: str, to_text: str):
  print(' '.join([
    f'{from_text[from_start:from_end]}-{to_text[to_start:to_end]}'
    for [from_start, from_end, to_start, to_end] in alignments
  ]))
