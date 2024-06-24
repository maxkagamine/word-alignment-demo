'''
An algorithm for merging word alignments in a way that produces the most compact
representation possible, such that for any selected span of the "from" text, the
resulting aligned span(s) in the "to" text remains the same, and vice versa.

Two alignments are eligible to be merged if:

- One is entirely contained within the other on both sides; or
- Both point to the same span on one side and are overlapping, adjacent, or
  separated by only whitespace on the other.

WARNING: The time complexity of this algorithm is O(scary). Even on a fairly
powerful machine, a non-trivial aligned sentence pair took 4.5 min to simplify.
I attempted to reduce the number of iterations necessary by breaking the problem
into smaller sub-graphs of intersecting alignments but couldn't get acceptable
results. There may be a way to optimize this, but I've spent too much time on it
already, so I've instead opted for a simpler, repeated-forward-pass approach
which, while it doesn't produce the optimal result in all cases, is good enough
for real-world sentence pairs.

Examples:

  a-ab b-ab c-ab ab-c c-abc          a-ab a-bc b-bc c-bc d-bc b-abc

  ab-ab c-ab ab-c c-abc              a-abc b-abc b-bc c-bc d-bc
  a-ab bc-ab ab-c c-abc              a-ab ab-bc b-abc c-bc d-bc
  a-ab b-ab ab-c c-abc               a-ab a-bc b-abc bc-bc d-bc
                                     a-ab a-bc b-abc c-bc d-bc
  abc-ab ab-c c-abc                  a-ab a-bc b-abc b-bc cd-bc
  ab-abc c-ab c-abc
  ab-ab c-abc ab-c                   ab-abc b-bc c-bc d-bc
                                     a-abc b-abc bc-bc d-bc
  abc-abc c-ab                       a-abc b-abc c-bc d-bc
  ab-abc c-abc                       a-abc b-abc b-bc cd-bc
                                     a-ab abc-bc b-abc d-bc
  abc-abc                            a-ab a-bc b-abc bcd-bc
                                     a-ab a-bc b-abc cd-bc
                                     a-ab ab-bc b-abc cd-bc

                                     ab-abc bc-bc d-bc
                                     a-abc b-abc bcd-bc
                                     ab-abc c-bc d-bc
                                     a-abc b-abc cd-bc
                                     ab-abc b-bc cd-bc
                                     a-ab abcd-bc b-abc

                                     ab-abc bcd-bc
                                     ab-abc cd-bc

Running those examples from the REPL (uncomment the print statements below):

  simplify([0,1,0,2,1,2,0,2,2,3,0,2,0,2,2,3,2,3,0,3], 'abcd', 'abcd')
  simplify([0,1,0,2,0,1,1,3,1,2,1,3,2,3,1,3,3,4,1,3,1,2,0,3], 'abcd', 'abcd')
'''
from collections import deque
import re

def simplify(alignments: list[int], from_text: str, to_text: str) -> list[int]:
  best = group_alignments(alignments)
  queue = deque([best])

  # Perform a breadth-first search to look for the most compact representation
  # “I'm sure this is fine, it's only, like, O(n!ⁿꜝ).” --Famous last words
  while len(queue) > 0:
    current = queue.popleft()
    # print(_debug(current, from_text, to_text))

    # Iterate over every combination of two alignment groups
    for i, left in enumerate(current[:len(current) - 1]):
      for right in current[i + 1:]:
        # See if these groups can be merged
        merged = merge_alignments(left, right, from_text, to_text)
        if merged is None:
          continue

        # Replace the two groups with the merged one
        simplified = [*(x for x in current if x != left and x != right), merged]
        simplified.sort()

        # Skip if already in the queue
        if simplified in queue:
          continue

        # Check if the new alignments is better
        if compare(simplified, best) < 0:
          best = simplified

        # Even if not, we may still get a better result continuing down this
        # path than with the current-best
        queue.append(simplified)
        # print('  ' + _debug(simplified, from_text, to_text))

  return ungroup_alignments(best)

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

def compare(a: list[tuple[int, int, int, int]], b: list[tuple[int, int, int, int]]) -> int:
  if len(a) < len(b):
    return -1
  if len(a) > len(b):
    return 1
  # Prefer shorter spans, just for cleaner visualizations
  aSpanLen = sum((x[1] - x[0]) + (x[3] - x[2]) for x in a)
  bSpanLen = sum((x[1] - x[0]) + (x[3] - x[2]) for x in b)
  return aSpanLen - bSpanLen

def group_alignments(alignments: list[int]) -> list[tuple[int, int, int, int]]:
  return [tuple(alignments[i*4:(i+1)*4]) for i in range(len(alignments)//4)]

def ungroup_alignments(alignments: list[tuple[int, int, int, int]]) -> list[int]:
  return [x for group in alignments for x in group]

def _debug(alignments: list[tuple[int, int, int, int]], from_text: str, to_text: str) -> str:
  return ' '.join([
    f'{from_text[from_start:from_end]}-{to_text[to_start:to_end]}'
    for [from_start, from_end, to_start, to_end] in alignments
  ])
