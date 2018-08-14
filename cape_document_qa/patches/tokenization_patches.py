# Copyright 2018 BLEMUNDSBURY AI LIMITED
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List, Tuple
from docqa.data_processing import text_utils

dash_re = re.compile("-|\u2212|\u2014|\u2013")
patched_extra_split_chars = ("-", "£", "€", "¥", "¢", "₹", "\u2212", "\u2014", "\u2013", "/", "~",
                     '"', "'", "\ud01C", "\u2019", "\u201D", "\u2018", "\u00B0", '.')

"""We patch to ensure dashes are handled properly, as the original author folds dashes together, beaking the original"""

def convert_to_spans(raw_text: str, sentences: List[List[str]]) -> List[List[Tuple[int, int]]]:
    """ Convert a tokenized version of `raw_text` into a series character spans referencing the `raw_text` """
    cur_idx = 0
    all_spans = []
    for sent in sentences:
        spans = []
        for token in sent:
            # (our) Tokenizer might transform double quotes, for this case search over several
            # possible encodings
            if text_utils.double_quote_re.match(token):
                span = text_utils.double_quote_re.search(raw_text[cur_idx:])
                tmp = cur_idx + span.start()
                l = span.end() - span.start()
            elif dash_re.match(token):
                span = dash_re.search(raw_text[cur_idx:])
                tmp = cur_idx + span.start()
                l = span.end() - span.start()
            else:
                tmp = raw_text.find(token, cur_idx)
                l = len(token)
            if tmp < cur_idx:
                raise ValueError(token)
            cur_idx = tmp
            spans.append((cur_idx, cur_idx + l))
            cur_idx += l
        all_spans.append(spans)
    return all_spans


text_utils.NltkAndPunctTokenizer.convert_to_spans = staticmethod(convert_to_spans)
patched_extra_split_tokens = ("``",
                      "(?<=[^_])_(?=[^_])",  # dashes w/o a preceeding or following dash, so __wow___ -> ___ wow ___
                      "''", "[" + "".join(patched_extra_split_chars) + "]")
text_utils.extra_split_chars_re = re.compile("(" + "|".join(patched_extra_split_tokens) + ")")
