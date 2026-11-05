"""Prompt building and tokenization, shared by both stages (README 6.1 and 6.4).

Every script that turns a FinQA question into model input must go through
`encode`, so that token indices mean the same thing in every file.
"""

import re

# README 6.1. `{context}` is the report: pre_text, table, post_text.
PROMPT_TEMPLATE = (
    "You are a financial analyst. Use only the report below.\n"
    "\n"
    "### Report\n"
    "{context}\n"
    "\n"
    "### Question\n"
    "{question}\n"
    "\n"
    "List the numbers from the report that you use, then give the final answer.\n"
    "Answer:"
)

# Everything before the report, used to find where the report starts.
_PROMPT_PREFIX = PROMPT_TEMPLATE.split("{context}")[0]


def _is_junk(line):
    """FinQA has many text lines that are only '.' (or only dots); drop them."""
    return re.search(r"[A-Za-z0-9]", line) is None


def table_to_text(table):
    """One row per line, cells joined with ' | ' (README 6.1)."""
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in table)


def build_context(pre_text, table, post_text):
    """The report as one string: pre_text lines, the table, post_text lines.

    `pre_text` and `post_text` are FinQA's lists of sentences. Each sentence
    goes on its own line; lines with no letters or digits are dropped.
    """
    parts = [line.strip() for line in pre_text if not _is_junk(line)]
    parts.append(table_to_text(table))
    parts += [line.strip() for line in post_text if not _is_junk(line)]
    return "\n".join(p for p in parts if p)


def build_prompt(question, context):
    """The README 6.1 prompt as plain text (before any chat template)."""
    return PROMPT_TEMPLATE.format(context=context, question=question.strip())


def to_model_text(prompt, tokenizer, chat_template=True):
    """The exact string the model reads.

    With `chat_template=True` the prompt is the user turn of the model's chat
    template, followed by the start of the assistant turn. The generated
    answer is appended directly after this string.
    """
    if not chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokens_in_span(offsets, char_start, char_end):
    """Indices of the tokens that overlap the characters [char_start, char_end).

    `offsets` is a fast tokenizer's offset_mapping: one (start, end) per token.
    """
    return [i for i, (s, e) in enumerate(offsets) if e > s and s < char_end and e > char_start]


def encode(tokenizer, question, context, chat_template=True):
    """Tokenize one prompt and locate the report inside it.

    Returns a dict with
      text       the exact model input string
      input_ids  its token ids
      ctx_start, ctx_end
                 token span of the report, half-open: input_ids[ctx_start:ctx_end]
                 are the context tokens of README 6.4. A token that straddles
                 the edge of the report (for example "." merged with the
                 following newlines) counts as context.
      ctx_chars  (start, end) character span of the report in `text`
    """
    prompt = build_prompt(question, context)
    text = to_model_text(prompt, tokenizer, chat_template)

    pos = text.find(prompt)
    if pos < 0 or text.find(prompt, pos + 1) >= 0:
        raise ValueError("the prompt must appear exactly once in the model text")
    char_start = pos + len(_PROMPT_PREFIX)
    char_end = char_start + len(context)

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    in_context = tokens_in_span(offsets, char_start, char_end)
    if not in_context:
        raise ValueError("no tokens found inside the report")
    return {
        "text": text,
        "input_ids": ids,
        "ctx_start": in_context[0],
        "ctx_end": in_context[-1] + 1,
        "ctx_chars": (char_start, char_end),
    }
