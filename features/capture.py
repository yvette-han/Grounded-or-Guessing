"""Attention rows one layer at a time, to fit small GPUs (shared by B0, B1-B2 and R2).

`model(..., output_attentions=True)` keeps every layer's full attention map
[heads, seq, seq] until the pass ends (~12 GB in float32 for a 3,000-token
prompt + answer on Qwen2.5-1.5B), and the LM head adds a [seq, vocab] logits
tensor (~1.8 GB). Here a forward hook on each layer's attention module copies
only the rows we need as soon as that layer has computed them, the full map is
freed before the next layer runs, and the pass stops before the LM head. Peak
extra memory is about one layer's attention map (~0.4 GB at 3,000 tokens).

Needs the model loaded with attn_implementation="eager" (only eager attention
returns the weights). The rows are the same numbers output_attentions gives
(tests/test_capture.py checks this on a tiny random Qwen2 model).
"""

import numpy as np


def _decoder(model):
    """The transformer stack without the LM head (Qwen2ForCausalLM.model, etc.)."""
    get = getattr(model, "get_decoder", None)
    dec = get() if callable(get) else None
    return dec if dec is not None else model.model


def attention_rows(model, input_ids, rows):
    """Attention of the given query rows, every layer and head.

    input_ids: LongTensor [1, seq] on the model's device. rows: list of query positions.
    Returns float32 numpy [layers, heads, len(rows), seq].
    """
    import torch

    decoder = _decoder(model)
    layers = decoder.layers
    rows_t = torch.as_tensor(rows, dtype=torch.long, device=input_ids.device)
    kept = [None] * len(layers)

    def hook_for(i):
        def hook(_module, _inputs, output):
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if weights is None:
                raise RuntimeError("attention weights not returned: load the model with "
                                   "attn_implementation='eager'")
            kept[i] = weights[0].index_select(1, rows_t).float().cpu()  # [heads, n_rows, seq]
        return hook

    handles = [layer.self_attn.register_forward_hook(hook_for(i)) for i, layer in enumerate(layers)]
    try:
        with torch.inference_mode():
            decoder(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(kept).numpy().astype(np.float32, copy=False)
