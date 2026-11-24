"""features/capture.py must give exactly the rows output_attentions gives (tiny random Qwen2, CPU)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from features.capture import attention_rows  # noqa: E402


def tiny_qwen2():
    cfg = transformers.Qwen2Config(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).eval()


def test_attention_rows_match_output_attentions():
    model = tiny_qwen2()
    ids = torch.randint(0, 97, (1, 23))
    rows = [5, 6, 17, 22]
    with torch.inference_mode():
        full = model(input_ids=ids, output_attentions=True).attentions
    expected = torch.stack([a[0][:, rows, :] for a in full]).float().numpy()
    got = attention_rows(model, ids, rows)
    assert got.shape == (3, 4, 4, 23)       # layers, heads, rows, seq
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(got.sum(-1), 1.0, atol=1e-5)


def test_attention_rows_removes_its_hooks():
    model = tiny_qwen2()
    attention_rows(model, torch.randint(0, 97, (1, 9)), [3])
    assert all(len(layer.self_attn._forward_hooks) == 0 for layer in model.model.layers)
