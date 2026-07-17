"""
textenc.py — frozen T5 encoder for lyric conditioning.

T5 is FROZEN. We never train it, so encoding is a one-time cost paid into the
disk cache. t5-base (d=768) over t5-small (d=512): lyrics are long and
semantically dense, and since it costs no training compute the larger model is
free quality.

The attention mask must be propagated all the way into cross-attention. If the
model attends to padding it learns to ignore lyrics entirely -- a silent
failure that looks like "conditioning just doesn't work".
"""

import torch
from transformers import AutoTokenizer, T5EncoderModel

from dconfig import DCFG as CFG


class LyricEncoder:
    def __init__(self, model_id: str = None, device: str = "cuda",
                 max_len: int = None):
        self.model_id = model_id or CFG.t5_id
        self.device = device
        self.max_len = max_len or CFG.text_max_len

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.enc = T5EncoderModel.from_pretrained(self.model_id).to(device).eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.d_text = self.enc.config.d_model

    @torch.no_grad()
    def encode(self, texts):
        """
        list[str] -> (ctx (B,L,d_text) float32, mask (B,L) bool)
        mask True = real token. Trailing pad columns are trimmed.
        """
        if isinstance(texts, str):
            texts = [texts]
        b = self.tok(texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=self.max_len)
        ids = b["input_ids"].to(self.device)
        att = b["attention_mask"].to(self.device)
        out = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        return out.float(), att.bool()

    def __repr__(self):
        return f"LyricEncoder({self.model_id}, d_text={self.d_text}, max_len={self.max_len})"


if __name__ == "__main__":
    import transformers
    transformers.logging.set_verbosity_error()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    le = LyricEncoder(device=dev)
    print(le)

    ctx, msk = le.encode(["genre: blues\nslow guitar",
                          "genre: metal\nfast aggressive drums and distorted guitar"])
    print(f"ctx {tuple(ctx.shape)} {ctx.dtype} | mask {tuple(msk.shape)} "
          f"real tokens per example: {msk.sum(1).tolist()}")
    assert ctx.shape[-1] == le.d_text
    assert msk[:, 0].all(), "first token must be real (cross-attn NaN guard)"

    # Different text must produce different embeddings -- otherwise conditioning
    # is a no-op and no amount of training will fix it.
    a, _ = le.encode(["genre: blues"])
    b, _ = le.encode(["genre: metal"])
    n = min(a.shape[1], b.shape[1])
    d = (a[:, :n] - b[:, :n]).abs().mean().item()
    print(f"blues vs metal embedding delta: {d:.4f}")
    assert d > 1e-3, "distinct lyrics produced identical embeddings"

    # Same text twice must be deterministic (frozen + eval, no dropout)
    c, _ = le.encode(["genre: blues"])
    assert torch.allclose(a, c, atol=1e-5), "encoder is not deterministic"
    print("OK — frozen, deterministic, discriminative")
