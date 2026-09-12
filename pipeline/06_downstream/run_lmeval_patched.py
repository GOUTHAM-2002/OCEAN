"""Wrapper around lm-eval CLI that forces resize_token_embeddings(mean_resizing=False).
The World Tour adapters added a pad token (vocab 32000->32001); transformers 4.46
defaults mean_resizing=True, which computes a full embedding covariance to init the
one new row (minutes of CPU, and the adapter overwrites that row anyway). Forcing it
off makes adapter load instant. Identical eval results (the pad row is never scored)."""
import transformers
_orig = transformers.PreTrainedModel.resize_token_embeddings
def _patched(self, *a, **k):
    k.setdefault("mean_resizing", False)
    return _orig(self, *a, **k)
transformers.PreTrainedModel.resize_token_embeddings = _patched

from lm_eval.__main__ import cli_evaluate
cli_evaluate()
