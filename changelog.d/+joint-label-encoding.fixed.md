- UCR/UEA classification labels are now encoded to a shared `0..K-1` space fitted on train U test; `num_classes` counts all classes and is restored from cache on every rank.

  **Behaviour change:** `num_classes` semantics changed (it now counts classes across train U test, not just train), and UCR raw label values changed to `0..K-1` codes instead of the raw ARFF values.

  **Action required:** delete any existing UCR/UEA cache directories before upgrading. The cache key was not changed for this fix, so a stale cache written by an older version will raise `KeyError: 'num_classes'` on `setup()` instead of being regenerated automatically.
