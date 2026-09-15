- UCR/UEA classification labels are now encoded to a shared `0..K-1` space fitted on train ∪ test; `num_classes` counts all classes and is restored from cache on every rank. Existing UCR/UEA caches are regenerated automatically.

  **Behaviour change:** `num_classes` semantics changed (it now counts classes across train ∪ test, not just train), and UCR raw label values changed to `0..K-1` codes instead of the raw ARFF values.
