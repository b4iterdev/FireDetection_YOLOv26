## 2026-05-28
- Abstract input source methods need explicit return annotations and ellipsis bodies for pyright to treat overrides correctly.
- Returning `np.ndarray | None` from source `read()` methods lets pyright narrow `frame` after `assert frame is not None`.
