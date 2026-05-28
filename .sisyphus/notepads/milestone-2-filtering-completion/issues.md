## 2026-05-28
- Pyright initially treated `BaseSource.read()` as returning `None` even after adding a tuple annotation while the body used `pass`.
- `tests/test_sources.py` needed an explicit non-None assertion before accessing `frame.shape`.
