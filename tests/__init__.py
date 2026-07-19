# Real package on purpose: yahoo_fantasy_api pollutes site-packages with a top-level
# 'tests' package; this one must win the import so `from tests.conftest import ...` works.
