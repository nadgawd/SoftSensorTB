"""Shared test setup."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# backend.database builds its engine at import time from DATABASE_URL, which
# .env points at the developer's real database. Tests get a throwaway one.
_TEST_DB = Path(tempfile.mkdtemp(prefix="sst_test_")) / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"
