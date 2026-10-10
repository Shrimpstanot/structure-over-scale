"""Regression tests for the Axolotl 0.13 dataset-lock compatibility patch."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from axolotl_lock_patch import (  # noqa: E402
    PATCH_MARKER,
    install_axolotl_lock_cleanup_patch,
)


def fake_axolotl_modules(loader_cls: type) -> dict[str, types.ModuleType]:
    axolotl = types.ModuleType("axolotl")
    utils = types.ModuleType("axolotl.utils")
    data = types.ModuleType("axolotl.utils.data")
    lock = types.ModuleType("axolotl.utils.data.lock")
    lock.FileLockLoader = loader_cls
    return {
        "axolotl": axolotl,
        "axolotl.utils": utils,
        "axolotl.utils.data": data,
        "axolotl.utils.data.lock": lock,
    }


class AxolotlLockPatchTest(unittest.TestCase):
    def test_missing_counter_is_ignored_and_install_is_idempotent(self):
        class Loader:
            def cleanup(self):
                raise FileNotFoundError("process_counter.txt")

        with patch.dict(sys.modules, fake_axolotl_modules(Loader)):
            install_axolotl_lock_cleanup_patch()
            installed_cleanup = Loader.cleanup
            install_axolotl_lock_cleanup_patch()

            self.assertIs(Loader.cleanup, installed_cleanup)
            self.assertTrue(getattr(Loader.cleanup, PATCH_MARKER))
            self.assertIsNone(Loader().cleanup())

    def test_successful_cleanup_return_value_is_preserved(self):
        result = object()

        class Loader:
            def cleanup(self):
                return result

        with patch.dict(sys.modules, fake_axolotl_modules(Loader)):
            install_axolotl_lock_cleanup_patch()
            self.assertIs(Loader().cleanup(), result)

    def test_non_file_not_found_errors_are_not_hidden(self):
        class Loader:
            def cleanup(self):
                raise ValueError("invalid process counter")

        with patch.dict(sys.modules, fake_axolotl_modules(Loader)):
            install_axolotl_lock_cleanup_patch()
            with self.assertRaisesRegex(ValueError, "invalid process counter"):
                Loader().cleanup()


if __name__ == "__main__":
    unittest.main()
