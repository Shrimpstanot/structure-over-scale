"""Runtime compatibility patch for Axolotl's dataset-lock cleanup.

Axolotl 0.13.0.dev0 calls ``FileLockLoader.cleanup()`` from a ``finally`` block.
If loading fails before the process counter is created, cleanup raises a second
``FileNotFoundError`` and can mask the original dataset-loading exception.

Newer Axolotl releases ignore that missing-file error. Apply the same narrow
behavior at runtime so the immutable training image does not need to be rebuilt
or have an entire site-packages module bind-mounted over it.
"""

from __future__ import annotations


PATCH_MARKER = "_structure_over_scale_missing_counter_safe"


def install_axolotl_lock_cleanup_patch() -> None:
    """Make Axolotl dataset-lock cleanup tolerate an absent counter file."""
    from axolotl.utils.data.lock import FileLockLoader

    original_cleanup = FileLockLoader.cleanup
    if getattr(original_cleanup, PATCH_MARKER, False):
        return

    def cleanup_missing_counter_safe(self):
        try:
            return original_cleanup(self)
        except FileNotFoundError:
            return None

    cleanup_missing_counter_safe.__name__ = original_cleanup.__name__
    cleanup_missing_counter_safe.__doc__ = original_cleanup.__doc__
    setattr(cleanup_missing_counter_safe, PATCH_MARKER, True)
    FileLockLoader.cleanup = cleanup_missing_counter_safe
