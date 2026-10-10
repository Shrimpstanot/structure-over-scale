"""Install runtime compatibility patches before Axolotl starts training."""

import runpy

from axolotl_lock_patch import install_axolotl_lock_cleanup_patch
from quanta_axolotl import install_quanta_loader_patch


install_axolotl_lock_cleanup_patch()
install_quanta_loader_patch()
runpy.run_module("axolotl.cli.train", run_name="__main__")
