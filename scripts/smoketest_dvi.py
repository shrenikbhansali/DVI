"""Compatibility wrapper for the historical smoketest script.

The full implementation now lives in :mod:`train_bestcase`. This module
provides the same CLI so existing instructions remain valid.
"""
from train_bestcase import main

if __name__ == "__main__":
    main()
