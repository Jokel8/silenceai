# Backwards-compatibility shim: original filename contained a typo. Use `test-controller.py` (dash) or `test_controller.py` (underscore) instead.
# Try importing the corrected module name, fallback to sys.path import if needed.
try:
    from test_controller import *  # relative import may fail in some contexts
except Exception:
    try:
        from .test_controller import *
    except Exception:
        # last resort: print message when executed directly
        print("Please run 'test-controller.py' or 'test_controller.py' instead of this file.")
