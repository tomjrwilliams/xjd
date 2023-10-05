

import importlib.util
import sys

# ---------------------------------------------------------------

def import_from_file(name, loc, reload = False):
    spec = importlib.util.spec_from_file_location(name, loc)
    assert spec is not None, dict(name=name, loc=loc)
    assert spec.loader is not None, dict(name=name, loc=loc)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    
# ---------------------------------------------------------------
