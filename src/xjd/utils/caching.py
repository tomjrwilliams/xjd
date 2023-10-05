

CACHE: dict = {}
CACHE_ORDER: dict = {}

# because vs code ikernel doesn't like functools.lru_cache
def lru_cache(size=None, drop_keys = False, key_order = None):
    def decorator(f):
        def f_decorated(*args, **kwargs):
            key = tuple(
                (args, *(tuple(kv) for kv in kwargs.items()))
            )
            if f not in CACHE:
                CACHE[f] = {}
            if key not in CACHE[f]:
                print("{} miss:".format(f.__name__), key)
                CACHE[f][key] = f(*args, **kwargs)
            return CACHE[f][key]
        return f_decorated
    return decorator
