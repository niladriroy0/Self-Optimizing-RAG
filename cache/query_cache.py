import hashlib

CACHE = {}

def get_cache_key(query):
    return hashlib.md5(query.encode()).hexdigest()


def get_cached(query):
    key = get_cache_key(query)

    if key in CACHE:
        print("⚡ Exact Cache Hit")
        return CACHE[key]

    return None


def set_cache(query, response):
    key = get_cache_key(query)
    CACHE[key] = response