import itertools

def unique_perms(series):
    return {p for p in itertools.permutations(series)}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
