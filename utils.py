from functools import wraps

def timeit(func):
    """ Python decorator for timing function execution """
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time.time()
        ret = func(*args,**kwargs)
        end = time.time()
        print(f'{end-start:.3f}s taken for {func.__name__}')
        return ret
    return wrapper
