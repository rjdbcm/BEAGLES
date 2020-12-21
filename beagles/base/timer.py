from time import perf_counter

class Timer(object):
    def __init__(self):
        self.timer = perf_counter

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs

    @classmethod
    def __call__(cls, func, *args, numerator=1):
        with cls() as t:
            x = func(*args)
        t = numerator / t.elapsed_secs
        return x, t