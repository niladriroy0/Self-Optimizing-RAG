import time


class LatencyTracker:
    def __init__(self):
        self.times = {}

    def start(self, key):
        self.times[key] = time.time()

    def end(self, key):
        if key in self.times:
            self.times[key] = time.time() - self.times[key]

    def get_all(self):
        return self.times