import time

class Reward:
    def __init__(self, signal, threshold=0.9, rate_of_increase=1):
        self.score = 0
        self.signal = signal
        self.threshold = threshold
        self.rate_of_increase = rate_of_increase
        self._increase_score = True
        self._increase_started_time = None
        pass

    def update(self):
        if self.signal.current_sample > self.threshold:
            if self._increase_score:
                self.score += 1
                print('Increase! Current:', self.score)
                self._increase_started_time = time.time()
                self._increase_score = False
            if time.time() - self._increase_started_time > self.rate_of_increase:
                self._increase_score = True
        pass

    def get_score(self):
        return self.score
