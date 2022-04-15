import time

class Reward:
    def __init__(self, signal_ind, threshold=0.9, rate_of_increase=0.25, fs=1000):
        self.score = 0
        self.fs = fs
        self.signal_ind = signal_ind
        self.threshold = threshold
        self.rate_of_increase = rate_of_increase
        self._increase_score = True
        self._increase_started_time = None
        self.enable = False
        self.n_acc = 0
        self.mean_acc = 0
        self.reward_factor = 1
        pass

    def update(self, sample, chunk_size):
        if self.enable:
            current_sample = sample * self.reward_factor
            if not isinstance(current_sample, float):
                current_sample = current_sample[0]
            if current_sample > self.threshold:
                self.score += chunk_size/self.fs/self.rate_of_increase
                # print(f"SCORE: {chunk_size/self.fs/self.rate_of_increase}, CHUNK: {chunk_size}, FS: {self.fs}, RATE: {self.rate_of_increase}, cSCORE: {self.score}, TH: {self.threshold}, CUR_SAMP: {current_sample}")
        pass

    def set_enabled(self, flag):
        self.enable = flag

    def get_score(self):
        return round(self.score)

    def reset(self):
        self.score = 0
