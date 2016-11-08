from numpy import array, nan, zeros, eye, dot


class Rejection():
    def __init__(self, val, rank=1, type_str='unknown', topographies=None):
        """
        :param val: np.array
        :param args: np.array args
        :param rank: rank of rejection
        :param type_str: source name of rejection
        :param topographies:  np.array with dim = (n x rank). It contains rank topography vectors with dim = n
                              (usually number of channels). If it's None, nan array will be created.
        :param kwargs: np.array kwargs
        """
        self.val = array(val)
        self.type_str = type_str
        self.rank = rank
        if topographies is not None:
            topographies = array(topographies)
            if topographies.ndim == 1:
                topographies = topographies.reshape((topographies.shape[0], 1))
            assert topographies.shape == (self.val.shape[0], self.rank), \
                'Bad topographies shape {}. Expected {}.'.format(topographies.shape, (self.val.shape[0], self.rank))
            self.topographies = topographies
        else:
            self.topographies = nan * zeros((self.val.shape[0], self.rank))



class Rejections():
    def __init__(self, n_channels, rejections_list=None, ica=None):
        self.n = n_channels
        self.has_ica = ica is not None
        self.list = [ica] if self.has_ica else []
        self.list += rejections_list or []

    def get_prod(self):
        prod = eye(self.n)
        for rejection in self.list:
            prod = dot(prod, rejection.val)
        return prod

    def get_list(self):
        return [rejection.val for rejection in self.list]

    def update_list(self, rejections, append=False):
        if append:
            self.list += rejections
        else:
            self.list = (self.list[0] if self.has_ica else []) + rejections

    def update_ica(self, ica):
        if self.has_ica:
            self.list[0] = ica
        else:
            self.has_ica = ica is not None
            self.list = ([ica] if self.has_ica else []) + self.list

    def drop(self, ind):
        if ind == 0:
            self.has_ica = False
        self.list.pop(ind)

    def __len__(self):
        return len(self.list)

    def __repr__(self):
        rep = 'Rejections object (dim = {}):\n'.format(self.n)
        rep += '\n'.join(['\t{}. rank: {}\ttype: {}\t'.format(j+1, rejection.rank, rejection.type_str)
                   for j, rejection in enumerate(self.list)]) if len(self)>0 else '\tempty'
        return rep


if __name__ == '__main__':
    a = Rejections(3, ica=Rejection(eye(3)*4), rejections_list=[Rejection(eye(3)*2)])
    print(a)

