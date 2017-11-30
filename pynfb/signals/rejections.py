from numpy import eye, dot
from ..signal_processing.filters import SpatialRejection


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

    def expand_by_mask(self, mask):
        list = [rej.expand_by_mask(mask) for rej in self.list]
        n = len(mask)
        return Rejections(n, list[1:] if self.has_ica else list, ica=list[0] if self.has_ica else None)

    def shrink_by_mask(self, mask):
        list = [rej.shrink_by_mask(mask) for rej in self.list]
        n = sum(mask)
        return Rejections(n, list[1:] if self.has_ica else list, ica=list[0] if self.has_ica else None)

    def __len__(self):
        return len(self.list)

    def __repr__(self):
        rep = 'Rejections object (dim = {}):\n'.format(self.n)
        rep += '\n'.join(['\t{}. rank: {}\ttype: {}\t'.format(j+1, rejection.rank, rejection.type_str)
                   for j, rejection in enumerate(self.list)]) if len(self)>0 else '\tempty'
        return rep


if __name__ == '__main__':
    a = Rejections(3, ica=SpatialRejection(eye(3) * 4), rejections_list=[SpatialRejection(eye(3) * 2)])
    print(a)

