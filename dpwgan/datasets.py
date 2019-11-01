import collections

import numpy as np
import pandas as pd
import torch


def to_onehot(data, codes):
    indices = [np.where(codes == val)[0][0] for val in data]
    indices = torch.LongTensor(list([val] for val in indices))
    onehot = torch.FloatTensor(indices.size(0), len(codes)).zero_()
    onehot.scatter_(1, indices, 1)
    return onehot


def from_onehot(data, codes):
    return codes[[np.where(data[i] == 1)[0][0] for i in range(len(data))]]


class CategoricalDataset(object):
    """Class to convert between pandas DataFrame with categorical variables
    and a torch Tensor with onehot encodings of each variable

    Parameters
    ----------
    data : pandas.DataFrame
    """
    def __init__(self, data):
        self.data = data
        self.codes = collections.OrderedDict(
            (var, np.unique(data[var])) for var in data
        )
        self.dimensions = [len(code) for code in self.codes.values()]

    def to_onehot_flat(self):
        """Returns a torch Tensor with onehot encodings of each variable
        in the original data set

        Returns
        -------
        torch.Tensor
        """
        return torch.cat([to_onehot(self.data[var], code)
                          for var, code
                          in self.codes.items()], 1)

    def from_onehot_flat(self, data):
        """Converts from a torch Tensor with onehot encodings of each variable
        to a pandas DataFrame with categories

        Parameters
        ----------
        data : torch.Tensor

        Returns
        -------
        pandas.DataFrame
        """
        categorical_data = pd.DataFrame()
        index = 0
        for var, code in self.codes.items():
            var_data = data[:, index:(index+len(code))]
            categorical_data[var] = from_onehot(var_data, code)
            index += len(code)
        return categorical_data
