import torch
import torch.nn.functional as F


class MultiCategoryGumbelSoftmax(torch.nn.Module):
    """Gumbel softmax for multiple output categories

    Parameters
    ----------
    input_dim : int
        Dimension for input layer
    output_dims : list of int
        Dimensions of categorical output variables
    tau : float
        Temperature for Gumbel softmax
    """
    def __init__(self, input_dim, output_dims, tau=2/3):
        super(MultiCategoryGumbelSoftmax, self).__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(input_dim, output_dim)
            for output_dim in output_dims
        )
        self.tau = tau

    def forward(self, x):
        xs = tuple(layer(x) for layer in self.layers)
        logits = tuple(F.log_softmax(x, dim=1) for x in xs)
        categorical_outputs = tuple(
            F.gumbel_softmax(logit, tau=self.tau, hard=True, eps=1e-10)
            for logit in logits
        )
        return torch.cat(categorical_outputs, 1)
