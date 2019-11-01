import pandas as pd
import torch

from dpwgan import DPWGAN, MultiCategoryGumbelSoftmax


def create_categorical_gan(noise_dim, hidden_dim, output_dims):
    generator = torch.nn.Sequential(
        torch.nn.Linear(noise_dim, hidden_dim),
        torch.nn.ReLU(),
        MultiCategoryGumbelSoftmax(hidden_dim, output_dims)
    )

    discriminator = torch.nn.Sequential(
        torch.nn.Linear(sum(output_dims), hidden_dim),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(hidden_dim, 1)
    )

    def noise_function(n):
        return torch.randn(n, noise_dim)

    gan = DPWGAN(
        generator=generator,
        discriminator=discriminator,
        noise_function=noise_function
    )

    return gan


def percentage_crosstab(variable_one, variable_two):
    return 100*pd.crosstab(variable_one, variable_two).apply(
        lambda r: r/r.sum(), axis=1
    )
