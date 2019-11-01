"""
This script generates synthetic data based on a toy categorical dataset
and prints crosstabs for the original and synthetic datasets.
"""

import pandas as pd
import torch
import logging

from dpwgan import CategoricalDataset
from dpwgan.utils import create_categorical_gan, percentage_crosstab

NOISE_DIM = 10
HIDDEN_DIM = 20
SIGMA = 1


def generate_data():
    df = pd.DataFrame(
        {'weather': ['sunny']*10000+['cloudy']*10000+['rainy']*10000,
         'status': ['on time']*8000+['delayed']*2000
         + ['on time']*3000+['delayed']*5000+['canceled']*2000
         + ['on time']*2000+['delayed']*4000+['canceled']*4000}
    )
    return df


def main():
    torch.manual_seed(123)
    # set logging level to INFO to display training
    logging.basicConfig(level=logging.INFO)
    real_data = generate_data()
    dataset = CategoricalDataset(real_data)
    data_tensor = dataset.to_onehot_flat()

    gan = create_categorical_gan(NOISE_DIM, HIDDEN_DIM, dataset.dimensions)
    gan.train(data=data_tensor,
              epochs=20,
              n_critics=5,
              batch_size=128,
              learning_rate=1e-3,
              weight_clip=1/HIDDEN_DIM,
              sigma=SIGMA)
    flat_synth_data = gan.generate(len(real_data))
    synth_data = dataset.from_onehot_flat(flat_synth_data)
    print('Real data crosstab:')
    print(percentage_crosstab(real_data['weather'], real_data['status']))
    print('Synthetic data crosstab:')
    print(percentage_crosstab(synth_data['weather'], synth_data['status']))


if __name__ == '__main__':
    main()
