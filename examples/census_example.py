"""
This script generates synthetic data based on ACS PUMS data.
Run `create_census_data.py` to download and create the input data set
before running this script.
"""

import logging
import os

import pandas as pd
import torch

from dpwgan import CategoricalDataset
from dpwgan.utils import create_categorical_gan
from create_census_data import CENSUS_FILE

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

NOISE_DIM = 20
HIDDEN_DIM = 20
SIGMA = 1


def main():
    torch.manual_seed(123)
    # set logging level to INFO to display training
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Preparing data set...')
    try:
        census = pd.read_csv(CENSUS_FILE, dtype=str)
    except FileNotFoundError:
        print('Error: Census data file does not exist.\n'
              'Please run `create_census_data.py` first.')
        return
    census = census.fillna('N/A')
    dataset = CategoricalDataset(census)
    data = dataset.to_onehot_flat()

    gan = create_categorical_gan(NOISE_DIM, HIDDEN_DIM, dataset.dimensions)

    logger.info('Training GAN...')
    gan.train(data=data,
              epochs=50,
              n_critics=5,
              learning_rate=1e-4,
              weight_clip=1/HIDDEN_DIM,
              sigma=SIGMA)

    logger.info('Generating synthetic data...')
    flat_synthetic_data = gan.generate(len(census))
    synthetic_data = dataset.from_onehot_flat(flat_synthetic_data)

    filename = os.path.join(THIS_DIR, 'synthetic_pums_il.csv')
    with open(filename, 'w') as f:
        synthetic_data.to_csv(f, index=False)

    logger.info('Synthetic data saved to {}'.format(filename))


if __name__ == '__main__':
    main()
