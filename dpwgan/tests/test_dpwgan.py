import numpy as np
import pandas as pd
import torch

from dpwgan import CategoricalDataset
from dpwgan.utils import create_categorical_gan, percentage_crosstab

NOISE_DIM = 5
HIDDEN_DIM = 10
SIGMA = 1


def generate_data():
    df = pd.DataFrame(
        {'weather': ['sunny']*10000+['cloudy']*10000,
         'status': ['on time']*8000+['delayed']*2000
         + ['on time']*4000+['delayed']*6000}
    )
    return df


def test_dpwgan():
    torch.manual_seed(123)
    real_data = generate_data()
    dataset = CategoricalDataset(real_data)
    data_tensor = dataset.to_onehot_flat()

    gan = create_categorical_gan(NOISE_DIM, HIDDEN_DIM, dataset.dimensions)
    gan.train(data=data_tensor,
              epochs=10,
              n_critics=5,
              batch_size=128,
              learning_rate=1e-3,
              weight_clip=1/HIDDEN_DIM,
              sigma=SIGMA)
    flat_synth_data = gan.generate(len(real_data))
    synth_data = dataset.from_onehot_flat(flat_synth_data)
    real = percentage_crosstab(real_data['weather'], real_data['status'])
    synth = percentage_crosstab(synth_data['weather'], synth_data['status'])
    # ensure difference in each cell of the cross tab is at most 5% (absolute)
    assert np.max(np.abs(real.values - synth.values)) < 5
