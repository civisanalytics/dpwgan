import pandas as pd

import dpwgan


def test_CategoricalDataset():
    df = pd.DataFrame({'foo': ['spam', 'eggs', 'sausage', 'spam'],
                       'bar': ['purple', 'purple', 'rain', 'rain']})
    dataset = dpwgan.CategoricalDataset(df)
    tensor = dataset.to_onehot_flat()
    df_new = dataset.from_onehot_flat(tensor)
    assert df_new.equals(df)
