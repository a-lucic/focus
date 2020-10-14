import numpy as np
import pandas as pd

# Currently this only works for numerical data.

def read_tsv_file(input_file, negative_is_missing=False):
  dataset = pd.read_csv(input_file, sep='\t', index_col=0)
  columns = dataset.columns
  np_values = dataset.values.astype(np.float64)
  
  if negative_is_missing:
    not_missing_mask = np.greater_equal(np_values, 0.)
  else:
    not_missing_mask = np.ones_like(np_values)

  return columns, np_values, not_missing_mask


def get_indices(input_file):
    dataset = pd.read_csv(input_file, sep='\t')
    indices = np.array(dataset.id)

    return indices

