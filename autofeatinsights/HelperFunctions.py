import pandas as pd
import numpy as np

def get_df_with_prefix(dataset: str, node_id: str, targetColumn = None):
    prefix = dataset + "/" + node_id + "."
    if targetColumn:
        dataframe = pd.read_csv('data/benchmark/' + dataset + "/" + node_id).set_index(targetColumn).add_prefix(prefix).reset_index()
    else:
        dataframe = pd.read_csv('data/benchmark/' + dataset + "/" + node_id).add_prefix(prefix)
    return dataframe

def pearson_correlation(x, y):
    x_dev = x - np.mean(x, axis=0)
    y_dev = y - np.mean(y)
    sq_dev_x = x_dev * x_dev
    sq_dev_y = y_dev * y_dev
    sum_dev = y_dev.T.dot(x_dev).reshape((x.shape[1],))
    denominators = np.sqrt(np.sum(sq_dev_y) * np.sum(sq_dev_x, axis=0))

    results = np.array(
        [(sum_dev[i] / denominators[i]) if denominators[i] > 0.0 else 0 for i
         in range(len(denominators))])
    return results