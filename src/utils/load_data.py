import pandas as pd
import numpy as np


def ini_od_dist(train_path):
    # find the most visited destination in train data
    df = pd.read_csv(train_path)
    num_trips = len(df)
    df['od'] = df.apply(lambda row: '%d_%d' % (row['ori'], row['des']), axis=1)
    df = df[['od', 'path']]
    df = df.groupby('od').count()
    df['path'] = df['path'] / num_trips
    print(df['path'].sum())
    return df.index.tolist(), df['path'].tolist()


def load_path_feature(path_feature_path):
    # 这样子等于多一个mask 就是走这条路你到不了那个地方
    path_feature = np.load(path_feature_path)
    path_feature_flat = path_feature.reshape(-1, path_feature.shape[2])
    path_feature_max, path_feature_min = np.max(path_feature_flat, 0), np.min(path_feature_flat, 0)
    print('path_feature', path_feature.shape)
    return path_feature, path_feature_max, path_feature_min


def load_link_feature(edge_path):
    edge_df = pd.read_csv(edge_path, usecols=['highway', 'length', 'n_id'], dtype={'highway': str})
    # if highway is a list, we define it as the first element of the list
    edge_df['highway'] = edge_df['highway'].apply(lambda loc: (loc.split(',')[0])[2:-1] if ',' in loc else loc)
    level2idx = {'residential': 0, 'primary': 1, 'unclassified': 2, 'tertiary': 3, 'living_street': 4, 'secondary': 5}
    edge_df['highway_idx'] = edge_df['highway'].apply(lambda loc: level2idx.get(loc,2))
    highway_idx = np.eye(6)[edge_df['highway_idx'].values]
    edge_feature = np.concatenate([np.expand_dims(edge_df['length'].values, 1), highway_idx], 1)
    edge_feature_max, edge_feature_min = np.max(edge_feature, 0), np.min(edge_feature, 0)
    print('edge_feature', edge_feature.shape)
    return edge_feature, edge_feature_max, edge_feature_min


def minmax_normalization(feature, xmax, xmin):
    feature = (feature - xmin) / (xmax - xmin + 1e-8)
    feature = 2 * feature - 1
    return feature


def load_train_sample(train_path):
    df = pd.read_csv(train_path, nrows=1000)
    test_traj = [path.split('_') for path in df['path'].tolist()]
    return test_traj, df[['ori', 'des']].values


def load_test_traj(test_path):
    df = pd.read_csv(test_path)
    df.sort_values(by=["des", "ori"], inplace=True)
    test_traj = [path.split('_') for path in df['path'].tolist()]
    return test_traj, df[['ori', 'des']].values