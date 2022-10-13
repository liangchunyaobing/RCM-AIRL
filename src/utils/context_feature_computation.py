import pandas as pd
import numpy as np
from yen_ksp import ksp_yen, construct_graph
import time


def create_edge_dict(edge_path, hide_link=None):
    edge_df = pd.read_csv(edge_path, usecols=['highway', 'length', 'n_id'], dtype={'highway': str})
    edge_df['highway'] = edge_df['highway'].apply(lambda loc: (loc.split(',')[0])[2:-1] if ',' in loc else loc)
    level2idx = {'residential': 0, 'primary': 1, 'unclassified': 2, 'tertiary': 3, 'living_street': 4, 'secondary': 5}
    edge_df['highway_idx'] = edge_df['highway'].apply(lambda loc: level2idx.get(loc, 2))
    edge2attr = {}
    if hide_link is not None:
        for index, row in edge_df.iterrows():
            if row['n_id'] == hide_link: continue
            edge2attr[row['n_id']] = {'length': row['length'], 'highway': row['highway_idx']}
    else:
        for index, row in edge_df.iterrows():
            edge2attr[row['n_id']] = {'length': row['length'], 'highway': row['highway_idx']}
    return edge2attr, len(level2idx.keys())


def load_transit(transit_path, new_transit_path=None, hide_link=None):
    transit_np = np.load(transit_path)
    netconfig = pd.DataFrame(transit_np, columns=['from', 'con', 'to'])
    if hide_link is not None:
        netconfig = netconfig.loc[(netconfig["from"] != hide_link) & (netconfig["to"] != hide_link)].copy()
        netconfig.reset_index(inplace=True, drop=True)
        np.save(new_transit_path, netconfig.values)
    transit_dict = {}
    for index, row in netconfig.iterrows():
        if row['from'] not in transit_dict.keys():
            transit_dict[row['from']] = {}
        transit_dict[row['from']][row['to']] = row['con']
    return transit_dict


def create_path_features(path, edge2attr, transit_dict, graph, num_level):
    n_road = len(path)
    cost = sum([edge2attr[p]['length'] for p in path])
    n_left, n_right, n_u = 0, 0, 0
    n_road_level = np.zeros(num_level)
    for i in range(1, len(path)):
        direct_label = transit_dict[path[i - 1]][path[i]]
        if direct_label == 2:
            n_right += 1
        elif direct_label == 6:
            n_left += 1
        elif direct_label == 4:
            n_u += 1
        n_road_level[edge2attr[path[i]]['highway']] += 1
    features = [n_road, cost, n_left, n_right, n_u] + n_road_level.tolist()
    return features


def create_path_level_features(edge2attr, transit_dict, graph, num_level, feature_path, hide_link=None):
    """output feature matrix [n_link, n_link, n_features]"""
    """[i, j, k] element of the output matrix is the k-th feature from the-ith link to the j-th destination link"""
    edge_len = len(edge2attr.keys())
    od_features = np.zeros((edge_len, edge_len, num_level + 6))
    for ori in range(edge_len):
        print(ori)
        for des in range(edge_len):
            if ori == des:
                path_features = create_path_features([ori], edge2attr, transit_dict, graph, num_level) + [1]
                od_features[ori, ori, :] = path_features
            candidate_path = ksp_yen(graph, ori, des, 1)
            if len(candidate_path) == 0:
                continue
            path_features = create_path_features(candidate_path[0]['path'], edge2attr, transit_dict, graph,
                                                 num_level) + [1]
            od_features[ori, des, :] = path_features
    print(od_features.shape)
    np.save(feature_path, od_features)


if __name__ == '__main__':
    node_p = "../data/node.txt"
    edge_p = "../data/edge.txt"
    network_p = "../data/transit.npy"
    feature_p = "../data/feature_od.npy"
    start_time = time.time()
    df = pd.read_csv(node_p)
    print(len(df))
    edge2attr, level_num = create_edge_dict(edge_p)
    print(level_num)
    print('done create edge dict...')
    transit_dict = load_transit(network_p)
    print('done load transit...')
    graph = construct_graph(edge_p, network_p)
    print('done construct graph...')
    create_path_level_features(edge2attr, transit_dict, graph, level_num, feature_p)
    print("feature od time", time.time()-start_time)