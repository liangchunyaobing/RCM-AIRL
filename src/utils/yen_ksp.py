from __future__ import generators
from operator import itemgetter
import numpy as np
import pandas as pd


class priorityDictionary(dict):
    def __init__(self):
        '''Initialize priorityDictionary by creating binary heap of pairs
        (value,key).  Note that changing or removing a dict entry will not
        remove the old pair from the heap until it is found by smallest() or
		until the heap is rebuilt.'''
        self.__heap = []
        dict.__init__(self)

    def smallest(self):
        '''Find smallest item after removing deleted items from heap.'''
        if len(self) == 0:
            raise IndexError("smallest of empty priorityDictionary")
        heap = self.__heap
        while heap[0][1] not in self or self[heap[0][1]] != heap[0][0]:
            lastItem = heap.pop()
            insertionPoint = 0
            while 1:
                smallChild = 2 * insertionPoint + 1
                if smallChild + 1 < len(heap) and \
                        heap[smallChild] > heap[smallChild + 1]:
                    smallChild += 1
                if smallChild >= len(heap) or lastItem <= heap[smallChild]:
                    heap[insertionPoint] = lastItem
                    break
                heap[insertionPoint] = heap[smallChild]
                insertionPoint = smallChild
        return heap[0][1]

    def __iter__(self):
        '''Create destructive sorted iterator of priorityDictionary.'''

        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]

        return iterfn()

    def __setitem__(self, key, val):
        '''Change value stored in dictionary and add corresponding pair to heap.
        Rebuilds the heap if the number of deleted items grows too large, to
        avoid memory leakage.'''
        dict.__setitem__(self, key, val)
        heap = self.__heap
        if len(heap) > 2 * len(self):
            self.__heap = [(v, k) for k, v in self.items()]
            self.__heap.sort()  # builtin sort likely faster than O(n) heapify
        else:
            newPair = (val, key)
            insertionPoint = len(heap)
            heap.append(None)
            while insertionPoint > 0 and \
                    newPair < heap[(insertionPoint - 1) // 2]:
                heap[insertionPoint] = heap[(insertionPoint - 1) // 2]
                insertionPoint = (insertionPoint - 1) // 2
            heap[insertionPoint] = newPair

    def setdefault(self, key, val):
        '''Reimplement setdefault to call our customized __setitem__.'''
        if key not in self:
            self[key] = val
        return self[key]


class Graph:
    ## The dictionary of the graph. Each key represents a node and the value of
    # the associated node is a dictionary of all the edges. The key of the edges
    # dictionary is the node the edge terminates at and the value is the cost of
    # the edge.

    ## Initializes the graph with an indentifier and Graphviz object.
    #
    # @post The graph will contain the data specified by the identifier, if that
    # data exist. If not, then the graph will be empty.
    #
    # @param self The object pointer.
    # @param name The identifier for the graph by which the data is stored as or
    # will be stored as.
    #
    def __init__(self, name=None):
        self.INFINITY = 1000000
        self.UNDEFINED = None
        self._data = {}
        return

    ## Gets the edges of a specified node.
    #
    # @param self The object pointer.
    # @param node The node whose edges are being queried.
    # @retval {} A dictionary of the edges and thier cost if the node exist
    # within the graph or None if the node is not in the graph.
    #
    def __getitem__(self, node):
        if node in self._data.keys():
            return self._data[node]
        else:
            return None

    ## Iterator for the digraph object.
    #
    # @param self The object pointer.
    # @retval iter An iterator that can be used to process each node of the
    # graph.
    #
    def __iter__(self):
        return self._data.__iter__()

    ## Adds a node to the graph.
    #
    # @param self The object pointer.
    # @param node The name of the node that is to be added to the graph.
    # @retval bool True if the node was added or False if the node already
    # existed in the graph.
    #
    def add_node(self, node):
        if node in self._data.keys():
            return False
        self._data[node] = {}
        return True

    ## Adds a edge to the graph.
    #
    # @post The two nodes specified exist within the graph and their exist an
    # edge between them of the specified value.
    #
    # @param self The object pointer.
    # @param node_from The node that the edge starts at.
    # @param node_to The node that the edge terminates at.
    # @param cost The cost of the edge, if the cost is not specified a random
    # cost is generated from 1 to 10.
    #
    def add_edge(self, node_from, node_to, cost=None):
        # if not cost:
        #     cost = random.randrange(1, 11)
        self.add_node(node_from)
        self.add_node(node_to)
        self._data[node_from][node_to] = cost
        return

    ## Removes an edge from the graph.
    #
    # @param self The object pointer.
    # @param node_from The node that the edge starts at.
    # @param node_to The node that the edge terminates at.
    # @param cost The cost of the edge, if the cost is not specified all edges
    # between the nodes are removed.
    # @retval int The cost of the edge that was removed. If the nodes of the
    # edge does not exist, or the cost of the edge was found to be infinity, or
    # if the specified edge does not exist, then -1 is returned.
    #
    def remove_edge(self, node_from, node_to, cost=None):
        if node_from not in self._data.keys():
            return -1

        if node_to in self._data[node_from].keys():
            if not cost:
                cost = self._data[node_from][node_to]

                if cost == self.INFINITY:
                    return -1
                else:
                    self._data[node_from][node_to] = self.INFINITY
                    return cost
            elif self._data[node_from][node_to] == cost:
                self._data[node_from][node_to] = self.INFINITY

                return cost
            else:
                return -1
        else:
            return -1


## @package YenKSP
# Computes K-Shortest Paths using Yen's Algorithm.
#
# Yen's algorithm computes single-source K-shortest loopless paths for a graph
# with non-negative edge cost. The algorithm was published by Jin Y. Yen in 1971
# and implores any shortest path algorithm to find the best path, then proceeds
# to find K-1 deviations of the best path.

## Computes K paths from a source to a sink in the supplied graph.
#
# @param graph A digraph of class Graph.
# @param start The source node of the graph.
# @param sink The sink node of the graph.
# @param K The amount of paths being computed.
#
# @retval [] Array of paths, where [0] is the shortest, [1] is the next
# shortest, and so on.
#

def ksp_yen(graph, node_start, node_end, max_k=5):
    distances, previous = dijkstra(graph, node_start)

    A = [{'cost': distances[node_end],
          'path': path(previous, node_start, node_end)}]
    B = []

    if not A[0]['path']: return A

    for k in range(1, max_k):
        for i in range(0, len(A[-1]['path']) - 1):
            node_spur = A[-1]['path'][i]
            path_root = A[-1]['path'][:i + 1]

            edges_removed = []
            for path_k in A:
                curr_path = path_k['path']
                if len(curr_path) > i and path_root == curr_path[:i + 1]:
                    # graph.remove edge 这一步是做什么
                    cost = graph.remove_edge(curr_path[i], curr_path[i + 1])
                    if cost == -1:
                        continue
                    edges_removed.append([curr_path[i], curr_path[i + 1], cost])

            path_spur = dijkstra(graph, node_spur, node_end)
            # print(path_spur)
            if path_spur['path']:
                path_total = path_root[:-1] + path_spur['path']
                dist_total = distances[node_spur] + path_spur['cost']
                potential_k = {'cost': dist_total, 'path': path_total}

                if not (potential_k in B):
                    B.append(potential_k)

            for edge in edges_removed:
                graph.add_edge(edge[0], edge[1], edge[2])

        if len(B):
            B = sorted(B, key=itemgetter('cost'))
            A.append(B[0])
            B.pop(0)
        else:
            break

    return A


def jaccard_score(path1, path2, edge2length):
    set1, set2 = set(path1), set(path2)
    intersect_set = set1.intersection(set2)
    intersect_len = sum([edge2length[path] for path in list(intersect_set)])
    path1_len = sum([edge2length[path] for path in path1])
    path2_len = sum([edge2length[path] for path in path2])
    return intersect_len / (path1_len + path2_len - intersect_len)


def ksp_yen_overlap(graph, node_start, node_end, edge2length, max_k=5, JCD_thres=0.8):
    distances, previous = dijkstra(graph, node_start)

    A = [{'cost': distances[node_end],
          'path': path(previous, node_start, node_end)}]
    B = []

    if not A[0]['path']: return A

    k = 1
    while k < max_k:
        for i in range(0, len(A[-1]['path']) - 1):
            node_spur = A[-1]['path'][i]
            path_root = A[-1]['path'][:i + 1]

            edges_removed = []
            for path_k in A:
                curr_path = path_k['path']
                if len(curr_path) > i and path_root == curr_path[:i + 1]:
                    # graph.remove edge 这一步是做什么
                    cost = graph.remove_edge(curr_path[i], curr_path[i + 1])
                    if cost == -1:
                        continue
                    edges_removed.append([curr_path[i], curr_path[i + 1], cost])

            path_spur = dijkstra(graph, node_spur, node_end)
            # print(path_spur)
            if path_spur['path']:
                path_total = path_root[:-1] + path_spur['path']
                dist_total = distances[node_spur] + path_spur['cost']
                potential_k = {'cost': dist_total, 'path': path_total}

                if not (potential_k in B):
                    B.append(potential_k)

            for edge in edges_removed:
                graph.add_edge(edge[0], edge[1], edge[2])

        if len(B):
            B = sorted(B, key=itemgetter('cost'))
            candidate_path = B[0]['path']
            max_JCD = 0
            for cost_path in A:
                JCD = jaccard_score(candidate_path[1:-1], cost_path['path'][1:-1], edge2length)
                if JCD > max_JCD:
                    max_JCD = JCD
                if max_JCD > JCD_thres: break
            if max_JCD < JCD_thres:
                A.append(B[0])
                print(k)
                k += 1
            B.pop(0)
        else:
            break

    return A


def dijkstra(graph, node_start, node_end=None):
    # print(node_start, node_end)
    distances = {}
    previous = {}
    Q = priorityDictionary()

    for v in graph:
        distances[v] = graph.INFINITY
        previous[v] = graph.UNDEFINED
        Q[v] = graph.INFINITY

    distances[node_start] = 0
    Q[node_start] = 0

    # 这里的v, u就是数字
    for v in Q:
        if v == node_end:
            break

        for u in graph[v]:
            cost_vu = distances[v] + graph[v][u]

            if cost_vu < distances[u]:
                distances[u] = cost_vu
                Q[u] = cost_vu
                previous[u] = v

    # print(node_end)
    if node_end is not None:
        # print({'cost': distances[node_end], 'path': path(previous, node_start, node_end)})
        return {'cost': distances[node_end], 'path': path(previous, node_start, node_end)}
    else:
        # print('no path is found')
        # return{'cost': graph.INFINITY, 'path': None}
        return (distances, previous)


def construct_graph(edge_path, transit_path, hide_link=None):
    """hide link is used to consider counterfactual scenarios when a link is closed"""
    g = Graph()
    edge_df = pd.read_csv(edge_path, header=0, usecols=['length', 'n_id'])
    if hide_link is not None:
        edge_df = edge_df.loc[edge_df['n_id'] != hide_link].copy()
    edge_df.reset_index(inplace=True, drop=True)
    print(edge_df.head())
    length_list, id_list = edge_df['length'].tolist(), edge_df['n_id'].tolist()
    edge2length = {id_list[i]: length_list[i] for i in range(len(id_list))}
    transit_np = np.load(transit_path)
    netconfig = pd.DataFrame(transit_np, columns=["from", "con", "to"])
    if hide_link is not None:
        netconfig = netconfig.loc[(netconfig["from"] != hide_link) & (netconfig["to"] != hide_link)].copy()
    for index, row in netconfig.iterrows():
        g.add_edge(row['from'], row['to'], edge2length[row['to']])
    print('done add edge...')
    return g


## Finds a paths from a source to a sink using a supplied previous node list.
#
# @param previous A list of node predecessors.
# @param node_start The source node of the graph.
# @param node_end The sink node of the graph.
#
# @retval [] Array of nodes if a path is found, an empty list if no path is
# found from the source to sink.
#
def path(previous, node_start, node_end):
    route = []

    node_curr = node_end
    while True:
        route.append(node_curr)
        if previous[node_curr] == node_start:
            route.append(node_start)
            break
        elif previous[node_curr] is None:
            return []

        node_curr = previous[node_curr]

    route.reverse()
    return route
