import numpy as np


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def str_list_to_int2(line):
    return [int(line[0]), int(line[1]), float(line[-1])]

def read_edges(train_filename, test_filename):
    """read data from files

    Args:
        train_filename: training file name
        test_filename: test file name

    Returns:
        node_num: int, number of nodes in the graph
        graph: dict, node_id -> list of neighbors in the graph
    """

    graph = {}
    nodes = set()
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename) if test_filename != "" else []

    for edge in train_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    for edge in test_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []
    nodes = list(nodes)
    return len(nodes),nodes, graph
class idxMap(dict):
    def __init__(self):
        super().__init__(self)
    def get(self,k,d=0): # 重写类
        return self[k] if k in self else d
def read_rating_edges(train_filename, test_filename):
    """read user-item rating data from files

    Args:
        train_filename: training file name
        test_filename: test file name

    Returns:
        node_num: int, number of nodes in the graph
        graph: dict, node_id -> list of neighbors in the graph
    """

    uneisi = {}
    uidx_map = idxMap()
    iidx_map = idxMap()
    sidx_map = idxMap()
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename) if test_filename != "" else []
    uidx = 1
    iidx = 1
    sidx = 1
    for edge in train_edges:
        user = edge[0]
        item = edge[1]
        score = float(edge[2])
        if uidx_map.get(user) == 0: # new user
            uidx_map[user] = uidx
            uneisi[uidx] = {}
            uidx = uidx + 1
        if iidx_map.get(item) == 0: # new item
            iidx_map[item] = iidx
            iidx = iidx + 1
        if sidx_map.get(score) == 0:
            sidx_map[score] = sidx
            sidx = sidx + 1
        user = uidx_map.get(user)
        item = iidx_map.get(item)
        score = sidx_map.get(score)
        uneisi[user][item]=score
        edge[0] = user
        edge[1] = item
        edge[2] = score

    for edge in test_edges:
        user = edge[0]
        item = edge[1]
        score = float(edge[2])
        if uidx_map.get(user) == 0:
            uidx_map[user] = uidx
            uidx = uidx + 1
        if iidx_map.get(item) == 0:
            iidx_map[item] = iidx
            iidx = iidx + 1
        if sidx_map.get(score) == 0:
            sidx_map[score] = sidx
            sidx = sidx + 1
        user = uidx_map.get(user)
        item = iidx_map.get(item)
        score = sidx_map.get(score)
        edge[0] = user
        edge[1] = item
        edge[2] = score
    return uneisi, train_edges, test_edges,uidx_map, uidx, iidx, sidx


def read_uu_edges(train_filename, test_filename, umap,uidx):
    """read data from files

    Args:
        train_filename: training file name
        test_filename: test file name

    Returns:
        node_num: int, number of nodes in the graph
        graph: dict, node_id -> list of neighbors in the graph
    """

    graph = {}
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename) if test_filename != "" else []

    for edge in train_edges:
        u1 = edge[0]
        u2 = edge[1]
        if umap.get(u1) == 0:
            umap[u1] = uidx
            uidx = uidx + 1
        if umap.get(u2) == 0:
            umap[u2] = uidx
            uidx = uidx + 1
        u1 = umap.get(u1)
        u2 = umap.get(u2)
        if graph.get(u1) is None:
            graph[u1] = []
        if graph.get(u2) is None:
            graph[u2] = []
        graph[u1].append(u2)
        graph[u2].append(u1)

    for edge in test_edges:
        u1 = edge[0]
        u2 = edge[1]
        if umap.get(u1) == 0:
            umap[u1] = uidx
            uidx = uidx + 1
        if umap.get(u2) == 0:
            umap[u2] = uidx
            uidx = uidx + 1
        u1 = umap.get(u1)
        u2 = umap.get(u2)
    return graph,uidx

def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int2(line.split()) for line in lines]
    return edges


def read_embeddings(filename, n_node, n_embed):
    """read pretrained node embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        embedding_matrix = dict()
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0])] = str_list_to_float(emd[1:])
        return embedding_matrix


def reindex_node_id(edges):
    """reindex the original node ID to [0, node_num)

    Args:
        edges: list, element is also a list like [node_id_1, node_id_2]
    Returns:
        new_edges: list[[1,2],[2,3]]
        new_nodes: list [1,2,3]
    """

    node_set = set()
    for edge in edges:
        node_set = node_set.union(set(edge))

    node_set = list(node_set)
    new_nodes = set()
    new_edges = []
    for edge in edges:
        new_edges.append([node_set.index(edge[0]), node_set.index(edge[1])])
        new_nodes = new_nodes.add(node_set.index(edge[0]))
        new_nodes = new_nodes.add(node_set.index(edge[1]))

    new_nodes = list(new_nodes)
    return new_edges, new_nodes


def generate_neg_links(train_filename, test_filename, test_neg_filename):
    """
    generate neg links for link prediction evaluation
    Args:
        train_filename: the training edges
        test_filename: the test edges
        test_neg_filename: the negative edges for test
    """

    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename)
    neighbors = {}  # dict, node_ID -> list_of_neighbors
    for edge in train_edges + test_edges:
        if neighbors.get(edge[0]) is None:
            neighbors[edge[0]] = []
        if neighbors.get(edge[1]) is None:
            neighbors[edge[1]] = []
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])
    nodes = set([x for x in range(len(neighbors))])

    # for each edge in the test set, sample a negative edge
    neg_edges = []

    for i in range(len(test_edges)):
        edge = test_edges[i]
        start_node = edge[0]
        neg_nodes = list(nodes.difference(set(neighbors[edge[0]] + [edge[0]])))
        neg_node = np.random.choice(neg_nodes, size=1)[0]
        neg_edges.append([start_node, neg_node])
    neg_edges_str = [str(x[0]) + "\t" + str(x[1]) + "\n" for x in neg_edges]
    with open(test_neg_filename, "w+") as f:
        f.writelines(neg_edges_str)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()
