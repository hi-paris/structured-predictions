import numpy as np

from grakel import Graph

import networkx as nx

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

import os

def plot_qm9(G, ax, text=None, pos=None, draw_edge_feature=True):
    #G = to_networkx(y, use_edge_feature, thres)

    edge_style_map = {"0": "-", "1": "--", "2": ":"}
    node_atom_map = {"0": "C", "1": "N", "2": "O", "3": "F"}
    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    # print(G.nodes.data('feature'))
    node_labels = dict(G.nodes.data("feature"))
    # print(node_labels)
    for key, val in node_labels.items():
        node_labels[key] = node_atom_map[val]

    # if use_edge_feature:
    # print(G.edges(data=True))
    # print(G.edges.data("bond"))
    if draw_edge_feature:
        edge_styles = [edge_style_map[edge[-1]] for edge in G.edges.data("bond")]
        nx.draw(G, pos, labels=node_labels, style=edge_styles, width=3, ax=ax)
    else:
        nx.draw(G, pos, labels=node_labels, width=3, ax=ax)
    ax.set_title(text)

    return pos


def gkl_to_nx(Y_gkl):
    Y_dict = from_grkl_to_dict(Y_gkl)
    Y_nx = [to_networkx(y_dict) for y_dict in Y_dict]
    return Y_nx


def from_grkl_to_dict(Y_grkl):
    Y_dict = []
    n = len(Y_grkl)
    for i in range(n):
        Y_dict_new = {}

        A = Y_grkl[i].get_adjacency_matrix()
        Y_dict_new['A'] = A

        n_vert = A.shape[0]

        F = np.zeros((n_vert, 4))
        vert_labels = Y_grkl[i].get_labels(label_type='vertex')
        for j in range(n_vert):
            F[j, vert_labels[j]] = 1
        Y_dict_new['F'] = F

        if n_vert == 1:
            E = [[[0., 0., 0., 1.]]]
        else:
            E = np.zeros((n_vert, n_vert, 3))
            E = np.dstack((E, np.ones((n_vert, n_vert))))
            edge_labels = Y_grkl[i].get_labels(label_type='edge')
            edges1, edges2 = np.where(A == 1)
            for j in range(len(edges1)):
                v1, v2 = edges1[j], edges2[j]
                E[v1, v2, edge_labels[(v1, v2)]] = 1
                E[v1, v2, 3] = 0
        Y_dict_new['E'] = E
        Y_dict.append(Y_dict_new)
    
    Y = np.array(Y_dict)
        
    return Y


def to_networkx(y, use_edge_feature=True, thres=None):
    if use_edge_feature:
        E = y['E']
        adj = np.argmax(E, axis=-1)
        idx_edge = E.shape[-1] - 1
        A = np.asarray(adj != idx_edge, dtype=int)
    else:
        A = y['A']
        A = A.copy()
        np.fill_diagonal(A, 0.0)
        # print(A)
        A = np.where(A > thres, 1, 0)

    # # print(A)
    # # print(F)
    # A = A.copy()
    # np.fill_diagonal(A, 0.0)
    # # print(A)
    # A = np.where(A > thres, 1, 0)
    F = y['F']
    F = np.argmax(F, axis=1)
    # E = np.argmax(E, axis=-1)

    rows, cols = np.where(A == 1)
    edges = list(zip(rows.tolist(), cols.tolist()))
    G = nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(list(range(len(F))))

    F_dic = {}
    for k, l in enumerate(F):
        F_dic[k] = str(l.item())

    nx.set_node_attributes(G, F_dic, name="feature")

    if use_edge_feature:
        E_dict = {}
        for i, j in edges:
            E_dict[(i, j)] = {"bond": str(adj[i, j])}

        nx.set_edge_attributes(G, E_dict)

    numeric_indices = [index for index in range(G.number_of_nodes())]
    node_indices = sorted([node for node in G.nodes()])
    # print(numeric_indices)
    # print(node_indices)
    assert numeric_indices == node_indices

    return G


# def graph_edit_distance_parrallel(y_preds, y_trgts):
#     G_preds = [to_networkx(y_pred["A"], y_pred["F"], y_pred["E"]) for y_pred in y_preds]
#     G_trgts = [
#         to_networkx(y_target["A"], y_target["F"], y_target["E"]) for y_target in y_trgts
#     ]

#     ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
#     ged.set_attr_graph_used("feature", "bond")
#     return ged.compare(G_preds, G_trgts)


def eval_graph(G_preds, G_trgts, with_edge_feature=True, n_jobs=None):
    res_total = {}

    node_match = lambda x, y: x["feature"] == y["feature"]
    edge_match = lambda x, y: x["bond"] == y["bond"]
    if n_jobs is None:
        n_jobs = os.cpu_count()
    if with_edge_feature:
        geds = Parallel(n_jobs=n_jobs)(
            delayed(nx.graph_edit_distance)(
                G_pred, G_trgt, node_match=node_match, edge_match=edge_match
            )
            for G_pred, G_trgt in zip(G_preds, G_trgts)
        )
    else:
        geds = Parallel(n_jobs=n_jobs)(
            delayed(nx.graph_edit_distance)(G_pred, G_trgt, node_match=node_match)
            for G_pred, G_trgt in zip(G_preds, G_trgts)
        )

    res_total["edit_distance"] = np.mean(geds)
    res_total["eds"] = geds

    return res_total


def draw_samples_with_candidates(G_trgts, G_preds, Y_pred_topk, scores_topk, eds, save_dir, figure_name):
    num_samples = 40
    rng = np.random.default_rng(seed=42)
    samples_indexs = rng.choice(range(len(G_trgts)), size=num_samples, replace=False)
    num_candi = 5
    fig, axs = plt.subplots(
        num_samples, num_candi + 2, figsize=(4 * (num_candi + 2), 4 * num_samples)
    )
    for j, i in enumerate(samples_indexs):
        pos_test = plot_qm9(G_trgts[i], axs[j, num_candi + 1], text="True")
        plot_qm9(
            G_preds[i], axs[j, num_candi], text=f"Pred ({eds[i]})"
        )

        for k in range(num_candi):
            y_cand_gkl = Y_pred_topk[i, k]
            y_dict_cand = from_grkl_to_dict([y_cand_gkl])[0]
            y_cand = to_networkx(y_dict_cand)
            weight = scores_topk[i, k]
            plot_qm9(y_cand, axs[j, -(k + 3)], text=str(weight))

    fig.tight_layout()
    save_path = os.path.join(save_dir, f'{figure_name}.pdf')
    fig.savefig(save_path)
    plt.close()


def draw_samples(G_trgts, G_preds, eds, save_dir, figure_name):
    num_samples = 40
    rng = np.random.default_rng(seed=42)
    samples_indexs = rng.choice(range(len(G_trgts)), size=num_samples, replace=False)
    fig, axs = plt.subplots(
        num_samples, 2, figsize=(4 * 2, 4 * num_samples)
    )
    for j, i in enumerate(samples_indexs):

        pos_test = plot_qm9(G_trgts[i], axs[j, 1], text="True")
        plot_qm9(
            G_preds[i], axs[j, 0], text=f"Pred ({eds[i]})"
        )

    fig.tight_layout()
    save_path = os.path.join(save_dir, f'{figure_name}.pdf')
    fig.savefig(save_path)
    plt.close()
