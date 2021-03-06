from __future__ import absolute_import
import numpy as np


def knn_threshold(data, count, epsilon):
    return data[count - 1] + epsilon


def epsilon_threshold(data, count, epsilon):
    return data[count - 1] * (1 + epsilon)


# def get_recall_values(dataset_distances, run_distances, count, threshold,
#                       epsilon=1e-3):
#     recalls = np.zeros(len(run_distances))  # len of nq
#     for i in range(len(run_distances)):
#         t = threshold(dataset_distances[i], count, epsilon)
#         actual = 0
#         for d in run_distances[i][:count]:
#             if d <= t:
#                 actual += 1
#         recalls[i] = actual
#     return (np.mean(recalls) / float(count),
#             np.std(recalls) / float(count),
#             recalls)


def get_recall_values(dataset_neighbors, run_neighbors, count, threshold,
                      epsilon=1e-3):
    recalls = np.zeros(len(run_neighbors))  # len of nq
    for i in range(len(run_neighbors)):
        # t = threshold(dataset_distances[i], count, epsilon)
        inter = np.intersect1d(run_neighbors[i][:count], dataset_neighbors[i][:count])
        # for d in run_neighbors[i][:count]:
        #     if d <= t:
        #         actual += 1
        recalls[i] = inter.shape[0]
    return (np.mean(recalls) / float(count),
            np.std(recalls) / float(count),
            recalls)


def knn(dataset_distances, run_distances, count, metrics, epsilon=1e-3):
    if 'knn' not in metrics:
        print('Computing knn metrics')
        knn_metrics = metrics.create_group('knn')
        mean, std, recalls = get_recall_values(dataset_distances,
                                               run_distances, count,
                                               knn_threshold, epsilon)
        knn_metrics.attrs['mean'] = mean
        knn_metrics.attrs['std'] = std
        knn_metrics['recalls'] = recalls
    else:
        print("Found cached result")
    return metrics['knn']


def epsilon(dataset_distances, run_distances, count, metrics, epsilon=0.01):
    s = 'eps' + str(epsilon)
    if s not in metrics:
        print('Computing epsilon metrics')
        epsilon_metrics = metrics.create_group(s)
        mean, std, recalls = get_recall_values(dataset_distances,
                                               run_distances, count,
                                               epsilon_threshold, epsilon)
        epsilon_metrics.attrs['mean'] = mean
        epsilon_metrics.attrs['std'] = std
        epsilon_metrics['recalls'] = recalls
    else:
        print("Found cached result")
    return metrics[s]


def rel(dataset_distances, run_distances, metrics):
    if 'rel' not in metrics.attrs:
        print('Computing rel metrics')
        total_closest_distance = 0.0
        total_candidate_distance = 0.0
        for true_distances, found_distances in zip(dataset_distances,
                                                   run_distances):
            for rdist, cdist in zip(true_distances, found_distances):

                if np.isnan(rdist) or np.isnan(cdist) or np.isinf(rdist) or np.isinf(cdist):
                    continue

                total_closest_distance += rdist
                total_candidate_distance += cdist
        if total_closest_distance < 0.01:
            metrics.attrs['rel'] = float("inf")
        else:
            metrics.attrs['rel'] = total_candidate_distance / \
                total_closest_distance

            if metrics.attrs['rel'] - 1.0 < 0:
                metrics.attrs['rel'] = 1.0
    else:
        print("Found cached result")
    return metrics.attrs['rel']


def queries_per_second(queries, attrs):
    return 1.0 / attrs["best_search_time"]


def index_size(queries, attrs):
    # TODO(erikbern): should replace this with peak memory usage or something
    return attrs.get("index_size", 0)


def build_time(queries, attrs):
    return attrs["build_time"]


def candidates(queries, attrs):
    return attrs["candidates"]


def dist_computations(queries, attrs):
    return attrs.get("dist_comps", 0) / (attrs['run_count'] * len(queries))


all_metrics = {
    "k-nn": {
        "description": "Recall",
        "function": lambda true_distances, run_distances, metrics, run_attrs: knn(true_distances, run_distances, run_attrs["count"], metrics).attrs['mean'],  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03]
    },
    "k-nn_0.8": {
        "description": "Recall 0.8+",
        "function": lambda true_distances, run_distances, metrics, run_attrs: knn(true_distances, run_distances, run_attrs["count"], metrics).attrs['mean'],  # noqa
        "worst": float("-inf"),
        "lim": [0.80, 1.03]
    },
    "epsilon": {
        "description": "Epsilon 0.01 Recall",
        "function": lambda true_distances, run_distances, metrics, run_attrs: epsilon(true_distances, run_distances, run_attrs["count"], metrics).attrs['mean'],  # noqa
        "worst": float("-inf")
    },
    "epsilon_0.8": {
        "description": "Epsilon 0.01 Recall 0.8+",
        "function": lambda true_distances, run_distances, metrics, run_attrs: epsilon(true_distances, run_distances, run_attrs["count"], metrics).attrs['mean'],  # noqa
        "worst": float("-inf"),
        "lim": [0.80, 1.03]
    },
    "largeepsilon": {
        "description": "Epsilon 0.1 Recall",
        "function": lambda true_distances, run_distances, metrics, run_attrs: epsilon(true_distances, run_distances, run_attrs["count"], metrics, 0.1).attrs['mean'],  # noqa
        "worst": float("-inf")
    },
    "largeepsilon_0.8": {
        "description": "Epsilon 0.1 Recall 0.8+",
        "function": lambda true_distances, run_distances, metrics, run_attrs: epsilon(true_distances, run_distances, run_attrs["count"], metrics, 0.1).attrs['mean'],  # noqa
        "worst": float("-inf"),
        "lim": [0.80, 1.03]
    },
    "rel": {
        "description": "Relative Error",
        "function": lambda true_distances, run_distances, metrics, run_attrs: rel(true_distances, run_distances, metrics),  # noqa
        "worst": float("inf")
    },
    "qps": {
        "description": "Queries per second (1/s)",
        "function": lambda true_distances, run_distances, metrics, run_attrs: queries_per_second(true_distances, run_attrs),  # noqa
        "worst": float("-inf")
    },
    "distcomps": {
        "description": "Distance computations",
        "function": lambda true_distances, run_distances,  metrics, run_attrs: dist_computations(true_distances, run_attrs), # noqa
        "worst": float("inf")
    },
    "build": {
        "description": "Refresh time (s)",
        "function": lambda true_distances, run_distances, metrics, run_attrs: build_time(true_distances, run_attrs), # noqa
        "worst": float("inf")
    },
    "candidates": {
        "description": "Candidates generated",
        "function": lambda true_distances, run_distances, metrics, run_attrs: candidates(true_distances, run_attrs), # noqa
        "worst": float("inf")
    },
    "indexsize": {
        "description": "Index size (kB)",
        "function": lambda true_distances, run_distances, metrics, run_attrs: index_size(true_distances, run_attrs),  # noqa
        "worst": float("inf")
    },
    "queriessize": {
        "description": "Index size (kB)/Queries per second (s)",
        "function": lambda true_distances, run_distances, metrics, run_attrs: index_size(true_distances, run_attrs) / queries_per_second(true_distances, run_attrs), # noqa
        "worst": float("inf")
    }
}
