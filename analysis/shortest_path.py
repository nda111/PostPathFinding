from typing import List, Tuple, Any


def floyd_warshall_path(centroids: List[Tuple[Any, int]]):
    num_centroids = len(centroids)
    weights = {}
    for i1 in range(num_centroids):
        weights[i1] = {}
        for i2 in range(num_centroids):
            if i1 == i2:
                weights[i1][i2] = 0
            else:
                weights[i1][i2] = centroids[i2][1]  # density of the point

    '''{
        0: {1: None, 2: None}, 
        1: {0: None, 2: None}, 
        2: {0: None, 1: None}
    }'''
    path = {i1: {i2: None for i2 in range(num_centroids) if i1 != i2} for i1 in range(num_centroids)}
    for inter in range(num_centroids):
        for depart in range(num_centroids):
            for arrival in range(num_centroids):
                new_weight = weights[depart][inter] + weights[inter][arrival]
                if new_weight < weights[depart][arrival]:
                    weights[depart][arrival] = new_weight
                    path[depart][arrival] = inter

    return path


def find_min_path(table, depart: int, arrival: int):
    if depart == arrival:
        return []

    inter = table[depart][arrival]
    if inter is None:
        return [depart, arrival]
    else:
        return find_min_path(table, depart, inter) + find_min_path(table, inter, arrival)

