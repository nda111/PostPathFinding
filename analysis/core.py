from pyspark import SparkContext
from load_chunks import load_text
from kmeans import kmeans
from shortest_path import floyd_warshall_path, find_min_path


def analyze(k: int):
    sc = SparkContext()

    print('Load dataset')
    dataset = load_text(sc, '../chunks')

    print('Cluster dataset')
    clusters = kmeans(sc, dataset, k)

    print('Find path map')
    path_map = floyd_warshall_path(clusters)

    return [cluster[0] for cluster in clusters], path_map


if __name__ == '__main__':
    clusters, path_map = analyze(k=5)
    print()

    print('Centroids:', clusters)

    s, e = 0, 2
    print()
    print(f'Find path from {s} to {e}.')
    print(find_min_path(path_map, s, e))

