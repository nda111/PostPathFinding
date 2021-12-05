from _import import *

target_files = [
    'sample.csv'
]

preprocess(target_files)
clusters, path_map = analyze(k=3)

print(find_min_path(path_map, 0, 2))
