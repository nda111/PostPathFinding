import os
import os.path
import torch
from pyspark import SparkContext, RDD


def __line_to_tensor(line: str):
    t = torch.tensor([float(item) for item in line.split()]).view(2, 2)

    t1, t2 = t
    return t1, t2


def load_text(sc: SparkContext, dir_name: str) -> RDD:
    file_names = [f'{dir_name}{os.path.sep}{file_name}' for file_name in os.listdir(dir_name)]

    coord_as_text = sc.parallelize([])
    for file_name in file_names:
        chunk = sc.textFile(file_name)
        coord_as_text = coord_as_text.union(chunk)

    return coord_as_text.flatMap(lambda line: __line_to_tensor(line))

