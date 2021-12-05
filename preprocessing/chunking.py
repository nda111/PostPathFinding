import os
from typing import Union
from math import ceil
import pandas as pd


def chunk_data(target: Union[str, pd.DataFrame], chunk_size: int, result_path: str):
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if type(target) == str:
        print(f'Chunking begin with {target}.')
        df = pd.read_csv(target)
    else:
        print(f'Chunking begin with a dataframe.')
        df = target
    num_chunk = ceil(len(df) / chunk_size)

    start, end = 0, chunk_size
    for chunk_id in range(num_chunk):
        with open(f'{result_path}{os.path.sep}{chunk_id}.txt', 'w+') as chuck_file:
            lines = [
                '\t'.join([str(item) for item in row[1]]) + '\n'
                for row in df[start:end].iterrows()
            ]
            chuck_file.writelines(lines)

        start = end
        end += chunk_size

    del df
    print(f'Chunking done.')

