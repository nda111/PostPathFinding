from typing import List
import pandas as pd
from reduce import filter_date, select_columns
from transform import address_to_coordinates
from chunking import chunk_data


def preprocess(file_names: List[str]):
    result = pd.DataFrame()
    for file_name in file_names:
        # 0. 파일 로드
        print(f'Load: {file_name}')
        df0 = pd.read_csv(file_name)
        df0.fillna('', inplace=True)
        print(f'Total of {len(df0)} rows.\n')

        # 1. 필요없는 컬럼 드랍
        print(f'Drop columns from: {file_name}')
        df1 = select_columns(df0).copy()
        del df0
        print(f'Total of {len(df1)} rows.\n')

        # 2. 날짜로 필터링하기
        print(f'Filter by date: {file_name}')
        df2 = filter_date(df1)
        del df1
        print(f'Total of {len(df2)} rows.\n')

        df2.to_csv('df2.csv')

        # 3. 주소 -> 좌표
        print(f'Replace address with coordinates: {file_name}')
        df3 = address_to_coordinates(df2)
        del df2
        print(f'Total of {len(df3)} rows.\n')

        # 4. 1~3을 모든 파일에 대해서, concat
        print(f'Finishing: {file_name}')
        result = result.append(df3)
        del df3
        print(f'Total of {len(result)} rows.\n')

    chunk_data(result, chunk_size=100, result_path='../chunks')


if __name__ == '__main__':
    preprocess(['../gen_sample.csv'])
