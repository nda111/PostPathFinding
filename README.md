# Post Path Finding

## Task
The goal of this project is to find the best path to deliver a parcal in Republic of Korea
on some special situations when the traffic explodes i.e. the seol-nal, the chu-seok.

## Data
The [domestic postal logistics data](https://kdx.kr/data/view/27134) from [KDX](https://kdx.kr/main) includes 
about *6 million rows and following 23 columns* which is **289.91GB** in total:
|Column|Type & Length|Description|
|:-|:-|:-|
|DELIV_DATE|VARCHAR(8)|The date that the parcal delivered.|
|DELIV_PO_REGI_POCD|VARCHAR(5)|The national symbol for the registration of the house delivery service.|
|DISC_NO|VARCHAR(64)|The parcal identification number.|
|DELIV_AREA_NO|VARCHAR(5)|The shipped area ID number.|
|MAIL_KIND_SP_CD|VARCHAR(2)|The type classification code by domestic parcal.|
|MAIL_SP_CD|VARCHAR(2)|The classification code by domestic parcal.|
|SPCL_TRT_CD|VARCHAR(2)|The special treatment code.|
|DOM_EM_YN|VARCHAR(1)|The doministic special treatment status.|
|RCV_PO_REGI_POCD|VARCHAR(5)|The national symbol for registration.|
|VST_PRCL_YN|VARCHAR(1)|Visiting parcel status.|
|ELECT_MAIL_YN|VARCHAR(1)|Whether it's election mail or not.|
|CNTR_REGI_YN|VARCHAR(1)|Whether to register a contract.|
|PRSN_APPO_YN|VARCHAR(1)|Whether you're designated or not.|
|CONG_MAIL_SP_CD|VARCHAR(2)|The classification code for congratulatory mail.|
|MAIL_WGHT|NUMBER(8.3)|The weight of the parcal.|
|MAIL_VOLM|VARCHAR(10)|The volume of the parcal.|
|RCV_DATE|VARCHAR(8)|The date that the parcal has registered.|
|SEMI_REGI_YN|VARCHAR(1)|Semi-registration status.|
|BLD_MNGM_NO|NUMBER(25)|The building management number.|
|DELIV_CTRL_ZONE_NM|VARCHAR(100)|The name of the delivery area.|
|DELIV_EMD_NM|VARCHAR(200)|The name of eup, myeon, dong to deliver.|
|DLNM_DOLI_NM|VARCHAR(200)|The name of dong, li to deliver.|
|EUPD_IVNM|VARCHAR(100)|The name of the reception area.|

we sampled about 10% of total rows because we lacked computing power to process
the whole 6 million rows.

## Implementation
### Preprocessing
We preprocessed the data in a form of `torch.Tensor` and `str` in `pandas.DataFrame`.
#### Data reduction
1. Dropped all columns except for five columns: `DELIV_DATE`, `DELIV_CTRL_ZONE_NM`, `DELIV_EMD_NM`, `DLNM_DOLI_NM` and `EUPD_IVNM`. 
    ```python
    def select_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df[['DELIV_DATE', 'DELIV_CTRL_ZONE_NM', 'DELIV_EMD_NM', 'DLNM_DOLI_NM',\
                   'EUPD_IVNM']]
    ```
2. Filtered date 4 weeks before from the "special dates". And drop the date column.
    ```python
    def filter_date(df: pd.DataFrame) -> pd.DataFrame:
        one_month = dt.timedelta(days=28)

        ranges = '20041022 20040928 20050209 20050918 20060129 20061006 20070218 20070925 ' \
                 '20080207 20080914 20090126 20091003 20100214 20100922'
        ranges = [__date_string_to_date(date) for date in ranges.split()]
        ranges = [(date - one_month, date) for date in ranges]

        result = pd.DataFrame(columns=['DELIV_DATE', 'DELIV_CTRL_ZONE_NM', 'DELIV_EMD_NM', \
                                       'DLNM_DOLI_NM', 'EUPD_IVNM'])
        for row in tqdm(df.iterrows()):
            date = __date_string_to_date(list(row[1])[0])
            if date is None:
                continue

            for r in ranges:
                if r[0] < date < r[1]:
                    result = result.append(row[1], ignore_index=True)
                    break

        return result[['DELIV_CTRL_ZONE_NM', 'DELIV_EMD_NM', 'DLNM_DOLI_NM', 'EUPD_IVNM']]
    ```
#### Transformation
1. Transform eup, myun, dong and dong, li into physical coordnates on earth (longitude and latitude).
    ```python
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    __coder = Nominatim(user_agent='South Korea')

    __save_path = 'addr_coord_map.tar'
    if os.path.exists(__save_path):
        with open(__save_path, 'r+') as file:
            __addr_coord_map = eval('\n'.join(file.readlines()))
    else:
        __addr_coord_map = {}

    def __merge_address(deliv_ctrl_zone_nm: str, deliv_emd_nm: str, deliv_dl_nm: str) -> str:
        deliv_ctrl_zone_nm = str(deliv_ctrl_zone_nm).strip()
        deliv_emd_nm = str(deliv_emd_nm).strip()
        deliv_dl_nm = str(deliv_dl_nm).strip()

        ctrl_zone, emd, dl = '', '', ''
        if deliv_emd_nm != '':
            emd = deliv_emd_nm
        if deliv_dl_nm != '':
            dl = deliv_dl_nm

        if emd != '' or dl != '':
            return f'{emd} {dl}'
        elif deliv_ctrl_zone_nm != '':
            return deliv_ctrl_zone_nm
        else:
            return ''

    def __search_coordinates(address: str) -> Union[None, Tuple[float, float]]:
        """
        주소를 좌표로 변환한다. 주소를 찾지 못하면 None 을 반환한다.

        :param address: 대한민국 주소
        :return: 위도, 경도 순서쌍의 부동소수점 실수 표현
        """

        if address in __addr_coord_map:
            return __addr_coord_map[address]
        else:
            try:
                data = __coder.geocode(address)
                if data is None:
                    coord = None
                else:
                    coord = (data.latitude, data.longitude)
            except GeocoderTimedOut:
                coord = None
            except GeocoderServiceError:
                coord = None

            __addr_coord_map[address] = coord
            return coord

    def address_to_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame()
        for row in tqdm(df.iterrows()):
            row = list(row[1])
            receiver = row[-1]
            sender = row[:-1]

            rec_coord = __receiving_address_to_coordinates(receiver)
            snd_coord = __sending_address_to_coordinates(sender[0], sender[1], sender[2])

            if rec_coord is not None and snd_coord is not None:
                row = {
                    'sender_latitude': snd_coord[0],
                    'sender_longitude': snd_coord[1],
                    'receiver_latitude': rec_coord[0],
                    'receiver_longitude': rec_coord[1],
                }

                result = result.append(row, ignore_index=True)

        __save_request_history()
        return result
    ```

#### Data chunking
1. Split the rows into some chunk files.
    ```python
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
    ```

### Analysis
1. Load chunk files from HDFS and convert coordinate columns into a `torch.FloatTensor` with shape `(1, 2)`.
    ```python
    from pyspark import SparkContext, RDD


    def __line_to_tensor(line: str):
        t = torch.tensor([float(item) for item in line.split()]).view(2, 2)

        t1, t2 = t
        return t1, t2


    def load_text(sc: SparkContext, dir_name: str) -> RDD:
        file_names = [f'{dir_name}{os.path.sep}{file_name}' 
                      for file_name in os.listdir(dir_name)]

        coord_as_text = sc.parallelize([])
        for file_name in file_names:
            chunk = sc.textFile(file_name)
            coord_as_text = coord_as_text.union(chunk)

        return coord_as_text.flatMap(lambda line: __line_to_tensor(line))
    ```
2. Cluster the coordinates into k clusters. This part was implemented with `PySpark`.
    ```python
    def find_nearest(point: torch.Tensor, centroids: List[torch.Tensor]) -> int:
        min_idx, min_dist = -1, 10E10
        for idx, centroid in enumerate(centroids):
            sub = point - centroid
            dist_square = torch.sum(sub * sub)
            if dist_square < min_dist:
                min_dist = dist_square
                min_idx = idx

        return min_idx


    def kmeans(sc: SparkContext, dataset: RDD, k: int):
        if k < 2 or dataset.count() <= k:
            raise ValueError

        # 1. 리스트로 전환
        dataset_list = dataset.collect()

        # 2. 센트로이드 랜덤 픽
        all_indices = list(range(len(dataset_list)))
        prev_centroids, centroids = [], []
        for i in range(k):
            idx = torch.randint(low=0, high=len(all_indices), size=(1,))[0]
            centroids.append(dataset_list[all_indices[idx]])
            prev_centroids.append(dataset_list[all_indices[idx]])
            all_indices.remove(all_indices[idx])

        iteration = 0
        while True:
            iteration += 1
            print(f'\titeration {iteration}')

            # 3. 거리구하기: RDD, 거리가 작은 센트로이드 ID를 key로 리듀스
            nearests = dataset.map(lambda x: (find_nearest(x, centroids), x))

            # 4. 클러스터별로 센트로이드 다시 계산
            counts = nearests.countByKey()
            devided = nearests.map(lambda x: (x[0], x[1] / counts[x[0]]))
            sum = devided.reduceByKey(lambda result, point: result + point)
            centroids = sum.map(lambda x: x[1]).collect()

            for centroid in centroids:
                for i in range(len(prev_centroids)):
                    if torch.all(torch.eq(centroid, prev_centroids[i])):
                        del prev_centroids[i]
                        break

            if len(prev_centroids) == 0:
                break
            else:
                prev_centroids = centroids

        idx_counts = sc.parallelize([pair for pair in counts.items()])
        result = idx_counts.union(sum)

        result = result.reduceByKey(lambda count, point: (point, count)).map(lambda x: x[1])
        return result.collect()
    ```
3. Make a full-path map of any clusters with Floyd-Warshall's algorithm.
    ```python
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
        path = {i1: {i2: None 
                    for i2 in range(num_centroids) if i1 != i2} 
                for i1 in range(num_centroids)}
        for inter in range(num_centroids):
            for depart in range(num_centroids):
                for arrival in range(num_centroids):
                    new_weight = weights[depart][inter] + weights[inter][arrival]
                    if new_weight < weights[depart][arrival]:
                        weights[depart][arrival] = new_weight
                        path[depart][arrival] = inter

        return path
    ```
4. Figure the best from `depart-th` cluster to `arrival-th` cluster out from the path map.
    ``` python
    def find_min_path(table, depart: int, arrival: int):
        if depart == arrival:
            return []

        inter = table[depart][arrival]
        if inter is None:
            return [depart, arrival]
        else:
            return find_min_path(table, depart, inter) + find_min_path(table, inter, arrival)
    ```

## Future Tasks
- This project was run on HDFS with `PySpark` but it was single-node cluster HDFS.
The `PySpark` is not suitable to single node computing and small amount of data.
It would be better running on a HDFS with many data nodes with better computing power.
- On this project, the data was only about departure area and arrival area. But in real word, it
doesn't work in that way. There are so many intermediate hubs to pass to arrive.
On the next project, if we could collect same data about not only the *end-end* data 
but also for the intermediate hubs, the prediction of best path will show better performance.

## Referenced Libraries
- `Pandas` for data preprocessing.
- `PyTorch` for numeric data handling.
- `PySpark` for map-reduce implementation on HDFS.
- `GeoPy` for transfromation from area name to physical coordinates.


## Collaborators
|Student ID|Name|Contact|Page|
|:-|:-|:-|:-|
|201831847|Lim Hyeonjin|vldzmf001@naver.com|-|
|201835476|Yu Geunhyeok|nda111@naver.com|[GitHub](https://github.com/nda111)|
|201835489|Lee Sora|ehrqor39@naver.com|[GitHub](https://github.com/ehrqor39)|
