from typing import List
from pyspark import SparkContext, RDD
import torch


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

