import pandas as pd
import numpy as np

# CSV 불러오기
df = pd.read_csv('merged_gpu_pod_list.csv')

# 시작과 끝 시각 찾기 (초 단위 UNIX timestamp)
start_time = int(df['creation_time'].min())
end_time = int(df['deletion_time'].max())

# 30분 간격 (초 단위)
interval = 30 * 60

# 시계열 집계용 리스트
time_bins = []
gpu_usages = []

for t in range(start_time, end_time, interval):
    t_start = t
    t_end = t + interval

    # 해당 구간에 존재하는 job 필터링
    active_jobs = df[(df['creation_time'] < t_end) & (df['deletion_time'] > t_start)]

    # 사용 중인 GPU 총합 (milli 단위 → GPU 단위로 변환)
    total_gpu = active_jobs['gpu_milli'].sum() / 1000.0

    # 결과 저장
    time_bins.append(t_start)  # timestamp 그대로 저장
    gpu_usages.append(total_gpu)

# 결과 DataFrame
result_df = pd.DataFrame({
    'timestamp': time_bins,
    'gpu_usage': gpu_usages
})

# 저장 (옵션)
result_df.to_csv('gpu_usage_30min.csv', index=False)

print(result_df.head())
