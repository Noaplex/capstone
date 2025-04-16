import pandas as pd
import glob

# 병합할 파일 경로 패턴
file_pattern = 'openb_pod_list_gpushare*.csv'
file_list = glob.glob(file_pattern)

# 모든 CSV 파일을 읽어서 리스트에 담기
df_list = [pd.read_csv(file) for file in file_list]

# 병합
merged_df = pd.concat(df_list, ignore_index=True)

# 하나의 CSV 파일로 저장
output_file = 'merged_gpu_pod_list.csv'
merged_df.to_csv(output_file, index=False)

print(f"병합 완료! 저장된 파일: {output_file}")
print(f"총 row 수: {len(merged_df)}")
