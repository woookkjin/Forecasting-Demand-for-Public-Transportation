## 대중교통 승차 수요 예측

```python 3.9.6
### 환경
- windows 
- vscode
- kernel : python 3.9.6

# 라이브러리
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

# train_data불러오기
train_data = pd.read_csv("/Users/chojungseok/Desktop/23-2/machine_learning/머신러닝 프로젝트/data/train_data_modified.csv")

# 컬럼명세
#    - GID: 격자번호
#    - DATE: 일자
#    - TIME: 시간(시)
#    - RIDE_DEMAND: 승차 수요 **(Target Variable)**
#    - ALIGHT_DEMAND: 하차 수요
#    - (주의) 하차수요가 아닌 승차수요를 예측하는 모델을 제출하여야 함

### 결측치 확인 (없음)
train_data.info()

# train_data의 기초 통계량 요약
train_data['TIME'].value_counts()

## 날짜에 따른 승차와 하차 추이 그래프
grouped_date = train_data.groupby('DATE').sum()

# 그래프를 그립니다.
plt.figure(figsize=(10, 6))
plt.plot(grouped_date.index, grouped_date['RIDE_DEMAND'], label='Ride Demand', marker='o')
plt.plot(grouped_date.index, grouped_date['ALIGHT_DEMAND'], label='Alight Demand', marker='x')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('날짜에 따른 탑승/하차 수요')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()

## 시간에 따른 승차와 하차 추이 그래프
grouped_time = train_data.groupby('TIME').sum()

# 그래프를 그립니다.
plt.figure(figsize=(10, 6))
plt.plot(grouped_time.index, grouped_time['RIDE_DEMAND'], label='Ride Demand', marker='o')
plt.plot(grouped_time.index, grouped_time['ALIGHT_DEMAND'], label='Alight Demand', marker='x')
plt.xlabel('TIME')
plt.ylabel('Demand')
plt.title('시간에 따른 승차/하차 수요')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()

# DATE 열을 날짜 형식으로 변환
train_data['DATE'] = pd.to_datetime(train_data['DATE'])

# 날짜로부터 요일을 추출하여 요일 열을 추가
train_data['DAY_OF_WEEK'] = train_data['DATE'].dt.day_name()

# 데이터프레임 출력
train_data

# 주어진 데이터프레임에서 'DAY_OF_WEEK' 열을 Categorical 데이터 타입으로 변환하고 요일 순서를 지정합니다.
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
train_data['DAY_OF_WEEK'] = pd.Categorical(train_data['DAY_OF_WEEK'], categories=weekday_order, ordered=True)

# 'DAY_OF_WEEK' 열을 기준으로 그룹화하여 합계를 계산합니다.
grouped_week = train_data.groupby('DAY_OF_WEEK').sum()

# 그래프를 그립니다.
plt.figure(figsize=(10, 6))
plt.plot(grouped_week.index, grouped_week['RIDE_DEMAND'], label='Ride Demand', marker='o')
plt.plot(grouped_week.index, grouped_week['ALIGHT_DEMAND'], label='Alight Demand', marker='x')
plt.xlabel('요일')
plt.ylabel('수요')
plt.title('요일에 따른 승차/하차 수요')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()

# 'DAY_OF_WEEK' 열을 기준으로 그룹화하여 합계를 계산합니다.
grouped_gid = train_data.groupby('gid').sum()

# 그래프를 그립니다.
plt.figure(figsize=(10, 6))
plt.plot(grouped_gid.index, grouped_gid['RIDE_DEMAND'], label='Ride Demand', marker='o')
plt.plot(grouped_gid.index, grouped_gid['ALIGHT_DEMAND'], label='Alight Demand', marker='x')
plt.xlabel('구역')
plt.ylabel('수요')
plt.title('구역에 따른 승차/하차 수요')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()
