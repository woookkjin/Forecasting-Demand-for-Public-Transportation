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
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

### train데이터와 날씨 데이터 호출
train = pd.read_csv("C:/Users/user/Desktop/머신러닝/train_data_modified.csv")
weather = pd.read_csv("C:/Users/user/Desktop/머신러닝/OBS_ASOS_TIM_20240225172450.csv", encoding = "cp949")
train.head()

#- 컬럼명세
#   - GID: 격자번호
#    - DATE: 일자
#    - TIME: 시간(시)
#    - RIDE_DEMAND: 승차 수요 **(Target Variable)**
#    - ALIGHT_DEMAND: 하차 수요
#    - (주의) 하차수요가 아닌 승차수요를 예측하는 모델을 제출하여야 함

# 행 열 확인
train.shape

### 결측치 없음
train.info()

# 'train' 데이터프레임의 기술 통계 요약
train.describe()

### 시간은 05시부터 익일 01시까지 1시간 단위로 존재
train['TIME'].value_counts()

import matplotlib.pyplot as plt

## 날짜에 따른 승차와 하차 추이 그래프
grouped_date = train.groupby('DATE').sum()

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

import matplotlib.pyplot as plt

# 시간에 따른 승차와 하차 추이 그래프
grouped_time = train.groupby('TIME').sum()

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

# 'DAY_OF_WEEK' 열을 기준으로 그룹화하여 합계를 계산합니다.
grouped_gid = train.groupby('gid').sum()

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

# weather 파일의 데이터 확인
weather.head()

# 'DATE' 열과 'TIME' 열을 결합하여 'datetime' 열을 생성
# 'TIME' 열의 값을 문자열로 변환하고, 시간 부분을 두 자리로 맞추고, 분 부분을 ':00'으로 설정
train['datetime'] = train['DATE'] + ' ' + train['TIME'].astype(str).str.zfill(2) + ':00'

# '지점'을 'num', '지점명'을 'name', '일시'를 'datetime', '기온(°C)'를 'temp', '강수량(mm)'를 'prec'로 변경합니다.
weather = weather.rename(columns={'지점': 'num', '지점명': 'name', '일시': 'datetime', '기온(°C)': 'temp', '강수량(mm)':'prec'})

