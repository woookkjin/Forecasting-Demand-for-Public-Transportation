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

## train데이터와 날씨 데이터 호출
train = pd.read_csv("/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/train_data_modified.csv")
weather = pd.read_csv("/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/OBS_ASOS_TIM_20231125142449.csv", encoding = "cp949")

train.head()

# 컬럼명세
#    - GID: 격자번호
#    - DATE: 일자
#    - TIME: 시간(시)
#    - RIDE_DEMAND: 승차 수요 **(Target Variable)**
#    - ALIGHT_DEMAND: 하차 수요
#    - (주의) 하차수요가 아닌 승차수요를 예측하는 모델을 제출하여야 함

# 데이터프레임 확인
train.shape

# 결측치 없음
train.info()

# 기초통계량 요약
train.describe()

# 시간은 05시부터 익일 01시까지 1시간 단위로 존재
train['TIME'].value_counts()

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

## 시간에 따른 승차와 하차 추이 그래프
grouped_time = train.groupby('TIME').sum()

plt.figure(figsize=(10, 6))
plt.plot(grouped_time.index, grouped_time['RIDE_DEMAND'], ='Ride Demand', marker='o')
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

# weather 데이터프레임의 처음 5개 행을 출력합니다.
weather.head()

# 'DATE' 열과 'TIME' 열을 결합하여 'datetime' 열을 생성
# 'TIME' 열의 값을 문자열로 변환하고, 시간 부분을 두 자리로 맞추고, 분 부분을 ':00'으로 설정
train['datetime'] = train['DATE'] + ' ' + train['TIME'].astype(str).str.zfill(2) + ':00'

# weather 데이터프레임의 컬럼 이름을 변경합니다:
# '지점' -> 'num', '지점명' -> 'name', '일시' -> 'datetime', '기온(°C)' -> 'temp', '강수량(mm)' -> 'prec'
weather = weather.rename(columns={'지점': 'num', '지점명': 'name', '일시': 'datetime', '기온(°C)': 'temp', '강수량(mm)':'prec'})
weather.head()

# 강수량 nan은 비가 오지 않은것 이므로 0으로 채움
train['prec'].fillna(0, inplace=True)

# 비가 온날과 안온날을 구분해주는 함수 생성
def rainy(x):
    if x == 0:
        return 0
    else:
        return 1

train['rainy_day'] = train['prec'].apply(rainy)
train.head()

# 'DATE' 열의 값을 날짜 형식으로 변환하여 'base_date' 열을 생성합니다.
# 변환할 때 사용되는 날짜 형식은 "%Y-%m-%d"이며, 이는 연도-월-일 형식을 나타냅니다.
train["base_date"] = pd.to_datetime(train["DATE"], format="%Y-%m-%d")
train.head()

# 년 / 월 / 일로 변경 및 요일 변수 
train['YEAR'] = train['base_date'].dt.year
train['MONTH'] = train['base_date'].dt.month
train['DAY'] = train['base_date'].dt.day
train['weekday'] = train['base_date'].dt.weekday

# 주말 여부
def weekend(x):
    if x ==5:
        return 1
    elif x==6:
        return 1
    else:
        return 0

train['weekend'] = train['weekday'].apply(weekend)
train.head()

# 공휴일 여부
def holiday(x):
    if x in ['2023-06-06','2023-08-15']:
        return 1
    else:
        return 0
train['holiday'] = train['DATE'].apply(holiday)
train.head()

# 관계있는 요일 그룹화
def weekdays(data):
    data['weekday_group'] = None
    data.loc[data['weekday'].isin([0,1, 2, 3,4]), 'weekday_group'] = '평일'
    data.loc[data['weekday'].isin([5, 6]), 'weekday_group'] = ' 주말'

weekdays(train)
train.head()

# 버스 정류장 데이터 호출
bus_stop = pd.read_csv("/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/대전광역시_버스정류장 현황_20221215.csv", encoding='cp949')

bus_stop.head()

# 대전광역시 1KM격자 데이터 생성
grid = gpd.read_file("/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/국토통계_인구정보-총 인구 수(전체)-(격자) 1KM_대전광역시_202304 (1)/nlsp_020001001.shp", encoding = 'utf-8')

grid.head()

# shapely.geometry 라이브러리에서 Point 클래스를 가져옴
from shapely.geometry import Point

# '경도'와 '위도' 열을 사용하여 '버스정류장좌표' 열을 생성
# 각 행에 대해 Point 객체를 생성하고, '버스정류장좌표' 열에 할당
bus_stop['버스정류장좌표'] = bus_stop.apply(lambda row: Point(row['경도'], row['위도']), axis=1)

# GeoDataFrame을 생성하여 '버스정류장좌표' 열을 지오메트리로 사용
bus_stop_gps = gpd.GeoDataFrame(bus_stop, geometry='버스정류장좌표')

# 좌표 시스템을 EPSG 코드 4326으로 설정
bus_stop_gps.set_crs(epsg=4326, inplace=True)

# 좌표 시스템을 EPSG 코드 5179로 변환
gdf_bus_pickup = bus_stop_gps.to_crs(epsg=5179)

# 'gdf_bus_pickup'과 'grid' 간의 지오메트리 연산을 수행하고, 'left' 조인 방식으로 연결
# 'within' 연산을 사용하여 'gdf_bus_pickup'의 점이 'grid' 다각형 내에 있는 경우 연결
bus_stop_join = gpd.sjoin(gdf_bus_pickup, grid, how='left', op='within')
bus_stop_join.head()

# bus_stop_join 데이터프레임에서 'gid' 컬럼의 각 값별로 빈도수를 계산하고, 이를 gid_counts 변수에 저장
gid_counts = bus_stop_join['gid'].value_counts()
gid_counts_df = pd.DataFrame(gid_counts).reset_index()
gid_counts_df.columns = ['gid', 'count']
gid_counts_df.head()

### 버스 정류장 갯수를 변수로 사용하기 위해 데이터프레임 결합
train = pd.merge(train, gid_counts_df, how= 'left', on = 'gid')
train.head()

### 버스 정류장이 없는 곳은 0입력
train['count'].fillna(0, inplace=True)

from sklearn.preprocessing import LabelEncoder

# 라벨 인코딩
str_col = ['gid', 'weekday_group']  # 라벨 인코딩을 적용할 컬럼 이름 리스트를 정의

for i in str_col:  # 리스트에 있는 각 컬럼 이름에 대해 반복문을 실행
    le = LabelEncoder()  # LabelEncoder 클래스의 인스턴스를 생성
    le = le.fit(train[i])  # train 데이터프레임의 지정된 컬럼에 LabelEncoder를 적용
    train[i] = le.transform(train[i])  # 적용된 LabelEncoder를 사용하여 컬럼의 값을 변환하고 데이터프레임을 업데이트
train.head()

grid_1KM = "/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/(B100)국토통계_인구정보-생산가능 인구 수(전체)-(격자) 1KM_대전광역시_202310/nlsp_020001007.shp"
data = gpd.read_file(grid_1KM, encoding='utf-8')
data.head()

str_col = ['gid']
for i in str_col :
    le = LabelEncoder()
    le = le.fit(data[i])
    data[i] = le.transform(data[i])
data.head()

train.drop(['base_date', 'datetime', 'name', 'num','YEAR','DATE','temp'], axis=1, inplace=True)

lard = gpd.read_file("/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/(B100)국토통계_인구정보-생산가능 인구 수(전체)-(격자) 1KM_대전광역시_202310/LARD_ADM_SECT_SGG_30.shp")

# Haversine 공식을 이용하여 위도와 경도에 기반한 거리를 계산하는 함수 정의
def haversine_array(lat1, lng1, lat2, lng2):
    # 위도 및 경도를 라디안으로 변환
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # 지구 반지름 (킬로미터 단위)
    
    # Haversine 공식 적용
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

# 총 하차 수 상위 10개 gid 추출하여 정렬
base_gid = train.groupby(['gid'])['ALIGHT_DEMAND'].agg([('ALIGHT_SUM', 'sum')]).sort_values(by='ALIGHT_SUM', ascending=False).head(10).index

# 데이터의 중심점(geometry_centroid)에 대한 geometry 속성 계산
data['geometry_centroid'] = data['geometry'].centroid

# 각 중심점이 어떤 구(polygon)에 속하는지 찾기
for idx, row in data.iterrows():
    point = row['geometry_centroid']
    
    for poly_idx, polygon in lard.iterrows():
        if (polygon['geometry']).contains(point):
            data.loc[idx, 'polygon_idx'] = poly_idx
            break

# 데이터에 위도(Latitude) 및 경도(Longitude) 열 추가
data['Latitude'] = data['geometry_centroid'].apply(lambda point: point.y)
data['Longitude'] = data['geometry_centroid'].apply(lambda point: point.x)

# 각 상위 10개 gid와의 거리 계산 및 열 추가
for i in base_gid:
    base_lat, base_lng = data[data['gid'] == i]['geometry_centroid'].y, data[data['gid'] == i]['geometry_centroid'].x
    col_name = f'distance_{i}'
    data[col_name] = data.apply(lambda row: haversine_array(base_lat, base_lng, row['geometry_centroid'].y, row['geometry_centroid'].x), axis=1)

# 데이터프레임에서 불필요한 열 제거
gid_dis = data.drop(['lbl', 'val', 'geometry', 'geometry_centroid'], axis=1)

# 기존 train 및 test 데이터프레임에 거리 정보 추가
train = pd.merge(train, gid_dis, on='gid', how='left')

# 결측값을 경계 외(5)로 채우기
train['polygon_idx'].fillna(5, inplace=True)
train.head()
train.columns

# 상관계수 계산
correlation = train[['RIDE_DEMAND', 'TIME', 'RIDE_DEMAND', 'ALIGHT_DEMAND', 'prec', 'rainy_day']].corr()

# 상관관계 그래프 그리기
plt.figure(figsize=(5, 4))
sns.heatmap(correlation, annot=True, cmap='coolwarm', cbar=False)
plt.title('Correlation RIDE_DEMAND')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 상관계수 계산
correlation = train[['RIDE_DEMAND', 'MONTH', 'DAY', 'weekday', 'weekend', 'holiday']].corr()

# 상관관계 그래프 그리기
plt.figure(figsize=(5, 4))
sns.heatmap(correlation, annot=True, cmap='coolwarm', cbar=False)
plt.title('Correlation RIDE_DEMAND')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 상관계수 계산
correlation = train[['RIDE_DEMAND','weekday_group', 'count', 'polygon_idx', 'Latitude', 'Longitude']].corr()

# 상관관계 그래프 그리기
plt.figure(figsize=(5, 4))
sns.heatmap(correlation, annot=True, cmap='coolwarm', cbar=False)
plt.title('Correlation RIDE_DEMAND')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 상관계수 계산
correlation = train[['RIDE_DEMAND','distance_159', 'distance_240', 'distance_262', 'distance_89',
       'distance_264', 'distance_241', 'distance_181', 'distance_202',
       'distance_109', 'distance_158']].corr()

# 상관관계 그래프 그리기
plt.figure(figsize=(5, 4))
sns.heatmap(correlation, annot=True, cmap='coolwarm', cbar=False)
plt.title('Correlation RIDE_DEMAND')
plt.show()

# 상관관계 계산 및 그래프 그리기
correlation = train.corr()['RIDE_DEMAND']

# 상관관계 그래프 그리기
plt.figure(figsize=(15, 10))
sns.barplot(x=correlation.values, y=correlation.index)
plt.title('Correlation with RIDE_DEMAND')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Columns')
plt.show()

# 특성과 타겟 변수 선택
features = train.drop(['RIDE_DEMAND'], axis=1)  # 타겟 변수와 날짜 열을 제외합니다.
target = train['RIDE_DEMAND']

# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# RandomForestRegressor 모델 생성 및 하이퍼파라미터 설정
random_forest_model = RandomForestRegressor(n_estimators=1200, max_depth=30, min_samples_split=2, min_samples_leaf=1, random_state=42)

# 모델 훈련
random_forest_model.fit(X_train, y_train)

# 훈련 세트에서 예측 수행
train_predictions = random_forest_model.predict(X_train)

# 모델 평가 (훈련 세트)
train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)

print(f'MAE (Training Set): {train_mae}')
print(f'MSE (Training Set): {train_mse}')
print(f'R2 Score (Training Set): {train_r2}')

# 테스트 세트에서 예측 수행
test_predictions = random_forest_model.predict(X_test)

# 모델 평가 (테스트 세트)
mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print(f'MAE (Test Set): {mae}')
print(f'MSE (Test Set): {mse}')
print(f'R2 Score (Test Set): {r2}')

# 모델을 저장
joblib.dump(random_forest_model, 'random_forest_model.pkl')

import joblib

# 모델을 로드
loaded_model = joblib.load('random_forest_model.pkl')

# 실제값 예측
test = pd.read_csv("/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/test_data_modified.csv")
test.head()

# Target열 생성
test['RIDE_DEMAND'] = pd.NA
test.head()

weather = pd.read_csv("/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/OBS_ASOS_TIM_20231125142449.csv", encoding = "cp949")
weather.head()

# 'DATE' 열과 'TIME' 열을 결합하여 'datetime' 열을 생성
# 'TIME' 열의 값을 문자열로 변환하고, 시간 부분을 두 자리로 맞추고, 분 부분을 ':00'으로 설정
test['datetime'] = test['DATE'] + ' ' + test['TIME'].astype(str).str.zfill(2) + ':00'

weather = weather.rename(columns={'지점': 'num', '지점명': 'name', '일시': 'datetime', '기온(°C)': 'temp', '강수량(mm)':'prec'})
weather = weather.loc[: , ['num', 'name','datetime', 'temp','prec']]
weather.head()

# 날씨를 변수로 사용하기 위해서 데이터 프레임 결합
test = pd.merge(test, weather, on='datetime', how = 'left')

# 강수량 nan은 바가 오지 않은것 이므로 0으로 채움

# 비 오늘 날
def rainy(x):
    if x == 0:
        return 0
    else:
        return 1
test['rainy_day'] = test['prec'].apply(rainy)
test.head()

# 'DATE' 열의 값을 날짜 형식으로 변환하여 'base_date' 열을 생성합니다.
# 변환할 때 사용되는 날짜 형식은 "%Y-%m-%d"이며, 이는 연도-월-일 형식을 나타냅니다.
test["base_date"] = pd.to_datetime(test["DATE"], format="%Y-%m-%d")
test.head()

# 년 / 월 / 일로 변경 및 요일 변수 
test['YEAR'] = test['base_date'].dt.year
test['MONTH'] = test['base_date'].dt.month
test['DAY'] = test['base_date'].dt.day
test['weekday'] = test['base_date'].dt.weekday

# 주말 여부
def weekend(x):
    if x ==5:
        return 1
    elif x==6:
        return 1
    else:
        return 0

test['weekend'] = test['weekday'].apply(weekend)

# 공휴일 여부
def holiday(x):
    if x in ['2023-06-06','2023-08-15']:
        return 1
    else:
        return 0

test['holiday'] = test['DATE'].apply(holiday)
test.head()

# 관계있는 요일 그룹화
def weekdays(data):
    data['weekday_group'] = None
    data.loc[data['weekday'].isin([0,1, 2, 3,4]), 'weekday_group'] = '평일'
    data.loc[data['weekday'].isin([5, 6]), 'weekday_group'] = ' 주말'
weekdays(test)

bus_stop = pd.read_csv("/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/대전광역시_버스정류장 현황_20221215.csv", encoding='cp949')
bus_stop.head()

grid = gpd.read_file("/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/국토통계_인구정보-총 인구 수(전체)-(격자) 1KM_대전광역시_202304 (1)/nlsp_020001001.shp", encoding = 'utf-8')
grid.head()

# shapely.geometry 라이브러리에서 Point 클래스를 가져옴
from shapely.geometry import Point

# '경도'와 '위도' 열을 사용하여 '버스정류장좌표' 열을 생성
# 각 행에 대해 Point 객체를 생성하고, '버스정류장좌표' 열에 할당
bus_stop['버스정류장좌표'] = bus_stop.apply(lambda row: Point(row['경도'], row['위도']), axis=1)

# GeoDataFrame을 생성하여 '버스정류장좌표' 열을 지오메트리로 사용
bus_stop_gps = gpd.GeoDataFrame(bus_stop, geometry='버스정류장좌표')

# 좌표 시스템을 EPSG 코드 4326으로 설정
bus_stop_gps.set_crs(epsg=4326, inplace=True)

# 좌표 시스템을 EPSG 코드 5179로 변환
gdf_bus_pickup = bus_stop_gps.to_crs(epsg=5179)

# 'gdf_bus_pickup'과 'grid' 간의 지오메트리 연산을 수행하고, 'left' 조인 방식으로 연결
# 'within' 연산을 사용하여 'gdf_bus_pickup'의 점이 'grid' 다각형 내에 있는 경우 연결
bus_stop_join = gpd.sjoin(gdf_bus_pickup, grid, how='left', op='within')

bus_stop_join.head()

gid_counts = bus_stop_join['gid'].value_counts()
gid_counts_df = pd.DataFrame(gid_counts).reset_index()
gid_counts_df.columns = ['gid', 'count']
gid_counts_df.head()

test = pd.merge(test, gid_counts_df, how= 'left', on = 'gid')
test.head()

test['count'].fillna(0, inplace=True)
from sklearn.preprocessing import LabelEncoder

# 라벨 인코딩
str_col = ['gid', 'weekday_group']
for i in str_col :
    le = LabelEncoder()
    le = le.fit(test[i])
    test[i] = le.transform(test[i])
test.head()

grid_1KM = "/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/(B100)국토통계_인구정보-생산가능 인구 수(전체)-(격자) 1KM_대전광역시_202310/nlsp_020001007.shp"
data = gpd.read_file(grid_1KM, encoding='utf-8')
data.head()

str_col = ['gid']
for i in str_col :
    le = LabelEncoder()
    le = le.fit(data[i])
    data[i] = le.transform(data[i])
data.head()

test.drop(['base_date', 'datetime', 'name', 'num','YEAR','DATE','temp'], axis=1, inplace=True)
test.head()

lard = gpd.read_file("/Users/chojungseok/Desktop/머신러닝/머신러닝 공모전 team_S.S.A/data/(B100)국토통계_인구정보-생산가능 인구 수(전체)-(격자) 1KM_대전광역시_202310/LARD_ADM_SECT_SGG_30.shp")

# 그리스 총 인구수 상위 10과의 거리를 계산하는 함수
def haversine_array(lat1, lng1, lat2, lng2):
    # 위도 및 경도를 라디안으로 변환
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # 지구 반지름 (킬로미터 단위)
    
    # Haversine 공식 적용
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

# 총 하차 수 상위 10개 gid 추출하여 정렬
base_gid = test.groupby(['gid'])['ALIGHT_DEMAND'].agg([('ALIGHT_SUM', 'sum')]).sort_values(by='ALIGHT_SUM', ascending=False).head(10).index

# 데이터의 중심점(geometry_centroid)에 대한 geometry 속성 계산
data['geometry_centroid'] = data['geometry'].centroid

# 각 중심점이 어떤 구(polygon)에 속하는지 찾기
for idx, row in data.iterrows():
    point = row['geometry_centroid']
    
    for poly_idx, polygon in lard.iterrows():
        if (polygon['geometry']).contains(point):
            data.loc[idx, 'polygon_idx'] = poly_idx
            break

# 데이터에 위도(Latitude) 및 경도(Longitude) 열 추가
data['Latitude'] = data['geometry_centroid'].apply(lambda point: point.y)
data['Longitude'] = data['geometry_centroid'].apply(lambda point: point.x)

# 각 상위 10개 gid와의 거리 계산 및 열 추가
for i in base_gid:
    base_lat, base_lng = data[data['gid'] == i]['geometry_centroid'].y, data[data['gid'] == i]['geometry_centroid'].x
    col_name = f'distance_{i}'
    data[col_name] = data.apply(lambda row: haversine_array(base_lat, base_lng, row['geometry_centroid'].y, row['geometry_centroid'].x), axis=1)

# 데이터프레임에서 불필요한 열 제거
gid_dis = data.drop(['lbl', 'val', 'geometry', 'geometry_centroid'], axis=1)

# 기존 train 및 test 데이터프레임에 거리 정보 추가
test = pd.merge(test, gid_dis, on='gid', how='left')

# 결측값을 경계 외(5)로 채우기
test['polygon_idx'].fillna(5, inplace=True)
test.head()

test = pd.DataFrame(test)

# 훈련된 rf 모델을 사용하여 예측 수행
new_predictions = loaded_model.predict(test.drop('RIDE_DEMAND',axis=1))

# 예측값을 'RIDE_DEMAND_x' 열에 추가합니다
test['RIDE_DEMAND'] = new_predictions

# 이제 'new_data' DataFrame에는 'RIDE_DEMAND_x' 열이 예측값으로 채워진 상태입니다
# 필요에 따라 'new_data'를 출력하거나 추가적으로 사용할 수 있습니다
test

# 후처리(음수값 나오면 0으로 아니면 정수값으로, 음수 없어서 반올림만) 중요
test["RIDE_DEMAND"] = test["RIDE_DEMAND"].apply(lambda x: max(0, np.round(x)))
test

test.to_csv("Final_exam_pred.csv", index = False)
