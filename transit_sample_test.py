#%%
'''
DATA SAMPLE TEST
기본 데이터 기반 - 
1. 행정동 별 데이터 정제
2. 구 별 데이터 정제
'''

#%%
'''IMPORTS'''
import os
import time
import psutil
import folium 
import duckdb
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import branca.colormap as cm
import matplotlib.pyplot as plt

from folium import plugins
from datetime import datetime
from pyproj import Transformer # 좌표 변환 (EPSG:5179 -> WGS84) 
from branca.colormap import linear
from folium.plugins import HeatMap
from shapely.geometry import Polygon, MultiPolygon

import warnings
warnings.filterwarnings('ignore')

#%%
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.last_checkpoint_time = None  # 이전 체크포인트 시간 추가
        self.process = psutil.Process(os.getpid())
        self.checkpoints = []  # 체크포인트 기록 저장
        
    def start(self):
        """모니터링 시작"""
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time  # 초기화
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"시작 메모리: {self.start_memory:.2f} MB")
        print("="*60)
        
    def checkpoint(self, step_name):
        """중간 체크포인트 - 개선된 버전"""
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # 총 경과 시간
        total_elapsed = current_time - self.start_time
        
        # 이전 체크포인트부터의 시간 (단계별 시간)
        step_time = current_time - self.last_checkpoint_time
        
        # 체크포인트 기록 저장
        checkpoint_data = {
            'step': step_name,
            'step_time': step_time,
            'total_elapsed': total_elapsed,
            'memory_usage': current_memory,
            'memory_increase': current_memory - self.start_memory
        }
        self.checkpoints.append(checkpoint_data)
        
        print(f"[{step_name}]")
        print(f"단계 실행시간: {step_time:.2f}초")
        print(f"총 경과시간: {total_elapsed:.2f}초")
        print(f"현재 메모리: {current_memory:.2f} MB")
        print(f"메모리 증가: {current_memory - self.start_memory:.2f} MB")
        print("-" * 40)
        
        # 다음 체크포인트를 위해 시간 업데이트
        self.last_checkpoint_time = current_time
        
    def end(self):
        """모니터링 종료 - 상세 리포트 포함"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        total_time = end_time - self.start_time
        memory_usage = end_memory - self.start_memory
        
        print("="*60)
        print("최종 성능 리포트")
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 실행 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        print(f"메모리 사용량: {memory_usage:.2f} MB")
        print("="*60)
        
        # 단계별 상세 리포트
        print("\n단계별 실행시간 상세:")
        print("-" * 60)
        for i, checkpoint in enumerate(self.checkpoints, 1):
            percentage = (checkpoint['step_time'] / total_time) * 100
            print(f"{i}. {checkpoint['step']}")
            print(f"   실행시간: {checkpoint['step_time']:.2f}초 ({percentage:.1f}%)")
            print(f"   메모리: {checkpoint['memory_usage']:.2f} MB")
        print("-" * 60)
        
        # 가장 시간이 오래 걸린 단계
        slowest_step = max(self.checkpoints, key=lambda x: x['step_time'])
        print(f"\n가장 오래 걸린 단계: {slowest_step['step']} ({slowest_step['step_time']:.2f}초)")
        print("="*60)

#%%
monitor = PerformanceMonitor()
monitor.start()

duckdb.sql("PRAGMA threads=2")  # 병렬 제한

def load_month_data(year, month, monitor=None):
    query = f"""
        SELECT 
            date, o_admi_cd, o_cell_id, o_cell_x, o_cell_y, o_cell_tp,
            d_admi_cd, d_cell_id, d_cell_x, d_cell_y, d_cell_tp,
            move_purpose, move_dist, move_time, total_cnt
        FROM '/home1/rldnjs16/transit/dataset/data_month/year={year:04}/month={month:02}/data.parquet'
    """
    print(f"{month}월 데이터 로딩 중...")
    table = duckdb.query(query).arrow()
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    print(f"데이터 로드 완료: {len(df):,}행 x {len(df.columns)}열")
    if monitor:
        monitor.checkpoint(f"DuckDB {month}월 데이터 로딩")

    return df

df2406 = load_month_data(2024, 6, monitor)
df2407 = load_month_data(2024, 7, monitor)
df2408 = load_month_data(2024, 8, monitor)
df2409 = load_month_data(2024, 9, monitor)
df2410 = load_month_data(2024, 10, monitor)
df2411 = load_month_data(2024, 11, monitor)
df2412 = load_month_data(2024, 12, monitor)
df2501 = load_month_data(2025, 1, monitor)
df2502 = load_month_data(2025, 2, monitor)
df2503 = load_month_data(2025, 3, monitor)
df2504 = load_month_data(2025, 4, monitor)
df2505 = load_month_data(2025, 5, monitor)

df = pd.concat([df2406, df2407, df2408, df2409, df2410, df2411, df2412, df2501, df2502, df2503, df2504, df2505], ignore_index=True)

monitor.checkpoint("DuckDB 데이터 로딩 (Arrow 방식)")

#%%
# 데이터 필터링 및 그룹핑
df_ = df[df['move_purpose'] == 1]
df_g = df_.groupby('o_cell_id').agg({
    'move_dist': 'sum',
    'move_time': 'sum',
    'total_cnt': 'sum',
    'o_cell_x' : 'first',
    'o_cell_y' : 'first'
}).reset_index()
monitor.checkpoint("데이터 필터링 및 그룹핑")

#%%
# 좌표 변환
transformer = Transformer.from_crs("epsg:5179", "epsg:4326", always_xy=True)
df_g['lon'], df_g['lat'] = transformer.transform(
    df_g['o_cell_x'].values,
    df_g['o_cell_y'].values
)
monitor.checkpoint("좌표 변환 (EPSG:5179 -> WGS84)")

#%%
# 지도 생성
# 1. 기본 
# min_time, max_time = df_g['move_time'].min(), df_g['move_time'].max()
# colormap = cm.linear.YlOrRd_09.scale(min_time, max_time)

# 2. 로그 스케일
df_g['move_time_log'] = np.log1p(df_g['move_time'])  # log(1+x) 안정적
min_t = df_g['move_time_log'].min()
max_t = df_g['move_time_log'].max()
colormap = cm.linear.YlOrRd_09.scale(min_t, max_t)

# 3. 이상치(분위수 상하위 1%) 제거
# vmin = df_g['move_time'].quantile(0.10)
# vmax = df_g['move_time'].quantile(0.90)

# colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)

m = folium.Map(location=[37.5665, 126.9780], zoom_start=10)
monitor.checkpoint("지도 객체 생성 및 색상 설정")

#%%
# 원 추가
for _, row in df_g.iterrows():
    folium.Circle(
        location=[row['lat'], row['lon']],
        radius=max(row['total_cnt'] / 25, 5),
        color=colormap(row['move_time_log']),
        fill=True,
        fill_opacity=0.7,
        popup=(
            f"격자이름: {row['o_cell_id']}<br>"
            f"격자: ({row['o_cell_x']}, {row['o_cell_y']})<br>"
            f"이동인구수: {row['total_cnt']}<br>"
            f"이동시간: {row['move_time']}분"
        )
    ).add_to(m)

colormap.caption = "이동시간 (분)"
colormap.add_to(m)
monitor.checkpoint("지도 시각화 (원 및 범례 추가)")

#%%
# 저장
m.save("/home1/rldnjs16/transit/map_visualization/seoul_transit_moving_map.html")
monitor.checkpoint("seoul_transit_moving_map HTML 파일 저장1")




# %%
df_g['move_time'].hist(bins=100)
plt.title("Original")
plt.show()

df_g['move_time_log'].hist(bins=100)
plt.title("Log scale")
plt.show()

#%%
# 행정동 별 그룹핑 및 시각화
df_g_h = df_.groupby(['o_admi_cd', 'o_cell_id']).agg({
    'move_dist': 'sum', # mean
    'move_time': 'sum',
    'total_cnt': 'sum',
    'o_cell_x' : 'mean',
    'o_cell_y' : 'mean'
}).reset_index()


# %%
transformer = Transformer.from_crs("epsg:5179", "epsg:4326", always_xy=True)
df_g_h['lon'], df_g_h['lat'] = transformer.transform(
    df_g_h['o_cell_x'].values,
    df_g_h['o_cell_y'].values
)

#%%
df_g_h_g = df_g_h.groupby('o_admi_cd').agg(
    {
    'move_dist': 'sum', # mean
    'move_time': 'sum',
    'total_cnt': 'sum',
    'o_cell_x' : 'mean',
    'o_cell_y' : 'mean'
    }
).reset_index()

#%%
transformer = Transformer.from_crs("epsg:5179", "epsg:4326", always_xy=True)
df_g_h_g['lon'], df_g_h_g['lat'] = transformer.transform(
    df_g_h_g['o_cell_x'].values,
    df_g_h_g['o_cell_y'].values
)

# %%
# 2. 로그 스케일
df_g_h_g['move_time_log'] = np.log1p(df_g_h_g['move_time'])  # log(1+x) 안정적
min_t = df_g_h_g['move_time_log'].min()
max_t = df_g_h_g['move_time_log'].max()
colormap = cm.linear.YlOrRd_09.scale(min_t, max_t)

m = folium.Map(location=[37.5665, 126.9780], zoom_start=10)
monitor.checkpoint("지도 객체 생성 및 색상 설정")

#%%
# 원 추가
for _, row in df_g_h_g.iterrows():
    folium.Circle(
        location=[row['lat'], row['lon']],
        radius=max(row['total_cnt'] / 25, 5),
        color=colormap(row['move_time_log']),
        fill=True,
        fill_opacity=0.7,
        popup=(
            f"행정동이름: {row['o_admi_cd']}<br>"
            f"위경도: ({row['lat']}, {row['lon']})<br>"
            f"이동인구수: {row['total_cnt']}<br>"
            f"이동시간: {row['move_time']}분"
        )
    ).add_to(m)

colormap.caption = "이동시간 (분)"
colormap.add_to(m)
monitor.checkpoint("지도 시각화 (원 및 범례 추가)")

#%%
m.save("/home1/rldnjs16/transit/map_visualization/seoul_transit_moving_map_admi.html")
monitor.checkpoint("seoul_transit_moving_map_admi HTML 파일 저장2")











# %%
# 1. 행정동 GeoJSON 로딩
gdf_adm = gpd.read_file("/home1/rldnjs16/transit/Administrative_boundaries/BND_ADM_DONG_PG/BND_ADM_DONG_PG.shp")
gdf_adm.to_file("BND_ADM_DONG_PG.geojson", driver='GeoJSON')
# gdf_adm: 이 데이터는 동만 제공하고 있으므로, 동이름이 중복되는 경우 merge 하는 과정에서 문제가 발생한다.

# 2. o_admi_cd를 기준으로 df_grouped와 merge
gdf_adm['ADM_CD'] = gdf_adm['ADM_CD'].astype(str)
df_g_h_g['o_admi_cd'] = df_g_h_g['o_admi_cd'].astype(str)
gdf_merged = gdf_adm.merge(df_g_h_g, left_on='ADM_CD', right_on='o_admi_cd', how='left')
print(gdf_merged['o_admi_cd'].isnull().mean())  # → 0.0이면 정상, 크면 문제 있음
# 더군다나 gdf_adm은 수도권 이 외의 데이터도 갖으므로, 더욱 문제가 됨

gdf_valid = gdf_merged[~gdf_merged['o_admi_cd'].isnull()]  # 이동 정보가 있는 행정동만
# 즉 이동 정보가 있는 행정동은 수도권에 해당할 것임
gdf_valid.shape
# 32개의 데이터만 있는데, 문제가 있어보임

#%%
# 동 이름 외에 지역명 전체를 가져오기 위한 작업
gdf = pd.read_excel(
    "/home1/rldnjs16/transit/Administrative_boundaries/센서스 공간정보 지역 코드.xlsx",
    skiprows=1,
    engine='openpyxl',
    dtype={'시도코드': str, '시군구코드': str, '읍면동코드': str}  # <- 여기서 'ADMI_CD'를 문자열로 강제 지정
)

gdf = gdf[(gdf['시도명칭'] == '서울특별시') | (gdf['시도명칭'] == '경기도') | (gdf['시도명칭'] == '인천광역시')]
gdf['NAME'] = gdf['시도코드'] + gdf['시군구코드'] + gdf['읍면동코드']

#%%
# 동일한 데이터이므로 행정동 코드 기반 merge
gdf_merge = gdf_adm.merge(gdf, left_on='ADM_CD', right_on='NAME', how='left')
gdf_merge_ = gdf_merge[~gdf_merge['시도코드'].isna()] # 수도권 데이터만 가져오기(gdf_merge는 수도권 외 데이터도 있으므로)

gdf_merge_['FULL_NAME'] = gdf_merge_['시도명칭'] +' ' + gdf_merge_['시군구명칭']+' ' + gdf_merge_['읍면동명칭']

gdf_merge_.drop(columns=['시도명칭', '시군구명칭', '읍면동명칭', 'BASE_DATE', 'NAME', '시도코드', '시군구코드', '읍면동코드'], inplace=True)

col_r = []
col_list = ['ADM_CD', 'ADM_NM', 'FULL_NAME']
for col in gdf_merge_.columns:
    if col not in col_list:
        col_r.append(col)
gdf_merge_ = gdf_merge_[col_list + col_r]

# 지역명 전체 가져오기 성공

#%%
# 우리의 데이터 행정동 코드로 변경하는 작업
# gdf_merge_의 행정동 코드와 우리의 데이터 행정동 코드가 다름
df_admi = pd.read_csv('/home1/rldnjs16/transit/ADMI_RE/ADMI_202406.csv')

df_admi['ADMI_CD']= df_admi['ADMI_CD'].astype('str')
gdf_merge_2 = gdf_merge_.merge(df_admi, left_on='FULL_NAME', right_on='FULL_NM', how='left')

gdf_merge_2.drop(columns=['ADM_CD', 'ADM_NM', 'FULL_NAME', 'SIDO_NM', 'SGG_NM', 'ADMI_NM', 'BASE_YM'], inplace=True)
gdf_merge_2 = gdf_merge_2[['FULL_NM', 'ADMI_CD', 'geometry']]
gdf_merged_3 = gdf_merge_2.merge(df_g_h_g, left_on='ADMI_CD', right_on='o_admi_cd', how='left')
gdf_merged_3.drop(columns='ADMI_CD',inplace=True)

#%%
# 중심 좌표 설정 (서울 중심)
m = folium.Map(location=[37.5665, 126.9780], zoom_start=10)

# 색상 맵핑 (이동시간 기준)
# min_time, max_time = gdf_merged_3['move_time'].min(), gdf_merged_3['move_time'].max()
# colormap = linear.YlOrRd_09.scale(min_time, max_time)

# df_g_h_g['move_time_log'] = np.log1p(df_g_h_g['move_time'])  # log(1+x) 안정적
min_t = df_g_h_g['move_time_log'].min()
max_t = df_g_h_g['move_time_log'].max()
colormap = cm.linear.YlOrRd_09.scale(min_t, max_t)
colormap.caption = '이동시간 (분)'
colormap.add_to(m)

# Choropleth (행정동 폴리곤 색 채우기)
folium.GeoJson(
    gdf_merged_3,
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['move_time_log']) if feature['properties']['move_time_log'] else '#ffffff',
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.6
    },
    tooltip=folium.features.GeoJsonTooltip(fields=['o_admi_cd', 'move_time_log', 'total_cnt'],
                                           aliases=['행정동코드', '이동시간(분)', '이동인구수'],
                                           localize=True)
).add_to(m)

# 중심좌표 (o_cell_x, o_cell_y → 위경도 변환)
transformer = Transformer.from_crs("epsg:5186", "epsg:4326")  # UTMK → 위경도
gdf_merged_3['lat'], gdf_merged_3['lon'] = transformer.transform(gdf_merged_3['o_cell_y'].values,
                                                             gdf_merged_3['o_cell_x'].values)

gdf_merged_3['centroid'] = gdf_merged_3['geometry'].centroid

# 원 표시 (이동인구수 반영)
for _, row in gdf_merged_3.dropna(subset=['total_cnt']).iterrows():
    folium.Circle(
        location=[row['lat'], row['lon']],
        radius=max(row['total_cnt'] / 50, 1),
        color=colormap(row['move_time_log']),
        fill=True,
        fill_opacity=0.7,
        popup=(f"행정동: {row['o_admi_cd']}<br>"
               f"이동인구수: {row['total_cnt']:,}명<br>"
               f"이동시간: {row['move_time_log']:.1f}분")
    ).add_to(m)

# 결과 저장
m.save("/home1/rldnjs16/transit/map_visualization/seoul_transit_moving_map_adm_gdf.html")
m

monitor.checkpoint("seoul_mobility_by_adm_gdf HTML 파일 저장")
monitor.end()
print('finished')
