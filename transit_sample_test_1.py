#%%
import os
import time
import psutil
import folium
import duckdb
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

from folium import plugins
from datetime import datetime
from pyproj import Transformer # 좌표 변환 (EPSG:5179 -> WGS84) 

#%%
# 성능 모니터링 클래스
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process(os.getpid())
        
    def start(self):
        """모니터링 시작"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"시작 메모리: {self.start_memory:.2f} MB")
        print("="*60)
        
    def checkpoint(self, step_name):
        """중간 체크포인트"""
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        elapsed_time = current_time - self.start_time
        
        print(f"{step_name}")
        print(f"경과 시간: {elapsed_time:.2f}초")
        print(f"현재 메모리: {current_memory:.2f} MB")
        print(f"메모리 증가: {current_memory - self.start_memory:.2f} MB")
        print("-" * 40)
        
    def end(self):
        """모니터링 종료"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        total_time = end_time - self.start_time
        memory_usage = end_memory - self.start_memory
        
        print("="*60)
        print("최종 성능 리포트")
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 실행 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        print(f"시작 메모리: {self.start_memory:.2f} MB")
        print(f"종료 메모리: {end_memory:.2f} MB")
        print(f"메모리 사용량: {memory_usage:.2f} MB")
        print(f"최대 메모리: {self.process.memory_info().rss / 1024 / 1024:.2f} MB")
        print("="*60)

#%%
# 글로벌 모니터 인스턴스 생성
monitor = PerformanceMonitor()

# 모니터링 시작
monitor.start()

# 데이터 로드
print("데이터 로딩 중...")
query = """
    SELECT 
        date, o_cell_id, o_cell_x, o_cell_y,
        move_purpose, move_dist, move_time, total_cnt
    FROM '/home1/rldnjs16/transit/dataset/data_month/year=2024/month=01/summary_data_commute.parquet'
"""

df2401 = duckdb.query(query).to_df()

#%%
print(f"데이터 로드 완료: {len(df2401):,}행 x {len(df2401.columns)}열")
monitor.checkpoint("DuckDB로 필요한 컬럼만 로딩 완료 - 데이터 로드 완료")

df2401['date'] = pd.to_datetime(df2401['date'], format='%Y%m%d')
df2401['weekday'] = df2401['date'].dt.weekday # 요일: Monday=0, Sunday=6
df2401['is_weekend'] = df2401['weekday'].apply(lambda x: 1 if x >= 5 else 0) # 평일/주말 컬럼 생성 (0=평일, 1=주말)

df2401_ = df2401[(df2401['move_purpose'] == 1) & df2401['is_weekend'] == 0] # 이동 목적 출근, 평일
# df2401_move = df2401_.groupby('o_cell_id')['move_time'].sum().sort_values(ascending=False)
df2401_move = df2401_.groupby('o_cell_id')[['move_time', 'total_cnt']].sum().sort_values(by='move_time', ascending=False).reset_index()

#%%
#%%
# 1. 먼저 데이터 확인 및 준비
print("=== 데이터 정보 확인 ===")
print(f"총 격자 수: {len(df2401_move)}")
print(f"이동시간 범위: {df2401_move['move_time'].min():.1f} ~ {df2401_move['move_time'].max():.1f}")
print(f"이동인구 범위: {df2401_move['total_cnt'].min():.0f} ~ {df2401_move['total_cnt'].max():.0f}")
print("\n상위 5개 격자:")
print(df2401_move.head())

#%%
# 2. 원본 데이터에서 좌표 정보 가져오기
coord_data = df2401_[['o_cell_id', 'o_cell_x', 'o_cell_y']].drop_duplicates()
df_with_coords = df2401_move.merge(coord_data, on='o_cell_id', how='left')

print(f"\n좌표 결합 후 데이터 수: {len(df_with_coords)}")
print(f"좌표 X 범위: {df_with_coords['o_cell_x'].min()} ~ {df_with_coords['o_cell_x'].max()}")
print(f"좌표 Y 범위: {df_with_coords['o_cell_y'].min()} ~ {df_with_coords['o_cell_y'].max()}")

#%%
# 3. 좌표 변환 함수 (EPSG:5179 -> WGS84)
def transform_coordinates(x, y):
    """한국 표준 좌표계를 WGS84로 변환"""
    try:
        transformer = Transformer.from_crs('EPSG:5179', 'EPSG:4326', always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon
    except:
        return None, None

#%%
# 4. 좌표 변환 및 데이터 정제
print("\n=== 좌표 변환 중 ===")
valid_data_with_coords = []

for idx, row in df_with_coords.iterrows():
    lat, lon = transform_coordinates(row['o_cell_x'], row['o_cell_y'])
    
    if lat is not None and lon is not None:
        # 수도권 범위 체크 (대략적)
        if 36.8 <= lat <= 38.2 and 126.3 <= lon <= 127.8:
            # 행 데이터에 변환된 좌표 추가
            row_dict = row.to_dict()
            row_dict['lat'] = lat
            row_dict['lon'] = lon
            valid_data_with_coords.append(row_dict)
    
    if idx % 1000 == 0:
        print(f"처리 중... {idx}/{len(df_with_coords)}")

df_final = pd.DataFrame(valid_data_with_coords)
print(f"\n변환 완료! 유효한 데이터: {len(df_final)}개")

#%%
# 5. 시각화를 위한 데이터 정규화
def normalize_data(series, min_val=0.1, max_val=1.0):
    """데이터를 min_val과 max_val 사이로 정규화"""
    min_s, max_s = series.min(), series.max()
    if max_s == min_s:
        return pd.Series([0.5] * len(series), index=series.index)
    normalized = (series - min_s) / (max_s - min_s)
    return normalized * (max_val - min_val) + min_val

# 색상용 정규화 (이동시간)
df_final['color_intensity'] = normalize_data(df_final['move_time'], 0.2, 1.0)

# 크기용 정규화 (이동인구)
df_final['size_factor'] = normalize_data(df_final['total_cnt'], 5, 25)

#%%
# 6. Folium 지도 생성
def create_interactive_map():
    """인터랙티브 지도 생성"""
    
    # 수도권 중심 지도
    m = folium.Map(
        location=[37.5665, 126.9780],  # 서울 중심
        zoom_start=10,
        tiles='CartoDB positron'  # 깔끔한 배경
    )
    
    # 색상 함수 정의
    def get_color(intensity):
        """이동시간 강도에 따른 색상 반환"""
        if intensity >= 0.8:
            return '#d73027'  # 진한 빨강 (매우 높음)
        elif intensity >= 0.6:
            return '#fc8d59'  # 주황 (높음)
        elif intensity >= 0.4:
            return '#fee08b'  # 노랑 (보통)
        elif intensity >= 0.2:
            return '#d9ef8b'  # 연두 (낮음)
        else:
            return '#91bfdb'  # 파랑 (매우 낮음)
    
    # 격자별 원 추가
    for idx, row in df_final.iterrows():
        lat, lon = row['lat'], row['lon']
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=row['size_factor'],
            popup=folium.Popup(
                f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <b>격자 ID:</b> {row['o_cell_id']}<br>
                    <b>이동시간:</b> {row['move_time']:,.0f}분<br>
                    <b>이동인구:</b> {row['total_cnt']:,.0f}명<br>
                    <b>좌표:</b> ({lat:.4f}, {lon:.4f})
                </div>
                """,
                max_width=200
            ),
            tooltip=f"이동시간: {row['move_time']:,.0f}분 | 인구: {row['total_cnt']:,.0f}명",
            color='white',
            weight=1,
            fillColor=get_color(row['color_intensity']),
            fillOpacity=0.7
        ).add_to(m)
    
    # 범례 추가
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 140px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>이동시간 (색상)</b></p>
    <p><i class="fa fa-circle" style="color:#d73027"></i> 매우 높음</p>
    <p><i class="fa fa-circle" style="color:#fc8d59"></i> 높음</p>
    <p><i class="fa fa-circle" style="color:#fee08b"></i> 보통</p>
    <p><i class="fa fa-circle" style="color:#d9ef8b"></i> 낮음</p>
    <p><i class="fa fa-circle" style="color:#91bfdb"></i> 매우 낮음</p>
    <br>
    <p><b>원 크기</b> = 이동인구</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

#%%
# 7. 히트맵 버전 생성 함수
def create_heatmap_version():
    """히트맵 버전 지도"""
    m_heat = folium.Map(
        location=[37.5665, 126.9780],
        zoom_start=10,
        tiles='CartoDB dark_matter'
    )
    
    # 히트맵 데이터 준비 (이동시간 기준)
    heat_data = []
    for idx, row in df_final.iterrows():
        lat, lon = row['lat'], row['lon']  # DataFrame에서 직접 가져오기
        heat_data.append([lat, lon, row['color_intensity']])
    
    # 히트맵 레이어 추가
    plugins.HeatMap(
        heat_data,
        radius=20,
        blur=15,
        max_zoom=17,
        gradient={
            0.0: 'blue',
            0.3: 'cyan', 
            0.5: 'lime',
            0.7: 'yellow',
            1.0: 'red'
        }
    ).add_to(m_heat)
    
    return m_heat

#%%
# 8. 지도 생성 및 저장
print("\n=== 지도 생성 중 ===")
map_viz = create_interactive_map()
map_viz.save('/home1/rldnjs16/transit/map_visualization/seoul_commute_grid_map.html')
print("지도 저장 완료: seoul_commute_grid_map.html")

# 히트맵 버전도 저장
heat_map = create_heatmap_version()
heat_map.save('/home1/rldnjs16/transit/map_visualization/seoul_commute_heatmap.html')
print("히트맵 버전도 저장: seoul_commute_heatmap.html")

#%%
# 9. 통계 요약
print("\n=== 데이터 통계 요약 ===")
print(f"총 시각화된 격자 수: {len(df_final):,}개")
print(f"평균 이동시간: {df_final['move_time'].mean():,.1f}분")
print(f"평균 이동인구: {df_final['total_cnt'].mean():,.0f}명")
print(f"최대 이동시간 격자: {df_final.loc[df_final['move_time'].idxmax(), 'o_cell_id']}")
print(f"최대 이동인구 격자: {df_final.loc[df_final['total_cnt'].idxmax(), 'o_cell_id']}")

#%%
# 10. 상관관계 분석
correlation = df_final['move_time'].corr(df_final['total_cnt'])
print(f"\n이동시간 vs 이동인구 상관계수: {correlation:.3f}")

print("\n🎉 모든 시각화 완료!")
print("📊 생성된 파일:")
print("   - seoul_commute_grid_map.html (격자별 원 시각화)")
print("   - seoul_commute_heatmap.html (히트맵 버전)")
