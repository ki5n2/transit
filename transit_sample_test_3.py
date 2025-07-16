#%%
import os
import duckdb
import pandas as pd
import numpy as np
import umap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import folium

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pyproj import Transformer

# ==============================
# 데이터 로딩
# ==============================

def load_month_data(month):
    query = f"""
        SELECT 
            o_cell_id, o_cell_x, o_cell_y,
            move_dist, move_time, total_cnt
        FROM '/home1/rldnjs16/transit/dataset/data_month/year=2024/month={month:02}/data.parquet'
        WHERE move_purpose = 1
    """
    print(f"{month}월 데이터 로딩 중...")
    df = duckdb.query(query).to_df()
    print(f"{month}월: {len(df):,} rows")
    return df

# 6~8월 데이터 불러오기
df = pd.concat([load_month_data(m) for m in [6, 7, 8, 9, 10, 11, 12]], ignore_index=True)

# ==============================
# 군집 대상 Feature 구성
# ==============================

df_grouped = df.groupby('o_cell_id').agg({
    'move_dist': 'sum',
    'move_time': 'sum',
    'total_cnt': 'sum',
    'o_cell_x': 'first',
    'o_cell_y': 'first'
}).reset_index()

X_raw = df_grouped[['move_dist', 'move_time', 'total_cnt']].values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ==============================
# Autoencoder 정의
# ==============================

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# ==============================
# Autoencoder 학습
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

model.train()
for epoch in range(300):
    optimizer.zero_grad()
    x_recon, _ = model(X_tensor)
    loss = criterion(x_recon, X_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ==============================
# 인코딩 및 UMAP
# ==============================

model.eval()
with torch.no_grad():
    _, latent = model(X_tensor)
    latent_np = latent.cpu().numpy()

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(latent_np)

# ==============================
# 클러스터링
# ==============================

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_umap)

df_grouped['cluster'] = cluster_labels
df_grouped['umap_x'] = X_umap[:, 0]
df_grouped['umap_y'] = X_umap[:, 1]

# ==============================
# UMAP 시각화 (평면)
# ==============================

plt.figure(figsize=(8, 6))
for c in np.unique(cluster_labels):
    plt.scatter(X_umap[cluster_labels == c, 0],
                X_umap[cluster_labels == c, 1],
                label=f'Cluster {c}')
plt.legend()
plt.title("UMAP + KMeans Clustering")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)
plt.show()

# ==============================
# 지도 시각화 (Folium)
# ==============================

# 좌표 변환: EPSG:5179 -> WGS84
transformer = Transformer.from_crs("epsg:5179", "epsg:4326", always_xy=True)
df_grouped['lon'], df_grouped['lat'] = transformer.transform(
    df_grouped['o_cell_x'].values,
    df_grouped['o_cell_y'].values
)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

for _, row in df_grouped.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        color=colors[row['cluster'] % len(colors)],
        fill=True,
        fill_opacity=0.7,
        popup=(f"격자: {row['o_cell_id']}<br>"
               f"Cluster: {row['cluster']}<br>"
               f"이동인구수: {row['total_cnt']:,}명<br>"
               f"이동거리: {row['move_dist']:.1f}m<br>"
               f"이동시간: {row['move_time']:.1f}분")
    ).add_to(m)

# 결과 저장
m.save("/home1/rldnjs16/transit/map_visualization/seoul_cluster_umap_map.html")
print("지도 저장 완료: seoul_cluster_umap_map.html")
