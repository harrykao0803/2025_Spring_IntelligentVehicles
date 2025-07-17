import numpy as np
import matplotlib.pyplot as plt

# 載入某一張高度圖
height_map = np.load("./road_height_output/scene-0039_pred/7c8ce823cc374a8f9376df6d8f0fed89.npy")

# 檢查 shape 與範圍
print("Shape:", height_map.shape)
print("最小值:", np.min(height_map), "最大值:", np.max(height_map))

# 可視化
plt.imshow(height_map, cmap='viridis')
plt.colorbar(label="Height (Z)")
plt.title("Drivable Surface Height Map")
plt.show()
