import os
import numpy as np
import cv2
from tqdm import tqdm

# 語義對應名稱（僅供參考）
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

# FlashOCC 預設的語義顏色表
color_map = np.array([
    [0, 0, 0, 255],    # others
    [255, 120, 50, 255], [255, 192, 203, 255], [255, 255, 0, 255], [0, 150, 245, 255],
    [0, 255, 255, 255], [200, 180, 0, 255], [255, 0, 0, 255], [255, 240, 150, 255],
    [135, 60, 0, 255], [160, 32, 240, 255], [255, 0, 255, 255], [175, 0, 75, 255],
    [75, 0, 75, 255], [150, 240, 80, 255], [230, 230, 250, 255],
    [0, 175, 0, 255], [255, 255, 255, 255]
], dtype=np.uint8)


# | Index | Class Name            | RGB Color (R,G,B) | 顏色描述                |
# | ----- | --------------------- | ----------------- | ------------------- |
# | 0     | others                | (0, 0, 0)         | 黑色 (black)          |
# | 1     | barrier               | (255, 120, 50)    | 橘紅色 (orange-red)    |
# | 2     | bicycle               | (255, 192, 203)   | 粉紅色 (pink)          |
# | 3     | bus                   | (255, 255, 0)     | 鮮黃色 (bright yellow) |
# | 4     | car                   | (0, 150, 245)     | 天藍色 (sky blue)      |
# | 5     | construction\_vehicle | (0, 255, 255)     | 青色 (cyan)           |
# | 6     | motorcycle            | (200, 180, 0)     | 暗黃色 (dark yellow)   |
# | 7     | pedestrian            | (255, 0, 0)       | 紅色 (red)            |
# | 8     | traffic\_cone         | (255, 240, 150)   | 淺黃色 (light yellow)  |
# | 9     | trailer               | (135, 60, 0)      | 棕色 (brown)          |
# | 10    | truck                 | (160, 32, 240)    | 紫色 (purple)         |
# | 11    | driveable\_surface    | (255, 0, 255)     | 品紅色 (magenta)       |
# | 12    | other\_flat           | (175, 0, 75)      | 暗紅色 (dark red)      |
# | 13    | sidewalk              | (75, 0, 75)       | 深紫色 (dark purple)   |
# | 14    | terrain               | (150, 240, 80)    | 淺綠色 (light green)   |
# | 15    | manmade               | (230, 230, 250)   | 淡紫白 (lavender)      |
# | 16    | vegetation            | (0, 175, 0)       | 綠色 (green)          |
# | 17    | free                  | (255, 255, 255)   | 白色 (white)          |


def occ2bev_img(sem3d):
    """sem3d: (Y, X, Z)，預測語義"""
    H, W, D = sem3d.shape
    free_id = len(occ_class_names) - 1  # 通常為 'free'
    semantics_2d = np.ones((H, W), dtype=np.int32) * free_id
    for i in range(D):
        semantics_i = sem3d[..., i]
        non_free_mask = (semantics_i != free_id)
        semantics_2d[non_free_mask] = semantics_i[non_free_mask]
    # 使用語義顏色表
    viz = color_map[semantics_2d][..., :3]  # RGB
    viz = cv2.resize(viz, dsize=(800, 800), interpolation=cv2.INTER_NEAREST)
    return viz

def process_scene(scene_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tokens = sorted(os.listdir(scene_dir))
    for token in tqdm(tokens, desc=f"Processing {os.path.basename(scene_dir)}"):
        pred_path = os.path.join(scene_dir, token, "pred.npz")
        if not os.path.exists(pred_path):
            continue
        data = np.load(pred_path)
        sem3d = data["pred"]  # shape: (Y, X, Z)
        bev_img = occ2bev_img(sem3d)
        cv2.imwrite(os.path.join(output_dir, f"{token}.jpg"), bev_img)

if __name__ == "__main__":
    # 設定要處理哪個 scene 資料夾
    scene_dir = "../work_dirs/flashocc_r50/results/scene-0003"
    output_dir = "bev_vis/gen-scene-0003"
    process_scene(scene_dir, output_dir)
    print(f"✅ 已儲存 BEV 圖至：{output_dir}")
