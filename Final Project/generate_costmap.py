import os
import numpy as np
from PIL import Image

# Semantic class names
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

color_map = np.array([
    [100, 100, 100, 255],
    [255, 120, 50, 255],
    [255, 192, 203, 255],
    [255, 255, 0, 255],
    [0, 150, 245, 255],
    [0, 255, 255, 255],
    [200, 180, 0, 255],
    [255, 0, 0, 255],
    [255, 240, 150, 255],
    [135, 60, 0, 255],
    [160, 32, 240, 255],
    [255, 0, 255, 255],
    [175, 0, 75, 255],
    [75, 0, 75, 255],
    [150, 240, 80, 255],
    [230, 230, 250, 255],
    [0, 175, 0, 255],
    [120, 255, 255, 255],
], dtype=np.uint8)[:, :3]  # RGB only

# Cost table per semantic class (lower is more drivable)
id2cost = np.array([
    200,  # 0: others
    255,  # 1: barrier
    255,  # 2: bicycle
    255,  # 3: bus
    255,  # 4: car
    255,  # 5: construction_vehicle
    255,  # 6: motorcycle
    255,  # 7: pedestrian
    230,  # 8: traffic_cone
    255,  # 9: trailer
    255,  # 10: truck
      1,  # 11: driveable_surface
     40,  # 12: other_flat
     80,  # 13: sidewalk
     60,  # 14: terrain
    120,  # 15: manmade
     50,  # 16: vegetation
      1,  # 17: free
], dtype=np.uint8)

# Semantic priority for 3D-to-BEV projection
semantic_priority = {
    'car': 100, 'truck': 95, 'bus': 95,
    'motorcycle': 90, 'pedestrian': 85, 'bicycle': 80,
    'traffic_cone': 75, 'construction_vehicle': 70,
    'sidewalk': 60, 'terrain': 50, 'driveable_surface': 40,
    'others': 0, 'free': 0
}
priority_table = np.zeros(len(occ_class_names))
for i, name in enumerate(occ_class_names):
    priority_table[i] = semantic_priority.get(name, 0)


def generate_bev_image(semantics, mask, save_path, z_base=(0, 1), z_foreground=(2, 10)):
    H, W, D = semantics.shape  # 取得語意體素的尺寸（高、寬、深）
    default_background = [200, 200, 200]  # 預設背景色（灰色）
    bev_image = np.full((H, W, 3), default_background, dtype=np.uint8)  # 初始化 BEV 圖像，全部設為灰色背景

    ignore_classes = [0, 17]  # 忽略的類別，例如 "others" 和 "free"

    # 建立類別索引對應的優先權表
    priority_table = np.zeros(len(occ_class_names))  # 初始化優先權表
    for i, name in enumerate(occ_class_names):
        priority_table[i] = semantic_priority.get(name, 0)  # 若無定義則優先權為 0

    # 第一步：渲染地板圖層（從 z=0 到 z=1）
    z0, z1 = z_base
    for x in range(H):
        for y in range(W):
            valid = mask[x, y, z0:z1 + 1]  # 取得該點 z 軸範圍內的有效遮罩
            if not np.any(valid):  # 如果都無有效體素就跳過
                continue
            labels = semantics[x, y, z0:z1 + 1][valid]  # 取得有效位置的語意標籤
            mask_valid = ~np.isin(labels, ignore_classes)  # 去除忽略的類別
            filtered_labels = labels[mask_valid]
            if len(filtered_labels) == 0:
                continue
            base_class = filtered_labels[-1]  # 使用最上層的有效類別當作底層地板
            bev_image[x, y] = color_map[base_class]  # 塗上對應的顏色

    # 第二步：覆蓋物體圖層（從 z=2 到 z=15）
    zf0, zf1 = z_foreground
    for x in range(H):
        for y in range(W):
            valid = mask[x, y, zf0:zf1 + 1]  # 取得物體層的有效體素
            if not np.any(valid):
                continue
            labels = semantics[x, y, zf0:zf1 + 1][valid]  # 取得語意標籤
            mask_valid = ~np.isin(labels, ignore_classes)  # 過濾掉不需要的類別
            filtered_labels = labels[mask_valid]
            if len(filtered_labels) == 0:
                continue
            priorities = priority_table[filtered_labels]  # 查詢每個類別的優先權
            major_class = filtered_labels[np.argmax(priorities)]  # 取出優先權最高的類別
            bev_image[x, y] = color_map[major_class]  # 覆蓋原圖的顏色

    # 儲存生成的 BEV 圖片
    img = Image.fromarray(bev_image)  # 將 ndarray 轉為圖片格式
    img.save(save_path)  # 儲存圖片
    print(f"[✓] 已儲存 BEV 到 {save_path}（地板 z={z0}-{z1}，物體 z={zf0}-{zf1}）")

    return bev_image  # 返回生成的 BEV 圖像


def semantic_bev_to_cost_map(bev_map):
    return id2cost[bev_map]  # shape: (H, W), uint8


def generate_cost_map_from_semantics(semantics, mask, save_path, z_base=(0, 1), z_foreground=(2, 10)):
    bev_semantics = generate_bev_image(semantics, mask, save_path, z_base, z_foreground)
    cost_map = semantic_bev_to_cost_map(bev_semantics)

    img = Image.fromarray(cost_map, mode='L')
    img.save(save_path)
    print(f"[✓] Saved cost map to {save_path}")


# def generate_cost_map(semantics, mask, save_path, z_base=(0, 1), z_foreground=(2, 15)):
#     H, W, D = semantics.shape
#     cost_map = np.full((H, W), 255, dtype=np.uint8)
#     ignore_classes = [0, 17]

#     # Base layer (z=0~1)
#     z0, z1 = z_base
#     for x in range(H):
#         for y in range(W):
#             valid = mask[x, y, z0:z1 + 1]
#             if not np.any(valid):
#                 continue
#             labels = semantics[x, y, z0:z1 + 1][valid]
#             labels = labels[~np.isin(labels, ignore_classes)]
#             if len(labels) == 0:
#                 continue
#             base_class = labels[-1]
#             cost_map[x, y] = id2cost[base_class]

#     # Foreground layer (z=2~15)
#     zf0, zf1 = z_foreground
#     for x in range(H):
#         for y in range(W):
#             valid = mask[x, y, zf0:zf1 + 1]
#             if not np.any(valid):
#                 continue
#             labels = semantics[x, y, zf0:zf1 + 1][valid]
#             labels = labels[~np.isin(labels, ignore_classes)]
#             if len(labels) == 0:
#                 continue
#             priorities = priority_table[labels]
#             major_class = labels[np.argmax(priorities)]
#             cost_map[x, y] = id2cost[major_class]

#     img = Image.fromarray(cost_map, mode='L')
#     img.save(save_path)
#     print(f"[✓] Saved cost map to {save_path}")


def process_scene(scene_root, output_dir, mode="gt"):
    subfolders = [f for f in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, f))]
    for folder in subfolders:
        if mode == "gt":
            npz_path = os.path.join(scene_root, folder, "labels.npz")
        else:
            npz_path = os.path.join(scene_root, folder, "pred.npz")

        if not os.path.exists(npz_path):
            print(f"[Skipped] {npz_path} not found")
            continue

        try:
            if mode == "gt":
                data = np.load(npz_path)
                semantics = data["semantics"]
                mask = data["mask_lidar"].astype(bool)
            elif mode == "pred":
                data = np.load(npz_path)
                gt_path = os.path.join("../../data/nuscenes/gts/scene-0039", folder, "labels.npz")
                camera_mask = np.load(gt_path)["mask_camera"]
                semantics = data["pred"]
                mask = (semantics <= 17) & camera_mask
            else:
                print(f"[Error] Unknown mode: {mode}")
                continue

            output_path = os.path.join(output_dir, f"{folder}.png")
            os.makedirs(output_dir, exist_ok=True)
            generate_cost_map_from_semantics(semantics, mask, output_path)

        except Exception as e:
            print(f"[Error] Failed processing {folder}: {e}")
            continue
        
        
if __name__ == "__main__":
    gt_input = "../../data/nuscenes/gts/scene-0039"
    gt_output = "./cost_output/scene-0039_gt"
    gt_input = "../../data/nuscenes/gts/scene-0039"

    process_scene(gt_input, gt_output, mode="gt")

    pred_input = "../../work_dirs/flashocc_r50/results/scene-0039"
    pred_output = "./cost_output/scene-0039_pred"
    pred_input = "../../work_dirs/flashocc_r50/results/scene-0039"

    process_scene(pred_input, pred_output, mode="pred")
