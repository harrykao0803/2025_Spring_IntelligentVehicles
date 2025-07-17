import os
import numpy as np
from PIL import Image
from collections import Counter

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



# 類別名稱	顏色（RGB）	顏色說明
# others	#000000	黑色（背景/未定義）
# barrier	#FF7832	橘紅色（路障）
# bicycle	#FFC0CB	粉紅色（腳踏車）
# bus	#FFFF00	黃色（公車）
# car	#0096F5	藍色（汽車）
# construction_vehicle	#00FFFF	青色（工程車）
# motorcycle	#C8B400	黃褐色（機車）
# pedestrian	#FF0000	紅色（行人）
# traffic_cone	#FFF096	淡黃橘（交通錐）
# trailer	#873C00	深褐色（拖車）
# truck	#A020F0	紫色（貨車）
# driveable_surface	#FF00FF	桃紅（可行駛區）
# other_flat	#AF004B	暗紅紫（其他平面）
# sidewalk	#4B004B	深紫色（人行道）
# terrain	#96F050	淡綠色（自然地面）
# manmade	#E6E6FA	淺紫灰（人工物）
# vegetation	#00AF00	深綠色（植被）
# free	#78FFFF	天藍色（自由空間）


from collections import Counter
import numpy as np

def fill_unlabeled_center_area(bev_image, semantic_map, center_size=50, default_color=(200, 200, 200), neighbor_range=3):
    H, W = semantic_map.shape
    half_size = center_size // 2
    cx, cy = H // 2, W // 2

    # 優先順序表
    priority_order = {
        'car': 100, 'truck': 95, 'bus': 90,
        'motorcycle': 85, 'pedestrian': 80, 'bicycle': 75,
        'traffic_cone': 70, 'construction_vehicle': 65,
        'driveable_surface': 60, 'sidewalk': 55, 'terrain': 50,
        'manmade': 40, 'vegetation': 35, 'other_flat': 30,
        'barrier': 25, 'trailer': 20, 'others': 0, 'free': 0
    }
    label_priority = np.array([priority_order.get(name, 0) for name in occ_class_names])

    for i in range(cx - half_size, cx + half_size):
        for j in range(cy - half_size, cy + half_size):
            if np.all(bev_image[i, j] == default_color):
                neighbor_labels = []

                # 搜尋一個範圍內的鄰近語意
                for dx in range(-neighbor_range, neighbor_range + 1):
                    for dy in range(-neighbor_range, neighbor_range + 1):
                        x, y = i + dx, j + dy
                        if 0 <= x < H and 0 <= y < W:
                            label = semantic_map[x, y]
                            if label >= 0 and not np.all(bev_image[x, y] == default_color):
                                neighbor_labels.append(label)

                if len(neighbor_labels) == 0:
                    continue  # 還是找不到可用標籤就略過

                counts = Counter(neighbor_labels)
                top_labels = [k for k, v in counts.items() if v == max(counts.values())]

                if len(top_labels) > 1:
                    top_labels.sort(key=lambda x: -label_priority[x])

                chosen = top_labels[0]
                bev_image[i, j] = color_map[chosen]
                semantic_map[i, j] = chosen



def generate_bev_image(semantics, mask, save_path, z_base=(0, 1), z_foreground=(2, 10)):
    H, W, D = semantics.shape
    default_background = [200, 200, 200]
    bev_image = np.full((H, W, 3), default_background, dtype=np.uint8)

    ignore_classes = [0, 17]
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

    z0, z1 = z_base
    for x in range(H):
        for y in range(W):
            valid = mask[x, y, z0:z1 + 1]
            if not np.any(valid):
                continue
            labels = semantics[x, y, z0:z1 + 1][valid]
            mask_valid = ~np.isin(labels, ignore_classes)
            filtered_labels = labels[mask_valid]
            if len(filtered_labels) == 0:
                continue
            base_class = filtered_labels[-1]
            bev_image[x, y] = color_map[base_class]

    zf0, zf1 = z_foreground
    semantic_map = -np.ones((H, W), dtype=int)  # 新增語意地圖記錄每個像素的語意標籤
    for x in range(H):
        for y in range(W):
            valid = mask[x, y, zf0:zf1 + 1]
            if not np.any(valid):
                continue
            labels = semantics[x, y, zf0:zf1 + 1][valid]
            mask_valid = ~np.isin(labels, ignore_classes)
            filtered_labels = labels[mask_valid]
            if len(filtered_labels) == 0:
                continue
            priorities = priority_table[filtered_labels]
            major_class = filtered_labels[np.argmax(priorities)]
            bev_image[x, y] = color_map[major_class]
            semantic_map[x, y] = major_class  # 儲存語意標籤

    # 呼叫補全函式
    fill_unlabeled_center_area(bev_image, semantic_map, center_size=66, default_color=default_background)

    img = Image.fromarray(bev_image)
    img.save(save_path)
    print(f"[✓] 已儲存 BEV 到 {save_path}（地板 z={z0}-{z1}，物體 z={zf0}-{zf1}）")


# 定義生成 BEV 圖像的函式
# def generate_bev_image(semantics, mask, save_path, z_base=(0, 1), z_foreground=(2, 10)):
#     H, W, D = semantics.shape  # 取得語意體素的尺寸（高、寬、深）
#     default_background = [200, 200, 200]  # 預設背景色（灰色）
#     bev_image = np.full((H, W, 3), default_background, dtype=np.uint8)  # 初始化 BEV 圖像，全部設為灰色背景

#     ignore_classes = [0, 17]  # 忽略的類別，例如 "others" 和 "free"

#     # 定義語意類別的優先權，數字越大優先權越高
#     semantic_priority = {
#         'car': 100, 'truck': 95, 'bus': 95,
#         'motorcycle': 90, 'pedestrian': 85, 'bicycle': 80,
#         'traffic_cone': 75, 'construction_vehicle': 70,
#         'sidewalk': 60, 'terrain': 50, 'driveable_surface': 40,
#         'others': 0, 'free': 0
#     }

#     # 建立類別索引對應的優先權表
#     priority_table = np.zeros(len(occ_class_names))  # 初始化優先權表
#     for i, name in enumerate(occ_class_names):
#         priority_table[i] = semantic_priority.get(name, 0)  # 若無定義則優先權為 0

#     # 第一步：渲染地板圖層（從 z=0 到 z=1）
#     z0, z1 = z_base
#     for x in range(H):
#         for y in range(W):
#             valid = mask[x, y, z0:z1 + 1]  # 取得該點 z 軸範圍內的有效遮罩
#             if not np.any(valid):  # 如果都無有效體素就跳過
#                 continue
#             labels = semantics[x, y, z0:z1 + 1][valid]  # 取得有效位置的語意標籤
#             mask_valid = ~np.isin(labels, ignore_classes)  # 去除忽略的類別
#             filtered_labels = labels[mask_valid]
#             if len(filtered_labels) == 0:
#                 continue
#             base_class = filtered_labels[-1]  # 使用最上層的有效類別當作底層地板
#             bev_image[x, y] = color_map[base_class]  # 塗上對應的顏色

#     # 第二步：覆蓋物體圖層（從 z=2 到 z=15）
#     zf0, zf1 = z_foreground
#     for x in range(H):
#         for y in range(W):
#             valid = mask[x, y, zf0:zf1 + 1]  # 取得物體層的有效體素
#             if not np.any(valid):
#                 continue
#             labels = semantics[x, y, zf0:zf1 + 1][valid]  # 取得語意標籤
#             mask_valid = ~np.isin(labels, ignore_classes)  # 過濾掉不需要的類別
#             filtered_labels = labels[mask_valid]
#             if len(filtered_labels) == 0:
#                 continue
#             priorities = priority_table[filtered_labels]  # 查詢每個類別的優先權
#             major_class = filtered_labels[np.argmax(priorities)]  # 取出優先權最高的類別
#             bev_image[x, y] = color_map[major_class]  # 覆蓋原圖的顏色

#     # 儲存生成的 BEV 圖片
#     img = Image.fromarray(bev_image)  # 將 ndarray 轉為圖片格式
#     img.save(save_path)  # 儲存圖片
#     print(f"[✓] 已儲存 BEV 到 {save_path}（地板 z={z0}-{z1}，物體 z={zf0}-{zf1}）")


# 處理整個場景資料夾的函式
def process_scene(scene_root, output_dir, mode="gt"):
    # 找出所有子資料夾（每個場景資料）
    subfolders = [f for f in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, f))]
    for folder in subfolders:
        # 可取消註解只處理單一資料夾
        # if folder != "163b70e627854893b88575caf85a56ea":
        #     break

        # 根據模式決定載入的 npz 檔案
        if mode == "gt":
            npz_path = os.path.join(scene_root, folder, "labels.npz")
        else:
            npz_path = os.path.join(scene_root, folder, "pred.npz")

        # 若檔案不存在就跳過
        if not os.path.exists(npz_path):
            print(f"[跳過] {npz_path} 不存在")
            continue

        try:
            # ground truth 模式
            if mode == "gt":
                data = np.load(npz_path)
                semantics = data["semantics"]  # 讀取語意體素
                mask = data["mask_lidar"].astype(bool)  # 使用 LiDAR 的有效遮罩

            # 預測結果模式
            elif mode == "pred":
                data = np.load(npz_path)
                # 從對應的 ground truth 讀取 camera 遮罩
                gt_path = os.path.join("../../data/nuscenes/gts/scene-0039", folder, "labels.npz")
                camera_mask = np.load(gt_path)["mask_camera"]
                semantics = data["pred"]  # 讀取模型預測語意體素
                mask = (semantics <= 17) & camera_mask  # 過濾無效預測 + 使用 camera 遮罩

            else:
                print(f"[錯誤] 未知模式: {mode}")
                continue

            # 設定輸出路徑並建立資料夾
            output_path = os.path.join(output_dir, f"{folder}.png")
            os.makedirs(output_dir, exist_ok=True)

            # 生成並儲存 BEV 圖像
            generate_bev_image(semantics, mask, output_path)

        except Exception as e:
            # 處理過程中若有錯誤則印出錯誤訊息
            print(f"[錯誤] 處理 {folder} 時發生錯誤: {e}")
            continue

# 主程式進入點
if __name__ == "__main__":
    # # 設定 ground truth 的輸入與輸出資料夾
    # gt_input = "../../data/nuscenes/gts/scene-0039"
    # gt_output = "./bev_output/scene-0039_gt"
    # process_scene(gt_input, gt_output, mode="gt")  # 處理 ground truth

    # 設定預測結果的輸入與輸出資料夾
    pred_input = "../../work_dirs/flashocc_r50/results/scene-0039"
    pred_output = "../../Harry_code/output/bev_output_origin/scene-0039_pred"
    process_scene(pred_input, pred_output, mode="pred")  # 處理預測結果



