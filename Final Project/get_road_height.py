import os
import numpy as np
from tqdm import tqdm

def generate_surface_height_map(semantics: np.ndarray, mask: np.ndarray, drivable_class=11):
    """
    從語意預測與 mask 中產生 200x200 的路面高度圖
    - drivable_surface 填入最低 z，並做高度平移
    - 其他填入 -1 表示不可通行
    """
    H, W, Z = semantics.shape

    # 自動補上 mask 維度（若為 H×W）
    if mask.shape == (H, W):
        mask = np.broadcast_to(mask[..., None], (H, W, Z))
    elif mask.shape != semantics.shape:
        raise ValueError(f"mask shape {mask.shape} 不符合 semantics shape {semantics.shape}")

    surface_map = np.full((H, W), -1.0, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            for k in range(Z):  # 從底層往上找
                if not mask[i, j, k]:
                    continue
                if semantics[i, j, k] == drivable_class:
                    surface_map[i, j] = float(k)

    # 平移最低高度為 0
    valid = surface_map >= 0
    if np.any(valid):
        surface_map[valid] -= surface_map[valid].min()

    return surface_map

def process_all_frames(pred_root, gt_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    frame_ids = sorted([f for f in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, f))])

    print(f"🚗 準備處理 {len(frame_ids)} 個 frames...")

    for frame_id in tqdm(frame_ids):
        pred_path = os.path.join(pred_root, frame_id, "pred.npz")
        gt_path = os.path.join(gt_root, frame_id, "labels.npz")
        output_path = os.path.join(output_root, frame_id + ".npy")

        if not os.path.exists(pred_path):
            print(f"[跳過] 找不到 {pred_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"[跳過] 找不到對應的 GT {gt_path}")
            continue

        try:
            pred_npz = np.load(pred_path)
            gt_npz = np.load(gt_path)

            semantics = pred_npz["pred"]  # shape: (H, W, Z)
            mask = gt_npz["mask_camera"]  # shape: (H, W, Z) 或 (H, W)

            height_map = generate_surface_height_map(semantics, mask, drivable_class=11)
            np.save(output_path, height_map)

        except Exception as e:
            print(f"[❌ 錯誤] Frame {frame_id} 發生例外：{e}")
            continue

if __name__ == "__main__":
    # 路徑設定（請依實際環境修改）
    pred_root = "../../work_dirs/flashocc_r50/results/scene-0039"
    gt_root = "../../data/nuscenes/gts/scene-0039"
    output_root = "./road_height_output/scene-0039_pred"

    process_all_frames(pred_root, gt_root, output_root)
