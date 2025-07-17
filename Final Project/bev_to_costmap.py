import os
import numpy as np
from PIL import Image

# Define RGB → class mapping
color_map = np.array([
    [100, 100, 100],   # 0: others
    [255, 120, 50],    # 1: barrier
    [255, 192, 203],   # 2: bicycle
    [255, 255, 0],     # 3: bus
    [0, 150, 245],     # 4: car
    [0, 255, 255],     # 5: construction_vehicle
    [200, 180, 0],     # 6: motorcycle
    [255, 0, 0],       # 7: pedestrian
    [255, 240, 150],   # 8: traffic_cone
    [135, 60, 0],      # 9: trailer
    [160, 32, 240],    # 10: truck
    [255, 0, 255],     # 11: driveable_surface
    [175, 0, 75],      # 12: other_flat
    [75, 0, 75],       # 13: sidewalk
    [150, 240, 80],    # 14: terrain
    [230, 230, 250],   # 15: manmade
    [0, 175, 0],       # 16: vegetation
    [120, 255, 255],   # 17: free
], dtype=np.uint8)

# Define class → cost mapping
id2cost = np.array([
    200, 255, 255, 255, 255, 255, 255, 255, 230, 255,
    255, 1, 40, 80, 60, 120, 50, 1
], dtype=np.uint8)

# Build RGB to class index lookup dictionary
color_to_class = {tuple(color_map[i]): i for i in range(len(color_map))}

def convert_bev_to_cost_map(bev_image_path, output_path):
    bev_img = Image.open(bev_image_path).convert('RGB')
    bev_np = np.array(bev_img)

    h, w, _ = bev_np.shape
    cost_map = np.full((h, w), 255, dtype=np.uint8)

    for rgb, class_id in color_to_class.items():
        mask = np.all(bev_np == rgb, axis=-1)
        cost_map[mask] = id2cost[class_id]

    # Save cost map image
    Image.fromarray(cost_map, mode='L').save(output_path)
    print(f"[✓] Saved cost map: {output_path}")

def batch_convert_bev_to_cost(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)
            convert_bev_to_cost_map(input_path, output_path)

if __name__ == "__main__":
    input_pred_dir = "./bev_output/scene-0039_pred"        # TODO: 替換成你的 BEV 圖資料夾
    output_pred_dir = "./cost_output/scene-0039_pred"        # TODO: 輸出 cost map 的資料夾
    input_gt_dir = "./bev_output/scene-0039_gt"        # TODO: 替換成你的 BEV 圖資料夾
    output_gt_dir = "./cost_output/scene-0039_gt"        # TODO: 輸出 cost map 的資料夾

    batch_convert_bev_to_cost(input_pred_dir, output_pred_dir)
    batch_convert_bev_to_cost(input_gt_dir, output_gt_dir)
