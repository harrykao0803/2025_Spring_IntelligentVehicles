import os
import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from pathlib import Path

def dilate_grayscale_cost_map(cost_map, car_height=15, car_width=8):
    """
    對灰階 cost map 進行最大池化膨脹處理（保留 cost 漸層）
    """
    return maximum_filter(cost_map, size=(car_height, car_width))

def process_cost_map_folder(input_dir, output_dir, car_height=15, car_width=8):
    """
    批次處理資料夾中的 PNG cost map，進行膨脹，並儲存到 output_dir
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".png"):
            input_path = input_dir / filename
            output_path = output_dir / filename


            # 讀入灰階 cost map（uint8: 0-255）
            cost_map = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            # 灰階膨脹（保留 cost 漸層）
            dilated_map = dilate_grayscale_cost_map(cost_map, car_height, car_width)


            # 儲存結果
            cv2.imwrite(str(output_path), dilated_map)
            print(f"✅ Processed: {filename}")

# 🟢 執行：指定兩個資料夾
process_cost_map_folder(
    input_dir="./cost_output/scene-0039_gt",
    output_dir="./dilated_costmap/scene-0039_gt",
    car_height=15,
    car_width=8
)

process_cost_map_folder(
    input_dir="./cost_output/scene-0039_pred",
    output_dir="./dilated_costmap/scene-0039_pred",
    car_height=15,
    car_width=8
)
