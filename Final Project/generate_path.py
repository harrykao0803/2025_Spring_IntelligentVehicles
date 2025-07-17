import numpy as np
import cv2
import matplotlib.pyplot as plt
from customized_astar import customized_astar
import os

# =============================
# 設定與資料讀取
# =============================
costmap_pth = f"./dilated_costmap/scene-0039_pred/"          # xxxx.png
height_map_pth = f"./road_height_output/scene-0039_pred/"     # xxxx.npy
output_pth = f"./local_path_output/scene-0039_pred/"         # 儲存路徑的資料夾
os.makedirs(output_pth, exist_ok=True)

for image in os.listdir(costmap_pth):
    if image.endswith(".png"):
        costmap_img_pth = os.path.join(costmap_pth, image)
        height_map_img_pth = os.path.join(height_map_pth, image.replace(".png", ".npy"))

    cost_map_full = cv2.imread(costmap_img_pth, cv2.IMREAD_GRAYSCALE)
    height_map_full = np.load(height_map_img_pth)

    # =============================
    # 擷取 local 區域
    # =============================
    cx, cy = 100, 100  # 車輛中心
    local_cost = cost_map_full[cy : cy+20, cx-8 : cx+8]
    local_height = height_map_full[cy : cy+20, cx-8 : cx+8]
    local_origin = (8, 0)  # local 座標系中的車輛位置

    # 全域 global goal
    global_goal = (100, 199)

    # =============================
    # 動態挑選最佳 local goal
    # =============================
    candidates = []
    for ly in range(local_cost.shape[0]):
        for lx in range(local_cost.shape[1]):
            gx, gy = cx - 8 + lx, cy + ly
            if height_map_full[gy, gx] < 0:  # 不可通行
                continue
            if cost_map_full[gy, gx] >= 240:
                continue
            dist = np.linalg.norm(np.array([gx, gy]) - np.array(global_goal))  # 計算到 global goal 的歐氏距離
            candidates.append((dist, (lx, ly)))

    if not candidates:
        print("找不到合法的 local goal！")
        exit()

    candidates.sort()
    local_goal = candidates[0][1]  # 找出距離 global goal 最近的 local point 作為 local goal

    # # For debug: 印出所有候選 local goal
    # for dist, (lx, ly) in candidates:
    #     gx, gy = cx - 8 + lx, cy + ly
    #     print(f"候選 local goal: ({{lx}}, {{ly}}) -> 全域座標: ({{gx}}, {{gy}}), 距離: {{dist:.2f}}")

    # =============================
    # 執行 A*
    # =============================
    path, goal_node, node_map = customized_astar(
        local_cost,
        local_height,
        local_origin,
        local_goal,
        cost_cutoff=240,
        alpha=1.0,      # 路徑長度權重
        beta=1.0,       # 路徑起伏權重
        gamma=1.0,      # 路徑 cost 權重
        w1=1.0, 
        visualize=False  # 不需要在 local 區域內視覺化         
    )

    # =============================
    # 視覺化：畫在整張 cost_map 上
    # =============================
    visual_full = cv2.cvtColor(cost_map_full, cv2.COLOR_GRAY2BGR)

    # 畫路徑
    for (lx, ly) in path:
        gx, gy = cx - 8 + lx, cy + ly
        visual_full[gy, gx] = (0, 0, 255)

    # 起點與終點
    visual_full[cy, cx] = (0, 255, 0)  # start: green
    goal_gx, goal_gy = cx - 8 + local_goal[0], cy + local_goal[1]
    visual_full[goal_gy, goal_gx] = (255, 0, 0)  # goal: blue

    # # 畫 local goal
    # plt.imshow(visual_full[..., ::-1])
    # plt.title("A* Local Path Toward Global Goal (100,199)")
    # plt.axis('off')
    # plt.show()

    output_img_path = os.path.join(output_pth, image)
    cv2.imwrite(output_img_path, visual_full)
    print(f"✅ 已儲存: {output_img_path}")