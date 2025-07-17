import numpy as np
import cv2
import networkx as nx

def cost_map_to_graph(cost_map: np.ndarray, connectivity=4):
    """
    將灰階 cost map 轉為 weighted graph。
    每個像素為一個節點，相鄰像素連邊，邊權為目的地的 cost。
    
    Args:
        cost_map (np.ndarray): 2D 灰階 cost map。
        connectivity (int): 4 or 8，表示相鄰邊的方式。
    
    Returns:
        G (nx.DiGraph): 有向加權圖。
    """
    H, W = cost_map.shape
    G = nx.DiGraph()  # 有向圖（從 A 指向 B，花費為 B 的成本）

    for x in range(H):
        for y in range(W):
            current_node = (x, y)
            current_cost = cost_map[x, y]

            # 4 或 8 鄰接方式
            directions = [(-1,0), (1,0), (0,-1), (0,1)]
            if connectivity == 8:
                directions += [(-1,-1), (-1,1), (1,-1), (1,1)]

            for dx, dy in directions:
                nx_, ny_ = x + dx, y + dy  # ✅ 完全不會衝突
                if 0 <= nx_ < H and 0 <= ny_ < W:
                    neighbor_cost = cost_map[nx_, ny_]
                    G.add_edge(current_node, (nx_, ny_), weight=neighbor_cost)

    return G
# 讀取 cost map
cost_map = cv2.imread("./cost_output/scene-0003_gt/1ac0914c98b8488cb3521efeba354496.png", cv2.IMREAD_GRAYSCALE)

# 轉成圖
graph = cost_map_to_graph(cost_map)

# 使用 Dijkstra 找最短路徑（例如從左上到右下）
start = (0, 0)
end = (cost_map.shape[0] - 1, cost_map.shape[1] - 1)

shortest_path = nx.dijkstra_path(graph, source=start, target=end, weight='weight')
print("最短路徑長度:", nx.dijkstra_path_length(graph, source=start, target=end, weight='weight'))
