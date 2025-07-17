# # Code with no visualization support
# import numpy as np
# import heapq

# class Node:
#     def __init__(self, x, y, distance, bumpiness, total_cost, h_cost, parent=None):
#         self.x = x
#         self.y = y
#         self.distance = distance
#         self.bumpiness = bumpiness
#         self.total_cost = total_cost
#         self.g_cost = None  # will be computed below
#         self.h_cost = h_cost
#         self.parent = parent

#     def compute_g(self, alpha, beta, gamma):
#         self.g_cost = alpha * self.distance + beta * self.bumpiness + gamma * self.total_cost

#     def f_cost(self):
#         return self.g_cost + self.h_cost

#     def __lt__(self, other):
#         return self.f_cost() < other.f_cost()

# def euclidean(p1, p2):
#     return np.linalg.norm(np.array(p1) - np.array(p2))

# def estimate_local_bumpiness(height_map, x, y, directions):
#     h0 = height_map[y, x]
#     if h0 < 0:
#         return 0
#     bump_sum = 0
#     count = 0
#     for dx, dy in directions:
#         nx, ny = x + dx, y + dy
#         if 0 <= nx < height_map.shape[1] and 0 <= ny < height_map.shape[0]:
#             h1 = height_map[ny, nx]
#             if h1 >= 0:
#                 bump_sum += abs(h1 - h0)
#                 count += 1
#     if count == 0:
#         return 0
#     return bump_sum / count

# def reconstruct_path(node):
#     path = []
#     while node:
#         path.append((node.x, node.y))
#         node = node.parent
#     return path[::-1]

# def customized_astar(cost_map, height_map, start, goal,
#                      cost_cutoff=240, alpha=1.0, beta=2.0, gamma=1.0, w1=1.0):
#     H, W = cost_map.shape
#     visited = set()
#     node_map = {}

#     directions = [(0,1), (-1,1), (1,1), (-1,0), (1,0)]  # down, down-left, down-right, left, right

#     sx, sy = start
#     gx, gy = goal

#     start_node = Node(sx, sy, 0.0, 0.0, cost_map[sy, sx], 0.0, None)
#     start_node.compute_g(alpha, beta, gamma)
#     node_map[(sx, sy)] = start_node

#     heap = [start_node]

#     while heap:
#         current = heapq.heappop(heap)
#         if (current.x, current.y) in visited:
#             continue
#         visited.add((current.x, current.y))

#         if (current.x, current.y) == (gx, gy):
#             return reconstruct_path(current), current, node_map

#         for dx, dy in directions:
#             nx, ny = current.x + dxcustomized_astar_local_bump
#                 c = cost_map[ny, nx]
#                 if c >= cost_cutoff:
#                     continue
#                 h0 = height_map[current.y, current.x]
#                 h1 = height_map[ny, nx]
#                 if h0 < 0 or h1 < 0:
#                     continue
#                 step_distance = np.hypot(dx, dy)
#                 height_diff = abs(h1 - h0)
#                 distance = current.distance + step_distance
#                 bumpiness = current.bumpiness + height_diff
#                 total_cost = current.total_cost + c

#                 h = euclidean((nx, ny), (gx, gy)) + w1 * estimate_local_bumpiness(height_map, nx, ny, directions)

#                 new_node = Node(nx, ny, distance, bumpiness, total_cost, h, current)
#                 new_node.compute_g(alpha, beta, gamma)

#                 if (nx, ny) not in node_map or new_node.g_cost < node_map[(nx, ny)].g_cost:
#                     node_map[(nx, ny)] = new_node
#                     heapq.heappush(heap, new_node)

#     return [], None, node_map




# Code with visualization support
import numpy as np
import heapq
import cv2
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y, distance, bumpiness, total_cost, h_cost, parent=None):
        self.x = x
        self.y = y
        self.distance = distance
        self.bumpiness = bumpiness
        self.total_cost = total_cost
        self.g_cost = None  # will be computed below
        self.h_cost = h_cost
        self.parent = parent

    def compute_g(self, alpha, beta, gamma):
        self.g_cost = alpha * self.distance + beta * self.bumpiness + gamma * self.total_cost

    def f_cost(self):
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        return self.f_cost() < other.f_cost()

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def estimate_local_bumpiness(height_map, x, y, directions):
    h0 = height_map[y, x]
    if h0 < 0:
        return 0
    min_bump_diff = 0
    count = 0
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < height_map.shape[1] and 0 <= ny < height_map.shape[0]:
            h1 = height_map[ny, nx]
            if h1 >= 0:
                bump_diff = abs(h1 - h0)
                min_bump_diff = min(min_bump_diff, bump_diff)
                count += 1
    if count == 0:
        return 0
    return min_bump_diff

def reconstruct_path(node):
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

def customized_astar(cost_map, height_map, start, goal,
                     cost_cutoff=240, alpha=1.0, beta=2.0, gamma=1.0, w1=1.0, visualize=False):
    H, W = cost_map.shape
    visited = set()
    node_map = {}

    directions = [(0,1), (-1,1), (1,1), (-1,0), (1,0)]  # down, down-left, down-right, left, right

    sx, sy = start
    gx, gy = goal

    start_node = Node(sx, sy, 0.0, 0.0, cost_map[sy, sx], 0.0, None)
    start_node.compute_g(alpha, beta, gamma)
    node_map[(sx, sy)] = start_node

    heap = [start_node]

    if visualize:
        plt.ion()
        vis_map = cv2.cvtColor(cost_map.copy(), cv2.COLOR_GRAY2BGR)
        vis_map[sy, sx] = (0, 255, 0)  # Start: green
        vis_map[gy, gx] = (255, 0, 0)  # Goal: blue
        history = set()

    while heap:
        current = heapq.heappop(heap)
        if (current.x, current.y) in visited:
            continue
        visited.add((current.x, current.y))

        if visualize:
            vis_map[current.y, current.x] = (255, 255, 255)  # 已展開: white
            for dx, dy in directions:
                nx, ny = current.x + dx, current.y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if cost_map[ny, nx] < cost_cutoff and height_map[ny, nx] >= 0:
                        vis_map[ny, nx] = (0, 0, 255)  # 合法鄰居: red
            vis_map[current.y, current.x] = (0, 165, 255)  # 當前節點: orange
            vis_map_copy = vis_map.copy()
            path = reconstruct_path(current)
            for (px, py) in path:
                vis_map_copy[py, px] = (255, 255, 0)  # path: light blue
            plt.imshow(vis_map_copy[..., ::-1])
            plt.title(f"Expand node ({current.x},{current.y})")
            plt.axis('off')
            plt.draw()
            input("Press Enter to continue...")

        if (current.x, current.y) == (gx, gy):
            path = reconstruct_path(current)
            if visualize:
                cv2.imwrite(f'./case_study_astar_path/astar_path_{alpha}_{beta}_{gamma}_{w1}.png', vis_map_copy)
            return path, current, node_map

        for dx, dy in directions:
            nx, ny = current.x + dx, current.y + dy
            if 0 <= nx < W and 0 <= ny < H:
                if (nx, ny) in visited:
                    continue
                c = cost_map[ny, nx]
                if c >= cost_cutoff:
                    continue
                h0 = height_map[current.y, current.x]
                h1 = height_map[ny, nx]
                if h0 < 0 or h1 < 0:
                    continue
                step_distance = np.hypot(dx, dy)
                height_diff = abs(h1 - h0)
                distance = current.distance + step_distance
                bumpiness = current.bumpiness + height_diff
                total_cost = current.total_cost + c

                h = euclidean((nx, ny), (gx, gy)) + w1 * estimate_local_bumpiness(height_map, nx, ny, directions)

                new_node = Node(nx, ny, distance, bumpiness, total_cost, h, current)
                new_node.compute_g(alpha, beta, gamma)

                if (nx, ny) not in node_map or new_node.g_cost < node_map[(nx, ny)].g_cost:
                    node_map[(nx, ny)] = new_node
                    heapq.heappush(heap, new_node)

    return [], None, node_map
