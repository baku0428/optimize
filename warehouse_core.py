import numpy as np
from collections import deque


# =========================================================
# 1) Warehouse
#    0 = Walkway
#    1 = Shelf
# =========================================================

warehouse = np.array([
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
], dtype=int)


# =========================================================
# 2) Entrance / Exit
# =========================================================

entrance = (0, 0)
exit_point = (17, 19)


# 4-direction movement
MOVEMENTS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
]


# =========================================================
# 3) Basic validation
# =========================================================

def is_valid_position(position):
    row, col = position
    rows, cols = warehouse.shape

    return 0 <= row < rows and 0 <= col < cols


def is_walkway(position):
    return (
        is_valid_position(position)
        and warehouse[position] == 0
    )


def validate_points(pickup_points):

    if not is_walkway(entrance):
        raise ValueError("Entrance must be on walkway (0)")

    if not is_walkway(exit_point):
        raise ValueError("Exit must be on walkway (0)")

    for point in pickup_points:

        if not is_valid_position(point):
            raise ValueError(
                f"Pickup {point} is outside warehouse"
            )

        if warehouse[point] != 1:
            raise ValueError(
                f"Pickup {point} must be on shelf (1)"
            )

        if len(get_pickable_cells(point)) == 0:
            raise ValueError(
                f"Pickup {point} has no adjacent walkway"
            )


# =========================================================
# 4) Generate Pickup Points
# =========================================================

def generate_pickup_points(n, seed=20):

    shelf_positions = list(
        zip(*np.where(warehouse == 1))
    )

    # เลือกเฉพาะ shelf ที่มีทางเดินติดอยู่
    shelf_positions = [
        point
        for point in shelf_positions
        if len(get_pickable_cells(point)) > 0
    ]

    if n > len(shelf_positions):
        raise ValueError(
            f"Cannot generate {n} pickup points. "
            f"Only {len(shelf_positions)} pickable shelves available."
        )

    rng = np.random.default_rng(seed)

    selected_indices = rng.choice(
        len(shelf_positions),
        size=n,
        replace=False
    )

    # แปลงเป็น int ธรรมดา (ไม่ให้แสดงเป็น np.int64)
    return [
        (int(shelf_positions[index][0]), int(shelf_positions[index][1]))
        for index in selected_indices
    ]


# =========================================================
# 5) หา Walkway ที่สามารถยืนหยิบสินค้าได้
# =========================================================

def get_pickable_cells(pickup_point):
    """
    Pickup อยู่บน shelf (1)

    คืนค่าช่อง walkway (0) ที่ติดกับ shelf
    ใน 4 ทิศ
    """

    row, col = pickup_point

    cells = []

    for dr, dc in MOVEMENTS:

        neighbor = (
            row + dr,
            col + dc
        )

        if is_walkway(neighbor):
            cells.append(
                (int(neighbor[0]), int(neighbor[1]))
            )

    return cells


# =========================================================
# 6) BFS
# =========================================================

def bfs(start, goal):
    """
    หา shortest path ระหว่าง walkway 2 จุด

    return:
        distance
        path

    path เช่น:
        [(0,0), (1,0), (1,1), ...]
    """

    if not is_walkway(start):
        return float("inf"), []

    if not is_walkway(goal):
        return float("inf"), []

    queue = deque([start])

    parent = {
        start: None
    }

    while queue:

        current = queue.popleft()

        if current == goal:

            # reconstruct path
            path = []

            node = goal

            while node is not None:
                path.append(node)
                node = parent[node]

            path.reverse()

            distance = len(path) - 1

            return distance, path

        row, col = current

        for dr, dc in MOVEMENTS:

            neighbor = (
                row + dr,
                col + dc
            )

            if (
                is_walkway(neighbor)
                and neighbor not in parent
            ):
                parent[neighbor] = current
                queue.append(neighbor)

    return float("inf"), []


# =========================================================
# 7) BFS จากจุดหนึ่งไปหลายเป้าหมาย
# =========================================================

def shortest_path_to_any(start, goals):

    best_distance = float("inf")
    best_path = []
    best_goal = None

    for goal in goals:

        distance, path = bfs(
            start,
            goal
        )

        if distance < best_distance:

            best_distance = distance
            best_path = path
            best_goal = goal

    return (
        best_distance,
        best_path,
        best_goal
    )


# =========================================================
# 8) ระยะทางระหว่าง Pickup สองจุด
# =========================================================

def pickup_to_pickup_distance(
    pickup_a,
    pickup_b
):
    """
    หา shortest distance ระหว่าง
    walkway รอบ Pickup A
    กับ walkway รอบ Pickup B
    """

    cells_a = get_pickable_cells(pickup_a)
    cells_b = get_pickable_cells(pickup_b)

    best_distance = float("inf")

    for cell_a in cells_a:

        for cell_b in cells_b:

            distance, _ = bfs(
                cell_a,
                cell_b
            )

            if distance < best_distance:
                best_distance = distance

    return best_distance


# =========================================================
# 9) BFS จากหลายจุดเริ่มพร้อมกัน (Multi-source BFS)
#
# คืนระยะทางจาก "จุดเริ่มที่ใกล้ที่สุด" ไปยังทุกช่องทางเดิน
# ใช้สร้าง Distance Matrix ได้เร็วกว่าการเรียก bfs() ทีละคู่มาก
# (BFS 1 รอบต่อ 1 จุดหยิบ แทนที่จะเป็นหลายรอบต่อ 1 คู่)
# =========================================================

def bfs_distance_map(sources):

    distance_map = {}
    queue = deque()

    for source in sources:

        if is_walkway(source) and source not in distance_map:
            distance_map[source] = 0
            queue.append(source)

    while queue:

        current = queue.popleft()
        row, col = current

        for dr, dc in MOVEMENTS:

            neighbor = (
                row + dr,
                col + dc
            )

            if (
                is_walkway(neighbor)
                and neighbor not in distance_map
            ):
                distance_map[neighbor] = distance_map[current] + 1
                queue.append(neighbor)

    return distance_map


def min_distance_to(distance_map, cells):

    return min(
        (distance_map.get(cell, float("inf")) for cell in cells),
        default=float("inf")
    )


# =========================================================
# 10) Build Distance Matrix
#
# Index:
#
# 0       = Entrance
# 1..n    = Pickup
# n + 1   = Exit
#
# ระยะทาง Pickup A -> Pickup B
#   = ระยะสั้นสุดจากช่องยืนใดๆ ของ A ไปช่องยืนใดๆ ของ B
# =========================================================

def build_distance_matrix(pickup_points):

    validate_points(pickup_points)

    n = len(pickup_points)

    matrix = np.full(
        (n + 2, n + 2),
        np.inf
    )

    np.fill_diagonal(matrix, 0)

    # ช่องยืนของแต่ละโหนด: Entrance, Pickup 1..n, Exit
    node_cells = (
        [[entrance]]
        + [get_pickable_cells(p) for p in pickup_points]
        + [[exit_point]]
    )

    for i in range(n + 2):

        distance_map = bfs_distance_map(
            node_cells[i]
        )

        for j in range(i + 1, n + 2):

            distance = min_distance_to(
                distance_map,
                node_cells[j]
            )

            matrix[i][j] = distance
            matrix[j][i] = distance

    return matrix


# =========================================================
# 11) ระยะเดินจริง (ใช้ร่วมกันทุก Algorithm)
#
# ข้อจำกัดของ Distance Matrix ข้างบน:
#   มองแต่ละ Pickup เป็น 1 โหนด แล้วใช้ "คู่ช่องยืนที่ใกล้ที่สุด" ของแต่ละคู่แยกกัน
#   ถ้าชั้นวางมีช่องยืน 2 ฝั่ง (เช่น ซ้าย/ขวาของชั้น) matrix จะยอมให้
#   "เดินเข้าฝั่งซ้าย แล้วออกฝั่งขวา" ได้ฟรี = เหมือนเดินทะลุชั้นวาง
#   -> ระยะตาม matrix ต่ำกว่าความจริง และยิ่งจุดเยอะยิ่งคลาดเคลื่อนมาก
#
# ของจริง: หยิบ Pickup แต่ละจุดต้องยืนที่ "ช่องเดียว" (เข้า-ออกช่องเดียวกัน)
# evaluate_order() ใช้ Dynamic Programming เลือกช่องยืนของทุกจุด
# ให้ระยะรวมสั้นที่สุดภายใต้ลำดับที่กำหนด = จำนวนก้าวเดินจริง
# (ใช้เป็น fitness ได้เลย: เร็วเพราะ BFS ถูกคำนวณไว้ล่วงหน้าใน build_route_data)
# =========================================================

def build_route_data(pickup_points):
    """
    เตรียมข้อมูลสำหรับคำนวณระยะเดินจริง

    return dict:
        pickup_points : จุดหยิบ
        node_cells    : ช่องยืนของแต่ละโหนด [Entrance, P1..Pn, Exit]
        node_cell_idx : index ของช่องยืนใน cell_dist
        cells         : รายการช่องยืนทั้งหมด (ไม่ซ้ำ)
        cell_dist     : ระยะ BFS ระหว่างช่องยืนทุกคู่ (numpy array)
        matrix        : Distance Matrix ระดับ Pickup (เหมือน build_distance_matrix)
    """

    validate_points(pickup_points)

    node_cells = (
        [[entrance]]
        + [get_pickable_cells(p) for p in pickup_points]
        + [[exit_point]]
    )

    cells = list(dict.fromkeys(
        cell for group in node_cells for cell in group
    ))

    index = {cell: i for i, cell in enumerate(cells)}

    cell_dist = np.full((len(cells), len(cells)), np.inf)

    for i, cell in enumerate(cells):

        distance_map = bfs_distance_map([cell])

        for j, other in enumerate(cells):
            cell_dist[i][j] = distance_map.get(other, np.inf)

    node_cell_idx = [
        [index[cell] for cell in group]
        for group in node_cells
    ]

    size = len(node_cells)
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            matrix[i][j] = cell_dist[
                np.ix_(node_cell_idx[i], node_cell_idx[j])
            ].min()

    return {
        "pickup_points": list(pickup_points),
        "node_cells": node_cells,
        "node_cell_idx": node_cell_idx,
        "cells": cells,
        "cell_dist": cell_dist,
        "matrix": matrix,
    }


def evaluate_order(order, route_data):
    """
    ระยะเดินจริงของลำดับการหยิบ (order = index 0..n-1)

    return: (steps, standing_cells)
        steps          = จำนวนก้าวรวมจริง Entrance -> ... -> Exit
        standing_cells = ช่องที่ยืนหยิบของแต่ละจุด เรียงตาม order
    """

    C = route_data["cell_dist"]
    groups = route_data["node_cell_idx"]

    layers = (
        [groups[0]]
        + [groups[i + 1] for i in order]
        + [groups[-1]]
    )

    # cost[k] = ระยะสั้นสุดจาก Entrance มาถึงช่องยืนที่ k ของชั้นปัจจุบัน
    cost = np.zeros(1)
    back = []

    for prev, cur in zip(layers, layers[1:]):

        total = cost[:, None] + C[np.ix_(prev, cur)]
        back.append(total.argmin(axis=0))
        cost = total.min(axis=0)

    # ย้อนกลับหาช่องยืนที่เลือก
    k = 0
    chosen = []

    for layer, parent in zip(reversed(layers[:-1]), reversed(back)):
        k = parent[k]
        chosen.append(route_data["cells"][layer[k]])

    chosen.reverse()

    return int(cost[0]), chosen[1:]


def reconstruct_full_path(order, pickup_points, route_data=None):
    """
    สร้างเส้นทางเดินทุกช่องจากลำดับการหยิบ

    return: (path, steps, standing_cells)
        path           = พิกัดทุกช่องตั้งแต่ Entrance ถึง Exit
        steps          = จำนวนก้าวจริง = len(path) - 1
        standing_cells = ช่องที่ยืนหยิบของแต่ละจุด เรียงตาม order
    """

    if route_data is None:
        route_data = build_route_data(pickup_points)

    _, standing_cells = evaluate_order(order, route_data)

    chosen = [entrance] + standing_cells + [exit_point]

    path = [entrance]

    for a, b in zip(chosen, chosen[1:]):
        _, segment = bfs(a, b)
        path.extend(segment[1:])

    return path, len(path) - 1, standing_cells


# =========================================================
# 12) คำนวณ Distance จาก Pickup Order
# =========================================================

def calculate_order_distance(
    order,
    distance_matrix
):
    """
    order ใช้ index 0..n-1

    เช่น
    [2, 0, 1]

    = Pickup 3 -> Pickup 1 -> Pickup 2
    """

    n = len(order)

    if n == 0:
        return distance_matrix[0][-1]

    total = 0

    # Entrance -> First Pickup
    total += distance_matrix[
        0
    ][
        order[0] + 1
    ]

    # Pickup -> Pickup
    for i in range(n - 1):

        current = order[i] + 1
        next_point = order[i + 1] + 1

        total += distance_matrix[
            current
        ][
            next_point
        ]

    # Last Pickup -> Exit
    exit_index = len(distance_matrix) - 1

    total += distance_matrix[
        order[-1] + 1
    ][
        exit_index
    ]

    return total


# =========================================================
# 13) Print Distance Matrix
# =========================================================

def print_distance_matrix(
    distance_matrix
):

    n = len(distance_matrix) - 2

    labels = (
        ["IN"]
        + [
            f"P{i + 1}"
            for i in range(n)
        ]
        + ["OUT"]
    )

    print("\n===== DISTANCE MATRIX =====")

    print(
        "       ",
        end=""
    )

    for label in labels:
        print(
            f"{label:>7}",
            end=""
        )

    print()

    for i, row in enumerate(
        distance_matrix
    ):

        print(
            f"{labels[i]:>7}",
            end=""
        )

        for value in row:

            if np.isinf(value):
                text = "INF"
            else:
                text = str(int(value))

            print(
                f"{text:>7}",
                end=""
            )

        print()