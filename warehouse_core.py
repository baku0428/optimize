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
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
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

    return [
        shelf_positions[index]
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
            cells.append(neighbor)

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
# 9) Build Distance Matrix
#
# Index:
#
# 0       = Entrance
# 1..n    = Pickup
# n + 1   = Exit
# =========================================================

def build_distance_matrix(pickup_points):

    validate_points(pickup_points)

    n = len(pickup_points)

    matrix = np.full(
        (n + 2, n + 2),
        np.inf
    )

    np.fill_diagonal(matrix, 0)

    # -----------------------------------------------------
    # Entrance <-> Pickup
    # -----------------------------------------------------

    for i, pickup in enumerate(
        pickup_points,
        start=1
    ):

        pickup_cells = get_pickable_cells(
            pickup
        )

        distance, _, _ = shortest_path_to_any(
            entrance,
            pickup_cells
        )

        matrix[0][i] = distance
        matrix[i][0] = distance

    # -----------------------------------------------------
    # Pickup <-> Pickup
    # -----------------------------------------------------

    for i in range(n):

        for j in range(i + 1, n):

            distance = pickup_to_pickup_distance(
                pickup_points[i],
                pickup_points[j]
            )

            matrix[i + 1][j + 1] = distance
            matrix[j + 1][i + 1] = distance

    # -----------------------------------------------------
    # Pickup <-> Exit
    # -----------------------------------------------------

    for i, pickup in enumerate(
        pickup_points,
        start=1
    ):

        pickup_cells = get_pickable_cells(
            pickup
        )

        best_distance = float("inf")

        for cell in pickup_cells:

            distance, _ = bfs(
                cell,
                exit_point
            )

            best_distance = min(
                best_distance,
                distance
            )

        matrix[i][n + 1] = best_distance
        matrix[n + 1][i] = best_distance

    # -----------------------------------------------------
    # Entrance -> Exit
    # -----------------------------------------------------

    distance, _ = bfs(
        entrance,
        exit_point
    )

    matrix[0][n + 1] = distance
    matrix[n + 1][0] = distance

    return matrix


# =========================================================
# 10) คำนวณ Distance จาก Pickup Order
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
# 11) Print Distance Matrix
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