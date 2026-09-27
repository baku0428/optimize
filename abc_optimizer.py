"""
abc_optimizer.py
Artificial Bee Colony (ABC)
Warehouse Picking Route Optimization
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from warehouse_core import (
    warehouse,
    entrance,
    exit_point,
    generate_pickup_points,
    get_pickable_cells,
    bfs,
    shortest_path_to_any,
    build_distance_matrix,
    calculate_order_distance,
)


# =========================================================
# 1) Calculate Distance
# =========================================================

def calculate_distance(order, distance_matrix):
    """
    คำนวณระยะทางรวม:
    Entrance -> Pickup ตามลำดับ -> Exit
    """

    return calculate_order_distance(
        order,
        distance_matrix
    )


# =========================================================
# 2) Create Initial Solution
# =========================================================

def create_solution(n):
    """
    สร้างลำดับ Pickup แบบสุ่ม

    เช่น:
    [2, 0, 4, 1, 3]
    """

    solution = list(range(n))

    random.shuffle(solution)

    return solution


# =========================================================
# 3) Create Neighbor Solution
# =========================================================

def create_neighbor(solution):
    """
    Neighborhood Operators

    60% = Inversion (2-Opt)
    25% = Insertion
    15% = Swap
    """

    neighbor = solution.copy()

    n = len(neighbor)

    if n <= 2:
        return neighbor

    r = random.random()

    i, j = sorted(
        random.sample(
            range(n),
            2
        )
    )

    # -----------------------------------------------------
    # Inversion / 2-Opt
    # -----------------------------------------------------

    if r < 0.60:

        neighbor[i:j + 1] = reversed(
            neighbor[i:j + 1]
        )

    # -----------------------------------------------------
    # Insertion
    # -----------------------------------------------------

    elif r < 0.85:

        value = neighbor.pop(j)

        neighbor.insert(
            i,
            value
        )

    # -----------------------------------------------------
    # Swap
    # -----------------------------------------------------

    else:

        neighbor[i], neighbor[j] = (
            neighbor[j],
            neighbor[i]
        )

    return neighbor


# =========================================================
# 4) Artificial Bee Colony
# =========================================================

def artificial_bee_colony(
    distance_matrix,
    num_bees=30,
    max_iterations=500,
    limit=40,
    seed=42,
):
    """
    Artificial Bee Colony สำหรับ Open TSP

    Entrance
        ->
    Pickup ทุกจุด
        ->
    Exit

    Return:
        best_order
        best_distance
        runtime
        history
    """

    random.seed(seed)
    np.random.seed(seed)

    start_time = time.perf_counter()

    # จำนวน Pickup
    n = len(distance_matrix) - 2

    # =====================================================
    # Initial Population
    # =====================================================

    population = [
        create_solution(n)
        for _ in range(num_bees)
    ]

    distances = [
        calculate_distance(
            solution,
            distance_matrix
        )
        for solution in population
    ]

    trials = [0] * num_bees

    # =====================================================
    # Initial Best
    # =====================================================

    best_index = min(
        range(num_bees),
        key=lambda i: distances[i]
    )

    best_solution = population[
        best_index
    ].copy()

    best_distance = distances[
        best_index
    ]

    history = [
        best_distance
    ]

    # =====================================================
    # ABC Iterations
    # =====================================================

    for _ in range(max_iterations):

        # =================================================
        # Phase 1: Employed Bee
        # =================================================

        for i in range(num_bees):

            neighbor = create_neighbor(
                population[i]
            )

            neighbor_distance = calculate_distance(
                neighbor,
                distance_matrix
            )

            if neighbor_distance < distances[i]:

                population[i] = neighbor

                distances[i] = neighbor_distance

                trials[i] = 0

            else:

                trials[i] += 1

        # =================================================
        # Phase 2: Onlooker Bee
        # =================================================

        max_distance = max(distances)
        min_distance = min(distances)

        if max_distance == min_distance:

            fitness = [
                1.0
            ] * num_bees

        else:

            fitness = [
                (
                    max_distance - distance
                )
                +
                (
                    max_distance - min_distance
                ) * 0.1

                for distance in distances
            ]

        total_fitness = sum(
            fitness
        )

        probabilities = [
            value / total_fitness
            for value in fitness
        ]

        for _ in range(num_bees):

            chosen_index = random.choices(
                range(num_bees),
                weights=probabilities,
                k=1
            )[0]

            neighbor = create_neighbor(
                population[chosen_index]
            )

            neighbor_distance = calculate_distance(
                neighbor,
                distance_matrix
            )

            if (
                neighbor_distance
                <
                distances[chosen_index]
            ):

                population[
                    chosen_index
                ] = neighbor

                distances[
                    chosen_index
                ] = neighbor_distance

                trials[
                    chosen_index
                ] = 0

            else:

                trials[
                    chosen_index
                ] += 1

        # =================================================
        # Phase 3: Scout Bee
        # =================================================

        for i in range(num_bees):

            if trials[i] >= limit:

                population[i] = (
                    create_solution(n)
                )

                distances[i] = (
                    calculate_distance(
                        population[i],
                        distance_matrix
                    )
                )

                trials[i] = 0

        # =================================================
        # Update Global Best
        # =================================================

        current_best_index = min(
            range(num_bees),
            key=lambda i: distances[i]
        )

        if (
            distances[current_best_index]
            <
            best_distance
        ):

            best_distance = distances[
                current_best_index
            ]

            best_solution = population[
                current_best_index
            ].copy()

        history.append(
            best_distance
        )

    # =====================================================
    # Runtime
    # =====================================================

    runtime = (
        time.perf_counter()
        -
        start_time
    )

    return {

        "best_order":
            best_solution,

        "best_distance":
            int(best_distance),

        "runtime":
            runtime,

        "history":
            history,
    }


# =========================================================
# 5) Reconstruct Full Walking Path
# =========================================================

def reconstruct_full_path(
    order,
    pickup_points
):
    """
    สร้างเส้นทางเดินจริง

    Entrance
        ->
    Pickup ทุกจุด
        ->
    Exit
    """

    full_path = []

    current_position = entrance

    # =====================================================
    # Entrance -> Pickups
    # =====================================================

    for pickup_index in order:

        shelf_position = pickup_points[
            pickup_index
        ]

        target_cells = get_pickable_cells(
            shelf_position
        )

        _, segment_path, best_goal = (
            shortest_path_to_any(
                current_position,
                target_cells
            )
        )

        if not full_path:

            full_path.extend(
                segment_path
            )

        else:

            full_path.extend(
                segment_path[1:]
            )

        current_position = best_goal

    # =====================================================
    # Last Pickup -> Exit
    # =====================================================

    _, exit_path = bfs(
        current_position,
        exit_point
    )

    if exit_path:

        full_path.extend(
            exit_path[1:]
        )

    return full_path


# =========================================================
# 6) Plot ABC Route
# =========================================================

def plot_abc_route(
    pickup_points,
    full_path,
    best_distance,
    save_path="abc_route.png"
):
    """
    วาดเส้นทาง ABC ลงบนแผนที่โกดัง
    """

    rows, cols = warehouse.shape

    # =====================================================
    # Warehouse Colors
    # =====================================================

    warehouse_cmap = ListedColormap([
        "#FFFFFF",      # Walkway
        "#808080",      # Shelf
    ])

    plt.figure(
        figsize=(10, 10)
    )

    plt.imshow(
        warehouse,
        cmap=warehouse_cmap,
        origin="upper",
        interpolation="nearest"
    )

    # =====================================================
    # ABC Walking Route
    # =====================================================

    if full_path:

        path_rows = [
            int(point[0])
            for point in full_path
        ]

        path_cols = [
            int(point[1])
            for point in full_path
        ]

        plt.plot(
            path_cols,
            path_rows,
            linewidth=2.5,
            zorder=3,
            label="ABC Route"
        )

    # =====================================================
    # Entrance
    # =====================================================

    plt.scatter(
        entrance[1],
        entrance[0],
        s=380,
        marker="o",
        zorder=6
    )

    plt.text(
        entrance[1],
        entrance[0],
        "IN",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        fontweight="bold",
        zorder=7
    )

    # =====================================================
    # Exit
    # =====================================================

    plt.scatter(
        exit_point[1],
        exit_point[0],
        s=380,
        marker="o",
        zorder=6
    )

    plt.text(
        exit_point[1],
        exit_point[0],
        "OUT",
        ha="center",
        va="center",
        color="white",
        fontsize=8,
        fontweight="bold",
        zorder=7
    )

    # =====================================================
    # Pickup Points
    # =====================================================

    for index, point in enumerate(
        pickup_points,
        start=1
    ):

        row = int(point[0])
        col = int(point[1])

        plt.scatter(
            col,
            row,
            s=400,
            marker="s",
            edgecolors="black",
            linewidths=1.5,
            zorder=6
        )

        plt.text(
            col,
            row,
            str(index),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=7
        )

    # =====================================================
    # Axis
    # =====================================================

    plt.xticks(
        np.arange(cols)
    )

    plt.yticks(
        np.arange(rows)
    )

    plt.xlim(
        -0.5,
        cols - 0.5
    )

    plt.ylim(
        rows - 0.5,
        -0.5
    )

    # =====================================================
    # Grid
    # =====================================================

    plt.xticks(
        np.arange(
            -0.5,
            cols,
            1
        ),
        minor=True
    )

    plt.yticks(
        np.arange(
            -0.5,
            rows,
            1
        ),
        minor=True
    )

    plt.grid(
        which="minor",
        linewidth=0.4,
        alpha=0.3
    )

    plt.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    # =====================================================
    # Title
    # =====================================================

    plt.title(
        (
            f"ABC Route (N={len(pickup_points)}) "
            f"- {best_distance} steps"
        ),
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Row"
    )

    # =====================================================
    # Save
    # =====================================================

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\nABC route image saved: {save_path}"
    )

    plt.show()


# =========================================================
# 7) ABC Experiment
# =========================================================

def run_experiment():
    """
    ทดลอง:
    N = 10
    N = 20
    N = 30
    N = 50

    แสดง:
    - Best Distance
    - Runtime
    - Convergence
    """

    print(
        "\n"
        +
        "=" * 65
    )

    print(
        "        ABC EXPERIMENT & BENCHMARKING SUITE"
    )

    print(
        "=" * 65
    )

    test_n = [
        10,
        20,
        30,
        50
    ]

    results = {}

    plt.figure(
        figsize=(10, 6)
    )

    # =====================================================
    # Run Each Problem Size
    # =====================================================

    for n in test_n:

        print(
            f"\n[+] Testing N = {n} pickup points..."
        )

        pickup_points = generate_pickup_points(
            n=n,
            seed=20
        )

        distance_matrix = build_distance_matrix(
            pickup_points
        )

        # N >= 30 ใช้ iteration มากขึ้น
        if n >= 30:

            max_iterations = 600

        else:

            max_iterations = 400

        result = artificial_bee_colony(
            distance_matrix=distance_matrix,
            num_bees=30,
            max_iterations=max_iterations,
            limit=50,
            seed=42,
        )

        results[n] = result

        print(
            "    Best Distance:",
            result["best_distance"],
            "steps"
        )

        print(
            "    Runtime:",
            f'{result["runtime"]:.4f}',
            "seconds"
        )

        # =================================================
        # Convergence
        # =================================================

        plt.plot(
            result["history"],
            label=(
                f"N = {n} "
                f"(Best: {result['best_distance']})"
            )
        )

    # =====================================================
    # Result Table
    # =====================================================

    print(
        "\n"
        +
        "=" * 65
    )

    print(
        "ABC EXPERIMENT RESULTS"
    )

    print(
        "=" * 65
    )

    print(
        f"{'N Points':<10} | "
        f"{'Best Distance':<20} | "
        f"{'Runtime (Seconds)':<18}"
    )

    print(
        "-" * 65
    )

    for n in test_n:

        print(
            f"{n:<10} | "
            f"{results[n]['best_distance']:<20} | "
            f"{results[n]['runtime']:<18.4f}"
        )

    print(
        "=" * 65
    )

    # =====================================================
    # Convergence Graph
    # =====================================================

    plt.title(
        "ABC Convergence Curve on Different Problem Scales",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(
        "Iteration",
        fontsize=12
    )

    plt.ylabel(
        "Best Total Distance (Steps)",
        fontsize=12
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.6
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "abc_convergence.png",
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nABC convergence image saved: "
        "abc_convergence.png"
    )

    plt.show()

    return results
