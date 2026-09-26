"""
abc_optimizer.py
รับผิดชอบโดย: คนที่ 2 (Artificial Bee Colony + เก็บผลการทดลอง)
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt

# นำเข้าฟังก์ชันจาก warehouse_core โดยไม่ดัดแปลงไฟล์ core
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
# 1) ฟังก์ชันคำนวณและตัวดำเนินการของ ABC
# =========================================================

def calculate_distance(order, distance_matrix):
    """คำนวณระยะทางรวมจาก Entrance -> Pickup ตามลำดับ -> Exit"""
    return calculate_order_distance(order, distance_matrix)


def create_solution(n):
    """สร้าง Permutation เริ่มต้นแบบสุ่ม (Index 0 ถึง n-1)"""
    solution = list(range(n))
    random.shuffle(solution)
    return solution


def create_neighbor(solution):
    """
    Neighborhood Operator ผสมผสาน:
    - Inversion (2-Opt) เพื่อคลายเส้นทางที่ตัดกัน (ความน่าจะเป็น 60%)
    - Insertion เพื่อแทรกจุดหยิบสินค้าใหม่ (ความน่าจะเป็น 25%)
    - Swap เพื่อสลับจุดหยิบ 2 จุด (ความน่าจะเป็น 15%)
    """
    neighbor = solution.copy()
    n = len(neighbor)
    if n <= 2:
        return neighbor

    r = random.random()
    i, j = sorted(random.sample(range(n), 2))

    if r < 0.60:
        # Inversion (2-Opt)
        neighbor[i : j + 1] = reversed(neighbor[i : j + 1])
    elif r < 0.85:
        # Insertion
        val = neighbor.pop(j)
        neighbor.insert(i, val)
    else:
        # Swap
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

    return neighbor


# =========================================================
# 2) Main ABC Algorithm (เรียกจาก main.py ได้โดยตรง)
# =========================================================

def artificial_bee_colony(
    distance_matrix,
    num_bees=30,
    max_iterations=500,
    limit=40,
    seed=42,
):
    """
    อัลกอริทึม Artificial Bee Colony สำหรับปัญหา Open TSP (Entrance -> Pickups -> Exit)
    
    คืนค่า dict ที่มี:
      - best_order: ลำดับ index (0-based)
      - best_distance: ระยะทางรวมที่สั้นที่สุด
      - runtime: เวลาในการประมวลผล (วินาที)
      - history: บันทึก best distance ในแต่ละรอบเพื่อดู Convergence
    """
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.perf_counter()

    n = len(distance_matrix) - 2

    # 1. Initial Population
    population = [create_solution(n) for _ in range(num_bees)]
    distances = [calculate_distance(sol, distance_matrix) for sol in population]
    trials = [0] * num_bees

    # หาคำตอบที่ดีที่สุดในชุดเริ่มต้น
    best_idx = min(range(num_bees), key=lambda i: distances[i])
    best_solution = population[best_idx].copy()
    best_distance = distances[best_idx]

    history = [best_distance]

    # Iteration Loop
    for _ in range(max_iterations):
        # -------------------------------------------------
        # Phase 1: Employed Bee
        # -------------------------------------------------
        for i in range(num_bees):
            neighbor = create_neighbor(population[i])
            neighbor_dist = calculate_distance(neighbor, distance_matrix)

            if neighbor_dist < distances[i]:
                population[i] = neighbor
                distances[i] = neighbor_dist
                trials[i] = 0
            else:
                trials[i] += 1

        # -------------------------------------------------
        # Phase 2: Onlooker Bee
        # -------------------------------------------------
        # คำนวณความน่าจะเป็นตาม Fitness (ระยะทางยิ่งน้อย ความน่าจะเป็นยิ่งสูง)
        max_d = max(distances)
        min_d = min(distances)
        if max_d == min_d:
            fitness = [1.0] * num_bees
        else:
            # ใช้ min-max scaling เพื่อกระจายความน่าจะเป็น
            fitness = [(max_d - d) + (max_d - min_d) * 0.1 for d in distances]

        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]

        for _ in range(num_bees):
            # สุ่มเลือก Food Source ตามค่าน้ำหนัก Fitness
            chosen_idx = random.choices(
                range(num_bees), weights=probabilities, k=1
            )[0]
            neighbor = create_neighbor(population[chosen_idx])
            neighbor_dist = calculate_distance(neighbor, distance_matrix)

            if neighbor_dist < distances[chosen_idx]:
                population[chosen_idx] = neighbor
                distances[chosen_idx] = neighbor_dist
                trials[chosen_idx] = 0
            else:
                trials[chosen_idx] += 1

        # -------------------------------------------------
        # Phase 3: Scout Bee
        # -------------------------------------------------
        for i in range(num_bees):
            if trials[i] >= limit:
                population[i] = create_solution(n)
                distances[i] = calculate_distance(population[i], distance_matrix)
                trials[i] = 0

        # Update Best Solution
        current_best_idx = min(range(num_bees), key=lambda i: distances[i])
        if distances[current_best_idx] < best_distance:
            best_distance = distances[current_best_idx]
            best_solution = population[current_best_idx].copy()

        history.append(best_distance)

    runtime = time.perf_counter() - start_time

    return {
        "best_order": best_solution,
        "best_distance": int(best_distance),
        "runtime": runtime,
        "history": history,
    }


# =========================================================
# 3) ฟังก์ชันถอดรหัสเส้นทางเดินจริง (Full Walkway Path)
# =========================================================

def reconstruct_full_path(order, pickup_points):
    """
    แกะรอยพิกัดทางเดินก้าวต่อก้าว:
    Entrance -> ช่องข้าง Shelf ของสินค้าตัวแรก -> ... -> Exit
    """
    full_path = []
    current_pos = entrance

    for pickup_idx in order:
        shelf_pos = pickup_points[pickup_idx]
        target_cells = get_pickable_cells(shelf_pos)

        # หาเส้นทางที่สั้นที่สุดจากจุดปัจจุบันไปยังช่องยืนหยิบของ Shelf นั้น
        _, seg_path, best_goal = shortest_path_to_any(current_pos, target_cells)

        if not full_path:
            full_path.extend(seg_path)
        else:
            full_path.extend(seg_path[1:])  # ตัดจุดเชื่อมต่อซ้ำ

        current_pos = best_goal

    # จากจุดหยิบสุดท้ายไปยังทางออก (exit_point)
    _, exit_seg = bfs(current_pos, exit_point)
    if exit_seg:
        full_path.extend(exit_seg[1:])

    return full_path


# =========================================================
# 4) ระบบเก็บผลการทดลอง (Experiment Suite) สำหรับทำรายงาน
# =========================================================

def run_experiment():
    """
    การทดลองตามข้อกำหนดข้อ 7.2.4 ของรายงาน:
    เปรียบเทียบขนาดจำนวนจุด Pickup N = [10, 20, 30, 50]
    เก็บสถิติระยะทาง เวลาที่ใช้ และพลอตกราฟ Convergence
    """
    print("\n" + "=" * 65)
    print("        ABC EXPERIMENT & BENCHMARKING SUITE")
    print("=" * 65)

    test_n = [10, 20, 30, 50]
    results = {}

    plt.figure(figsize=(10, 6))

    for n in test_n:
        print(f"\n[+] กำลังทดสอบ N = {n} จุด...")
        pts = generate_pickup_points(n=n, seed=20)
        dist_mat = build_distance_matrix(pts)

        # รัน ABC ด้วยการตั้งค่าตามสเกล
        max_iter = 600 if n >= 30 else 400
        res = artificial_bee_colony(
            dist_mat,
            num_bees=30,
            max_iterations=max_iter,
            limit=50,
            seed=42,
        )

        results[n] = res
        print(f"    - ระยะทางที่ดีที่สุด (ก้าว): {res['best_distance']}")
        print(f"    - เวลาประมวลผล (วินาที): {res['runtime']:.4f} s")

        # พลอตกราฟ Convergence
        plt.plot(res["history"], label=f"N = {n} (Best: {res['best_distance']})")

    # สรุปผลเป็นตารางสำหรับนำไปใส่รายงาน
    print("\n" + "=" * 65)
    print("สรุปผลการทดลอง ABC สำหรับรายงาน (ข้อ 7.2.3 และ 7.2.4)")
    print("=" * 65)
    print(f"{'N Points':<10} | {'Best Distance (Steps)':<25} | {'Runtime (Seconds)':<18}")
    print("-" * 65)
    for n in test_n:
        print(
            f"{n:<10} | {results[n]['best_distance']:<25} | {results[n]['runtime']:<18.4f}"
        )
    print("=" * 65)

    # แสดงกราฟ Convergence
    plt.title("ABC Convergence Curve on Different Problem Scales", fontsize=14, fontweight="bold")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Total Distance (Steps)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# Execution Point
# =========================================================

if __name__ == "__main__":
    # 1. ทดสอบการรันค่าเริ่มต้น (N = 10 ตามโจทย์)
    pts_10 = generate_pickup_points(n=10, seed=20)
    mat_10 = build_distance_matrix(pts_10)

    res_10 = artificial_bee_colony(mat_10, seed=42)
    order_1based = [x + 1 for x in res_10["best_order"]]
    full_path = reconstruct_full_path(res_10["best_order"], pts_10)

    print("\n=== ผลลัพธ์ ABC เบื้องต้น (N=10) ===")
    print("5.1 ลำดับการหยิบสินค้า:", order_1based)
    print(f"5.3 ระยะทางรวมน้อยที่สุด: {res_10['best_distance']} ก้าว")
    print(f"    เวลาประมวลผล: {res_10['runtime']:.4f} วินาที")
    print(f"5.2 เส้นทางเดินจริงทั้งหมด ({len(full_path)} ก้าว):")
    print(" -> ".join([str(pos) for pos in full_path[:10]]) + " -> ... -> " + str(full_path[-1]))

    # 2. ทำการรัน Experiment Suite N = [10, 20, 30, 50]
    run_experiment()