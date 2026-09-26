"""
pso.py
Particle Swarm Optimization (PSO) สำหรับหาเส้นทางหยิบสินค้าในโกดัง
(Open TSP: Entrance -> Pickups ทุกจุด -> Exit)

การเข้ารหัสคำตอบ: Random Key Encoding
    - แต่ละอนุภาคมีตำแหน่งเป็นเวกเตอร์ค่าจริงยาว n (1 ค่าต่อ 1 จุดหยิบ)
    - ถอดรหัสเป็นลำดับการหยิบด้วย argsort (ค่าคีย์น้อย -> หยิบก่อน)
    - ทำให้ใช้สมการ velocity/position ของ PSO แบบต่อเนื่องได้ตามปกติ

การปรับปรุงให้เหมาะกับโจทย์ (use_local_search=True):
    - Memetic PSO: เมื่อ gbest ดีขึ้น จะปรับ gbest ด้วย 2-opt + Or-opt
      แล้วเขียนกลับเข้าเวกเตอร์ตำแหน่ง (encode กลับ) เพื่อให้ฝูงถูกดึงไปหาคำตอบที่ดีขึ้น
    - ค่า inertia weight ลดลงเชิงเส้น (สำรวจก่อน -> ค่อยเจาะจง)
    - จำกัดความเร็ว (velocity clamp) กันอนุภาคกระโดดไกลเกินไป
    - Stagnation restart: ถ้า gbest ไม่ดีขึ้นนาน จะสุ่มอนุภาคครึ่งที่แย่สุดใหม่
      (แล้วปรับด้วย local search) เพื่อกันฝูงรวมกลุ่มเร็วเกินไป (premature convergence)

เรียกใช้จาก main.py:  run_pso(distance_matrix)
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from warehouse_core import (
    warehouse,
    entrance,
    exit_point,
    generate_pickup_points,
    get_pickable_cells,
    bfs,
    build_distance_matrix,
    calculate_order_distance,
)


# =========================================================
# 1) Encode / Decode
# =========================================================

def decode(position):
    """Random key -> ลำดับการหยิบ (index 0..n-1)"""
    return [int(i) for i in np.argsort(position, kind="stable")]


def encode_into(position, order):
    """
    เขียนลำดับ order กลับเป็น random key โดยใช้ชุดค่าคีย์เดิมของอนุภาค
    (คีย์ที่น้อยที่สุดไปอยู่กับจุดที่หยิบก่อน ฯลฯ)
    """
    new_position = np.empty_like(position)
    new_position[order] = np.sort(position)
    return new_position


# =========================================================
# 2) Local Search: 2-opt + Or-opt  (ทำงานบนลำดับที่มี Entrance/Exit ตรึงหัวท้าย)
# =========================================================

def local_search(order, distance_matrix):
    """
    ปรับลำดับด้วย 2-opt และ Or-opt (ย้ายช่วงยาว 1-3 จุด) แบบ first-improvement
    จนกว่าจะไม่มีการปรับปรุง  matrix สมมาตร จึงคำนวณ delta ได้ตรง ๆ
    """
    D = distance_matrix
    n = len(order)
    if n < 3:
        return list(order), calculate_order_distance(order, D)

    # s = [IN, p1, ..., pn, OUT]  (index ใน matrix)
    s = [0] + [p + 1 for p in order] + [n + 1]
    improved = True

    while improved:
        improved = False

        # ---------- 2-opt ----------
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                a, b, c, d = s[i - 1], s[i], s[j], s[j + 1]
                delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
                if delta < -1e-9:
                    s[i:j + 1] = s[i:j + 1][::-1]
                    improved = True

        # ---------- Or-opt ----------
        for seg_len in (1, 2, 3):
            i = 1
            while i + seg_len <= n:
                seg = s[i:i + seg_len]
                prev, nxt = s[i - 1], s[i + seg_len]
                remove_gain = D[prev][seg[0]] + D[seg[-1]][nxt] - D[prev][nxt]

                rest = s[:i] + s[i + seg_len:]      # ลำดับหลังตัดช่วงออก
                best_delta, best_k, best_rev = -1e-9, None, False
                for k in range(len(rest) - 1):
                    u, v = rest[k], rest[k + 1]
                    base = D[u][v]
                    fwd = D[u][seg[0]] + D[seg[-1]][v] - base
                    rev = D[u][seg[-1]] + D[seg[0]][v] - base
                    if fwd - remove_gain < best_delta:
                        best_delta, best_k, best_rev = fwd - remove_gain, k, False
                    if rev - remove_gain < best_delta:
                        best_delta, best_k, best_rev = rev - remove_gain, k, True

                if best_k is not None:
                    ins = seg[::-1] if best_rev else seg
                    s = rest[:best_k + 1] + ins + rest[best_k + 1:]
                    improved = True
                i += 1

    new_order = [x - 1 for x in s[1:-1]]
    return new_order, calculate_order_distance(new_order, D)


# =========================================================
# 3) Main PSO
# =========================================================

def run_pso(
    distance_matrix,
    num_particles=40,
    max_iterations=300,
    w_max=0.9,
    w_min=0.4,
    c1=1.5,
    c2=1.5,
    v_max=0.5,
    use_local_search=True,
    stall_limit=25,
    seed=42,
):
    """
    คืนค่า dict:
      - best_order    : ลำดับ index (0-based) ของจุดหยิบ
      - best_distance : ระยะทางรวมตาม distance matrix
      - runtime       : เวลาประมวลผล (วินาที)
      - history       : best distance ในแต่ละรอบ (ไว้พลอต convergence)
    รูปแบบเดียวกับ artificial_bee_colony() เพื่อเทียบกันได้ตรง ๆ
    """
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()

    n = len(distance_matrix) - 2

    if n == 0:
        return {
            "best_order": [],
            "best_distance": int(distance_matrix[0][-1]),
            "runtime": time.perf_counter() - start_time,
            "history": [int(distance_matrix[0][-1])],
        }

    # ---------- เริ่มต้นฝูง ----------
    x = rng.random((num_particles, n))
    v = rng.uniform(-v_max, v_max, (num_particles, n))

    fitness = np.array(
        [calculate_order_distance(decode(p), distance_matrix) for p in x]
    )

    pbest_x = x.copy()
    pbest_f = fitness.copy()

    g = int(np.argmin(pbest_f))
    gbest_x = pbest_x[g].copy()
    gbest_f = float(pbest_f[g])

    history = [gbest_f]
    stall = 0

    # ---------- วนรอบ ----------
    for it in range(max_iterations):
        w = w_max - (w_max - w_min) * it / max(1, max_iterations - 1)

        r1 = rng.random((num_particles, n))
        r2 = rng.random((num_particles, n))

        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -v_max, v_max)
        x = x + v

        for i in range(num_particles):
            f = calculate_order_distance(decode(x[i]), distance_matrix)
            fitness[i] = f

            if f < pbest_f[i]:
                pbest_f[i] = f
                pbest_x[i] = x[i].copy()

                if f < gbest_f:
                    gbest_f = float(f)
                    gbest_x = x[i].copy()

                    # Memetic step: ปรับ gbest ด้วย local search แล้ว encode กลับ
                    if use_local_search:
                        order, f_ls = local_search(decode(gbest_x), distance_matrix)
                        if f_ls < gbest_f:
                            gbest_f = float(f_ls)
                            gbest_x = encode_into(gbest_x, order)
                            x[i] = gbest_x.copy()
                            pbest_x[i] = gbest_x.copy()
                            pbest_f[i] = gbest_f

        # ---------- Stagnation restart ----------
        if use_local_search:
            stall = 0 if gbest_f < history[-1] else stall + 1
            if stall >= stall_limit:
                worst = np.argsort(fitness)[num_particles // 2:]
                for i in worst:
                    x[i] = rng.random(n)
                    v[i] = rng.uniform(-v_max, v_max, n)
                    order, f_new = local_search(decode(x[i]), distance_matrix)
                    x[i] = encode_into(x[i], order)
                    fitness[i] = f_new
                    pbest_x[i] = x[i].copy()
                    pbest_f[i] = f_new
                    if f_new < gbest_f:
                        gbest_f, gbest_x = float(f_new), x[i].copy()
                stall = 0

        history.append(gbest_f)

    best_order = decode(gbest_x)
    runtime = time.perf_counter() - start_time

    return {
        "best_order": best_order,
        "best_distance": int(gbest_f),
        "runtime": runtime,
        "history": history,
    }


# =========================================================
# 4) ถอดรหัสเส้นทางเดินจริง (Exact, ตามลำดับที่ PSO หาได้)
# =========================================================
#
# distance matrix ถือว่าแต่ละ Pickup เป็นโหนดเดียว (ใช้ช่องยืนที่ใกล้ที่สุดของคู่นั้น ๆ)
# แต่ในความจริงต้อง "ยืนที่ช่องเดียว" ตอนหยิบ  ฟังก์ชันนี้ใช้ DP เลือกช่องยืนของแต่ละ
# จุดให้รวมสั้นที่สุด ภายใต้ลำดับที่กำหนด -> ได้ระยะทางเดินจริงของลำดับนั้น

def reconstruct_full_path(order, pickup_points):
    """
    return: (path, steps)
      path  = รายการพิกัดตั้งแต่ Entrance ถึง Exit
      steps = จำนวนก้าวจริง = len(path) - 1
    """
    cache = {}

    def route(a, b):
        if (a, b) not in cache:
            cache[(a, b)] = bfs(a, b)
        return cache[(a, b)]

    layers = [[entrance]]
    for idx in order:
        layers.append([tuple(map(int, c)) for c in get_pickable_cells(pickup_points[idx])])
    layers.append([exit_point])

    # DP: cost[k][c] = ระยะทางน้อยสุดที่ไปถึงช่อง c ของชั้น k
    cost = [{c: 0 for c in layers[0]}]
    back = [{}]
    for k in range(1, len(layers)):
        cost.append({})
        back.append({})
        for c in layers[k]:
            best, arg = float("inf"), None
            for p in layers[k - 1]:
                d = cost[k - 1][p] + route(p, c)[0]
                if d < best:
                    best, arg = d, p
            cost[k][c] = best
            back[k][c] = arg

    # ย้อนกลับหาช่องยืนที่เลือก
    chosen = [exit_point]
    for k in range(len(layers) - 1, 0, -1):
        chosen.append(back[k][chosen[-1]])
    chosen.reverse()

    path = [chosen[0]]
    for a, b in zip(chosen, chosen[1:]):
        path.extend(route(a, b)[1][1:])

    path = [(int(r), int(c)) for r, c in path]
    return path, len(path) - 1


# =========================================================
# 5) พลอตกราฟ
# =========================================================

def plot_route(path, pickup_points, order, title, filename):
    rows, cols = warehouse.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(warehouse, cmap="Greys", vmin=0, vmax=1.6)

    ys, xs = zip(*path)
    ax.plot(xs, ys, "-", color="tab:blue", linewidth=2, alpha=0.8)

    for rank, idx in enumerate(order, start=1):
        r, c = pickup_points[idx]
        ax.plot(c, r, "s", color="gold", markeredgecolor="k", markersize=15)
        ax.text(c, r, str(rank), ha="center", va="center", fontsize=8, fontweight="bold")

    ax.plot(entrance[1], entrance[0], "o", color="tab:green", markersize=14)
    ax.plot(exit_point[1], exit_point[0], "o", color="tab:red", markersize=14)
    ax.text(entrance[1], entrance[0], "IN", ha="center", va="center", fontsize=6, color="w")
    ax.text(exit_point[1], exit_point[0], "OUT", ha="center", va="center", fontsize=6, color="w")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


# =========================================================
# 6) Experiment Suite (ข้อ 7.2.3 / 7.2.4)
# =========================================================

def run_experiment(sizes=(10, 20, 30, 50), n_runs=10):
    """
    เทียบ PSO แบบมาตรฐาน (ไม่มี local search) กับ PSO + Local Search
    รันซ้ำ n_runs seed ต่อขนาด เพื่อดูค่าเฉลี่ยและความนิ่ง
    """
    print("\n" + "=" * 78)
    print("        PSO EXPERIMENT & BENCHMARKING SUITE")
    print("=" * 78)

    variants = {
        "PSO (plain)": False,
        "PSO + LocalSearch": True,
    }

    summary = {}
    curves = {}

    for n in sizes:
        pts = generate_pickup_points(n=n, seed=20)
        mat = build_distance_matrix(pts)
        iters = 300 if n <= 20 else 500

        for name, use_ls in variants.items():
            results = [
                run_pso(mat, max_iterations=iters, use_local_search=use_ls, seed=s)
                for s in range(n_runs)
            ]
            dists = [r["best_distance"] for r in results]
            times = [r["runtime"] for r in results]
            best_run = min(results, key=lambda r: r["best_distance"])

            summary[(n, name)] = {
                "best": min(dists),
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists)),
                "time": float(np.mean(times)),
            }
            curves[(n, name)] = best_run["history"]

    print(f"\n({n_runs} runs ต่อกรณี)")
    print(
        f"{'N':<4} | {'Method':<18} | {'Best':>5} | {'Mean':>7} | {'Std':>5} | {'Avg time (s)':>12}"
    )
    print("-" * 78)
    for n in sizes:
        for name in variants:
            s = summary[(n, name)]
            print(
                f"{n:<4} | {name:<18} | {s['best']:>5} | {s['mean']:>7.1f} | "
                f"{s['std']:>5.2f} | {s['time']:>12.3f}"
            )
        print("-" * 78)

    # Convergence plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, n in zip(axes.ravel(), sizes):
        for name in variants:
            ax.plot(curves[(n, name)], label=name)
        ax.set_title(f"N = {n}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best distance (steps)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    fig.suptitle("PSO Convergence (best run of each setting)", fontweight="bold")
    fig.tight_layout()
    fig.savefig("pso_convergence.png", dpi=130)
    plt.close(fig)
    print("\nบันทึกกราฟ: pso_convergence.png")


# =========================================================
# Execution Point
# =========================================================

if __name__ == "__main__":
    pts_10 = generate_pickup_points(n=10, seed=20)
    mat_10 = build_distance_matrix(pts_10)

    res = run_pso(mat_10, seed=42)
    path, real_steps = reconstruct_full_path(res["best_order"], pts_10)

    print("\n=== ผลลัพธ์ PSO (N=10) ===")
    print("5.1 ลำดับการหยิบสินค้า:", [x + 1 for x in res["best_order"]])
    print(f"5.3 ระยะทางรวมน้อยที่สุด (ตาม distance matrix): {res['best_distance']} ก้าว")
    print(f"    ระยะทางเดินจริง (เลือกช่องยืนด้วย DP):       {real_steps} ก้าว")
    print(f"    เวลาประมวลผล: {res['runtime']:.4f} วินาที")
    print(f"5.2 เส้นทางเดินทั้งหมด ({len(path)} ช่อง):")
    print(" -> ".join(str(p) for p in path))

    plot_route(
        path,
        pts_10,
        res["best_order"],
        f"PSO route (N=10) - {real_steps} steps",
        "pso_route.png",
    )
    print("\nบันทึกกราฟเส้นทาง: pso_route.png")

    run_experiment()
