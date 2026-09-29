"""
grasp.py
GRASP (Greedy Randomized Adaptive Search Procedure) สำหรับหาเส้นทางหยิบสินค้าในโกดัง
(Open TSP: Entrance -> Pickups ทุกจุด -> Exit)

แต่ละรอบของ GRASP มี 2 ขั้น:
    1) Construction  : สร้างลำดับแบบ greedy + สุ่ม
         - เริ่มที่ Entrance ทุกครั้ง แล้วเลือกจุดถัดไปจาก Restricted Candidate List (RCL)
         - RCL = จุดที่ยังไม่หยิบ ที่ระยะ <= d_min + alpha * (d_max - d_min)
         - alpha = 0 -> greedy ล้วน (nearest neighbor), alpha = 1 -> สุ่มล้วน
    2) Local Search  : ปรับลำดับด้วย 2-opt + Or-opt จนถึง local optimum
       เก็บคำตอบที่ดีที่สุดจากทุกรอบ

การปรับให้เหมาะกับโจทย์:
    - Entrance / Exit ถูกตรึงไว้หัวท้าย (ไม่ใช่ TSP วงปิด) local search จึงไม่ย้ายสองจุดนี้
    - ใช้ระยะ BFS จริงบนทางเดิน (ไม่ใช่ระยะเส้นตรง)
    - Optimize "ระยะเดินจริง" ไม่ใช่ระยะตาม Distance Matrix ระดับ Pickup
      (matrix ยอมให้เข้าฝั่งหนึ่งของชั้นแล้วออกอีกฝั่งได้ฟรี -> ต่ำกว่าความจริง)
      โจทย์นี้จริง ๆ คือ Generalized TSP: แต่ละ Pickup ต้องเลือกช่องยืน 1 ช่อง
        * Construction เลือกทั้ง "จุดถัดไป + ช่องยืน" จากตำแหน่งที่ยืนอยู่จริง
        * Local search สลับ 2 ขั้น: (ก) ตรึงช่องยืน แล้วทำ 2-opt/Or-opt บนลำดับ
          (ข) ตรึงลำดับ แล้วเลือกช่องยืนใหม่ด้วย DP (evaluate_order) วนจนไม่ดีขึ้น
    - Reactive GRASP: ไม่ fix ค่า alpha แต่สุ่มจากชุดค่า แล้วปรับความน่าจะเป็น
      ให้ alpha ที่ให้ผลดีถูกเลือกบ่อยขึ้นเอง (ไม่ต้องจูน alpha ด้วยมือ)

เรียกใช้จาก main.py:  run_grasp(build_route_data(pickup_points))
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from warehouse_core import (
    warehouse,
    entrance,
    exit_point,
    generate_pickup_points,
    build_route_data,
    evaluate_order,
    reconstruct_full_path,
)


# =========================================================
# 1) Greedy Randomized Construction
# =========================================================

def greedy_randomized_construction(route_data, alpha, rng):
    """
    สร้างลำดับการหยิบ (index 0..n-1) โดยเริ่มจากช่อง Entrance
    ต้นทุนของ Pickup j = ระยะจากช่องที่ยืนอยู่ ไปช่องยืนที่ใกล้ที่สุดของ j
    แล้วสุ่มเลือกจาก RCL และย้ายไปยืนที่ช่องนั้น
    """
    C = route_data["cell_dist"]
    groups = route_data["node_cell_idx"][1:-1]

    unvisited = list(range(len(groups)))
    current = route_data["node_cell_idx"][0][0]     # ช่อง Entrance
    order = []

    while unvisited:
        near_cell = [
            groups[j][int(np.argmin(C[current, groups[j]]))]
            for j in unvisited
        ]
        costs = C[current, near_cell]

        c_min, c_max = costs.min(), costs.max()
        threshold = c_min + alpha * (c_max - c_min)
        rcl = np.flatnonzero(costs <= threshold)

        pick = rcl[rng.integers(len(rcl))]

        order.append(unvisited.pop(pick))
        current = near_cell[pick]

    return order


# =========================================================
# 2) Local Search: 2-opt + Or-opt
#    ทำงานบน s = [IN, p1, ..., pn, OUT] โดย IN / OUT ตรึงอยู่กับที่
# =========================================================

def two_opt(s, D):
    """กลับช่วง s[i..j] ถ้าทำให้สั้นลง (first improvement)"""
    improved = False
    last = len(s) - 2

    for i in range(1, last):
        for j in range(i + 1, last + 1):
            a, b, c, d = s[i - 1], s[i], s[j], s[j + 1]
            delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]

            if delta < 0:
                s[i:j + 1] = s[i:j + 1][::-1]
                improved = True

    return improved


def or_opt(s, D):
    """ย้ายช่วงยาว 1-3 จุดไปแทรกตำแหน่งอื่น (กลับทิศได้)"""
    improved = False
    last = len(s) - 2

    for seg_len in (1, 2, 3):
        i = 1

        while i + seg_len - 1 <= last:
            seg = s[i:i + seg_len]
            prev, nxt = s[i - 1], s[i + seg_len]
            remove_gain = D[prev][seg[0]] + D[seg[-1]][nxt] - D[prev][nxt]

            rest = s[:i] + s[i + seg_len:]
            best_delta, best_k, best_rev = 0, None, False

            for k in range(len(rest) - 1):
                u, v = rest[k], rest[k + 1]
                base = D[u][v]
                forward = D[u][seg[0]] + D[seg[-1]][v] - base - remove_gain
                reverse = D[u][seg[-1]] + D[seg[0]][v] - base - remove_gain

                if forward < best_delta:
                    best_delta, best_k, best_rev = forward, k, False
                if reverse < best_delta:
                    best_delta, best_k, best_rev = reverse, k, True

            if best_k is not None:
                insert = seg[::-1] if best_rev else seg
                s[:] = rest[:best_k + 1] + insert + rest[best_k + 1:]
                improved = True

            i += 1

    return improved


def local_search(order, route_data):
    """
    สลับ 2 ขั้นจนระยะเดินจริงไม่ลดลง -> คืน (order, steps)
      (ก) ตรึงช่องยืนของแต่ละ Pickup -> ได้ matrix ที่ระยะตรงกับความจริง
          แล้ววน 2-opt + Or-opt บนลำดับ
      (ข) ตรึงลำดับ -> เลือกช่องยืนใหม่ที่ดีที่สุดด้วย DP
    ทุกรอบระยะไม่เพิ่มขึ้น จึงหยุดได้แน่นอน
    """
    C = route_data["cell_dist"]
    cell_index = {cell: i for i, cell in enumerate(route_data["cells"])}
    in_cell = route_data["node_cell_idx"][0][0]
    out_cell = route_data["node_cell_idx"][-1][0]
    n = len(order)

    steps, standing = evaluate_order(order, route_data)

    while True:
        # (ก) matrix ของช่องยืนที่ตรึงไว้: index 0 = IN, p+1 = Pickup p, n+1 = OUT
        fixed = [0] * n
        for p, cell in zip(order, standing):
            fixed[p] = cell_index[cell]
        nodes = [in_cell] + fixed + [out_cell]
        D = C[np.ix_(nodes, nodes)].tolist()

        s = [0] + [p + 1 for p in order] + [n + 1]
        while two_opt(s, D) | or_opt(s, D):
            pass
        new_order = [x - 1 for x in s[1:-1]]

        # (ข) เลือกช่องยืนใหม่ตามลำดับใหม่
        new_steps, new_standing = evaluate_order(new_order, route_data)

        if new_steps >= steps:
            return order, steps

        order, steps, standing = new_order, new_steps, new_standing


# =========================================================
# 3) Main GRASP
# =========================================================

def run_grasp(
    route_data,
    max_iterations=100,
    alpha=None,
    alphas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    reactive_block=10,
    use_local_search=True,
    seed=42,
):
    """
    route_data = ผลจาก build_route_data(pickup_points)

    alpha = None  -> Reactive GRASP (เลือก alpha จาก alphas แบบปรับน้ำหนัก)
    alpha = ค่าคงที่ -> GRASP แบบ alpha คงที่

    คืนค่า dict (รูปแบบเดียวกับ run_pso / artificial_bee_colony):
      - best_order    : ลำดับ index (0-based) ของจุดหยิบ
      - best_distance : จำนวนก้าวเดินจริง (Entrance -> ... -> Exit)
      - runtime       : เวลาประมวลผล (วินาที)
      - history       : best distance หลังแต่ละรอบ (ไว้พลอต convergence)
      - alpha_probs   : ความน่าจะเป็นสุดท้ายของแต่ละ alpha (Reactive)
    """
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()

    n = len(route_data["pickup_points"])

    if n == 0:
        steps, _ = evaluate_order([], route_data)
        return {
            "best_order": [],
            "best_distance": steps,
            "runtime": time.perf_counter() - start_time,
            "history": [steps],
            "alpha_probs": {},
        }

    reactive = alpha is None
    alpha_list = list(alphas) if reactive else [alpha]
    probs = np.full(len(alpha_list), 1 / len(alpha_list))
    alpha_sum = np.zeros(len(alpha_list))
    alpha_count = np.zeros(len(alpha_list))

    best_order, best_distance = None, float("inf")
    history = []

    for it in range(max_iterations):
        a_idx = rng.choice(len(alpha_list), p=probs)

        order = greedy_randomized_construction(route_data, alpha_list[a_idx], rng)

        if use_local_search:
            order, distance = local_search(order, route_data)
        else:
            distance, _ = evaluate_order(order, route_data)

        if distance < best_distance:
            best_order, best_distance = order, distance

        alpha_sum[a_idx] += distance
        alpha_count[a_idx] += 1
        history.append(best_distance)

        # ---------- Reactive: ปรับความน่าจะเป็นของ alpha ทุก ๆ block ----------
        # q_i = (best / mean_i)^10  (alpha ที่เฉลี่ยใกล้ best -> q สูง -> ถูกเลือกบ่อย)
        if reactive and (it + 1) % reactive_block == 0:
            mean = np.where(
                alpha_count > 0,
                alpha_sum / np.maximum(alpha_count, 1),
                best_distance,
            )
            q = (best_distance / mean) ** 10
            probs = q / q.sum()

    return {
        "best_order": best_order,
        "best_distance": int(best_distance),
        "runtime": time.perf_counter() - start_time,
        "history": history,
        "alpha_probs": dict(zip(alpha_list, np.round(probs, 3).tolist())),
    }


# =========================================================
# 4) พลอตเส้นทาง
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
# 5) Experiment Suite (ข้อ 7.2.3 / 7.2.4)
# =========================================================

def run_experiment(sizes=(10, 20, 30, 50), n_runs=10):
    """
    เทียบ 3 แบบ (รันซ้ำ n_runs seed ต่อขนาด):
      - Greedy NN        : alpha = 0, 1 รอบ, ไม่มี local search (baseline)
      - GRASP (no LS)    : Reactive construction อย่างเดียว
      - GRASP + LS       : Reactive GRASP เต็มรูปแบบ
    """
    print("\n" + "=" * 70)
    print("        GRASP EXPERIMENT & BENCHMARKING SUITE")
    print("=" * 70)

    variants = {
        "Greedy NN": dict(max_iterations=1, alpha=0.0, use_local_search=False),
        "GRASP (no LS)": dict(use_local_search=False),
        "GRASP + LS": dict(use_local_search=True),
    }

    summary = {}
    curves = {}

    for n in sizes:
        pts = generate_pickup_points(n=n, seed=20)
        data = build_route_data(pts)
        iters = 100 if n <= 20 else 200

        for name, kwargs in variants.items():
            kwargs = {"max_iterations": iters, **kwargs}
            results = [run_grasp(data, seed=s, **kwargs) for s in range(n_runs)]

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

    print(f"\n({n_runs} runs ต่อกรณี, หน่วยระยะ = จำนวนก้าวเดินจริง)")
    print(
        f"{'N':<4} | {'Method':<14} | {'Best':>5} | {'Mean':>7} | {'Std':>5} | "
        f"{'Avg time (s)':>12}"
    )
    print("-" * 70)
    for n in sizes:
        for name in variants:
            s = summary[(n, name)]
            print(
                f"{n:<4} | {name:<14} | {s['best']:>5} | {s['mean']:>7.1f} | "
                f"{s['std']:>5.2f} | {s['time']:>12.4f}"
            )
        print("-" * 70)

    # Convergence plot (Greedy NN มีแค่ 1 จุด จึงแสดงเป็นเส้นแนวนอน)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, n in zip(axes.ravel(), sizes):
        for name in ("GRASP (no LS)", "GRASP + LS"):
            ax.plot(curves[(n, name)], label=name)
        ax.axhline(summary[(n, "Greedy NN")]["best"], color="gray",
                   linestyle=":", label="Greedy NN")
        ax.set_title(f"N = {n}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best distance (steps)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    fig.suptitle("GRASP Convergence (best run of each setting)", fontweight="bold")
    fig.tight_layout()
    fig.savefig("grasp_convergence.png", dpi=130)
    plt.close(fig)
    print("\nบันทึกกราฟ: grasp_convergence.png")

    return summary


# =========================================================
# Execution Point
# =========================================================

if __name__ == "__main__":
    pts_10 = generate_pickup_points(n=10, seed=20)
    data_10 = build_route_data(pts_10)

    res = run_grasp(data_10, seed=42)
    path, real_steps, cells = reconstruct_full_path(res["best_order"], pts_10, data_10)

    print("\n=== ผลลัพธ์ GRASP (N=10) ===")
    print("5.1 ลำดับการหยิบสินค้า:", " -> ".join(f"P{x + 1}" for x in res["best_order"]))
    for rank, (idx, cell) in enumerate(zip(res["best_order"], cells), start=1):
        print(f"    {rank:>2}. P{idx + 1} {pts_10[idx]}  ยืนหยิบที่ {cell}")
    print(f"5.3 จำนวนก้าวรวม: {real_steps} ก้าว")
    print(f"    เวลาประมวลผล: {res['runtime']:.4f} วินาที")
    print(f"    Reactive alpha probs: {res['alpha_probs']}")
    print(f"5.2 เส้นทางเดินทั้งหมด ({len(path)} ช่อง):")
    print(" -> ".join(str(p) for p in path))

    plot_route(
        path,
        pts_10,
        res["best_order"],
        f"GRASP route (N=10) - {real_steps} steps",
        "grasp_route.png",
    )
    print("\nบันทึกกราฟ: grasp_route.png")

    run_experiment()
