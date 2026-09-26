from warehouse_core import (
    warehouse,
    entrance,
    exit_point,
    generate_pickup_points,
    get_pickable_cells,
    build_distance_matrix,
    print_distance_matrix,
)


# =========================================================
# Configuration
# =========================================================

N_PICKUPS = 10
SEED = 20


def main():

    print("=" * 60)
    print("WAREHOUSE PICKING ROUTE OPTIMIZATION")
    print("=" * 60)

    # =====================================================
    # 1) Generate Pickup Points
    # =====================================================

    pickup_points = generate_pickup_points(
        n=N_PICKUPS,
        seed=SEED
    )

    print("\nEntrance:")
    print(entrance)

    print("\nExit:")
    print(exit_point)

    print(
        f"\nPickup Points ({N_PICKUPS} points):"
    )

    for i, point in enumerate(
        pickup_points,
        start=1
    ):

        print(
            f"P{i}: {point}"
        )

        print(
            "    Pickable walkway:",
            get_pickable_cells(point)
        )

    # =====================================================
    # 2) Build Distance Matrix
    # =====================================================

    print(
        "\nBuilding Distance Matrix..."
    )

    distance_matrix = build_distance_matrix(
        pickup_points
    )

    # =====================================================
    # 3) Print Matrix
    # =====================================================

    print_distance_matrix(
        distance_matrix
    )

    # =====================================================
    # 4) Algorithms
    #
    # ตอนนี้ comment ไว้ก่อน
    # พอแต่ละคนทำเสร็จค่อยเปิด
    # =====================================================

    # -------------------------
    # GRASP
    # -------------------------

    # from grasp import run_grasp
    #
    # grasp_result = run_grasp(
    #     distance_matrix
    # )
    #
    # print("\n===== GRASP =====")
    # print(grasp_result)


    # -------------------------
    # ABC
    # -------------------------

    # from abc_optimizer import artificial_bee_colony
    #
    # abc_result = artificial_bee_colony(
    #     distance_matrix
    # )
    #
    # print("\n===== ABC =====")
    # print(abc_result)


    # -------------------------
    # PSO
    # -------------------------

    # from pso import run_pso
    #
    # pso_result = run_pso(
    #     distance_matrix
    # )
    #
    # print("\n===== PSO =====")
    # print(pso_result)


if __name__ == "__main__":
    main()