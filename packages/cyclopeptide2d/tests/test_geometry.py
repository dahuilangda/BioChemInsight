from cyclopeptide2d.geometry import DEFAULT_CONFIG, _layout_metrics


def test_angles_are_measured_without_ring_based_clamping():
    points = {0: (30, 0), 1: (0, 0), 2: (0, 30), 3: (-20, -15), 4: (0, -30), 5: (30, -30)}
    bonds = [(0, 1), (1, 3), (3, 4), (4, 5), (5, 0), (1, 2)]
    adjacency = [set() for _ in points]
    for a, b in bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    report = _layout_metrics(
        points,
        adjacency,
        [0, 1, 2],
        bonds,
        DEFAULT_CONFIG,
        small_rings=[(0, 1, 3, 4, 5)],
        close_ring=False,
    )
    assert report["angle_min"] == 90.0
    assert report["angle_max"] == 90.0
