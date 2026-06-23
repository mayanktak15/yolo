from src.visualization.trajectory import relative_speed


def test_relative_speed_uses_frame_delta() -> None:
    points = [(0, 0.0, 0.0), (2, 6.0, 8.0)]

    assert relative_speed(points) == 5.0

