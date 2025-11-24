import cv2
import numpy as np

from app.models.shapes import PolylineShape, Shape2D


def draw_shape(frame, shape, highlight: bool = False, highlight_color=(0, 255, 0)) -> None:
    if isinstance(shape, Shape2D):
        _draw_rect_shape(frame, shape, highlight, highlight_color)
    elif isinstance(shape, PolylineShape):
        _draw_polyline(frame, shape, highlight, highlight_color)


def _draw_rect_shape(frame, shape: Shape2D, highlight: bool, highlight_color) -> None:
    color = highlight_color if highlight else shape.color

    x, y = shape.center
    half_size = shape.size / 2.0
    angle_rad = np.deg2rad(shape.angle)

    points = np.array(
        [[-half_size, -half_size], [half_size, -half_size], [half_size, half_size], [-half_size, half_size]]
    )
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_points = points @ rotation_matrix.T + np.array([x, y])
    pts = rotated_points.astype(int).reshape((-1, 1, 2))

    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
    cv2.circle(frame, tuple(np.array([x, y]).astype(int)), 5, color, -1)


def _draw_polyline(frame, shape: PolylineShape, highlight: bool, highlight_color) -> None:
    if not shape.points:
        return
    color = highlight_color if highlight else shape.color
    pts = np.array(shape.points, dtype=int).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=shape.thickness, lineType=cv2.LINE_AA)
