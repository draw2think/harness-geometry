"""
Shared rendering helpers for GeoGebra test scripts.

Keeps manual draw and llm2draw visually consistent with one global style preset.
"""

from dataclasses import dataclass


@dataclass
class RenderStyle:
    polygon_rgb: tuple[int, int, int] = (173, 216, 230)
    line_rgb: tuple[int, int, int] = (180, 180, 180)
    segment_rgb: tuple[int, int, int] = (70, 70, 70)
    circle_rgb: tuple[int, int, int] = (80, 80, 80)
    angle_rgb: tuple[int, int, int] = (220, 120, 0)
    polygon_fill: float = 0.30
    line_thickness: int = 1
    segment_thickness: int = 2
    curve_thickness: int = 2
    show_labels: bool = False


DEFAULT_STYLE = RenderStyle()


def js(ggb, script: str):
    return ggb._driver.execute_script(script)


def set_color(ggb, name: str, r: int, g: int, b: int):
    js(ggb, f'ggbApplet.setColor("{name}", {r}, {g}, {b})')


def set_thickness(ggb, name: str, thickness: int):
    js(ggb, f'ggbApplet.setLineThickness("{name}", {thickness})')


def set_visible(ggb, name: str, visible: bool):
    js(ggb, f'ggbApplet.setVisible("{name}", {"true" if visible else "false"})')


def set_label_visible(ggb, name: str, visible: bool):
    js(ggb, f'ggbApplet.setLabelVisible("{name}", {"true" if visible else "false"})')


def set_filling(ggb, name: str, alpha: float):
    js(ggb, f'ggbApplet.setFilling("{name}", {alpha})')


def fit_view_square(ggb, padding: float = 1.2):
    """Fit viewport to all points and keep strict 1:1 unit scale."""
    state = ggb.get_construction_state()
    pts = [
        (info["x"], info["y"])
        for info in state["objects"].values()
        if info.get("type") == "point" and "x" in info and "y" in info
    ]
    if not pts:
        js(ggb, "ggbApplet.setCoordSystem(-8, 8, -8, 8)")
        return

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    xmin, xmax = min(xs) - padding, max(xs) + padding
    ymin, ymax = min(ys) - padding, max(ys) + padding

    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)
    ratio = xspan / yspan
    target_ratio = 1.0

    if ratio < target_ratio:
        extra = (yspan * target_ratio - xspan) / 2
        xmin -= extra
        xmax += extra
    elif ratio > target_ratio:
        extra = (xspan / target_ratio - yspan) / 2
        ymin -= extra
        ymax += extra

    js(ggb, f"ggbApplet.setCoordSystem({xmin:.2f}, {xmax:.2f}, {ymin:.2f}, {ymax:.2f})")


def apply_global_style(ggb, style: RenderStyle = DEFAULT_STYLE):
    """Apply one style preset to all current objects."""
    state = ggb.get_construction_state()
    for name, info in state["objects"].items():
        obj_type = info.get("type", "")
        if obj_type == "polygon":
            set_color(ggb, name, *style.polygon_rgb)
            set_filling(ggb, name, style.polygon_fill)
            set_thickness(ggb, name, style.segment_thickness)
        elif obj_type == "line":
            set_color(ggb, name, *style.line_rgb)
            set_thickness(ggb, name, style.line_thickness)
        elif obj_type == "segment":
            set_color(ggb, name, *style.segment_rgb)
            set_thickness(ggb, name, style.segment_thickness)
        elif obj_type in ("circle", "conic"):
            set_color(ggb, name, *style.circle_rgb)
            set_thickness(ggb, name, style.curve_thickness)
        elif obj_type == "angle":
            set_color(ggb, name, *style.angle_rgb)

        set_label_visible(ggb, name, style.show_labels)
