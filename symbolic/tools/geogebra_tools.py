"""
Global GeoGebra tool catalog + provider adapters.

This module centralizes tool definitions so different LLM vendors can share
the same geometry action space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ToolParam:
    param_type: str  # "string" | "number" | "integer"
    description: str
    required: bool = True


@dataclass
class ToolSpec:
    name: str
    description: str
    params: Dict[str, ToolParam]


GLOBAL_GEOGEBRA_TOOLS: List[ToolSpec] = [
    ToolSpec("add_point", "Place a free (unconstrained) point at coordinates (x, y).", {
        "name": ToolParam("string", "Point name, e.g. A"),
        "x": ToolParam("number", "X coordinate"),
        "y": ToolParam("number", "Y coordinate"),
    }),
    ToolSpec("add_slider", "Create a numeric slider with range [min, max] and step size.", {
        "name": ToolParam("string", "Slider name"),
        "min": ToolParam("number", "Minimum value"),
        "max": ToolParam("number", "Maximum value"),
        "step": ToolParam("number", "Step value"),
    }),
    ToolSpec("set_value", "Set the value of a free numeric object (slider, free number, or boolean). "
             "Use this to change a slider's current value, e.g. after add_slider to explore "
             "how dependent objects (intersections, graphs) change with different parameter values. "
             "For booleans: 0 = false, 1 = true.", {
        "name": ToolParam("string", "Object name (slider or free numeric)"),
        "value": ToolParam("number", "New value to set"),
    }),
    ToolSpec("add_point_on", "Create a point constrained to lie on a curve or path. The object must be a line, segment, ray, circle, arc, conic, or function — NOT a point or numeric. Optionally place it near (x, y) via SetCoords.", {
        "name": ToolParam("string", "Point name"),
        "obj": ToolParam("string", "Parent object — must be a line, segment, ray, circle, arc, or conic (NOT a point or numeric)"),
        "x": ToolParam("number", "Approx X for initial placement (optional)", required=False),
        "y": ToolParam("number", "Approx Y for initial placement (optional)", required=False),
    }),
    ToolSpec("add_intersect", "Create intersection point(s) of two geometric objects. Both obj1 and obj2 must be lines, segments, rays, circles, conics, or functions — NOT points or numerics. Omit index to get ALL intersections as a single point (best for segment-segment or line-line which have exactly one). Use index=1 or index=2 when two objects have multiple intersections (e.g. line-circle). The result is stored under the exact 'name' you provide — do NOT append _1 or _2 suffixes to access it.", {
        "name": ToolParam("string", "Result point name"),
        "obj1": ToolParam("string", "First object — must be a line, segment, ray, circle, or conic (NOT a point)"),
        "obj2": ToolParam("string", "Second object — must be a line, segment, ray, circle, or conic (NOT a point)"),
        "index": ToolParam("integer", "Which intersection to return (1 or 2). Omit for segment-segment or line-line intersections.", required=False),
    }),
    ToolSpec("add_roots", "Create root point(s) (x-intercepts) of a polynomial function.", {
        "name": ToolParam("string", "Result label"),
        "obj": ToolParam("string", "Polynomial function label"),
    }),
    ToolSpec("add_turning_point", "Create turning point(s) (local extrema) of a POLYNOMIAL function. "
        "Only works on smooth polynomials — NOT on piecewise, abs(), or If() functions. "
        "For non-polynomial functions, use query_function_max/query_function_min with an interval instead.", {
        "name": ToolParam("string", "Result label"),
        "obj": ToolParam("string", "Function label"),
    }),
    ToolSpec("add_best_fit_line", "Create best-fit (least-squares regression) line from a list of points.", {
        "name": ToolParam("string", "Line name"),
        "list_obj": ToolParam("string", "List object name containing the points"),
    }),
    ToolSpec("add_segment", "Draw a segment between two existing points. Both p1 and p2 must be point objects on the canvas — NOT lines, circles, or numerics.", {
        "name": ToolParam("string", "Segment name"),
        "p1": ToolParam("string", "First endpoint (must be a point)"),
        "p2": ToolParam("string", "Second endpoint (must be a point)"),
    }),
    ToolSpec("add_line", "Draw a line through two existing points (extends infinitely). Both p1 and p2 must be point objects — NOT segments, lines, or numerics. If you need a line through a point perpendicular/parallel to another line, use add_perpendicular_line or add_parallel_line instead.", {
        "name": ToolParam("string", "Line name"),
        "p1": ToolParam("string", "First point (must be a point object)"),
        "p2": ToolParam("string", "Second point (must be a point object)"),
    }),
    ToolSpec("add_ray", "Draw a ray starting at p1 and passing through p2 (extends infinitely beyond p2). Both must be point objects.", {
        "name": ToolParam("string", "Ray name"),
        "p1": ToolParam("string", "Start point (must be a point object)"),
        "p2": ToolParam("string", "Direction point (must be a point object)"),
    }),
    ToolSpec("add_vector", "Create a directed vector from p1 to p2. Both must be point objects.", {
        "name": ToolParam("string", "Vector name"),
        "p1": ToolParam("string", "Tail (start) point (must be a point object)"),
        "p2": ToolParam("string", "Head (end) point (must be a point object)"),
    }),
    ToolSpec("add_polygon", "Draw a polygon through the given points (minimum 3). Points must be a comma-separated string of point names, e.g. \"A,B,C\" or \"A,B,C,D\".", {
        "name": ToolParam("string", "Polygon name"),
        "points": ToolParam("string", "Comma-separated point names, e.g. A,B,C"),
    }),
    ToolSpec("add_regular_polygon", "Draw a regular polygon with n vertices, defined by two adjacent vertices p1 and p2.", {
        "name": ToolParam("string", "Polygon name"),
        "p1": ToolParam("string", "First vertex"),
        "p2": ToolParam("string", "Second (adjacent) vertex"),
        "n": ToolParam("integer", "Number of vertices (= number of sides)"),
    }),
    ToolSpec("add_vertex", "Get the nth vertex of a polygon or endpoint of a segment (1-based index). The object must be a polygon or segment — NOT a point, line, or circle. For polygons: returns the nth vertex. For segments: index 1 = start point, 2 = end point. Returns a point whose x/y appear in new_objects.", {
        "name": ToolParam("string", "Result point name, e.g. V3"),
        "polygon": ToolParam("string", "Polygon or Segment label (NOT a point, line, or circle)"),
        "index": ToolParam("integer", "Vertex index (1-based)"),
    }),
    ToolSpec("add_midpoint", "Create the midpoint between two points, OR the center of a conic (circle/ellipse). For two points: provide p1 and p2. For center of a circle/ellipse: provide only p1 as the conic name (omit p2).", {
        "name": ToolParam("string", "Midpoint name"),
        "p1": ToolParam("string", "First point, or conic name (circle/ellipse) to get its center"),
        "p2": ToolParam("string", "Second point. Omit when p1 is a conic.", required=False),
    }),
    ToolSpec("add_perpendicular_line", "Create a line through a point, perpendicular to a reference line/segment/ray. The 'point' must be a point object, and 'line' must be a line, segment, or ray — NOT a point. If you have two points A and B and want a perpendicular at A, first create the line: L=Line(A,B), then PerpendicularLine(A, L).", {
        "name": ToolParam("string", "Line name"),
        "point": ToolParam("string", "Point the new line passes through (must be a point object)"),
        "line": ToolParam("string", "Reference line, segment, or ray to be perpendicular to (NOT a point)"),
    }),
    ToolSpec("add_perpendicular_bisector", "Create the perpendicular bisector of segment p1-p2.", {
        "name": ToolParam("string", "Line name"),
        "p1": ToolParam("string", "First endpoint"),
        "p2": ToolParam("string", "Second endpoint"),
    }),
    ToolSpec("add_parallel_line", "Create a line through a point, parallel to a reference line/segment/ray. The 'line' param must be a line, segment, or ray — NOT a point.", {
        "name": ToolParam("string", "Line name"),
        "point": ToolParam("string", "Point the new line passes through (must be a point object)"),
        "line": ToolParam("string", "Reference line, segment, or ray to be parallel to (NOT a point)"),
    }),
    ToolSpec("add_angle_bisector", "Create the angle bisector of the angle at 'vertex' formed by arm points p1 and p2. Returns a single bisector line.", {
        "name": ToolParam("string", "Line name"),
        "p1": ToolParam("string", "First arm point"),
        "vertex": ToolParam("string", "Vertex (apex) of the angle"),
        "p2": ToolParam("string", "Second arm point"),
    }),
    ToolSpec("add_tangent", "Create tangent line(s) from an external point to a conic (circle, ellipse, hyperbola, parabola). "
        "ALWAYS use this tool for tangent lines — do NOT manually construct tangents with add_segment or add_line. "
        "May return one or two lines stored under the exact 'name' you provide. Do NOT append _1 or _2 suffixes — use the name directly. "
        "For a reliable single tangent contact point, prefer the Thales-circle construction (Midpoint + Circle + Intersect).", {
        "name": ToolParam("string", "Tangent line name (use this name directly, not name_1)"),
        "point": ToolParam("string", "External point"),
        "conic": ToolParam("string", "Circle, ellipse, hyperbola, or parabola"),
    }),
    ToolSpec("add_tangent_conic_conic",
        "Create common tangent line(s) between two circles (or ellipses). "
        "ALWAYS use this for common/external/internal tangents between two circles — "
        "do NOT approximate with add_segment or add_line. "
        "Returns up to 4 tangent lines, auto-named name_{1} through name_{4}. "
        "Two external tangents exist if circles don't overlap; "
        "two internal tangents exist if circles don't intersect. "
        "Use the auto-generated names (e.g. 'ct_{1}', 'ct_{2}') to reference individual lines.", {
        "name": ToolParam("string", "Base name for tangent lines (lines will be name_{1}, name_{2}, etc.)"),
        "conic1": ToolParam("string", "First circle or conic"),
        "conic2": ToolParam("string", "Second circle or conic"),
    }),
    ToolSpec("add_circle", "Circle by center + numeric radius, OR center + point on circle. Provide exactly one of 'radius' or 'point'.", {
        "name": ToolParam("string", "Circle name, e.g. circ_O"),
        "center": ToolParam("string", "Center point"),
        "radius": ToolParam("number", "Numeric radius. Omit when using point.", required=False),
        "point": ToolParam("string", "A point on the circle (radius = distance from center). Omit when using numeric radius.", required=False),
    }),
    ToolSpec("add_arc", "Circular arc sweeping CCW from start_pt toward end_pt, with radius = Distance(center, start_pt). Note: end_pt does NOT need to lie on the circle — it only determines the end angle. Use query_length on the arc to get arc length. Do NOT use query_distance for arc length (that gives chord length).", {
        "name": ToolParam("string", "Arc name, e.g. arc_AB"),
        "center": ToolParam("string", "Center point"),
        "start_pt": ToolParam("string", "Start point (must lie on circle; determines radius)"),
        "end_pt": ToolParam("string", "End point (determines end angle; CCW from start_pt)"),
    }),
    ToolSpec("add_sector", "Circular sector (pie slice) sweeping CCW from start_pt toward end_pt around center. Radius = Distance(center, start_pt). Use query_area on the sector to get sector area.", {
        "name": ToolParam("string", "Sector name, e.g. sector_OAB"),
        "center": ToolParam("string", "Center point"),
        "start_pt": ToolParam("string", "Start point (determines radius)"),
        "end_pt": ToolParam("string", "End point (determines end angle; CCW from start_pt)"),
    }),
    ToolSpec("add_semicircle",
        "Create a semicircle with the given segment as diameter. "
        "The arc is drawn on the LEFT side when walking from p1 to p2. "
        "Direction rules: "
        "(1) p1=left, p2=right (horizontal) → arc ABOVE; "
        "(2) p1=right, p2=left (horizontal) → arc BELOW; "
        "(3) p1=bottom, p2=top (vertical) → arc LEFT; "
        "(4) p1=top, p2=bottom (vertical) → arc RIGHT. "
        "To flip the arc, SWAP p1 and p2. "
        "Useful for Thales' theorem: any point on the semicircle sees the diameter at 90°.", {
        "name": ToolParam("string", "Semicircle name, e.g. semi_AB"),
        "p1": ToolParam("string", "First endpoint of diameter"),
        "p2": ToolParam("string", "Second endpoint of diameter"),
    }),
    ToolSpec("add_circle_3_points", "Circle through three points.", {
        "name": ToolParam("string", "Circle name"),
        "a": ToolParam("string", "Point A"),
        "b": ToolParam("string", "Point B"),
        "c": ToolParam("string", "Point C"),
    }),
    ToolSpec("add_incircle", "Create the inscribed circle (incircle) of a triangle — the unique circle tangent to all three sides. Use query_radius to get the inradius, query_center to get the incenter.", {
        "name": ToolParam("string", "Incircle name, e.g. incircle_ABC"),
        "a": ToolParam("string", "First vertex of the triangle"),
        "b": ToolParam("string", "Second vertex of the triangle"),
        "c": ToolParam("string", "Third vertex of the triangle"),
    }),
    ToolSpec("add_center", "Create the center point of a circle, ellipse, or hyperbola. Returns a point object whose x/y appear in new_objects.", {
        "name": ToolParam("string", "Center point name, e.g. ctr_O"),
        "conic": ToolParam("string", "Circle, ellipse, or hyperbola label"),
    }),
    ToolSpec("add_triangle_center", "Create a named triangle center point. n=1 Incenter, n=2 Centroid, n=3 Circumcenter, n=4 Orthocenter. Returns a point whose coordinates appear in new_objects.", {
        "name": ToolParam("string", "Result point name, e.g. incenter_ABC"),
        "a": ToolParam("string", "First vertex"),
        "b": ToolParam("string", "Second vertex"),
        "c": ToolParam("string", "Third vertex"),
        "n": ToolParam("integer", "Center index: 1=Incenter, 2=Centroid, 3=Circumcenter, 4=Orthocenter"),
    }),
    ToolSpec("add_ellipse", "Create an ellipse defined by two foci and one point on the ellipse.", {
        "name": ToolParam("string", "Conic name"),
        "f1": ToolParam("string", "Focus 1"),
        "f2": ToolParam("string", "Focus 2"),
        "p": ToolParam("string", "Point on ellipse"),
    }),
    ToolSpec("add_parabola", "Create a parabola from a focus point and a directrix line.", {
        "name": ToolParam("string", "Conic name"),
        "focus": ToolParam("string", "Focus point"),
        "directrix": ToolParam("string", "Directrix line"),
    }),
    ToolSpec("add_hyperbola", "Create a hyperbola defined by two foci and one point on the hyperbola.", {
        "name": ToolParam("string", "Conic name"),
        "f1": ToolParam("string", "Focus 1"),
        "f2": ToolParam("string", "Focus 2"),
        "p": ToolParam("string", "Point on hyperbola"),
    }),
    ToolSpec("add_angle", "Create a visual angle object at vertex. Value is returned as a display string in new_objects (e.g. '37.5°'). Prefer query_angle when you need a clean numeric float.", {
        "name": ToolParam("string", "Angle label"),
        "p1": ToolParam("string", "First arm point"),
        "vertex": ToolParam("string", "Vertex (apex) of the angle"),
        "p2": ToolParam("string", "Second arm point"),
    }),
    ToolSpec("add_distance", "Create a visual distance label between two points. Value is returned as a display string in new_objects (e.g. '5.83'). Prefer query_distance when you need a clean numeric float.", {
        "name": ToolParam("string", "Distance label"),
        "p1": ToolParam("string", "First point"),
        "p2": ToolParam("string", "Second point"),
    }),
    ToolSpec("add_area", "Create a visual area label for a polygon, circle, or sector. Value is returned as a display string in new_objects. Prefer query_area when you need a clean numeric float.", {
        "name": ToolParam("string", "Area label"),
        "obj": ToolParam("string", "Polygon, circle, or CircularSector label"),
    }),
    ToolSpec("add_slope", "Create a visual slope triangle for a line. Value is returned as a display string in new_objects. Prefer query_slope when you need a clean numeric float.", {
        "name": ToolParam("string", "Slope label"),
        "line": ToolParam("string", "Line label"),
    }),
    ToolSpec("transform_reflect_line", "Reflect an object across a line (mirror reflection).", {
        "name": ToolParam("string", "New object label"),
        "obj": ToolParam("string", "Object to reflect"),
        "line": ToolParam("string", "Mirror line"),
    }),
    ToolSpec("transform_reflect_point", "Reflect an object through a center point (point symmetry / 180° rotation).", {
        "name": ToolParam("string", "New object label"),
        "obj": ToolParam("string", "Object to reflect"),
        "point": ToolParam("string", "Center of reflection"),
    }),
    ToolSpec("transform_rotate", "Rotate an object by a given angle around a center point. The 'obj' must be an existing object on canvas, and 'center' must be a point. Angle can be in degrees with the ° symbol (e.g. 120°) or radians (e.g. pi/3). Without ° the angle is interpreted as radians.", {
        "name": ToolParam("string", "New object label"),
        "obj": ToolParam("string", "Object to rotate (must exist on canvas)"),
        "angle": ToolParam("string", "Angle expression — use ° for degrees (e.g. 120°) or radians (e.g. pi/3)"),
        "center": ToolParam("string", "Center of rotation (must be a point object)"),
    }),
    ToolSpec("transform_translate", "Translate (shift) an object by a vector.", {
        "name": ToolParam("string", "New object label"),
        "obj": ToolParam("string", "Object to translate"),
        "vector": ToolParam("string", "Translation vector label"),
    }),
    ToolSpec("transform_dilate", "Dilate (scale) an object by a factor around a center point.", {
        "name": ToolParam("string", "New object label"),
        "obj": ToolParam("string", "Object to dilate"),
        "factor": ToolParam("number", "Scale factor (>1 enlarges, 0<f<1 shrinks)"),
        "center": ToolParam("string", "Center of dilation"),
    }),
    # ── Analytic geometry / function tools ───────────────────────────────────
    ToolSpec("add_function", "Define a function of x and display its graph. The expression must use 'x' as the variable. "
        "For piecewise functions use If(): e.g. expr='If(x<0, -x, x^2)'. "
        "Optionally restrict the visible domain with start_x / end_x. "
        "Do NOT include '(x)' in the name — it is added automatically.", {
        "name": ToolParam("string", "Function name without (x), e.g. 'f' or 'g' — NOT 'f(x)'"),
        "expr": ToolParam("string", "Expression in x. Examples: 'x^2', 'sin(x)', '2*x+1', 'If(x<0, -x, x^2)'"),
        "start_x": ToolParam("number", "Left bound of visible domain (optional — omit to show full range)", required=False),
        "end_x": ToolParam("number", "Right bound of visible domain (optional — omit to show full range)", required=False),
    }),
    # ── Calculus / Analysis tools ────────────────────────────────────────────
    ToolSpec("add_derivative", "Create the derivative of a function and plot it. "
        "Returns a new function object. Use order=2 for second derivative, etc. "
        "The input must be an existing function name (defined via add_function). "
        "Example: add_derivative('df', 'f') creates df(x) = f'(x).", {
        "name": ToolParam("string", "Name for the derivative function, e.g. 'df'"),
        "function": ToolParam("string", "Existing function name to differentiate, e.g. 'f'"),
        "order": ToolParam("integer", "Derivative order (default 1). Use 2 for f''(x), etc.", required=False),
    }),
    ToolSpec("add_integral_function", "Create the antiderivative (indefinite integral) of a function and plot it. "
        "Returns a new function object F(x) such that F'(x) = f(x). "
        "The input must be an existing function name. "
        "For definite integrals (numeric value), use query_definite_integral instead.", {
        "name": ToolParam("string", "Name for the integral function, e.g. 'F'"),
        "function": ToolParam("string", "Existing function name to integrate, e.g. 'f'"),
    }),
    ToolSpec("add_inflection_point", "Find inflection point(s) of a function (where concavity changes). "
        "Works best on polynomials. Returns point(s) on the function graph. "
        "The input must be an existing function name.", {
        "name": ToolParam("string", "Result point name"),
        "function": ToolParam("string", "Function name, e.g. 'f'"),
    }),
    ToolSpec("add_asymptote", "Find asymptote line(s) of a rational function. "
        "Returns a list of lines (vertical, horizontal, oblique). "
        "Works on rational functions like (x^2-1)/(x-2). "
        "May not find all asymptotes for transcendental functions (e.g. ln(x)).", {
        "name": ToolParam("string", "Result name for asymptote list"),
        "function": ToolParam("string", "Function name, e.g. 'f'"),
    }),
    ToolSpec("add_curve", "Create a parametric curve {x(t), y(t)} over a parameter range [t_start, t_end]. "
        "IMPORTANT: do NOT use x, y, or z as the parameter variable — use 't' or 's'. "
        "Example: x_expr='3*cos(t)', y_expr='2*sin(t)', t_start='0', t_end='2*pi' draws an ellipse.", {
        "name": ToolParam("string", "Curve name"),
        "x_expr": ToolParam("string", "X-component as expression in t, e.g. '3*cos(t)'"),
        "y_expr": ToolParam("string", "Y-component as expression in t, e.g. '2*sin(t)'"),
        "t_start": ToolParam("string", "Start value of t (number or expression, e.g. '0' or '-pi')"),
        "t_end": ToolParam("string", "End value of t (number or expression, e.g. '2*pi' or '10')"),
        "param": ToolParam("string", "Parameter variable name — default 't'. Never use x, y, or z.", required=False),
    }),
    ToolSpec("add_inequality", "Create a shaded inequality region. GeoGebra auto-fills the feasible region. "
        "Supports: polynomial in one var (x^3 > x + 1), quadratic in two vars (x^2 + y^2 <= 4), "
        "linear in one var (y < 2*x + 1). Combine with && (AND) or || (OR). "
        "Use <=, >=, <, > for comparison. The shaded region is drawn automatically.", {
        "name": ToolParam("string", "Inequality object name"),
        "expr": ToolParam("string", "Inequality expression, e.g. 'y <= x^2', 'x^2 + y^2 < 4', '(x >= 0) && (y <= 3)'"),
    }),
    ToolSpec("add_integral_shade", "Shade the area under a function or between two functions over [x_start, x_end]. "
        "The shaded region is drawn automatically on the canvas and the numeric area is returned. "
        "Mode 1 (func to x-axis): provide func, x_start, x_end. "
        "Mode 2 (between two funcs): also provide func2 — shades the region between func and func2.", {
        "name": ToolParam("string", "Integral object name"),
        "func": ToolParam("string", "Function name — must already exist on canvas (e.g. 'f' if you defined f(x)=...)"),
        "x_start": ToolParam("string", "Left bound (number or expression, e.g. '0' or '-pi')"),
        "x_end": ToolParam("string", "Right bound (number or expression, e.g. '3' or 'pi')"),
        "func2": ToolParam("string", "Second function for between-curves mode (optional — omit to shade to x-axis)", required=False),
    }),
    ToolSpec("add_text", "Place a text label or annotation at a position on the canvas. "
        "The text is a visual element for labels, formulas, or annotations. "
        "For LaTeX formulas set latex=1 and use LaTeX syntax in the text (e.g. '\\frac{1}{2}').", {
        "name": ToolParam("string", "REQUIRED unique identifier for this text object (e.g. 'label_A', 'txt_zero')"),
        "text": ToolParam("string", "Text content — plain text or LaTeX formula"),
        "x": ToolParam("number", "X position on canvas"),
        "y": ToolParam("number", "Y position on canvas"),
        "latex": ToolParam("integer", "1 = render as LaTeX formula, 0 = plain text (default 0)", required=False),
    }),
    # ── Utility ──────────────────────────────────────────────────────────────
    ToolSpec("rename_object", "Rename an existing object on the canvas. "
        "Useful when GeoGebra auto-names objects with unwanted names "
        "(e.g. Rotate may create \"A'\" or Tangent may create \"name_{1}\"). "
        "After renaming, use the new name to reference the object in all subsequent calls.", {
        "name": ToolParam("string", "CURRENT name of the object (the name to change FROM)"),
        "new_name": ToolParam("string", "NEW name for the object (the name to change TO)"),
    }),
    ToolSpec("set_label_visible", "Show or hide an object's label in the Graphics View.", {
        "name": ToolParam("string", "Object label"),
        "visible": ToolParam("integer", "1 = show label, 0 = hide label"),
    }),
    ToolSpec("set_object_visible", "Show or hide an object in the Graphics View.", {
        "name": ToolParam("string", "Object label"),
        "visible": ToolParam("integer", "1 = show object, 0 = hide object"),
    }),
    ToolSpec("delete_object", "Delete an object from the construction by name. WARNING: also deletes all dependent objects (e.g. deleting a point removes lines/circles built from it).", {
        "name": ToolParam("string", "Object label to delete"),
    }),
]

# ── Query tools (boolean predicates + scalar readbacks) ─────────────────────
QUERY_GEOGEBRA_TOOLS: List[ToolSpec] = [
    # ── Boolean predicates ── each creates a named boolean object in GeoGebra
    ToolSpec("query_are_parallel", "Check if two linear objects (lines, segments, rays) are parallel. Returns boolean.", {
        "name": ToolParam("string", "Result label"),
        "obj1": ToolParam("string", "First line/segment/ray"),
        "obj2": ToolParam("string", "Second line/segment/ray"),
    }),
    ToolSpec("query_are_perpendicular", "Check if two linear objects are perpendicular. Returns boolean.", {
        "name": ToolParam("string", "Result label"),
        "obj1": ToolParam("string", "First line/segment/ray"),
        "obj2": ToolParam("string", "Second line/segment/ray"),
    }),
    ToolSpec("query_is_tangent", "Check if a line is tangent to a conic/circle. Returns boolean. Note argument order: line first, conic second.", {
        "name": ToolParam("string", "Result label"),
        "line": ToolParam("string", "Line (first argument)"),
        "conic": ToolParam("string", "Circle or conic (second argument)"),
    }),
    ToolSpec("query_is_in_region", "Check if a point lies inside (or on the boundary of) a polygon, circle, or closed region. Returns boolean. "
        "NOT for functions/curves — to check if a point (px, py) is on function f, compare f(px) with py instead.", {
        "name": ToolParam("string", "Result label"),
        "point": ToolParam("string", "Point to test"),
        "region": ToolParam("string", "Polygon, circle, or closed region"),
    }),
    ToolSpec("query_are_equal", "Check if two objects are geometrically identical — same position, shape, and size. Stricter than congruence: AreEqual((0,0),(0,0)) is true, but two congruent circles at different centers are NOT equal. Returns boolean.", {
        "name": ToolParam("string", "Result label"),
        "obj1": ToolParam("string", "First object"),
        "obj2": ToolParam("string", "Second object"),
    }),
    ToolSpec("query_are_collinear", "Check if three points lie on the same line. Returns boolean.", {
        "name": ToolParam("string", "Result label"),
        "a": ToolParam("string", "Point A"),
        "b": ToolParam("string", "Point B"),
        "c": ToolParam("string", "Point C"),
    }),
    ToolSpec("query_are_concyclic", "Check if exactly four points all lie on a common circle. Returns boolean. Requires exactly 4 points.", {
        "name": ToolParam("string", "Result label"),
        "a": ToolParam("string", "Point A"),
        "b": ToolParam("string", "Point B"),
        "c": ToolParam("string", "Point C"),
        "d": ToolParam("string", "Point D"),
    }),
    ToolSpec("query_are_congruent", "Check if two objects have the same shape and size, regardless of position or orientation (e.g., two circles with equal radii are congruent even if at different centers). Returns boolean.", {
        "name": ToolParam("string", "Result label"),
        "obj1": ToolParam("string", "First object"),
        "obj2": ToolParam("string", "Second object"),
    }),
    ToolSpec("query_is_defined", "Check if a named object is defined and valid (e.g., an intersection that actually exists). Returns boolean (1=defined, 0=undefined). Useful after constructions that may fail in degenerate cases.", {
        "name": ToolParam("string", "Result label"),
        "obj": ToolParam("string", "Object label to check"),
    }),
    ToolSpec("query_dependents", "List all objects that depend on a given object (the cascade set that would be deleted by delete_object). Returns a comma-separated string of dependent object names, or empty string if none. Use BEFORE delete_object to check impact.", {
        "name": ToolParam("string", "Result label"),
        "obj": ToolParam("string", "Object to query dependents of"),
    }),
    # ── Scalar queries ── create a named numeric object and return its value
    ToolSpec("query_length", "Get the length of a segment, arc, or vector (numeric result). Works on Segment, CircularArc, and Vector objects. For arc length: first use add_arc to create the arc, then call query_length on it. WARNING: Length(list) returns element count, NOT geometric length — do not pass a list.", {
        "name": ToolParam("string", "Result label"),
        "obj": ToolParam("string", "Segment, CircularArc, or Vector label"),
    }),
    ToolSpec("query_perimeter", "Get the perimeter of a polygon, circle, or ellipse (numeric result). For a circle this returns circumference = 2πr.", {
        "name": ToolParam("string", "Result label"),
        "obj": ToolParam("string", "Polygon, circle, or ellipse label"),
    }),
    ToolSpec("query_angle", "Measure the angle at vertex b (the apex), sweeping counter-clockwise from ray b→a to ray b→c. Returns degrees. IMPORTANT: point order matters — Angle(A,B,C) ≠ Angle(C,B,A). For the interior angle of triangle ABC at vertex B, use Angle(C,B,A) (go from one side to the other in the direction that gives the interior angle). All three must be existing point objects.", {
        "name": ToolParam("string", "Result label"),
        "a":    ToolParam("string", "First arm point — angle sweeps CCW from ray b→a (must be a point)"),
        "b":    ToolParam("string", "Vertex (apex) point"),
        "c":    ToolParam("string", "Second arm point — angle sweeps TO ray b→c (must be a point)"),
    }),
    ToolSpec("query_area", "Compute the area of a polygon, circle, ellipse, or CircularSector (numeric result).", {
        "name": ToolParam("string", "Result label"),
        "obj":  ToolParam("string", "Polygon, circle, ellipse, or CircularSector label"),
    }),
    ToolSpec("query_distance", "Shortest distance between two objects (numeric result). Supports: point↔point, point↔line, point↔segment, point↔conic, point↔function, line↔line (parallel). Returns straight-line distance — NOT arc length. For arc length use add_arc + query_length.", {
        "name": ToolParam("string", "Result label"),
        "obj1": ToolParam("string", "First object (point, line, segment, conic, function)"),
        "obj2": ToolParam("string", "Second object"),
    }),
    ToolSpec("query_slope", "Get the slope (rise/run = Δy/Δx) of a line (numeric result). Applies to Line objects; for a segment use its underlying line.", {
        "name": ToolParam("string", "Result label"),
        "obj":  ToolParam("string", "Line label"),
    }),
    ToolSpec("query_x_coord", "Get the x-coordinate of a point (numeric result).", {
        "name": ToolParam("string", "Result label"),
        "pt":   ToolParam("string", "Point label"),
    }),
    ToolSpec("query_y_coord", "Get the y-coordinate of a point (numeric result).", {
        "name": ToolParam("string", "Result label"),
        "pt":   ToolParam("string", "Point label"),
    }),
    ToolSpec("query_radius", "Get the radius of a circle or semicircle as a numeric value. Works on Circle objects (including Incircle results).", {
        "name": ToolParam("string", "Result label"),
        "conic": ToolParam("string", "Circle label"),
    }),
    ToolSpec("query_solve", "Solve an algebraic equation symbolically for a variable and return the solution string (e.g. '{x = 30}'). For simple linear/quadratic equations in geometry problems. Example: equation='2*x+30=90', variable='x' → '{x = 30}'.", {
        "name": ToolParam("string", "Result label"),
        "equation": ToolParam("string", "Equation to solve, e.g. 2*x + 30 = 90"),
        "variable": ToolParam("string", "Variable to solve for, e.g. x"),
    }),
    ToolSpec("query_nsolve", "Solve an equation numerically and return the solution string. Use as fallback when query_solve fails (e.g. trigonometric equations). Example: equation='cos(x)=0.5', variable='x'.", {
        "name": ToolParam("string", "Result label"),
        "equation": ToolParam("string", "Equation to solve numerically"),
        "variable": ToolParam("string", "Variable to solve for"),
    }),

    # ── Calculus queries ──────────────────────────────────────────────────
    ToolSpec("query_definite_integral", "Compute the definite integral ∫ₐᵇ f(x)dx and return its numeric value. "
        "The function must already exist on canvas (defined via add_function). "
        "Returns a single number (the signed area under the curve). "
        "For plotting the antiderivative, use add_integral_function instead.", {
        "function": ToolParam("string", "Existing function name, e.g. 'f'"),
        "start": ToolParam("number", "Lower bound a"),
        "end": ToolParam("number", "Upper bound b"),
    }),
    ToolSpec("query_function_max", "Find the point where a function reaches its local maximum in [start, end]. "
        "Returns a point (x, y) on the function graph. "
        "The function should be continuous with only one local max in the interval. "
        "For polynomials with multiple extrema, use add_turning_point instead.", {
        "function": ToolParam("string", "Existing function name, e.g. 'f'"),
        "start": ToolParam("number", "Left bound of search interval"),
        "end": ToolParam("number", "Right bound of search interval"),
    }),
    ToolSpec("query_function_min", "Find the point where a function reaches its local minimum in [start, end]. "
        "Returns a point (x, y) on the function graph. "
        "The function should be continuous with only one local min in the interval. "
        "For polynomials with multiple extrema, use add_turning_point instead.", {
        "function": ToolParam("string", "Existing function name, e.g. 'f'"),
        "start": ToolParam("number", "Left bound of search interval"),
        "end": ToolParam("number", "Right bound of search interval"),
    }),
]

# ── Render / styling tools (plug-in group for visual benchmarks) ─────────
#
# These tools control *appearance* without changing geometry.
# Include this group when the task requires styled output (e.g. GenExam).
# For pure geometry solving (e.g. GeoLaux), these can be omitted.
#
RENDER_GEOGEBRA_TOOLS: List[ToolSpec] = [
    ToolSpec("render_set_color", "Set the color of an existing object. Use an English color name. "
        "Common names: Black, Red, Blue, Green, Orange, Purple, Cyan, Gray, Brown, "
        "Magenta, Maroon, Gold, Pink, Yellow, White, Dark Blue, Dark Green, Light Gray.", {
        "obj": ToolParam("string", "Object name (must already exist on canvas)"),
        "color": ToolParam("string", "English color name, e.g. 'Red', 'Blue', 'Black', 'Orange'"),
    }),
    ToolSpec("render_set_line_style", "Set line style of a line, segment, ray, circle, arc, or function graph. "
        "0 = Solid (default), 1 = Dashed (long), 2 = Dashed (short), 3 = Dotted, 4 = Dash-dot.", {
        "obj": ToolParam("string", "Object name (line, segment, circle, arc, function, etc.)"),
        "style": ToolParam("integer", "Line style code: 0=Solid, 1=Dashed long, 2=Dashed short, 3=Dotted, 4=Dash-dot"),
    }),
    ToolSpec("render_set_line_thickness", "Set line thickness of an object. Default is about 2–5.", {
        "obj": ToolParam("string", "Object name"),
        "thickness": ToolParam("integer", "Thickness value (2=thin, 5=normal, 8=thick, 13=very thick)"),
    }),
    ToolSpec("render_set_point_style", "Change the visual marker of a point. "
        "0=Filled circle (default), 1=Cross, 2=Empty circle, 3=Plus, "
        "4=Filled diamond, 5=Empty diamond, 6=Triangle up, 7=Triangle down.", {
        "obj": ToolParam("string", "Point name"),
        "style": ToolParam("integer", "0=Filled dot, 1=Cross, 2=Empty circle, 3=Plus, 4=Diamond, 5=Empty diamond, 6=Triangle up"),
    }),
    ToolSpec("render_set_point_size", "Change the visual size of a point. Default is 5.", {
        "obj": ToolParam("string", "Point name"),
        "size": ToolParam("integer", "Point size (1=tiny, 3=small, 5=default, 7=large, 9=very large)"),
    }),
    ToolSpec("render_set_filling", "Set the fill opacity of a closed shape (polygon, circle, sector, inequality region). "
        "0.0 = fully transparent (no fill), 1.0 = fully opaque.", {
        "obj": ToolParam("string", "Object name (polygon, circle, sector, inequality, etc.)"),
        "opacity": ToolParam("number", "Fill opacity: 0.0 (transparent) to 1.0 (opaque). Typical: 0.3 for light fill."),
    }),
    ToolSpec("render_set_decoration", "Add tick marks or arrows to a segment, or arc decorations to an angle. "
        "Segment codes: 0=none, 1=one tick, 2=two ticks, 3=three ticks, 4=one arrow, 5=two arrows, 6=three arrows. "
        "Angle codes: 0=none, 1=double arc, 2=triple arc, 3=one tick, 4=two ticks, 5=three ticks.", {
        "obj": ToolParam("string", "Segment or Angle object name"),
        "decoration": ToolParam("integer", "Decoration code (meaning depends on object type — see description)"),
    }),
    ToolSpec("render_show_axes", "Show or hide the coordinate axes (both x and y together).", {
        "visible": ToolParam("integer", "1 = show axes, 0 = hide axes"),
    }),
    ToolSpec("render_show_grid", "Show or hide the coordinate grid.", {
        "visible": ToolParam("integer", "1 = show grid, 0 = hide grid"),
    }),
    ToolSpec("render_set_caption", "Set a custom caption for an object and switch its label display to show the caption. "
        "Use this to display custom text next to an object instead of its internal name.", {
        "obj": ToolParam("string", "Object name"),
        "caption": ToolParam("string", "Caption text to display, e.g. 'r = 5', '∠A', 'tangent line'"),
    }),
    ToolSpec("render_set_label_mode", "Control what is displayed as an object's label "
        "and automatically make the label visible. "
        "0 = Name only (e.g. 'A'), 1 = Name + Value (e.g. 'A = (1,2)'), "
        "2 = Value only (e.g. '(1,2)'), 3 = Caption (must set caption first).", {
        "obj": ToolParam("string", "Object name"),
        "mode": ToolParam("integer", "Label mode: 0=Name, 1=Name+Value, 2=Value, 3=Caption"),
    }),
    ToolSpec("render_set_coord_system", "Set the visible viewport bounds — controls which region of the coordinate plane is shown. "
        "This is the zoom/pan control. For analytic geometry, set bounds to show the relevant range.", {
        "x_min": ToolParam("number", "Left bound of visible x-range"),
        "x_max": ToolParam("number", "Right bound of visible x-range"),
        "y_min": ToolParam("number", "Bottom bound of visible y-range"),
        "y_max": ToolParam("number", "Top bound of visible y-range"),
    }),
    ToolSpec("render_add_right_angle_mark", "Add a right-angle square marker (□) at the vertex. "
        "GeoGebra automatically renders the □ symbol for Angle objects that are exactly 90°. "
        "This tool verifies the angle is 90° (±2° tolerance) before creating the mark — "
        "it will FAIL if the angle is not a right angle.", {
        "name": ToolParam("string", "Angle mark label"),
        "a": ToolParam("string", "First arm point"),
        "b": ToolParam("string", "Vertex — the corner where the □ mark appears"),
        "c": ToolParam("string", "Second arm point"),
    }),
]


def _json_schema(spec: ToolSpec) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    required: List[str] = []
    for key, p in spec.params.items():
        t = {"string": "string", "number": "number", "integer": "integer"}[p.param_type]
        props[key] = {"type": t, "description": p.description}
        if p.required:
            required.append(key)
    return {"type": "object", "properties": props, "required": required}


def build_openai_tools(include_render: bool = False) -> List[Dict[str, Any]]:
    # IMPORTANT: Render phase = GLOBAL + QUERY + RENDER (additive, not replacement).
    # GLOBAL contains display tools (set_label_visible, set_object_visible,
    # rename_object) that are essential during render. Do NOT pass only
    # RENDER tools to save tokens — the LLM will lose critical capabilities.
    specs = GLOBAL_GEOGEBRA_TOOLS + QUERY_GEOGEBRA_TOOLS
    if include_render:
        specs = specs + RENDER_GEOGEBRA_TOOLS
    tools: List[Dict[str, Any]] = []
    for spec in specs:
        tools.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": _json_schema(spec),
            },
        })
    return tools


def build_anthropic_tools(include_render: bool = False) -> List[Dict[str, Any]]:
    specs = GLOBAL_GEOGEBRA_TOOLS + QUERY_GEOGEBRA_TOOLS
    if include_render:
        specs = specs + RENDER_GEOGEBRA_TOOLS
    tools: List[Dict[str, Any]] = []
    for spec in specs:
        tools.append({
            "name": spec.name,
            "description": spec.description,
            "input_schema": _json_schema(spec),
        })
    return tools


def build_gemini_tools(include_render: bool = False):
    from google.genai import types

    specs = GLOBAL_GEOGEBRA_TOOLS + QUERY_GEOGEBRA_TOOLS
    if include_render:
        specs = specs + RENDER_GEOGEBRA_TOOLS
    decls = []
    for spec in specs:
        properties = {}
        required = []
        for key, p in spec.params.items():
            g_type = {"string": "STRING", "number": "NUMBER", "integer": "INTEGER"}[p.param_type]
            properties[key] = types.Schema(type=g_type, description=p.description)
            if p.required:
                required.append(key)
        decls.append(
            types.FunctionDeclaration(
                name=spec.name,
                description=spec.description,
                parameters=types.Schema(type="OBJECT", properties=properties, required=required),
            )
        )
    return [types.Tool(function_declarations=decls)]


def execute_geogebra_tool(ggb, tool_name: str, args: Dict[str, Any]) -> Tuple[str, bool, str]:
    """Execute one standardized tool call on GeoGebra."""

    def run(cmd: str) -> Tuple[str, bool, str]:
        result = ggb.eval_command(cmd)
        return cmd, bool(result.success), (result.error_message or "")

    def run_checked(cmd: str, name: str, hint: str = "") -> Tuple[str, bool, str]:
        """Run command, then verify the created object is actually defined.

        Many GeoGebra construction commands (AngleBisector, Tangent, Rotate,
        PerpendicularLine, …) silently return undefined objects when inputs
        are degenerate (coincident points, collinear points, etc.).
        This wrapper catches those cases and returns a clear error.
        """
        cmd_str, ok, err = run(cmd)
        if ok and not ggb.is_defined(name):
            detail = f"Object '{name}' is undefined after command (degenerate input?)"
            if hint:
                detail += f". Hint: {hint}"
            return cmd_str, False, detail
        return cmd_str, ok, err

    # ── Auto-correct _N suffix hallucination ──────────────────────────
    # LLMs sometimes append _1 / _2 to object names (e.g. "C_pts_1"
    # instead of "C_pts").  If the suffixed name doesn't exist but the
    # base name does, silently fix it.  Only applies to reference params
    # (not the "name" param which creates a new object).
    import re as _re
    _SKIP_KEYS = {"name", "label"}          # creation params — don't rewrite
    for key, val in list(args.items()):
        if key in _SKIP_KEYS or not isinstance(val, str):
            continue
        if _re.fullmatch(r'.+_[12]', val) and not ggb.is_defined(val):
            base = val.rsplit("_", 1)[0]
            if ggb.is_defined(base):
                args[key] = base

    if tool_name == "add_point":
        return run(f"{args['name']} = ({args['x']}, {args['y']})")
    if tool_name == "add_slider":
        return run(f"{args['name']} = Slider({args['min']}, {args['max']}, {args['step']})")
    if tool_name == "set_value":
        name, val = args["name"], args["value"]
        ggb.eval_command(f"SetValue({name}, {val})")
        # read back the actual value — detect clamping by slider range
        actual = ggb.get_value(name)
        if actual is not None and abs(actual - val) > 0.01:
            msg = f"{name} = {actual} (requested {val}, clamped to slider range)"
        elif actual is not None:
            msg = f"{name} = {actual}"
        else:
            msg = f"{name} set to {val}"
        return (f"SetValue({name}, {val})", True, msg)
    if tool_name == "add_point_on":
        cmd, ok, err = run(f"{args['name']} = Point({args['obj']})")
        if ok and "x" in args and "y" in args:
            ggb.eval_command(f"SetCoords({args['name']}, {args['x']}, {args['y']})")
        return cmd, ok, err
    if tool_name == "add_intersect":
        # ── Validate: Intersect needs two curves/lines, not points ──
        for arg_key in ("obj1", "obj2"):
            obj = args[arg_key]
            otype = ggb.get_object_type(obj) if ggb.is_defined(obj) else None
            if otype == "point":
                return (f"{args['name']} = Intersect({args['obj1']}, {args['obj2']})",
                        False,
                        f"Cannot Intersect with a point ({obj} is a point). "
                        f"Use Point(circle) to place a point on a curve, or "
                        f"use the point directly if it is already at the desired location.")
        if "index" in args and args["index"] is not None:
            idx = int(args["index"])
            cmd, ok, err = run(f"{args['name']} = Intersect({args['obj1']}, {args['obj2']}, {idx})")
            if ok and not ggb.is_defined(args['name']):
                return cmd, False, f"Intersection point undefined (no intersection at index {idx})"
        else:
            cmd, ok, err = run(f"{args['name']} = Intersect({args['obj1']}, {args['obj2']})")
            if ok and not ggb.is_defined(args['name']):
                return cmd, False, "Intersection point undefined (objects do not intersect)"
        return cmd, ok, err
    if tool_name == "add_roots":
        return run(f"{args['name']} = Root({args['obj']})")
    if tool_name == "add_turning_point":
        return run(f"{args['name']} = TurningPoint({args['obj']})")
    if tool_name == "add_best_fit_line":
        return run(f"{args['name']} = FitLine({args['list_obj']})")
    if tool_name == "add_segment":
        return run_checked(
            f"{args['name']} = Segment({args['p1']}, {args['p2']})",
            args['name'], "Segment requires 2 distinct points")
    if tool_name == "add_line":
        return run_checked(
            f"{args['name']} = Line({args['p1']}, {args['p2']})",
            args['name'], "Line requires 2 distinct points")
    if tool_name == "add_ray":
        return run_checked(
            f"{args['name']} = Ray({args['p1']}, {args['p2']})",
            args['name'], "Ray requires 2 distinct points")
    if tool_name == "add_vector":
        return run(f"{args['name']} = Vector({args['p1']}, {args['p2']})")
    if tool_name == "add_polygon":
        pts = str(args["points"]).replace(",", ", ")
        return run(f"{args['name']} = Polygon({pts})")
    if tool_name == "add_regular_polygon":
        return run(f"{args['name']} = Polygon({args['p1']}, {args['p2']}, {int(args['n'])})")
    if tool_name == "add_vertex":
        return run(f"{args['name']} = Vertex({args['polygon']}, {int(args['index'])})")
    if tool_name == "add_midpoint":
        if "p2" in args and args["p2"] is not None:
            return run(f"{args['name']} = Midpoint({args['p1']}, {args['p2']})")
        else:
            # Midpoint(Conic) → center of circle/ellipse
            return run(f"{args['name']} = Midpoint({args['p1']})")
    if tool_name == "add_perpendicular_line":
        return run_checked(
            f"{args['name']} = PerpendicularLine({args['point']}, {args['line']})",
            args['name'],
            "Check that the point and line are both defined")
    if tool_name == "add_perpendicular_bisector":
        return run_checked(
            f"{args['name']} = PerpendicularBisector({args['p1']}, {args['p2']})",
            args['name'],
            "PerpendicularBisector requires 2 distinct points")
    if tool_name == "add_parallel_line":
        # GeoGebra has no standalone ParallelLine command;
        # Line(Point, Line) is the polymorphic "parallel" variant of Line().
        return run_checked(
            f"{args['name']} = Line({args['point']}, {args['line']})",
            args['name'], "Check that point and reference line are both defined")
    if tool_name == "add_angle_bisector":
        return run_checked(
            f"{args['name']} = AngleBisector({args['p1']}, {args['vertex']}, {args['p2']})",
            args['name'],
            "AngleBisector requires 3 distinct non-collinear points")
    if tool_name == "add_tangent":
        name = args['name']
        cmd = f"{name} = Tangent({args['point']}, {args['conic']})"
        cmd_str, ok, err = run(cmd)
        if not ok:
            return cmd_str, ok, err
        # GeoGebra auto-splits into name_{1}, name_{2} — detect and report
        sub1 = f"{name}_{{1}}"   # e.g. "tanD_{1}" in GeoGebra's LaTeX naming
        sub2 = f"{name}_{{2}}"
        if ggb.is_defined(name):
            return cmd_str, True, ""
        elif ggb.is_defined(sub1):
            names_found = [sub1]
            if ggb.is_defined(sub2):
                names_found.append(sub2)
            return cmd_str, True, (
                f"GeoGebra created {len(names_found)} tangent line(s): "
                + ", ".join(names_found)
                + f". Use these names (not '{name}') to reference them.")
        else:
            return cmd_str, False, (
                f"Object '{name}' is undefined after command (degenerate input?). "
                "Hint: Tangent requires an external point; point may be inside the conic")
    if tool_name == "add_tangent_conic_conic":
        name = args['name']
        cmd = f"{name} = Tangent({args['conic1']}, {args['conic2']})"
        cmd_str, ok, err = run(cmd)
        if not ok:
            return cmd_str, ok, err
        # GeoGebra auto-splits into name_{1} .. name_{4}
        names_found = []
        for i in range(1, 5):
            sub = f"{name}_{{{i}}}"
            if ggb.is_defined(sub):
                names_found.append(sub)
        if ggb.is_defined(name):
            return cmd_str, True, ""
        elif names_found:
            return cmd_str, True, (
                f"GeoGebra created {len(names_found)} common tangent line(s): "
                + ", ".join(names_found)
                + f". Use these names (not '{name}') to reference them.")
        else:
            return cmd_str, False, (
                "No tangent lines created — circles may be concentric, "
                "identical, or one inside the other.")
    if tool_name == "add_circle":
        if "point" in args and args["point"] is not None:
            return run(f"{args['name']} = Circle({args['center']}, {args['point']})")
        else:
            return run(f"{args['name']} = Circle({args['center']}, {args['radius']})")
    if tool_name == "add_arc":
        return run_checked(
            f"{args['name']} = CircularArc({args['center']}, {args['start_pt']}, {args['end_pt']})",
            args['name'], "Arc requires distinct center, start, and end points")
    if tool_name == "add_sector":
        return run_checked(
            f"{args['name']} = CircularSector({args['center']}, {args['start_pt']}, {args['end_pt']})",
            args['name'], "Sector requires distinct center, start, and end points")
    if tool_name == "add_semicircle":
        name = args['name']
        p1n, p2n = args['p1'], args['p2']
        cmd_str, ok, err = run_checked(
            f"{name} = Semicircle({p1n}, {p2n})",
            name, "Semicircle requires 2 distinct endpoints")
        if ok:
            # Probe a point on the arc to report direction
            probe = f"smpb{name}"
            try:
                ggb.eval_command(f"{probe} = Point({name}, 0.5)")
                pc = ggb.get_coords(probe)
                p1c = ggb.get_coords(p1n)
                p2c = ggb.get_coords(p2n)
                ggb.delete_object(probe)
                if pc and p1c and p2c:
                    mx = (p1c[0] + p2c[0]) / 2
                    my = (p1c[1] + p2c[1]) / 2
                    dirs = []
                    if abs(pc[1] - my) > 0.01:
                        dirs.append("ABOVE" if pc[1] > my else "BELOW")
                    if abs(pc[0] - mx) > 0.01:
                        dirs.append("RIGHT" if pc[0] > mx else "LEFT")
                    if dirs:
                        err = (f"Arc bulges {' and '.join(dirs)} of diameter midpoint "
                               f"({mx:.1f},{my:.1f}). To flip, swap p1 and p2.")
            except Exception:
                pass  # probe failed — skip feedback, tool still succeeded
        return cmd_str, ok, err
    if tool_name == "add_circle_3_points":
        return run_checked(
            f"{args['name']} = Circle({args['a']}, {args['b']}, {args['c']})",
            args['name'], "Circle(3pts) requires 3 distinct non-collinear points")
    if tool_name == "add_incircle":
        return run_checked(
            f"{args['name']} = Incircle({args['a']}, {args['b']}, {args['c']})",
            args['name'], "Incircle requires 3 distinct non-collinear points")
    if tool_name == "add_center":
        return run(f"{args['name']} = Center({args['conic']})")
    if tool_name == "add_triangle_center":
        return run(f"{args['name']} = TriangleCenter({args['a']}, {args['b']}, {args['c']}, {int(args['n'])})")
    if tool_name == "add_ellipse":
        return run_checked(
            f"{args['name']} = Ellipse({args['f1']}, {args['f2']}, {args['p']})",
            args['name'], "Ellipse requires 2 distinct foci and a point on the curve")
    if tool_name == "add_parabola":
        return run_checked(
            f"{args['name']} = Parabola({args['focus']}, {args['directrix']})",
            args['name'], "Parabola requires a focus point and a directrix line")
    if tool_name == "add_hyperbola":
        return run_checked(
            f"{args['name']} = Hyperbola({args['f1']}, {args['f2']}, {args['p']})",
            args['name'], "Hyperbola requires 2 distinct foci and a point on the curve")
    if tool_name == "add_angle":
        import math
        p1, vtx, p2 = args['p1'], args['vertex'], args['p2']
        cmd, ok, err = run(f"{args['name']} = Angle({p1}, {vtx}, {p2})")
        if ok:
            value_rad = ggb.get_value(args['name'])
            if value_rad is not None and math.degrees(value_rad) > 180.0:
                ggb.eval_command(f"{args['name']} = Angle({p2}, {vtx}, {p1})")
        return cmd, ok, err
    if tool_name == "add_distance":
        return run(f"{args['name']} = Distance({args['p1']}, {args['p2']})")
    if tool_name == "add_area":
        return run(f"{args['name']} = Area({args['obj']})")
    if tool_name == "add_slope":
        return run(f"{args['name']} = Slope({args['line']})")
    if tool_name == "transform_reflect_line":
        return run_checked(
            f"{args['name']} = Reflect({args['obj']}, {args['line']})",
            args['name'], "Check that both object and mirror line are defined")
    if tool_name == "transform_reflect_point":
        return run_checked(
            f"{args['name']} = Reflect({args['obj']}, {args['point']})",
            args['name'], "Check that both object and center point are defined")
    if tool_name == "transform_rotate":
        return run_checked(
            f"{args['name']} = Rotate({args['obj']}, {args['angle']}, {args['center']})",
            args['name'], "Check that object, angle expression, and center are valid")
    if tool_name == "transform_translate":
        return run_checked(
            f"{args['name']} = Translate({args['obj']}, {args['vector']})",
            args['name'], "Check that both object and vector are defined")
    if tool_name == "transform_dilate":
        return run_checked(
            f"{args['name']} = Dilate({args['obj']}, {args['factor']}, {args['center']})",
            args['name'], "Check that object, factor, and center are valid")
    # ── Analytic / function construction tools ──────────────────────────
    if tool_name == "add_function":
        name = args['name']
        # Strip accidental "(x)" from name — common LLM error
        if name.endswith("(x)"):
            name = name[:-3]
        expr = args['expr']
        if args.get('start_x') is not None and args.get('end_x') is not None:
            cmd = f"{name} = Function({expr}, {args['start_x']}, {args['end_x']})"
        else:
            cmd = f"{name}(x) = {expr}"
        return run(cmd)

    # ── Calculus / Analysis execution ─────────────────────────────────────
    if tool_name == "add_derivative":
        order = args.get("order", 1) or 1
        fn = args["function"]
        if order == 1:
            return run(f"{args['name']}(x) = Derivative({fn})")
        else:
            return run(f"{args['name']}(x) = Derivative({fn}, {order})")
    if tool_name == "add_integral_function":
        return run(f"{args['name']}(x) = Integral({args['function']})")
    if tool_name == "query_definite_integral":
        fn = args["function"]
        a, b = args["start"], args["end"]
        temp = f"_qdi_{fn}_{a}_{b}".replace(".", "p").replace("-", "m")
        cmd, ok, err = run(f"{temp} = Integral({fn}, {a}, {b})")
        if ok:
            val = ggb.get_value(temp)
            return (f"Integral({fn}, {a}, {b})", True,
                    f"{val}" if val is not None else "computed (check canvas)")
        return cmd, ok, err
    if tool_name == "add_inflection_point":
        return run(f"{args['name']} = InflectionPoint({args['function']})")
    if tool_name == "add_asymptote":
        return run(f"{args['name']} = Asymptote({args['function']})")
    if tool_name == "query_function_max":
        fn, a, b = args["function"], args["start"], args["end"]
        temp = f"_qmax_{fn}"
        cmd, ok, err = run(f"{temp} = Max({fn}, {a}, {b})")
        if ok:
            x_val = ggb.get_value(f"x({temp})")
            y_val = ggb.get_value(f"y({temp})")
            return (f"Max({fn}, {a}, {b})", True,
                    f"({x_val}, {y_val})" if x_val is not None else "point found")
        return cmd, ok, err
    if tool_name == "query_function_min":
        fn, a, b = args["function"], args["start"], args["end"]
        temp = f"_qmin_{fn}"
        cmd, ok, err = run(f"{temp} = Min({fn}, {a}, {b})")
        if ok:
            x_val = ggb.get_value(f"x({temp})")
            y_val = ggb.get_value(f"y({temp})")
            return (f"Min({fn}, {a}, {b})", True,
                    f"({x_val}, {y_val})" if x_val is not None else "point found")
        return cmd, ok, err

    if tool_name == "add_curve":
        param = args.get('param') or 't'
        if param in ('x', 'y', 'z'):
            return (f"Curve(..., {param}, ...)", False,
                    f"Parameter variable '{param}' is reserved — use 't' or 's' instead")
        return run(
            f"{args['name']} = Curve({args['x_expr']}, {args['y_expr']}, "
            f"{param}, {args['t_start']}, {args['t_end']})")
    if tool_name == "add_inequality":
        return run(f"{args['name']}: {args['expr']}")
    if tool_name == "add_integral_shade":
        if args.get('func2'):
            cmd = (f"{args['name']} = IntegralBetween({args['func']}, "
                   f"{args['func2']}, {args['x_start']}, {args['x_end']})")
        else:
            cmd = f"{args['name']} = Integral({args['func']}, {args['x_start']}, {args['x_end']})"
        return run(cmd)
    if tool_name == "add_text":
        import re
        text = args['text']
        # Recover LaTeX commands whose backslash was eaten by JSON escape
        # (\t→tab, \f→formfeed, \b→backspace consumed at JSON decode layer)
        text = re.sub(r'\text\{',  r'\\text{',  text)
        text = re.sub(r'\frac\{',  r'\\frac{',  text)
        text = re.sub(r'(?<![a-zA-Z])rac\{', r'\\frac{', text)
        text = re.sub(r'(?<![a-zA-Z])ext\{',  r'\\text{', text)
        text = re.sub(r'(?<![a-zA-Z])egin\{', r'\\begin{', text)
        text = text.replace('\f', '\\f').replace('\t', '\\t').replace('\b', '\\b')
        # Fix multi-char subscripts/superscripts without braces:
        #   _sol → _{sol},  ^abc → ^{abc}
        # Only applies when _ or ^ is followed by 2+ alphanumeric chars without {
        text = re.sub(r'_([A-Za-z0-9]{2,})(?![{])', r'_{\1}', text)
        text = re.sub(r'\^([A-Za-z0-9]{2,})(?![{])', r'^{\1}', text)
        # Escape for GeoGebra command parser
        text = text.replace('\\', '\\\\').replace('"', '\\"')
        # Force LaTeX mode for proper formula rendering
        cmd = f'{args["name"]} = Text("{text}", ({args["x"]}, {args["y"]}), false, true)'
        return run(cmd)
    # ── Utility ──────────────────────────────────────────────────────────
    if tool_name == "rename_object":
        old = args['name']
        new = args['new_name']
        if not ggb.is_defined(old):
            return f"Rename({old}→{new})", False, f"Object '{old}' does not exist"
        if ggb.is_defined(new):
            return f"Rename({old}→{new})", False, (
                f"Cannot rename: '{new}' already exists on canvas. "
                f"Choose a different name or delete '{new}' first.")
        ok = ggb.rename_object(old, new)
        return f"Rename({old}→{new})", ok, "" if ok else f"GeoGebra refused rename '{old}'→'{new}'"
    if tool_name == "set_label_visible":
        ok = ggb.set_label_visible(args["name"], bool(int(args["visible"])))
        return f"SetLabelVisible({args['name']}, {bool(int(args['visible']))})", ok, "" if ok else "failed"
    if tool_name == "set_object_visible":
        ok = ggb.set_object_visible(args["name"], bool(int(args["visible"])))
        return f"SetVisible({args['name']}, {bool(int(args['visible']))})", ok, "" if ok else "failed"
    if tool_name == "delete_object":
        name = args["name"]
        if ggb.is_defined(name):
            ggb.delete_object(name)
            return f"Delete({name})", True, ""
        else:
            return f"Delete({name})", False, f"Object '{name}' not found"
    # ── Render / styling tools ───────────────────────────────────────────
    if tool_name == "render_set_color":
        obj, color = args['obj'], args['color']
        # Map English color names to RGB (GeoGebra evalCommand returns false for SetColor scripting)
        _COLOR_MAP = {
            "black": (0,0,0), "red": (255,0,0), "blue": (0,0,255), "green": (0,128,0),
            "orange": (255,165,0), "purple": (128,0,128), "cyan": (0,255,255),
            "gray": (128,128,128), "brown": (139,69,19), "magenta": (255,0,255),
            "maroon": (128,0,0), "gold": (255,215,0), "pink": (255,192,203),
            "yellow": (255,255,0), "white": (255,255,255), "dark blue": (0,0,139),
            "dark green": (0,100,0), "light gray": (192,192,192), "dark gray": (64,64,64),
            "indigo": (75,0,130), "violet": (238,130,238), "crimson": (220,20,60),
            "lime": (0,255,0), "turquoise": (64,224,208), "aqua": (0,255,255),
            "silver": (192,192,192), "light blue": (173,216,230),
        }
        rgb = _COLOR_MAP.get(color.lower())
        if rgb:
            ok = ggb.set_color(obj, *rgb)
        else:
            # Fallback: try eval_command for unknown color names
            result = ggb.eval_command(f'SetColor({obj}, "{color}")')
            ok = bool(result.success)
        return f"SetColor({obj}, \"{color}\")", ok, "" if ok else f"Unknown color '{color}'"
    if tool_name == "render_set_line_style":
        ok = ggb.set_line_style(args['obj'], int(args['style']))
        return f"SetLineStyle({args['obj']}, {args['style']})", ok, "" if ok else "failed"
    if tool_name == "render_set_line_thickness":
        ok = ggb.set_line_thickness(args['obj'], int(args['thickness']))
        return f"SetLineThickness({args['obj']}, {args['thickness']})", ok, "" if ok else "failed"
    if tool_name == "render_set_point_style":
        ok = ggb.set_point_style(args['obj'], int(args['style']))
        return f"SetPointStyle({args['obj']}, {args['style']})", ok, "" if ok else "failed"
    if tool_name == "render_set_point_size":
        ok = ggb.set_point_size(args['obj'], int(args['size']))
        return f"SetPointSize({args['obj']}, {args['size']})", ok, "" if ok else "failed"
    if tool_name == "render_set_filling":
        ok = ggb.set_filling(args['obj'], float(args['opacity']))
        return f"SetFilling({args['obj']}, {args['opacity']})", ok, "" if ok else "failed"
    if tool_name == "render_set_decoration":
        obj, dec = args['obj'], int(args['decoration'])
        if not ggb.is_defined(obj):
            return f"SetDecoration({obj}, {dec})", False, f"Object '{obj}' does not exist"
        # SetDecoration is a scripting command — evalCommand returns false
        # even on success. Use evalCommand without checking return value;
        # verify success by checking the object still exists (no crash).
        cmd = f"SetDecoration({obj}, {dec})"
        try:
            ggb.eval_command(cmd)
            ok = ggb.is_defined(obj)  # sanity check
        except Exception as e:
            return cmd, False, str(e)
        return cmd, ok, "" if ok else "SetDecoration may have failed"
    if tool_name == "render_show_axes":
        v = bool(int(args['visible']))
        ok = ggb.set_axes_visible(v, v)
        return f"ShowAxes({v})", ok, "" if ok else "failed"
    if tool_name == "render_show_grid":
        v = bool(int(args['visible']))
        ok = ggb.set_grid_visible(v)
        return f"ShowGrid({v})", ok, "" if ok else "failed"
    if tool_name == "render_set_caption":
        ok1 = ggb.set_caption(args['obj'], args['caption'])
        ok2 = ggb.set_label_style(args['obj'], 3)  # switch to Caption mode
        ok = ok1 and ok2
        return f"SetCaption({args['obj']}, \"{args['caption']}\")", ok, "" if ok else "failed"
    if tool_name == "render_set_label_mode":
        ok = ggb.set_label_style(args['obj'], int(args['mode']))
        # Auto-show label — setting mode without visibility is useless
        if ok:
            ggb.set_label_visible(args['obj'], True)
        return f"SetLabelMode({args['obj']}, {args['mode']})", ok, "" if ok else "failed"
    if tool_name == "render_set_coord_system":
        ok = ggb.set_coord_system(
            float(args['x_min']), float(args['x_max']),
            float(args['y_min']), float(args['y_max']))
        return (f"SetCoordSystem({args['x_min']}, {args['x_max']}, "
                f"{args['y_min']}, {args['y_max']})"), ok, "" if ok else "failed"
    if tool_name == "render_add_right_angle_mark":
        import math
        a, b, c = args['a'], args['b'], args['c']
        # Verify 90° precondition using a temporary object
        # GeoGebra rejects names starting with underscore, so use 'chkra' prefix
        chk = f"chkra{args['name']}"
        ggb.eval_command(f"{chk} = Angle({a}, {b}, {c})")
        value_rad = ggb.get_value(chk)
        ggb.delete_object(chk)
        swap = False
        if value_rad is not None:
            deg = math.degrees(value_rad)
            if deg > 180:
                # GeoGebra measures counter-clockwise; swap a,c to get the
                # acute (90°) angle instead of the reflex (270°) one.
                swap = True
                deg = 360 - deg
            if abs(deg - 90) > 2.0:
                return (f"Angle({a},{b},{c})", False,
                        f"Angle is {deg:.1f}°, not 90°. Right-angle mark requires a 90° angle.")
        elif value_rad is None:
            # Fallback: use ArePerpendicular if angle check failed
            pass
        # Create the Angle object — GeoGebra auto-renders □ for 90°
        if swap:
            a, c = c, a
        cmd, ok, err = run(f"{args['name']} = Angle({a}, {b}, {c})")
        if ok:
            # Hide the "90°" value label — the □ mark is sufficient
            ggb.set_label_visible(args['name'], False)
        return cmd, ok, err

    if tool_name.startswith("query_"):
        cmd, ok, err, _value = execute_query_tool(ggb, tool_name, args)
        return cmd, ok, err

    return f"[unknown tool: {tool_name}]", False, f"Tool '{tool_name}' not implemented"


def _solve_via_root(ggb, name: str, equation: str, variable: str) -> Tuple[str, bool, str, Any]:
    """Fallback solver: convert equation to f(x)=0 form and use Root()."""
    import re
    # GeoGebra disallows names starting with underscore
    fname = f"rf{name}"
    eq = equation.strip()
    if "=" in eq:
        lhs, rhs = eq.split("=", 1)
        expr = f"{lhs.strip()} - ({rhs.strip()})"
    else:
        expr = eq

    # ── Check: is the variable actually present in the expression? ──
    # If not, this is a constant expression — evaluate directly, no Root needed.
    # Use word-boundary match to avoid false positives (e.g. "ax" matching "a").
    if not re.search(rf'\b{re.escape(variable)}\b', expr):
        # Constant expression: just evaluate it
        cmd_eval = f"{name} = {expr}"
        r = ggb.eval_command(cmd_eval)
        if r.success:
            val = ggb._call_api("getValue", name) if hasattr(ggb, '_call_api') else None
            if val is None:
                try:
                    val = ggb._driver.execute_script(
                        f'return ggbApplet.getValue("{name}")')
                except Exception:
                    pass
            display = f"{name} = {val}" if val is not None else ggb.get_value_string(name)
            return f"[const eval] {cmd_eval}", True, "", display
        # If direct eval failed, fall through to Root approach

    # ── Use a safe parameter name to avoid GeoGebra slider/reserved clashes ──
    # Single letters (a-f, x, y, z) can conflict with GeoGebra sliders.
    safe_var = "tvar"
    safe_expr = re.sub(rf'\b{re.escape(variable)}\b', safe_var, expr)

    cmd_f = f"{fname}({safe_var}) = {safe_expr}"
    r = ggb.eval_command(cmd_f)
    if not r.success:
        return cmd_f, False, f"Root fallback: cannot define function — {r.error_message}", None
    cmd_r = f"{name} = Root({fname}, -1000, 1000)"
    r2 = ggb.eval_command(cmd_r)
    if not r2.success:
        return cmd_r, False, f"Root fallback: Root() failed — {r2.error_message}", None
    vs = ggb.get_value_string(name)
    # Root returns a point (x, 0); extract x-coordinate as the solution value
    x = ggb._call_api("getXcoord", name) if hasattr(ggb, '_call_api') else None
    if x is None:
        try:
            x = ggb._driver.execute_script(f'return ggbApplet.getXcoord("{name}")')
        except Exception:
            pass
    display = f"{name} = {x}" if x is not None else vs
    return f"[Root fallback] {cmd_r}", True, "", display


def execute_query_tool(ggb, tool_name: str, args: Dict[str, Any]) -> Tuple[str, bool, str, Any]:
    """
    Execute a query tool, returning (command, success, error, value).

    Boolean predicates return 1.0 (true) / 0.0 (false) via GeoGebra getValue().
    Scalar queries return a float.
    """
    # ── Auto-correct _N suffix hallucination (same as execute_geogebra_tool) ──
    import re as _re
    _SKIP_KEYS = {"name", "label"}
    for key, val in list(args.items()):
        if key in _SKIP_KEYS or not isinstance(val, str):
            continue
        if _re.fullmatch(r'.+_[12]', val) and not ggb.is_defined(val):
            base = val.rsplit("_", 1)[0]
            if ggb.is_defined(base):
                args[key] = base

    def run_query(cmd: str) -> Tuple[str, bool, str, Any]:
        result = ggb.eval_command(cmd)
        if not result.success:
            return cmd, False, result.error_message or "", None
        value = ggb.get_value(args["name"])
        return cmd, True, "", value

    def run_query_str(cmd: str) -> Tuple[str, bool, str, Any]:
        """Like run_query but returns getValueString() for list/equation results (e.g. Solve)."""
        result = ggb.eval_command(cmd)
        if not result.success:
            return cmd, False, result.error_message or "", None
        value = ggb.get_value_string(args["name"])
        if value and value.rstrip().endswith("= ?"):
            return cmd, False, "CAS not ready: Solve returned '?'", value
        return cmd, True, "", value

    if tool_name == "query_are_parallel":
        return run_query(f"{args['name']} = AreParallel({args['obj1']}, {args['obj2']})")
    if tool_name == "query_are_perpendicular":
        return run_query(f"{args['name']} = ArePerpendicular({args['obj1']}, {args['obj2']})")
    if tool_name == "query_is_tangent":
        return run_query(f"{args['name']} = IsTangent({args['line']}, {args['conic']})")
    if tool_name == "query_is_in_region":
        return run_query(f"{args['name']} = IsInRegion({args['point']}, {args['region']})")
    if tool_name == "query_are_equal":
        return run_query(f"{args['name']} = AreEqual({args['obj1']}, {args['obj2']})")
    if tool_name == "query_are_collinear":
        return run_query(f"{args['name']} = AreCollinear({args['a']}, {args['b']}, {args['c']})")
    if tool_name == "query_are_concyclic":
        return run_query(
            f"{args['name']} = AreConcyclic({args['a']}, {args['b']}, {args['c']}, {args['d']})"
        )
    if tool_name == "query_are_congruent":
        return run_query(f"{args['name']} = AreCongruent({args['obj1']}, {args['obj2']})")
    if tool_name == "query_length":
        return run_query(f"{args['name']} = Length({args['obj']})")
    if tool_name == "query_perimeter":
        return run_query(f"{args['name']} = Perimeter({args['obj']})")
    if tool_name == "query_angle":
        import math
        a, b, c = args['a'], args['b'], args['c']
        cmd_str = f"{args['name']} = Angle({a}, {b}, {c})"
        result = ggb.eval_command(cmd_str)
        if not result.success:
            return cmd_str, False, result.error_message or "", None
        value_rad = ggb.get_value(args["name"])
        if value_rad is None:
            return cmd_str, True, "", None
        value_deg = math.degrees(value_rad)
        # GeoGebra Angle(A, Apex, C) measures CCW in [0°, 360°].
        # If reflex (>180°), swap A↔C so the canvas arc shows the interior angle.
        if value_deg > 180.0:
            value_deg = 360.0 - value_deg
            swap_cmd = f"{args['name']} = Angle({c}, {b}, {a})"
            ggb.eval_command(swap_cmd)
            cmd_str = swap_cmd
        # value_deg = round(value_deg, 6)  # [OPTION] round to 6dp for cleaner feedback
        # Return full engine precision — Engine Faithfulness guarantee
        return cmd_str, True, "", value_deg
    if tool_name == "query_area":
        return run_query(f"{args['name']} = Area({args['obj']})")
    if tool_name == "query_distance":
        # LLM sometimes uses p1/p2 instead of obj1/obj2
        o1 = args.get('obj1') or args.get('p1', '')
        o2 = args.get('obj2') or args.get('p2', '')
        return run_query(f"{args['name']} = Distance({o1}, {o2})")
    if tool_name == "query_slope":
        return run_query(f"{args['name']} = Slope({args['obj']})")
    if tool_name == "query_x_coord":
        return run_query(f"{args['name']} = x({args['pt']})")
    if tool_name == "query_y_coord":
        return run_query(f"{args['name']} = y({args['pt']})")
    if tool_name == "query_radius":
        return run_query(f"{args['name']} = Radius({args['conic']})")
    if tool_name == "query_is_defined":
        return run_query(f"{args['name']} = IsDefined({args['obj']})")
    if tool_name == "query_dependents":
        deps = ggb.get_all_dependents(args["obj"])
        dep_str = ", ".join(deps) if deps else "(none)"
        cmd_str = f"[dependents of {args['obj']}]"
        return cmd_str, True, "", dep_str
    if tool_name == "query_solve":
        cmd = f"{args['name']} = Solve({args['equation']}, {args['variable']})"
        result = run_query_str(cmd)
        if result[1]:  # success
            return result
        # Fallback: CAS unavailable — use Root() numeric solver
        return _solve_via_root(ggb, args["name"], args["equation"], args["variable"])
    if tool_name == "query_nsolve":
        cmd = f"{args['name']} = NSolve({args['equation']}, {args['variable']})"
        result = run_query_str(cmd)
        if result[1]:
            return result
        return _solve_via_root(ggb, args["name"], args["equation"], args["variable"])

    # ── Calculus queries ──────────────────────────────────────────────────
    if tool_name == "query_definite_integral":
        fn = args["function"]
        a, b = args["start"], args["end"]
        temp = f"qdi_{fn}".replace(".", "p").replace("-", "m")
        cmd = f"{temp} = Integral({fn}, {a}, {b})"
        res = ggb.eval_command(cmd)
        if not res.success:
            return cmd, False, res.error_message or "", None
        val = ggb.get_value(temp)
        return cmd, True, "", val

    if tool_name == "query_function_max":
        fn, a, b = args["function"], args["start"], args["end"]
        temp = f"qmax_{fn}"
        cmd = f"{temp} = Max({fn}, {a}, {b})"
        res = ggb.eval_command(cmd)
        if not res.success:
            return cmd, False, res.error_message or "", None
        x_val = ggb.get_value(f"x({temp})")
        y_val = ggb.get_value(f"y({temp})")
        return cmd, True, "", (x_val, y_val)

    if tool_name == "query_function_min":
        fn, a, b = args["function"], args["start"], args["end"]
        temp = f"qmin_{fn}"
        cmd = f"{temp} = Min({fn}, {a}, {b})"
        res = ggb.eval_command(cmd)
        if not res.success:
            return cmd, False, res.error_message or "", None
        x_val = ggb.get_value(f"x({temp})")
        y_val = ggb.get_value(f"y({temp})")
        return cmd, True, "", (x_val, y_val)

    return f"[unknown query tool: {tool_name}]", False, f"Tool '{tool_name}' not implemented", None


# ── Canvas state tracker ─────────────────────────────────────────────────────

def build_rich_canvas(ggb) -> dict:
    """Return a compact dict of all objects currently on the GeoGebra canvas.

    For numeric objects (distances, angles, areas, etc.), the ``val`` field
    uses the full-precision value from ``get_value()`` instead of GeoGebra's
    display-rounded ``value_string``, ensuring the model always sees IEEE 754
    precision in the canvas context.
    """
    objs = ggb.get_construction_state().get("objects", {})
    out: dict = {}
    for name, info in objs.items():
        t = info.get("type", "?")
        entry: dict = {"type": t}
        if t == "point":
            if "x" in info:
                entry["x"] = round(info["x"], 6)
            if "y" in info:
                entry["y"] = round(info["y"], 6)
        if info.get("value_string"):
            vs = info["value_string"]
            # For numeric types, use full precision from get_value()
            if t in ("numeric", "angle"):
                try:
                    full_val = ggb.get_value(name)
                    if full_val is not None:
                        vs = f"{name} = {full_val}"
                except Exception:
                    pass
            entry["val"] = vs
        out[name] = entry
    return out


def _build_display_state(ggb) -> dict:
    """Build a compact display-state dict for render feedback.

    Returns ``{obj_name: {"type": ..., "label_visible": bool, "color": ...}, ...}``
    so the LLM can see which objects have visible labels and current styling.
    """
    objs = ggb.get_construction_state().get("objects", {})
    out: dict = {}
    for name, info in objs.items():
        t = info.get("type", "?")
        entry: dict = {"type": t}
        # Label visibility — GeoGebra JS API
        try:
            lv = ggb._execute_js(f'return ggbApplet.getLabelVisible("{name}")')
            entry["label_visible"] = bool(lv)
        except Exception:
            pass
        # Color
        try:
            c = ggb.get_color(name)
            if c:
                entry["color"] = c
        except Exception:
            pass
        out[name] = entry
    return out


class CanvasTracker:
    """Track canvas state across tool calls.

    Wraps ``execute_geogebra_tool`` / ``execute_query_tool`` with delta
    computation so callers don't need to manage the ``known`` set themselves.

    Usage::

        tracker = CanvasTracker()
        result, log_entry = tracker.execute(ggb, fn_name, args)
        # result  — dict to send back as tool response to the LLM
        # log_entry — dict for the process log (fn, cmd, ok, new_objects/value/canvas/error)
    """

    def __init__(self) -> None:
        self._known: set[str] = set()
        self.ok_n: int = 0
        self.fail_n: int = 0
        self.total_n: int = 0
        self._llm_viewport: tuple[float, float, float, float] | None = None

    # ------------------------------------------------------------------

    def execute(self, ggb, fn_name: str, args: Dict[str, Any]) -> Tuple[dict, dict]:
        """Execute a tool call and return (tool_response, log_entry).

        * For ``query_*`` tools: tool_response includes ``value`` + full ``canvas``.
        * For ``add_*/delete_*`` tools: tool_response includes ``new_objects`` (delta).
        """
        self.total_n += 1

        # ── Parameter normalization (Robustness Principle) ────────────
        # LLM confuses param names across tools (p1/obj1/point1 for the
        # same semantic role). Accept common aliases → map to canonical.
        # Long-term fix: unify param naming across all ToolSpecs.
        _ALIASES = {
            "query_distance": {"p1": "obj1", "p2": "obj2",
                               "point1": "obj1", "point2": "obj2",
                               "object1": "obj1", "object2": "obj2"},
            "query_angle":    {"p1": "a", "p2": "b", "p3": "c",
                               "point1": "a", "point2": "b", "point3": "c",
                               "vertex": "b"},
            "query_area":     {"polygon": "obj", "circle": "obj",
                               "object": "obj", "region": "obj"},
            "query_length":   {"segment": "obj", "arc": "obj",
                               "object": "obj"},
            "query_perimeter": {"polygon": "obj", "circle": "obj",
                                "object": "obj"},
        }
        if fn_name in _ALIASES:
            for alias, canonical in _ALIASES[fn_name].items():
                if alias in args and canonical not in args:
                    args[canonical] = args.pop(alias)

        # ── Validate required params before execution ─────────────────
        _all_specs = {s.name: s for s in GLOBAL_GEOGEBRA_TOOLS + QUERY_GEOGEBRA_TOOLS + RENDER_GEOGEBRA_TOOLS}
        spec = _all_specs.get(fn_name)
        if spec:
            missing = [k for k, p in spec.params.items() if p.required and k not in args]
            if missing:
                self.fail_n += 1
                err = (f"Missing required parameter(s): {missing}. "
                       f"Expected: {list(spec.params.keys())}")
                result = {"command": f"{fn_name}({args})", "success": False, "error": err}
                log = {"fn": fn_name, "cmd": f"{fn_name}({args})", "ok": False,
                       "error": err, "value": None}
                return result, log

        if fn_name.startswith("query_"):
            return self._exec_query(ggb, fn_name, args)
        return self._exec_mutate(ggb, fn_name, args)

    # ------------------------------------------------------------------

    def _exec_query(self, ggb, fn_name: str, args: Dict[str, Any]) -> Tuple[dict, dict]:
        cmd, ok, err, val = execute_query_tool(ggb, fn_name, args)
        if ok:
            self.ok_n += 1
            canvas = build_rich_canvas(ggb)
            self._known.update(canvas)
            result = {"command": cmd, "success": True, "error": "",
                      "value": val, "canvas": canvas}
            log = {"fn": fn_name, "cmd": cmd, "ok": True,
                   "value": val, "canvas": canvas}
        else:
            self.fail_n += 1
            result = {"command": cmd, "success": False, "error": err}
            log = {"fn": fn_name, "cmd": cmd, "ok": False,
                   "error": err, "value": val}
        return result, log

    def _exec_mutate(self, ggb, fn_name: str, args: Dict[str, Any]) -> Tuple[dict, dict]:
        cmd, ok, err = execute_geogebra_tool(ggb, fn_name, args)
        if ok and fn_name == "render_set_coord_system":
            try:
                self._llm_viewport = (
                    float(args['x_min']), float(args['x_max']),
                    float(args['y_min']), float(args['y_max']))
            except (KeyError, ValueError, TypeError):
                pass
        if ok:
            self.ok_n += 1
            fc_ = build_rich_canvas(ggb)
            removed = sorted(self._known - set(fc_))  # cascade-deleted objects
            self._known &= set(fc_)               # sync: drop deleted objects
            new = {n: v for n, v in fc_.items() if n not in self._known}
            self._known.update(fc_)

            # Classify by behaviour, not name prefix:
            #   display tools: modify visual properties, don't create objects
            #   construction tools: create/delete geometric objects
            _DISPLAY_TOOLS = {"set_label_visible", "set_object_visible", "rename_object"}
            is_display = fn_name.startswith("render_") or fn_name in _DISPLAY_TOOLS

            if is_display:
                display = _build_display_state(ggb)
                # LLM result: lightweight confirmation only (save tokens)
                result = {"success": True,
                          "applied": f"{fn_name}({args})"}
                # Process log: keep full display for debugging
                log = {"fn": fn_name, "cmd": cmd, "ok": True,
                       "applied": f"{fn_name}({args})",
                       "display": display}
            else:
                # LLM result: incremental + full canvas snapshot
                result = {"command": cmd, "success": True, "error": "",
                          "new_objects": new, "removed_objects": removed,
                          "canvas": fc_}
                # Process log: same
                log = {"fn": fn_name, "cmd": cmd, "ok": True,
                       "new_objects": new, "removed_objects": removed,
                       "canvas": fc_}
        else:
            self.fail_n += 1
            result = {"command": cmd, "success": False, "error": err}
            log = {"fn": fn_name, "cmd": cmd, "ok": False, "error": err}
        return result, log
