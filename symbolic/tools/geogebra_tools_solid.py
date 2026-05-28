"""
GeoGebra 3D (Solid Geometry) tool catalog + execution logic.

Extends the 2D tool set (geogebra_tools.py) with 3D construction, query,
and rendering tools.  Uses the SAME ToolSpec / CanvasTracker infrastructure.

Command syntax sourced from GeoGebra manual (fetch via: python setup.py download_manual):
  docs/geogebra-manual/commands/
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from symbolic.tools.geogebra_tools import (
    ToolParam,
    ToolSpec,
    build_rich_canvas,        # reuse
)

# ═══════════════════════════════════════════════════════════════════════════════
#  3D Construction ToolSpecs
#  Syntax verified against GeoGebra manual (not hallucinated).
# ═══════════════════════════════════════════════════════════════════════════════

SOLID_GEOGEBRA_TOOLS: List[ToolSpec] = [

    # ── Points & Vectors ────────────────────────────────────────────────────

    ToolSpec("add_point3d",
             "Place a free point at 3D coordinates (x, y, z).",
             {
                 "name": ToolParam("string", "Point name, e.g. A"),
                 "x":    ToolParam("number", "X coordinate"),
                 "y":    ToolParam("number", "Y coordinate"),
                 "z":    ToolParam("number", "Z coordinate"),
             }),

    ToolSpec("add_vector3d",
             "Create a 3D vector from origin to (x, y, z), or between two existing points.",
             {
                 "name": ToolParam("string", "Vector name"),
                 "x":    ToolParam("number", "X component (or leave 0 if using from/to)", required=False),
                 "y":    ToolParam("number", "Y component", required=False),
                 "z":    ToolParam("number", "Z component", required=False),
                 "from_pt": ToolParam("string", "Start point name (optional)", required=False),
                 "to_pt":   ToolParam("string", "End point name (optional)", required=False),
             }),

    # ── Planes ──────────────────────────────────────────────────────────────

    ToolSpec("add_plane",
             "Create an INFINITE mathematical plane — extends in all directions. "
             "Use this for geometric computation: cross-sections (IntersectPath), "
             "perpendicularity checks, angle calculations. "
             "NOT suitable for diagrams/illustrations where you want a bounded region "
             "— use add_finite_plane instead for visual clarity. "
             "Syntax: Plane(A, B, C). Also supports Plane(Polygon) or Plane(Point, Line).",
             {
                 "name": ToolParam("string", "Plane name"),
                 "a":    ToolParam("string", "First point or polygon"),
                 "b":    ToolParam("string", "Second point or line (optional if a is polygon)", required=False),
                 "c":    ToolParam("string", "Third point (optional)", required=False),
             }),

    ToolSpec("add_finite_plane",
             "Create a BOUNDED rectangular plane for visual illustration. "
             "Draws a parallelogram (4 vertices) centered at a point, spanning "
             "two direction vectors. Use this instead of add_plane when the goal "
             "is a readable diagram — e.g. tangent planes, cutting planes shown "
             "in textbook style. The plane is a Polygon, so it has finite extent "
             "and does not obscure other objects.",
             {
                 "name":   ToolParam("string", "Plane polygon name"),
                 "center": ToolParam("string", "Center point of the rectangle"),
                 "dir1":   ToolParam("string", "First direction: vector name, or 'x'/'y'/'z' for axis direction"),
                 "dir2":   ToolParam("string", "Second direction: vector name, or 'x'/'y'/'z' for axis direction"),
                 "size":   ToolParam("number", "Half-width of the rectangle along each direction (default 3)", required=False),
             }),

    ToolSpec("add_perpendicular_plane",
             "Create an INFINITE plane through a point, perpendicular to a line or vector. "
             "For visual diagrams, consider using add_finite_plane instead. "
             "Syntax: PerpendicularPlane(Point, Line) or PerpendicularPlane(Point, Vector).",
             {
                 "name": ToolParam("string", "Plane name"),
                 "point": ToolParam("string", "Point on the plane"),
                 "direction": ToolParam("string", "Line or vector perpendicular to the plane"),
             }),

    ToolSpec("add_plane_bisector",
             "Create the perpendicular bisector plane between two points or of a segment. "
             "Syntax: PlaneBisector(Point, Point) or PlaneBisector(Segment).",
             {
                 "name": ToolParam("string", "Plane name"),
                 "a": ToolParam("string", "First point or segment"),
                 "b": ToolParam("string", "Second point (omit if a is a segment)", required=False),
             }),

    # ── Solids ──────────────────────────────────────────────────────────────

    ToolSpec("add_pyramid",
             "Create a pyramid. "
             "Variants: Pyramid(Polygon, Point) — polygon base + apex; "
             "Pyramid(Polygon, Height) — polygon base + centered apex at given height; "
             "Pyramid(A, B, C, D) — four points (base ABC, apex D).",
             {
                 "name":   ToolParam("string", "Pyramid name"),
                 "base":   ToolParam("string", "Base polygon name, OR first vertex name"),
                 "top":    ToolParam("string", "Apex point name, OR second vertex, OR height (number as string)"),
                 "c":      ToolParam("string", "Third vertex (only for point-list form)", required=False),
                 "d":      ToolParam("string", "Fourth vertex / apex (only for point-list form)", required=False),
             }),

    ToolSpec("add_prism",
             "Create a prism. "
             "Variants: Prism(Polygon, Height) — right prism with given height; "
             "Prism(Polygon, Point) — prism with polygon base, first top vertex at Point; "
             "Prism(A, B, C, D) — points where AD is the extrusion vector.",
             {
                 "name":   ToolParam("string", "Prism name"),
                 "base":   ToolParam("string", "Base polygon name, OR first vertex"),
                 "top":    ToolParam("string", "First top point, OR height (number as string)"),
                 "c":      ToolParam("string", "Third vertex (point-list form)", required=False),
                 "d":      ToolParam("string", "Fourth vertex (point-list form)", required=False),
             }),

    ToolSpec("add_cone",
             "Create a cone. "
             "Variants: Cone(Circle, Height) — cone with circular base and height; "
             "Cone(Point, Point, Radius) — cone with apex at second point, "
             "circle center at first point, given radius.",
             {
                 "name":   ToolParam("string", "Cone name"),
                 "a":      ToolParam("string", "Circle (base) or center point"),
                 "b":      ToolParam("string", "Height value, or apex point"),
                 "radius": ToolParam("number", "Radius (only for Point,Point,Radius form)", required=False),
             }),

    ToolSpec("add_cylinder",
             "Create a 3D cylinder (filled solid, not just lateral surface). "
             "Variants: Cylinder(Circle, Height) — extrude a circle by a height; "
             "Cylinder(Point, Point, Radius) — two points as top/bottom center, given radius. "
             "Use query_surface_area / query_volume on the resulting solid for measurements.",
             {
                 "name":   ToolParam("string", "Cylinder name"),
                 "a":      ToolParam("string", "Circle (base) or bottom center point"),
                 "b":      ToolParam("string", "Height value, or top center point"),
                 "radius": ToolParam("number", "Radius (only for Point,Point,Radius form)", required=False),
             }),

    ToolSpec("add_sphere",
             "Create a sphere. "
             "Variants: Sphere(Point, Radius); Sphere(Point, Point) — center through point.",
             {
                 "name":   ToolParam("string", "Sphere name"),
                 "center": ToolParam("string", "Center point"),
                 "radius_or_point": ToolParam("string", "Radius (number) or point name on sphere"),
             }),

    ToolSpec("add_tetrahedron",
             "Create a regular tetrahedron. "
             "Variants: Tetrahedron(Point, Point) — edge defined by two points; "
             "Tetrahedron(Point, Point, Point) — three points of first (equilateral) face.",
             {
                 "name": ToolParam("string", "Tetrahedron name"),
                 "a":    ToolParam("string", "First point"),
                 "b":    ToolParam("string", "Second point"),
                 "c":    ToolParam("string", "Third point (optional)", required=False),
             }),

    ToolSpec("add_cube",
             "Create a cube. "
             "Variants: Cube(Point, Point) — edge defined by two points (third vertex auto-placed on a circle); "
             "Cube(Point, Point, Point) — three adjacent vertices of first face. "
             "The three points MUST form a square (perpendicular adjacent edges of equal length); "
             "an arbitrary triangle yields an undefined object. "
             "Prefer the 2-point variant if a third corner is not constrained to be square.",
             {
                 "name": ToolParam("string", "Cube name"),
                 "a":    ToolParam("string", "First point"),
                 "b":    ToolParam("string", "Second point"),
                 "c":    ToolParam("string", "Third point (optional)", required=False),
             }),

    # ── Cross-section & Intersection ────────────────────────────────────────

    ToolSpec("add_cross_section",
             "Create the intersection (cross-section) of a plane with a solid or polygon. "
             "Syntax: IntersectPath(Plane, Quadric) — e.g. plane cutting a cone/cylinder/sphere returns a conic; "
             "IntersectPath(Plane, Polygon) — returns a segment.",
             {
                 "name":  ToolParam("string", "Result name"),
                 "plane": ToolParam("string", "Plane name"),
                 "solid": ToolParam("string", "Solid, quadric, or polygon name"),
             }),

    # ── Net (unfolded surface) ──────────────────────────────────────────────

    ToolSpec("add_net",
             "Create the net (unfolded surface) of a convex polyhedron onto the plane of its base. "
             "Syntax: Net(Polyhedron, Number) — Number from 0 (folded) to 1 (fully unfolded).",
             {
                 "name":       ToolParam("string", "Net name"),
                 "polyhedron": ToolParam("string", "Polyhedron name"),
                 "unfold":     ToolParam("number", "Unfold progress: 0 (closed) to 1 (fully open)"),
             }),

    # ── Text / Label in 3D ────────────────────────────────────────────────

    ToolSpec("add_text_3d",
             "Place a text label or LaTeX formula at 3D coordinates. "
             "Supports LaTeX: \\frac{a}{b}, \\sqrt{x}, \\text{label}. "
             "The text is always rendered in LaTeX mode.",
             {
                 "name": ToolParam("string", "Text object name"),
                 "text": ToolParam("string", "Text content (LaTeX supported)"),
                 "x":    ToolParam("number", "X coordinate"),
                 "y":    ToolParam("number", "Y coordinate"),
                 "z":    ToolParam("number", "Z coordinate"),
             }),

    # ── Surface (parametric surface) ────────────────────────────────────────

    ToolSpec("add_surface_revolution",
             "Create a surface of revolution by rotating a function around the x-axis. "
             "Syntax: Surface(Function, Angle) — rotates from 0 to angle around x-axis.",
             {
                 "name":     ToolParam("string", "Surface name"),
                 "function": ToolParam("string", "Function name to revolve"),
                 "angle":    ToolParam("string", "Rotation angle, e.g. '2*pi' or '360°'"),
             }),
]

# ── 3D Query Tools ──────────────────────────────────────────────────────────

SOLID_QUERY_TOOLS: List[ToolSpec] = [

    ToolSpec("query_volume",
             "Get the volume of a solid (Pyramid, Prism, Cone, Cylinder, Sphere). "
             "Returns the numeric volume value.",
             {
                 "solid": ToolParam("string", "Solid object name"),
             }),

    ToolSpec("query_surface_area",
             "Get the total surface area of a 3D solid (cube, sphere, cylinder, prism, pyramid, cone). "
             "Returns 0 for 2D shapes such as circles or polygons—use query_area for those instead.",
             {
                 "solid": ToolParam("string", "3D solid object name"),
             }),

    ToolSpec("query_coords3d",
             "Get the 3D coordinates (x, y, z) of a point.",
             {
                 "point": ToolParam("string", "Point name"),
             }),
]

# ── 3D Render Tools ─────────────────────────────────────────────────────────

SOLID_RENDER_TOOLS: List[ToolSpec] = [

    ToolSpec("render_set_3d_view",
             "Configure 3D viewport: rotation angles, zoom, axis/plate visibility. "
             "All parameters are optional — only the provided ones are changed. "
             "xAngle/zAngle control camera orbit (degrees). "
             "scale controls zoom (default 50, larger = closer). "
             "show_axes: show/hide xyz axes. show_plate: show/hide xOy plane shadow. "
             "show_numbers: show/hide axis tick numbers. "
             "IMPORTANT: Avoid zAngle=-45 (diagonal direction causes vertex overlap). "
             "Best default: xAngle=20, zAngle=-50 (all edges separated). "
             "Safe range: xAngle=20-30, zAngle=-50 to -55.",
             {
                 "x_angle":      ToolParam("number", "Rotation around x-axis in degrees (default 20)", required=False),
                 "z_angle":      ToolParam("number", "Rotation around z-axis in degrees (default -60)", required=False),
                 "scale":        ToolParam("number", "Zoom level (default 50, try 70-100 for close-up)", required=False),
                 "show_axes":    ToolParam("string", "Show axes: 'true' or 'false'", required=False),
                 "show_plate":   ToolParam("string", "Show xOy plane shadow: 'true' or 'false'", required=False),
                 "show_numbers": ToolParam("string", "Show axis tick numbers: 'true' or 'false'", required=False),
             }),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Execution
# ═══════════════════════════════════════════════════════════════════════════════

def execute_solid_tool(ggb, tool_name: str, args: Dict[str, Any]) -> Tuple[str, bool, str]:
    """Execute one 3D tool call on GeoGebra.

    Returns (command_string, success, error_message).
    Same contract as execute_geogebra_tool in geogebra_tools.py.
    """

    def run(cmd: str) -> Tuple[str, bool, str]:
        result = ggb.eval_command(cmd)
        return cmd, bool(result.success), (result.error_message or "")

    def run_checked(cmd: str, name: str, hint: str = "") -> Tuple[str, bool, str]:
        cmd_str, ok, err = run(cmd)
        if ok and not ggb.is_defined(name):
            detail = f"Object '{name}' is undefined after command (degenerate input?)"
            if hint:
                detail += f". Hint: {hint}"
            return cmd_str, False, detail
        return cmd_str, ok, err

    # ── Points & Vectors ────────────────────────────────────────────────

    if tool_name == "add_point3d":
        return run_checked(
            f"{args['name']} = ({args['x']}, {args['y']}, {args['z']})",
            args['name'], "3D point requires numeric x, y, z")

    if tool_name == "add_vector3d":
        name = args['name']
        if args.get('from_pt') and args.get('to_pt'):
            return run_checked(
                f"{name} = Vector({args['from_pt']}, {args['to_pt']})",
                name)
        else:
            x = args.get('x', 0)
            y = args.get('y', 0)
            z = args.get('z', 0)
            return run_checked(f"{name} = Vector(({x}, {y}, {z}))", name)

    # ── Planes ──────────────────────────────────────────────────────────

    if tool_name == "add_plane":
        name = args['name']
        a, b, c = args['a'], args.get('b'), args.get('c')
        if b and c:
            # Plane(A, B, C)
            return run_checked(f"{name} = Plane({a}, {b}, {c})", name,
                               "Three points must be non-collinear")
        elif b:
            # Plane(Point, Line) or Plane(Line, Line)
            return run_checked(f"{name} = Plane({a}, {b})", name)
        else:
            # Plane(Polygon)
            return run_checked(f"{name} = Plane({a})", name)

    if tool_name == "add_finite_plane":
        # Build a parallelogram centered at `center` with two directions.
        # Creates 4 corner points (hidden) + Polygon.
        name = args['name']
        center = args['center']
        d1 = args['dir1']
        d2 = args['dir2']
        size = args.get('size', 3)
        # Map axis shortcuts to vectors
        axis_map = {'x': f'(1,0,0)', 'y': f'(0,1,0)', 'z': f'(0,0,1)'}
        v1 = axis_map.get(d1, d1)
        v2 = axis_map.get(d2, d2)
        # Create temp vectors if axis shortcuts used
        prefix = f"fp{name}"
        cmds = []
        if d1 in axis_map:
            cmds.append(f"{prefix}v1 = Vector({v1})")
            v1_ref = f"{prefix}v1"
        else:
            v1_ref = d1
        if d2 in axis_map:
            cmds.append(f"{prefix}v2 = Vector({v2})")
            v2_ref = f"{prefix}v2"
        else:
            v2_ref = d2
        # 4 corners: center ± size*v1 ± size*v2
        cmds.extend([
            f"{prefix}c1 = {center} + {size}*{v1_ref} + {size}*{v2_ref}",
            f"{prefix}c2 = {center} - {size}*{v1_ref} + {size}*{v2_ref}",
            f"{prefix}c3 = {center} - {size}*{v1_ref} - {size}*{v2_ref}",
            f"{prefix}c4 = {center} + {size}*{v1_ref} - {size}*{v2_ref}",
            f"{name} = Polygon(" + "{" + f"{prefix}c1, {prefix}c2, {prefix}c3, {prefix}c4" + "})",
        ])
        for cmd in cmds:
            r = ggb.eval_command(cmd)
            if not r.success:
                return cmd, False, f"add_finite_plane failed at: {cmd}"
        # Hide helper points and vectors
        for helper in [f"{prefix}v1", f"{prefix}v2",
                       f"{prefix}c1", f"{prefix}c2",
                       f"{prefix}c3", f"{prefix}c4"]:
            if ggb.exists(helper):
                ggb.set_label_visible(helper, False)
                ggb.set_object_visible(helper, False)
        if ggb.exists(name):
            # Override the listener's low filling — finite planes need
            # visible fill + clear border to look like textbook planes.
            ggb._execute_js(f'ggbApplet.setFilling("{name}", 0.15)')
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 3)')
            # Also thicken auto-created edge segments
            n_obj = ggb.get_object_number()
            for idx in range(n_obj):
                obj = ggb.get_object_name(idx)
                if (ggb.get_object_type(obj) == "segment"
                        and obj.startswith(prefix)):
                    ggb._execute_js(f'ggbApplet.setLineThickness("{obj}", 3)')
            return f"{name} = FinitePlane({center}, {d1}, {d2}, size={size})", True, ""
        return f"{name} = FinitePlane(...)", False, "Polygon not created"

    if tool_name == "add_perpendicular_plane":
        return run_checked(
            f"{args['name']} = PerpendicularPlane({args['point']}, {args['direction']})",
            args['name'], "Point and direction (line or vector) required")

    if tool_name == "add_plane_bisector":
        name = args['name']
        a, b = args['a'], args.get('b')
        if b:
            return run_checked(f"{name} = PlaneBisector({a}, {b})", name)
        else:
            return run_checked(f"{name} = PlaneBisector({a})", name)

    # ── Solids ──────────────────────────────────────────────────────────

    if tool_name == "add_pyramid":
        name = args['name']
        base, top = args['base'], args['top']
        c, d = args.get('c'), args.get('d')
        if c and d:
            # Pyramid(A, B, C, D)
            return run_checked(f"{name} = Pyramid({base}, {top}, {c}, {d})",
                               name, "Four points for a triangular pyramid")
        else:
            # Try numeric height first, fall back to point
            try:
                float(top)
                cmd = f"{name} = Pyramid({base}, {top})"
            except (ValueError, TypeError):
                cmd = f"{name} = Pyramid({base}, {top})"
            return run_checked(cmd, name,
                               "Pyramid(Polygon, Point/Height)")

    if tool_name == "add_prism":
        name = args['name']
        base, top = args['base'], args['top']
        c, d = args.get('c'), args.get('d')
        if c and d:
            return run_checked(f"{name} = Prism({base}, {top}, {c}, {d})",
                               name, "Points where last-first defines extrusion")
        else:
            try:
                float(top)
                cmd = f"{name} = Prism({base}, {top})"
            except (ValueError, TypeError):
                cmd = f"{name} = Prism({base}, {top})"
            return run_checked(cmd, name, "Prism(Polygon, Point/Height)")

    if tool_name == "add_cone":
        name = args['name']
        a, b = args['a'], args['b']
        radius = args.get('radius')
        if radius is not None:
            # Cone(Point, Point, Radius)
            return run_checked(f"{name} = Cone({a}, {b}, {radius})", name,
                               "Cone(center, apex, radius)")
        else:
            # Cone(Circle, Height)
            return run_checked(f"{name} = Cone({a}, {b})", name,
                               "Cone(Circle, Height) or Cone(Point, Point, Radius)")

    if tool_name == "add_cylinder":
        name = args['name']
        a, b = args['a'], args['b']
        radius = args.get('radius')
        if radius is not None:
            return run_checked(f"{name} = Cylinder({a}, {b}, {radius})", name,
                               "Cylinder(bottomCenter, topCenter, radius)")
        else:
            return run_checked(f"{name} = Cylinder({a}, {b})", name,
                               "Cylinder(Circle, Height)")

    if tool_name == "add_sphere":
        name = args['name']
        center = args['center']
        rp = args['radius_or_point']
        # Sphere(Point, Radius) or Sphere(Point, Point)
        return run_checked(f"{name} = Sphere({center}, {rp})", name,
                           "Sphere(center, radius) or Sphere(center, pointOnSphere)")

    if tool_name == "add_tetrahedron":
        name = args['name']
        a, b, c = args['a'], args['b'], args.get('c')
        if c:
            return run_checked(f"{name} = Tetrahedron({a}, {b}, {c})", name,
                               "Three points must form an equilateral triangle")
        else:
            return run_checked(f"{name} = Tetrahedron({a}, {b})", name)

    if tool_name == "add_cube":
        name = args['name']
        a, b, c = args['a'], args['b'], args.get('c')
        if c:
            return run_checked(f"{name} = Cube({a}, {b}, {c})", name,
                               "Three points must form a square")
        else:
            return run_checked(f"{name} = Cube({a}, {b})", name)

    # ── Cross-section & Net ─────────────────────────────────────────────

    if tool_name == "add_cross_section":
        return run_checked(
            f"{args['name']} = IntersectPath({args['plane']}, {args['solid']})",
            args['name'],
            "IntersectPath(Plane, Solid/Polygon) — plane must intersect the solid")

    if tool_name == "add_net":
        return run_checked(
            f"{args['name']} = Net({args['polyhedron']}, {args['unfold']})",
            args['name'],
            "Net(Polyhedron, unfoldProgress 0..1)")

    if tool_name == "add_text_3d":
        import re as _re
        text = args['text']
        # ── LaTeX backslash recovery (same logic as 2D add_text) ──
        # JSON decode eats \f→formfeed, \t→tab, \b→backspace.
        text = _re.sub(r'\text\{',  r'\\text{',  text)
        text = _re.sub(r'\frac\{',  r'\\frac{',  text)
        text = _re.sub(r'(?<![a-zA-Z])rac\{', r'\\frac{', text)
        text = _re.sub(r'(?<![a-zA-Z])ext\{',  r'\\text{', text)
        text = _re.sub(r'(?<![a-zA-Z])egin\{', r'\\begin{', text)
        text = text.replace('\f', '\\f').replace('\t', '\\t').replace('\b', '\\b')
        # Fix multi-char subscripts/superscripts without braces:
        #   _sol → _{sol},  ^abc → ^{abc}
        text = _re.sub(r'_([A-Za-z0-9]{2,})(?![{])', r'_{\1}', text)
        text = _re.sub(r'\^([A-Za-z0-9]{2,})(?![{])', r'^{\1}', text)
        # ── Escape for the two layers: JS string → GGB parser ──
        # GGB LaTeX needs one backslash: \sqrt, \frac
        # JS string literal (single-quoted) needs \\ for one backslash
        # So: Python \\ → JS \\ → GGB sees single \  ✓
        text_js = text.replace('\\', '\\\\')  # double for JS layer
        text_js = text_js.replace("'", "\\'")  # escape single quotes
        text_js = text_js.replace('"', '\\"')  # escape double quotes
        name = args['name']
        x, y, z = args['x'], args['y'], args['z']
        ggb_cmd = f'{name} = Text("{text}", ({x}, {y}, {z}), false, true)'
        # Use JS single-quote wrapper to avoid double-quote conflicts
        js = (f"return ggbApplet.evalCommand("
              f"'{name} = Text(\"{text_js}\", ({x}, {y}, {z}), false, true)')")
        try:
            ok = ggb._execute_js(js)
            return ggb_cmd, bool(ok), ""
        except Exception as e:
            return ggb_cmd, False, str(e)

    if tool_name == "add_surface_revolution":
        return run_checked(
            f"{args['name']} = Surface({args['function']}, {args['angle']})",
            args['name'],
            "Surface(Function, Angle) — revolve around x-axis")

    # ── Queries ─────────────────────────────────────────────────────────

    # ── Queries (dispatch to execute_solid_query) ─────────────────────
    if tool_name.startswith("query_"):
        cmd, ok, err, val = execute_solid_query(ggb, tool_name, args)
        return cmd, ok, err

    # ── Render ──────────────────────────────────────────────────────────

    if tool_name == "render_set_3d_view":
        import re as _re, base64 as _b64
        try:
            xml = ggb._execute_js("return ggbApplet.getXML()")

            # ── Rotation & zoom ─────────────────────────────────────
            if args.get("x_angle") is not None or args.get("z_angle") is not None:
                # Parse current values as defaults
                m = _re.search(r'xAngle="([^"]*)" zAngle="([^"]*)"', xml)
                cur_x = m.group(1) if m else "20"
                cur_z = m.group(2) if m else "-60"
                new_x = args.get("x_angle", cur_x)
                new_z = args.get("z_angle", cur_z)
                xml = _re.sub(r'xAngle="[^"]*" zAngle="[^"]*"',
                              f'xAngle="{new_x}" zAngle="{new_z}"', xml)

            if args.get("scale") is not None:
                xml = _re.sub(r'(coordSystem[^>]*?)scale="[^"]*"',
                              rf'\1scale="{args["scale"]}"', xml)

            # ── Axis visibility ─────────────────────────────────────
            if args.get("show_axes") is not None:
                vis = args["show_axes"].lower()
                xml = _re.sub(r'(<axis id="\d" show=")(?:true|false)',
                              rf'\1{vis}', xml)

            # ── Axis tick numbers ───────────────────────────────────
            if args.get("show_numbers") is not None:
                vis = args["show_numbers"].lower()
                xml = _re.sub(r'showNumbers="(?:true|false)"',
                              f'showNumbers="{vis}"', xml)

            # ── xOy plate shadow ────────────────────────────────────
            if args.get("show_plate") is not None:
                vis = args["show_plate"].lower()
                xml = _re.sub(r'<plate show="(?:true|false)"/>',
                              f'<plate show="{vis}"/>', xml)

            # Apply modified XML
            encoded = _b64.b64encode(xml.encode()).decode()
            ggb._execute_js(f'ggbApplet.setXML(atob("{encoded}"))')

            desc_parts = []
            for k in ["x_angle", "z_angle", "scale", "show_axes",
                       "show_plate", "show_numbers"]:
                if args.get(k) is not None:
                    desc_parts.append(f"{k}={args[k]}")
            return f"set3DView({', '.join(desc_parts)})", True, ""
        except Exception as e:
            return f"set3DView({args})", False, str(e)

    return f"{tool_name}({args})", False, f"Unknown solid tool: {tool_name}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Query execution wrapper (matches execute_query_tool contract)
# ═══════════════════════════════════════════════════════════════════════════════

def execute_solid_query(ggb, tool_name: str, args: Dict[str, Any]) -> Tuple[str, bool, str, Any]:
    """Execute a 3D query tool.

    Returns (command_string, success, error_message, value).
    """
    if tool_name == "query_volume":
        obj = args['solid']
        cmd = f"Volume({obj})"
        # Strategy: create explicit Volume() object first (works for all
        # solid types including Sphere/Cone/Cylinder which don't return
        # volume via getValue directly), then fallback to getValue on
        # the object itself (works for Prism/Pyramid).
        tmp = f"qv{obj}"      # no underscore prefix (GGB rejects _names)
        ggb.eval_command(f"{tmp} = Volume({obj})")
        val = ggb.get_value(tmp)
        ggb.delete_object(tmp)
        if val is None or val == 0:
            val = ggb.get_value(obj)
        if val is not None and val != 0:
            return cmd, True, "", round(val, 6)
        return cmd, False, f"Volume({obj}) returned {val}", None

    if tool_name == "query_surface_area":
        obj = args['solid']
        cmd = f"SurfaceArea({obj})"
        tmp = f"qsa{obj}"
        # GeoGebra doesn't have a direct SurfaceArea(); use numeric approach
        ggb.eval_command(f"{tmp} = Surface({obj})")
        val = ggb.get_value(tmp)
        ggb.delete_object(tmp)
        if val is not None and val != 0:
            return cmd, True, "", round(val, 6)
        return cmd, False, f"Surface area of {obj} returned {val}", None

    if tool_name == "query_coords3d":
        pt = args['point']
        coords = ggb.get_coords_3d(pt)
        cmd = f"coords3d({pt})"
        if coords:
            return cmd, True, "", {"x": round(coords[0], 6),
                                   "y": round(coords[1], 6),
                                   "z": round(coords[2], 6)}
        return cmd, False, f"Could not get 3D coords for '{pt}'", None

    return f"{tool_name}({args})", False, f"Unknown solid query: {tool_name}", None


# ═══════════════════════════════════════════════════════════════════════════════
#  Provider Adapters — build tool schemas for LLM APIs
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  Public routing — used by eval scripts to dispatch 2D vs 3D tool calls
# ═══════════════════════════════════════════════════════════════════════════════

# Names of all 3D solid tools
SOLID_TOOL_NAMES = {s.name for s in
                    SOLID_GEOGEBRA_TOOLS + SOLID_QUERY_TOOLS + SOLID_RENDER_TOOLS}


def exec_with_solid_routing(tracker, ggb, fn_name, args):
    """Execute a tool call, routing 3D solid tools to execute_solid_tool.

    2D tools go through CanvasTracker as before.
    3D tools use execute_solid_tool / execute_solid_query directly,
    but still update CanvasTracker's counters and canvas-delta tracking.

    Args:
        tracker: CanvasTracker instance
        ggb: GeoGebraAPI instance
        fn_name: tool function name
        args: tool arguments dict

    Returns:
        (result_dict, log_dict) — same contract as CanvasTracker.execute()
    """
    if fn_name not in SOLID_TOOL_NAMES:
        return tracker.execute(ggb, fn_name, args)

    # ── 3D solid tool path ─────────────────────────────────────────
    tracker.total_n += 1

    if fn_name.startswith("query_"):
        cmd, ok, err, val = execute_solid_query(ggb, fn_name, args)
        if ok:
            tracker.ok_n += 1
            canvas = build_rich_canvas(ggb)
            tracker._known.update(canvas)
            result = {"command": cmd, "success": True, "error": "",
                      "value": val, "canvas": canvas}
            log = {"fn": fn_name, "cmd": cmd, "ok": True,
                   "value": val, "canvas": canvas}
        else:
            tracker.fail_n += 1
            result = {"command": cmd, "success": False, "error": err}
            log = {"fn": fn_name, "cmd": cmd, "ok": False,
                   "error": err, "value": val}
        return result, log

    # Construction / render tool
    cmd, ok, err = execute_solid_tool(ggb, fn_name, args)
    if ok:
        tracker.ok_n += 1
        fc_ = build_rich_canvas(ggb)
        removed = sorted(tracker._known - set(fc_))
        tracker._known &= set(fc_)
        new = {n: v for n, v in fc_.items() if n not in tracker._known}
        tracker._known.update(fc_)

        _DISPLAY_TOOLS = {"set_label_visible", "set_object_visible", "rename_object"}
        is_display = fn_name.startswith("render_") or fn_name in _DISPLAY_TOOLS

        if is_display:
            result = {"success": True, "applied": f"{fn_name}({args})"}
            log = {"fn": fn_name, "cmd": cmd, "ok": True,
                   "applied": f"{fn_name}({args})"}
        else:
            result = {"command": cmd, "success": True, "error": "",
                      "new_objects": new, "removed_objects": removed}
            log = {"fn": fn_name, "cmd": cmd, "ok": True,
                   "new_objects": new, "removed_objects": removed,
                   "canvas": fc_}
    else:
        tracker.fail_n += 1
        result = {"command": cmd, "success": False, "error": err}
        log = {"fn": fn_name, "cmd": cmd, "ok": False, "error": err}
    return result, log


# ═══════════════════════════════════════════════════════════════════════════════
#  Provider Adapters — build tool schemas for LLM APIs
# ═══════════════════════════════════════════════════════════════════════════════

def _solid_specs(include_render: bool = False) -> List[ToolSpec]:
    """Return all solid ToolSpecs, optionally including render tools."""
    specs = list(SOLID_GEOGEBRA_TOOLS) + list(SOLID_QUERY_TOOLS)
    if include_render:
        specs += list(SOLID_RENDER_TOOLS)
    return specs


def build_openai_solid_tools(include_render: bool = False) -> List[Dict[str, Any]]:
    """OpenAI function-calling schema for 3D tools."""
    tools = []
    for s in _solid_specs(include_render):
        props = {}
        required = []
        for pname, p in s.params.items():
            props[pname] = {"type": p.param_type, "description": p.description}
            if p.required:
                required.append(pname)
        tools.append({
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        })
    return tools


def build_anthropic_solid_tools(include_render: bool = False) -> List[Dict[str, Any]]:
    """Anthropic tool schema for 3D tools."""
    tools = []
    for s in _solid_specs(include_render):
        props = {}
        required = []
        for pname, p in s.params.items():
            props[pname] = {"type": p.param_type, "description": p.description}
            if p.required:
                required.append(pname)
        tools.append({
            "name": s.name,
            "description": s.description,
            "input_schema": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        })
    return tools


def build_gemini_solid_tools(include_render: bool = False):
    """Gemini FunctionDeclaration list for 3D tools."""
    from google.genai import types
    decls = []
    for s in _solid_specs(include_render):
        props = {}
        required = []
        for pname, p in s.params.items():
            schema_type = {"string": "STRING", "number": "NUMBER",
                           "integer": "INTEGER"}.get(p.param_type, "STRING")
            props[pname] = types.Schema(
                type=schema_type, description=p.description)
            if p.required:
                required.append(pname)
        decls.append(types.FunctionDeclaration(
            name=s.name,
            description=s.description,
            parameters=types.Schema(
                type="OBJECT", properties=props, required=required),
        ))
    return [types.Tool(function_declarations=decls)]
