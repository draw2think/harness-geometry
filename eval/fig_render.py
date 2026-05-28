"""
Per-turn canvas renderer for paper figures.

Replays a problem's construction process, saving a canvas PNG after each turn.
Reads the existing result.json (process log) and re-executes tool calls on a
fresh GeoGebra instance, capturing intermediate states.

Usage:
    python eval/fig_render.py --result <path/to/result.json> --out figs/fig2_case/turns

Output:
    figs/fig2_case/turns/
        turn_0_input.png        ← empty canvas (before any construction)
        turn_1_canvas.png       ← after turn 1 tool calls
        turn_1_tools.txt        ← tool call summary for turn 1
        turn_2_canvas.png
        turn_2_tools.txt
        turn_3_canvas.png       ← final canvas
        turn_3_tools.txt
        summary.md              ← markdown summary of all turns
"""
import json
import sys
import os
from pathlib import Path
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from symbolic.integrations.geogebra_api import GeoGebraAPI
from symbolic.tools.geogebra_tools import execute_geogebra_tool, execute_query_tool


def _set_point_coords(ggb, name: str, x: float, y: float):
    """Move a free/semi-free point to exact coordinates (for faithful replay)."""
    try:
        ggb._execute_js(
            f'ggbApplet.setCoords("{name}", {x}, {y})'
        )
    except Exception:
        pass


def _enlarge_font(ggb, size=32):
    """Enlarge global label font via XML: get → replace <font size> → setXML."""
    import re
    full_xml = ggb._execute_js('return ggbApplet.getXML()')
    new_xml = re.sub(r'<font size="\d+"/>', f'<font size="{size}"/>', full_xml)
    ggb._driver.execute_script('ggbApplet.setXML(arguments[0])', new_xml)


def _set_line_styles(ggb):
    """Set absolute line/point styles after setXML reload."""
    for name in ggb.get_all_object_names() or []:
        t = ggb.get_object_type(name)
        if t in ("circle", "conic"):
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 5)')
        elif t == "arc":
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 8)')
        elif t in ("segment", "line", "ray", "vector"):
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 4)')
        elif t == "angle":
            ggb._execute_js(f'ggbApplet.setLineThickness("{name}", 5)')
        elif t == "point":
            ggb._execute_js(f'ggbApplet.setPointSize("{name}", 6)')


def _hide_numeric_objects(ggb, tc):
    """Hide numeric query objects (len_*, r, etc.) that clutter the canvas."""
    new_objs = tc.get("new_objects") or tc.get("canvas") or {}
    for name, info in new_objs.items():
        if isinstance(info, dict) and info.get("type") == "numeric":
            ggb.set_object_visible(name, False)


def replay_and_render(result_path: Path, out_dir: Path):
    """Replay tool calls from result.json, save per-turn canvas PNGs."""
    result = json.load(open(result_path))
    process = result.get("process", {})
    turns = process.get("turns", [])

    if not turns:
        print(f"No turns in {result_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Start fresh GeoGebra
    ggb = GeoGebraAPI(mode="selenium", headless=True)
    ggb.initialize()

    # Clean canvas: hide axes and grid
    ggb.set_axes_visible(False, False)
    ggb.set_grid_visible(False)

    # Save empty canvas
    ggb.export_png(out_dir / "turn_0_input.png")
    print(f"  [turn 0] empty canvas saved")

    summary_lines = [
        f"# Per-Turn Replay: {result.get('id', '?')}",
        f"Model: {result.get('model', '?')}",
        f"Total turns: {len(turns)}",
        f"Passed: {result.get('passed', '?')}",
        "",
    ]

    for turn_data in turns:
        turn_num = turn_data.get("turn", "?")
        tool_calls = turn_data.get("tool_calls", [])
        answer = turn_data.get("answer_raw")

        tool_lines = []
        for tc in tool_calls:
            fn = tc.get("fn", "?")
            cmd = tc.get("cmd", "?")
            ok = tc.get("ok", False)
            tag = "OK" if ok else "FAIL"

            # Re-execute the tool call (only successful ones)
            if ok:
                try:
                    ggb.eval_command(cmd)

                    # For Point(obj) commands, set coords to match original
                    if fn in ("add_point_on",) and tc.get("new_objects"):
                        for name, info in tc["new_objects"].items():
                            if isinstance(info, dict) and "x" in info and "y" in info:
                                _set_point_coords(ggb, name, info["x"], info["y"])

                    # Hide numeric query objects (len_*, r, etc.)
                    if fn.startswith("query_"):
                        _hide_numeric_objects(ggb, tc)

                except Exception as e:
                    tag = f"REPLAY_ERR: {e}"

            # Build tool line with fn name and value
            line = f"  [{tag:4s}] {fn}: {cmd}"
            if tc.get("value") is not None:
                line += f"  → {tc['value']}"
            tool_lines.append(line)

        # Step 1: Enlarge font + fix line styles via XML (setXML reloads all)
        _enlarge_font(ggb, size=32)
        _set_line_styles(ggb)

        # Step 2: Smart label cleanup AFTER setXML (so captions stick)
        try:
            all_names = ggb.get_all_object_names()
            for name in all_names:
                obj_type = ggb.get_object_type(name)
                # Hide: numeric, boolean, list objects
                if obj_type in ("numeric", "boolean", "list"):
                    ggb.set_label_visible(name, False)
                    continue
                # Angles: always show value (313°), hide name
                if obj_type == "angle":
                    ggb.set_label_style(name, 2)  # VALUE mode
                    ggb.set_label_visible(name, True)
                    continue
                # Hide: non-point geometric objects (circle, arc, line, segment, ray, etc.)
                if obj_type in ("circle", "conic", "arc", "line", "segment",
                                "ray", "vector", "polygon", "quadrilateral"):
                    ggb.set_label_visible(name, False)
                    continue
                # Objects with _ in name: use caption mode with _ → space
                if "_" in name:
                    ggb.set_caption(name, name.replace("_", " "))
                    ggb.set_label_style(name, 3)  # CAPTION mode
                    ggb.set_label_visible(name, True)
                    continue
                # Default: show label for points etc.
                ggb.set_label_visible(name, True)
        except Exception as e:
            print(f"    [warn] label cleanup: {e}")

        png_path = out_dir / f"turn_{turn_num}_canvas.png"
        try:
            ggb.fit_view(padding=2.5)
            ggb.export_png(png_path)
            print(f"  [turn {turn_num}] {len(tool_calls)} tools, canvas saved → {png_path.name}")
        except Exception as e:
            print(f"  [turn {turn_num}] screenshot failed: {e}")

        # Save tool summary
        tools_path = out_dir / f"turn_{turn_num}_tools.txt"
        tools_text = f"Turn {turn_num}: {len(tool_calls)} tool calls\n"
        tools_text += "\n".join(tool_lines)
        if answer:
            tools_text += f"\n\n  ANSWER: {json.dumps(answer)}"
        tools_path.write_text(tools_text)

        # Add to summary
        ok_count = sum(1 for tc in tool_calls if tc.get("ok"))
        fail_count = len(tool_calls) - ok_count
        summary_lines.append(f"## Turn {turn_num}")
        summary_lines.append(f"- Tools: {ok_count} OK / {fail_count} fail")
        summary_lines.append(f"- Canvas: `turn_{turn_num}_canvas.png`")
        for line in tool_lines[:8]:
            summary_lines.append(line)
        if len(tool_lines) > 8:
            summary_lines.append(f"  ... ({len(tool_lines) - 8} more)")
        if answer:
            summary_lines.append(f"- **ANSWER: {json.dumps(answer)}**")
        summary_lines.append("")

    # Save summary
    (out_dir / "summary.md").write_text("\n".join(summary_lines))

    ggb.cleanup()
    print(f"\n  Done. {len(turns)} turns rendered to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Render per-turn canvas for paper figures")
    parser.add_argument("--result", type=str, required=True,
                        help="Path to result.json (e.g. eval/mathverse/1795/gemini-3-flash-preview@medium_result.json)")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for turn PNGs")
    args = parser.parse_args()

    result_path = Path(args.result)
    out_dir = Path(args.out)

    if not result_path.exists():
        print(f"Error: {result_path} not found")
        sys.exit(1)

    replay_and_render(result_path, out_dir)


if __name__ == "__main__":
    main()
