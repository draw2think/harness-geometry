#!/usr/bin/env python3
"""Run isolated ToolSpec-overlay variants on MathVerse problem 3850.

This is a source-preserving runner for the harness-refinement toy study. It
reuses ``test_agentic_geo_constructer.py`` and changes only the ToolSpec text
exposed to Gemini at runtime. The underlying ``symbolic/tools`` source files
are left unchanged.

Default behavior is a dry run. Pass ``--run`` to call the model.

Examples:
    python code/eval/harness_refinement_3850_run.py
    python code/eval/harness_refinement_3850_run.py --variant v1_semicircle_minimal --run
    python code/eval/harness_refinement_3850_run.py --variant all --run --overwrite
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Variant:
    name: str
    note: str
    edits: tuple[dict[str, Any], ...]


SEMICIRCLE_MINIMAL = (
    {
        "tool": "add_semicircle",
        "param": "p1",
        "description": "First endpoint of diameter (on the boundary, not the centre)",
    },
    {
        "tool": "add_semicircle",
        "param": "p2",
        "description": "Second endpoint of diameter (on the boundary, not the centre)",
    },
)

SEMICIRCLE_VERBOSE = (
    {
        "tool": "add_semicircle",
        "param": "p1",
        "description": (
            "First endpoint of diameter; must be a named point on the boundary, "
            "not the centre. If the endpoint is missing, create it first."
        ),
    },
    {
        "tool": "add_semicircle",
        "param": "p2",
        "description": (
            "Second endpoint of diameter; must be a named point on the boundary, "
            "not the centre. If the endpoint is missing, create it first."
        ),
    },
)

ARC_SHORT = (
    {
        "tool": "add_arc",
        "description_append": (
            " For the shorter arc, choose start_pt and end_pt so the CCW sweep "
            "is the smaller sweep; otherwise swap them."
        ),
    },
)

SECTOR_SHORT = (
    {
        "tool": "add_sector",
        "description_append": (
            " For the smaller sector, choose start_pt and end_pt so the CCW "
            "sweep is the smaller sweep; otherwise swap them."
        ),
    },
)

# Negative control: edit a tool that has zero geometric relevance to the
# circle-packing area problem. Any trajectory shift vs v0_source under this
# variant is attributable to context-noise alone, not to semantic guidance.
NEGCONTROL = (
    {
        "tool": "add_function",
        "description_append": (
            " Useful for plotting algebraic curves over a domain."
        ),
    },
)

VARIANTS: dict[str, Variant] = {
    "v0_source": Variant(
        name="v0_source",
        note="Unmodified source ToolSpecs.",
        edits=(),
    ),
    "v1_semicircle_minimal": Variant(
        name="v1_semicircle_minimal",
        note="Minimal endpoint hint on add_semicircle p1/p2.",
        edits=SEMICIRCLE_MINIMAL,
    ),
    "v2_semicircle_verbose": Variant(
        name="v2_semicircle_verbose",
        note="Verbose endpoint guidance on add_semicircle p1/p2.",
        edits=SEMICIRCLE_VERBOSE,
    ),
    "v3_arc_short": Variant(
        name="v3_arc_short",
        note="Short-arc ordering hint on add_arc only.",
        edits=ARC_SHORT,
    ),
    "v4_sector_short": Variant(
        name="v4_sector_short",
        note="Small-sector ordering hint on add_sector only.",
        edits=SECTOR_SHORT,
    ),
    "v5_arc_sector_short": Variant(
        name="v5_arc_sector_short",
        note="Ordering hints on both add_arc and add_sector.",
        edits=ARC_SHORT + SECTOR_SHORT,
    ),
    "v6_semicircle_minimal_arc_sector": Variant(
        name="v6_semicircle_minimal_arc_sector",
        note="Minimal semicircle endpoint hint plus arc/sector ordering hints.",
        edits=SEMICIRCLE_MINIMAL + ARC_SHORT + SECTOR_SHORT,
    ),
    "v7_semicircle_verbose_arc_sector": Variant(
        name="v7_semicircle_verbose_arc_sector",
        note="Verbose semicircle endpoint guidance plus arc/sector ordering hints.",
        edits=SEMICIRCLE_VERBOSE + ARC_SHORT + SECTOR_SHORT,
    ),
    "v8_negcontrol": Variant(
        name="v8_negcontrol",
        note="Negative control: edit add_function (unused by 3850 trajectory).",
        edits=NEGCONTROL,
    ),
}


def _selected_variants(name: str) -> list[Variant]:
    if name == "all":
        return [VARIANTS[k] for k in sorted(VARIANTS)]
    if name not in VARIANTS:
        raise SystemExit(f"Unknown variant: {name}")
    return [VARIANTS[name]]


def _apply_variant_to_specs(variant: Variant):
    import symbolic.tools.geogebra_tools as tools_mod

    specs = copy.deepcopy(
        list(tools_mod.GLOBAL_GEOGEBRA_TOOLS)
        + list(tools_mod.QUERY_GEOGEBRA_TOOLS)
    )
    by_name = {spec.name: spec for spec in specs}

    for edit in variant.edits:
        tool = edit["tool"]
        if tool not in by_name:
            raise KeyError(f"Tool not found in ToolSpec catalog: {tool}")
        spec = by_name[tool]

        if "description" in edit:
            spec.description = edit["description"]
        if "description_append" in edit:
            suffix = edit["description_append"]
            if suffix not in spec.description:
                spec.description = spec.description.rstrip() + suffix
        if "param" in edit:
            param = edit["param"]
            if param not in spec.params:
                raise KeyError(f"Param not found: {tool}.{param}")
            spec.params[param].description = edit["description"]

    return specs


def _snapshot_source_specs(tools_mod) -> dict[str, dict]:
    """Freeze the in-memory ToolSpec catalog so we can verify it stays intact.

    The runner only patches the runtime ``build_gemini_tools`` factory and
    deepcopies before mutating; this snapshot lets us ASSERT that no code path
    accidentally mutated the source catalog after a run.
    """
    snap: dict[str, dict] = {}
    all_specs = (
        list(tools_mod.GLOBAL_GEOGEBRA_TOOLS)
        + list(tools_mod.QUERY_GEOGEBRA_TOOLS)
        + list(tools_mod.RENDER_GEOGEBRA_TOOLS)
    )
    for spec in all_specs:
        snap[spec.name] = {
            "description": spec.description,
            "params": {
                k: (p.description, p.param_type, p.required)
                for k, p in spec.params.items()
            },
        }
    return snap


def _verify_source_specs(tools_mod, snapshot: dict[str, dict]) -> None:
    all_specs = (
        list(tools_mod.GLOBAL_GEOGEBRA_TOOLS)
        + list(tools_mod.QUERY_GEOGEBRA_TOOLS)
        + list(tools_mod.RENDER_GEOGEBRA_TOOLS)
    )
    for spec in all_specs:
        ref = snapshot.get(spec.name)
        if ref is None:
            raise RuntimeError(
                f"Source schema polluted: new tool {spec.name!r} appeared mid-run"
            )
        if spec.description != ref["description"]:
            raise RuntimeError(
                f"Source schema polluted: {spec.name}.description mutated"
            )
        for k, p in spec.params.items():
            ref_p = ref["params"].get(k)
            if ref_p is None:
                raise RuntimeError(
                    f"Source schema polluted: {spec.name}.params[{k!r}] new key"
                )
            if (p.description, p.param_type, p.required) != ref_p:
                raise RuntimeError(
                    f"Source schema polluted: {spec.name}.params[{k!r}] mutated"
                )


def _snapshot_source_specs(tools_mod) -> dict[str, dict]:
    """Freeze the in-memory ToolSpec catalog (GLOBAL + QUERY only — RENDER is
    never disclosed to the model in construct mode and is out of scope for
    this ablation). Used to assert that no run mutates the source schema."""
    snap: dict[str, dict] = {}
    in_scope = (
        list(tools_mod.GLOBAL_GEOGEBRA_TOOLS)
        + list(tools_mod.QUERY_GEOGEBRA_TOOLS)
    )
    for spec in in_scope:
        snap[spec.name] = {
            "description": spec.description,
            "params": {
                k: (p.description, p.param_type, p.required)
                for k, p in spec.params.items()
            },
        }
    return snap


def _verify_source_specs(tools_mod, snapshot: dict[str, dict]) -> None:
    in_scope = (
        list(tools_mod.GLOBAL_GEOGEBRA_TOOLS)
        + list(tools_mod.QUERY_GEOGEBRA_TOOLS)
    )
    for spec in in_scope:
        ref = snapshot.get(spec.name)
        if ref is None:
            raise RuntimeError(
                f"Source schema polluted: new tool {spec.name!r} appeared mid-run"
            )
        if spec.description != ref["description"]:
            raise RuntimeError(
                f"Source schema polluted: {spec.name}.description mutated"
            )
        for k, p in spec.params.items():
            ref_p = ref["params"].get(k)
            if ref_p is None:
                raise RuntimeError(
                    f"Source schema polluted: {spec.name}.params[{k!r}] new key"
                )
            if (p.description, p.param_type, p.required) != ref_p:
                raise RuntimeError(
                    f"Source schema polluted: {spec.name}.params[{k!r}] mutated"
                )


def _make_gemini_builder(variant: Variant):
    patched_specs = _apply_variant_to_specs(variant)

    def build_gemini_tools_overlay(include_render: bool = False):
        from google.genai import types
        import symbolic.tools.geogebra_tools as tools_mod

        specs = list(patched_specs)
        if include_render:
            specs = specs + list(tools_mod.RENDER_GEOGEBRA_TOOLS)

        decls = []
        for spec in specs:
            properties = {}
            required = []
            for key, param in spec.params.items():
                g_type = {
                    "string": "STRING",
                    "number": "NUMBER",
                    "integer": "INTEGER",
                }[param.param_type]
                properties[key] = types.Schema(
                    type=g_type,
                    description=param.description,
                )
                if param.required:
                    required.append(key)
            decls.append(
                types.FunctionDeclaration(
                    name=spec.name,
                    description=spec.description,
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=properties,
                        required=required,
                    ),
                )
            )
        print(
            f"[harness_refinement] {variant.name}: built "
            f"{len(decls)} runtime ToolSpecs"
        )
        return [types.Tool(function_declarations=decls)]

    return build_gemini_tools_overlay


def _write_manifest(out_dir: Path, variant: Variant,
                    args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "variant": variant.name,
        "note": variant.note,
        "edits": list(variant.edits),
        "dataset": "mathverse",
        "problem_id": args.problem_id,
        "data_dir": str(args.data_dir),
        "model": args.model,
        "thinking": args.thinking,
        "ttft_timeout": args.ttft_timeout,
        "mode": "construct",
        "source_policy": (
            "Runtime ToolSpec overlay only; symbolic/tools source files are "
            "not modified by this runner. Source schema (GLOBAL + QUERY) is "
            "asserted unchanged after each run."
        ),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "variant_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )


def _run_variant(variant: Variant, args: argparse.Namespace) -> None:
    """Run one variant serially in this process.

    Patches ``build_gemini_tools`` on both the source tools module and the
    eval driver module, runs the eval driver, then unconditionally restores
    the original module-level bindings via ``try/finally``. After the run,
    asserts that the source ToolSpec catalog is byte-identical to the
    pre-run snapshot, so leftover state cannot leak into the next variant.

    With temperature=0 a single run per variant is treated as canonical;
    if a variant trajectory looks degenerate, iterate the description text
    in this file and rerun that variant rather than repeating it.
    """
    import symbolic.tools.geogebra_tools as tools_mod
    import test_agentic_geo_constructer as eval_mod

    snapshot = _snapshot_source_specs(tools_mod)

    builder = _make_gemini_builder(variant)
    orig_tools_builder = tools_mod.build_gemini_tools
    orig_eval_builder = eval_mod.build_gemini_tools

    out_dir = Path(args.output_root) / variant.name
    _write_manifest(out_dir, variant, args)

    argv = [
        "test_agentic_geo_constructer.py",
        "--dataset", "mathverse",
        "--data_dir", str(args.data_dir),
        "--id", str(args.problem_id),
        "--mode", "construct",
        "--model", args.model,
        "--thinking", args.thinking,
        "--ttft-timeout", str(args.ttft_timeout),
        "--workers", "1",
        "--out-dir", str(out_dir),
    ]
    if args.save_screenshot_per_turn:
        argv.append("--save-screenshot-per-turn")
    if not args.overwrite:
        argv.append("--skip-done")

    old_argv = sys.argv[:]
    try:
        print(f"\n{'=' * 72}")
        print(f"  Running {variant.name}")
        print(f"  Output: {out_dir}")
        print(f"{'=' * 72}")
        tools_mod.build_gemini_tools = builder
        eval_mod.build_gemini_tools = builder
        sys.argv = argv
        eval_mod.main()
    finally:
        sys.argv = old_argv
        tools_mod.build_gemini_tools = orig_tools_builder
        eval_mod.build_gemini_tools = orig_eval_builder

    _verify_source_specs(tools_mod, snapshot)
    print(f"[harness_refinement] {variant.name}: "
          "source schema integrity verified")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MathVerse 3850 runtime ToolSpec-overlay runner"
    )
    parser.add_argument("--variant", default="all", choices=["all", *VARIANTS.keys()])
    parser.add_argument("--run", action="store_true",
                        help="Actually call the model. Without this, only prints variants.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing per-variant result files.")
    parser.add_argument("--problem-id", default="3850")
    parser.add_argument("--data-dir", default="/data/mathverse", type=Path)
    parser.add_argument("--model", default="gemini-3-flash-preview@medium")
    parser.add_argument("--thinking", default="medium",
                        choices=["off", "minimal", "low", "medium", "high"])
    parser.add_argument("--ttft-timeout", type=int, default=120)
    parser.add_argument(
        "--output-root",
        default=(
            Path("Writing")
            / "figs"
            / "harness_ablation"
            / "runs"
            / "mathverse_3850"
        ),
        type=Path,
    )
    parser.add_argument("--save-screenshot-per-turn", action="store_true")
    args = parser.parse_args()

    selected = _selected_variants(args.variant)
    print("Selected harness-refinement variants:")
    for variant in selected:
        print(f"  {variant.name}: {variant.note}")
        for edit in variant.edits:
            target = edit["tool"] + (f".{edit['param']}" if "param" in edit else "")
            text = edit.get("description") or edit.get("description_append", "")
            print(f"    - {target}: {text}")

    if not args.run:
        print("\nDry run only. Add --run to execute model calls.")
        print(f"(would launch {len(selected)} variant(s), serially)")
        return

    # Refuse to write into the legacy 3850_v{1,2,3}/ trees so that
    # previously-collected runs are never overwritten.
    out_root = Path(args.output_root).resolve()
    for legacy in ("3850_v1", "3850_v2", "3850_v3"):
        legacy_path = (Path("Writing") / "figs" / "harness_ablation"
                       / legacy).resolve()
        if out_root == legacy_path or legacy_path in out_root.parents:
            raise SystemExit(
                f"Refusing to run: --output-root resolves to or under the "
                f"legacy directory {legacy_path}. Pick a different path."
            )

    for variant in selected:
        _run_variant(variant, args)


if __name__ == "__main__":
    main()
