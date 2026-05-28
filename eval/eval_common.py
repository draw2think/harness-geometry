"""Shared infrastructure for eval scripts.

Both eval/test_agentic_geo_constructer.py (agentic multi-turn eval) and
eval/eval_baseline.py (single-turn baseline eval) import from this module.

Contents
--------
  - Answer parsing    : parse_answer, eval_symbolic, validate
  - Image helpers     : load_image, image_part_gemini, image_content_openai,
                        image_content_anthropic
  - Question helpers  : format_choices

Dataset loaders live in eval/loaders/ (the DATASET_LOADERS registry).
"""

import json
import math
import os
import random
import re
from pathlib import Path

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False


# ── Question / choice formatting ─────────────────────────────────────────────

def format_choices(choices: list) -> str:
    """Format choices as '(A) x  (B) y  (C) z  (D) w'."""
    return "  ".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))


# ── Image helpers ────────────────────────────────────────────────────────────

def load_image(path: str) -> tuple[bytes, str]:
    """Read image file and return (raw_bytes, mime_type)."""
    p = Path(path)
    data = p.read_bytes()
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return data, mime


def image_part_gemini(path: str):
    """Gemini inline image Part."""
    from google.genai import types
    data, mime = load_image(path)
    return types.Part(inline_data=types.Blob(mime_type=mime, data=data))


def image_content_openai(path: str) -> dict:
    """OpenAI-compatible image content block (base64 data URL)."""
    import base64
    data, mime = load_image(path)
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}}


def image_content_anthropic(path: str) -> dict:
    """Anthropic image content block (base64)."""
    import base64
    data, mime = load_image(path)
    b64 = base64.b64encode(data).decode()
    return {"type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64}}


# ── Answer parsing + validation ──────────────────────────────────────────────

def _extract_boxed_value(text: str) -> str | None:
    """Extract the last balanced \\boxed{...} content; strip leading 'NAME = '."""
    # Find all \boxed{...} with possible nested braces (depth <= 3 is enough)
    results = []
    i = 0
    while True:
        j = text.find('\\boxed', i)
        if j < 0:
            break
        k = text.find('{', j)
        if k < 0:
            break
        depth = 1
        k += 1
        start = k
        while k < len(text) and depth > 0:
            if text[k] == '\\':
                k += 2  # skip escape
                continue
            if text[k] == '{':
                depth += 1
            elif text[k] == '}':
                depth -= 1
            k += 1
        if depth == 0:
            results.append(text[start:k-1])
        i = k
    if not results:
        return None
    val = results[-1].strip()
    # Strip "NAME = " prefix (e.g. "PO_1 = 2\\sqrt3", "QE=\\sqrt{19}/4")
    eq = val.rfind('=')
    if eq >= 0 and eq < len(val) - 1:
        val = val[eq+1:].strip()
    return val or None


def _extract_inline_value(text: str) -> str | None:
    """Extract final answer from trailing \\(...=VAL\\) or \\[...=VAL\\]."""
    # Use last inline / display math block that contains an '=' sign
    patterns = [
        r'\\\[(.+?)\\\]',
        r'\\\((.+?)\\\)',
    ]
    import re as _re
    candidates = []
    for pat in patterns:
        for m in _re.finditer(pat, text, _re.DOTALL):
            content = m.group(1)
            if '=' in content:
                candidates.append((m.start(), content))
    if not candidates:
        return None
    # Take the latest
    candidates.sort()
    _, last = candidates[-1]
    tail = last.rsplit('=', 1)[-1].strip()
    # Strip trailing punctuation and LaTeX text wrappers
    tail = _re.sub(r'\\text\s*\{[^}]*\}', '', tail).strip()
    tail = tail.rstrip('.,;。，')
    return tail or None


def parse_answer(text: str) -> dict | None:
    idx = text.find("ANSWER:")
    if idx < 0:
        # Fallbacks: accept \boxed{...} (common LaTeX math convention) and
        # trailing inline/display math such as "\(QE=\frac{\sqrt{19}}{4}\)".
        boxed = _extract_boxed_value(text)
        if boxed:
            # Try numeric; fall back to text/expression.
            try:
                return {"value": float(boxed), "type": "numerical"}
            except ValueError:
                return {"value": boxed, "type": "text"}
        inline = _extract_inline_value(text)
        if inline:
            try:
                return {"value": float(inline), "type": "numerical"}
            except ValueError:
                return {"value": inline, "type": "text"}
        return None
    after = text[idx + len("ANSWER:"):]
    brace = after.find("{")
    # Try JSON parse first
    if brace >= 0:
        raw = after[brace:]
        # Restore Python escape chars that were once LaTeX commands:
        # \f (formfeed \x0c) → \f, \b (backspace \x08) → \b,
        # \t (tab \x09) → \t, \r (CR \x0d) → \r, \n (LF \x0a) → \n
        # These get corrupted when SDK JSON-decodes the LLM response string.
        raw = raw.replace('\x0c', '\\f')   # \frac, \forall
        raw = raw.replace('\x08', '\\b')   # \bar, \binom
        raw = raw.replace('\x09', '\\t')   # \text, \theta
        raw = raw.replace('\x0d', '\\r')   # \right, \rho
        # Note: \n (newline) is ambiguous: it could be a real newline or \nu.
        # Don't restore \n to avoid breaking multi-line text.

        decoder = json.JSONDecoder()
        # LaTeX commands like \frac, \bar collide with JSON escapes
        # (\f = form feed, \b = backspace, \n = newline, \r = CR, \t = tab).
        # Pre-escape ALL backslashes that look like LaTeX commands (followed
        # by a letter), so \frac, \sqrt, \pi etc. survive JSON parsing.
        escaped = re.sub(r'\\(?=[a-zA-Z])', r'\\\\', raw)
        try:
            obj, _ = decoder.raw_decode(escaped)
            return obj
        except json.JSONDecodeError:
            pass
        # Fallback: try raw (may lose LaTeX backslashes but still parse)
        try:
            obj, _ = decoder.raw_decode(raw)
            return obj
        except json.JSONDecodeError:
            pass
    # Mid-fallback: broken JSON; extract the "value" field via regex.
    if brace is not None and brace >= 0:
        m = re.search(r'"value"\s*:\s*"?([^",}]+)"?', after)
        if m:
            val_str = m.group(1).strip()
            try:
                return {"value": float(val_str), "type": "numerical"}
            except ValueError:
                return {"value": val_str, "type": "text"}
    # Fallback: plain text answer such as "ANSWER: B" or "ANSWER: 42.5".
    val = after.strip().split("\n")[0].strip()
    if val:
        # Try numeric
        try:
            return {"value": float(val), "type": "numerical"}
        except ValueError:
            pass
        return {"value": val, "type": "text"}
    return None


def _normalize_latex(s: str) -> str:
    """Normalize LaTeX quirks before feeding to sympy parse_latex."""
    s = s.strip()
    s = s.replace('\x0c', chr(92) + 'f')  # form-feed -> backslash+f (restores \frac)
    # Unicode √ -> \sqrt  (models sometimes emit Unicode radical sign)
    s = s.replace('√', '\\sqrt')
    # Unicode π -> \pi
    s = s.replace('π', '\\pi')
    # \sqrt N or \sqrtN (no braces) -> \sqrt{N}  -- parse_latex requires braces
    s = re.sub(r'\\sqrt\s*(\d+(?:\.\d+)?)', r'\\sqrt{\1}', s)
    # Implicit multiplication: 2\sqrt{3} -> 2 \cdot \sqrt{3}
    s = re.sub(r'(\d)\s*\\sqrt', r'\1 \\sqrt', s)
    # Implicit multiplication: 9\pi -> 9 \pi  (parse_latex needs space/cdot)
    s = re.sub(r'(\d)\s*\\pi', r'\1 \\pi', s)
    # Strip trailing unit suffixes: cm², m, cm, km, mm, °, etc.
    s = re.sub(r'\s*(cm[²2³3]?|m[²2³3]?|km|mm|°)\s*$', '', s)
    return s


def verify_math(gold_str: str, model_str: str,
                float_rounding: int = 2) -> bool | None:
    """Compare two math expressions using HuggingFace math-verify.

    Wraps raw expressions in $...$ for LaTeX parsing, then uses sympy-based
    symbolic equivalence. Returns None if math-verify is not installed or
    both sides fail to parse.
    """
    if not _HAS_MATH_VERIFY:
        return None
    # Normalize Unicode → LaTeX and wrap in $...$ for parsing
    def _wrap(s):
        s = _normalize_latex(s)
        if not s:
            return s
        if s.startswith(('$', r'\boxed', r'\(', r'\[')):
            return s
        return f'${s}$'
    try:
        g = mv_parse(_wrap(gold_str))
        a = mv_parse(_wrap(model_str))
        if g and a:
            return mv_verify(g, a, float_rounding=float_rounding)
    except Exception:
        pass
    return None


def eval_symbolic(expr: str) -> float:
    """Evaluate a LaTeX or plain numeric expression to float.

    Uses sympy parse_latex as primary parser, falls back to manual
    regex-based conversion for edge cases.
    """
    s = str(expr).strip()
    # Fast path: plain number
    try:
        return float(s)
    except ValueError:
        pass
    # Primary: sympy parse_latex
    try:
        from sympy.parsing.latex import parse_latex
        parsed = parse_latex(_normalize_latex(s))
        return float(parsed.evalf())
    except Exception:
        pass
    # Fallback: manual regex conversion (handles non-standard notation)
    ns = {"sqrt": math.sqrt, "pi": math.pi, "e": math.e,
          "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "asin": math.asin, "acos": math.acos, "atan": math.atan,
          "atan2": math.atan2, "log": math.log, "exp": math.exp,
          "abs": abs, "__builtins__": {}}
    s2 = _normalize_latex(s)
    # \frac{A}{B} -> (A)/(B)
    s2 = re.sub(r'\\frac\s*\{\s*([^}]+?)\s*\}\s*\{\s*([^}]+?)\s*\}',
                r'(\1)/(\2)', s2)
    s2 = re.sub(r'\\?sqrt\s*\{\s*([^}]+?)\s*\}', r'sqrt(\1)', s2)
    s2 = s2.replace('\\pi', ' pi ')
    s2 = s2.replace('\\', '')
    s2 = re.sub(r'(\d)\s+(pi|sqrt\()', r'\1*\2', s2)
    s2 = re.sub(r'\)\s*\(', r')*(', s2)
    s2 = s2.replace('^', '**')
    return eval(s2, ns)  # noqa: S307


def _norm_choice_str(s: str) -> str:
    """Normalize for choice string matching: unify symbols, strip units/whitespace."""
    s = s.strip()
    # Unify Unicode ↔ LaTeX symbols
    s = s.replace('√', '\\sqrt').replace('π', '\\pi')
    # Strip LaTeX braces and whitespace
    s = re.sub(r'[\s{}]', '', s)
    # Strip units: cm², m, km, etc.
    s = re.sub(r'(cm[²2³3]?|m[²2³3]?|km|mm|°)$', '', s)
    return s.lower()


def _resolve_choice_letter(v, prob: dict) -> float | None:
    """Map a choice letter like 'B', '(B)', or '(C) 31.416' to its numeric value."""
    if not isinstance(v, str):
        return None
    m = re.match(r'^\(?([A-Ea-e])\)?(?:\s.*)?$', v.strip())
    if m:
        choices = prob.get("choices", [])
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            c = choices[idx]
            try:
                return float(c)
            except (ValueError, TypeError):
                pass
            # LaTeX symbolic choice like "4 \sqrt 6 + 2 \sqrt{14}"
            try:
                return eval_symbolic(c)
            except Exception:
                pass
    return None


def _strip_units(s: str) -> str:
    """Strip common units/LaTeX wrappers from a value string for numeric comparison."""
    s = str(s).strip()
    s = s.replace('掳', '°')                            # Repair mojibake degree signs.
    s = re.sub(r'\\mathrm\{[^}]*\}', '', s)           # \mathrm{~cm}
    s = re.sub(r'\\text\s*\{[^}]*\}', '', s)          # \text{ units }
    # Bare {unit} wrappers (non-LaTeX-standard but common in this dataset):
    # Strip units such as {inches}, {cm}, {m}, {meters} only when content is alphabetic.
    s = re.sub(r'\{(?:inches|inch|cm|mm|km|m|metres|meters|units|sq|degrees|deg)\}',
               '', s, flags=re.IGNORECASE)
    s = re.sub(r'\^\{[^}]*\}', '', s)                  # ^{2}
    s = re.sub(r'\^[²2³3]', '', s)                     # bare ^2 or ² after unit strip
    s = s.replace('$', '').replace('\\circ', '')
    s = re.sub(r'[°\s]*(cm[²2³3]?|mm|km|m|metres|meters|units|T)\s*$',
               '', s, flags=re.IGNORECASE)
    s = s.rstrip('°')                                  # standalone ° (no unit word)
    return s.strip()


def _clean_gt_value(raw: str) -> str:
    """Extract core numeric/symbolic value from a GT answer string.

    Examples:
        'Circumference $=50.27 \\mathrm{~cm}$' → '50.27'
        'Area $=9 \\pi \\mathrm{cm}^{2}$'      → '9 \\pi'
        '$r=24.4$'                               → '24.4'
        '$$r=7.00$$'                             → '7.00'
        '$D=13.37$'                              → '13.37'
        'S 23.5 ^\\circ W'                       → '23.5'
    """
    s = raw.strip()
    # Strip all $ delimiters (may appear at edges or mid-string: "Circumference $=50.27...$")
    s = s.replace('$', '').strip()
    # Strip label prefix: "Area =", "Circumference =", "r=", "C=", "Center ="
    s = re.sub(r'^[A-Za-z\s]*=\s*', '', s)
    # Strip LaTeX units
    s = _strip_units(s)
    # Strip bearing prefix
    s = re.sub(r'^[NSEW]\s*', '', s)
    return s.strip()


def _compass_to_degrees(raw: str) -> float | None:
    """Convert compass bearing notation to numeric degrees (0-360).

    Examples:
        'S 23.5 ^\\circ W' → 180 + 23.5 = 203.5
        'N 34° W'          → 360 - 34 = 326
        '090^\\circ T'     → 90
        '053 ^\\circ T'    → 53
    Returns None if not a compass bearing.
    """
    s = raw.strip().replace('$', '').replace('\\circ', '°').replace('^°', '°')
    s = re.sub(r'\s+', ' ', s).strip()
    # True bearing: NNN° T  (e.g. "090° T", "053° T")
    m = re.match(r'^(\d+(?:\.\d+)?)\s*°?\s*T$', s)
    if m:
        return float(m.group(1))
    # Quadrant bearing: S 23.5° W, N 34° W, etc.
    m = re.match(r'^([NSEW])\s*(\d+(?:\.\d+)?)\s*°?\s*([NSEW])$', s)
    if m:
        start, angle, end = m.group(1), float(m.group(2)), m.group(3)
        if start == 'N' and end == 'E': return angle
        if start == 'N' and end == 'W': return 360 - angle
        if start == 'S' and end == 'E': return 180 - angle
        if start == 'S' and end == 'W': return 180 + angle
    return None


def validate(answer: dict | None, prob: dict,
             judge_fn=None) -> tuple[bool, str]:
    if answer is None:
        return False, "no answer emitted"
    v = answer.get("value")

    # ── Fast path: MC letter matching ──
    # If prob has answer_label and model emitted a letter, compare directly.
    answer_label = prob.get("answer_label")
    if answer_label and isinstance(v, str):
        m = re.match(r'^\(?([A-Ea-e])\)?(?:\s.*)?$', v.strip())
        if m and m.group(1).upper() == answer_label.upper():
            return True, f"letter match: {answer_label}"
        if m:
            return False, f"letter {m.group(1).upper()} vs {answer_label}"

    # ── Numeric value → reverse-lookup choices for MC outputs. ──
    if answer_label and not isinstance(v, str):
        choices = prob.get("choices", [])
        try:
            actual = float(v)
            best_i, best_diff = None, float('inf')
            tol = prob.get("tolerance", 1.0)
            for i, c in enumerate(choices):
                try:
                    cv = float(_strip_units(str(c)))
                except (ValueError, TypeError):
                    # LaTeX choice like "\\frac{3}{5}"; try eval_symbolic.
                    try:
                        cv = eval_symbolic(str(c))
                    except Exception:
                        continue
                diff = abs(actual - cv)
                if diff <= tol and diff < best_diff:
                    best_diff = diff
                    best_i = i
            if best_i is not None:
                idx_label = chr(ord('A') + best_i)
                ok = idx_label == answer_label.upper()
                return ok, f"value→choice: {actual} → {idx_label}" + \
                    (f" vs {answer_label}" if not ok else "")
        except (ValueError, TypeError):
            pass

    # ── math-verify symbolic comparison (preferred for LaTeX expressions) ──
    exp_raw = prob.get("expected_raw") or prob.get("raw_answer")
    if exp_raw and v is not None and _HAS_MATH_VERIFY:
        mv_result = verify_math(str(exp_raw), str(v))
        if mv_result is True:
            return True, f"math-verify: {v!r} ≡ {exp_raw!r}"
        # mv_result False or None → fall through to legacy comparison

    # ── Numeric comparison ──
    exp = prob.get("expected")
    # If expected is None, try extracting from expected_raw
    if exp_raw is None:
        exp_raw = prob.get("expected_raw")
    if exp is None and exp_raw:
        cleaned = _clean_gt_value(exp_raw)
        try:
            exp = float(cleaned)
        except ValueError:
            try:
                exp = float(eval_symbolic(cleaned))
            except Exception:
                pass

    if exp is not None:
        tol = prob.get("tolerance", 1.0)
        try:
            actual = eval_symbolic(v) if isinstance(v, str) else float(v)
        except Exception:
            resolved = _resolve_choice_letter(v, prob)
            if resolved is not None:
                actual = resolved
            else:
                actual = None
        if actual is not None:
            ok = abs(actual - float(exp)) <= tol
            return ok, f"{round(actual, 4)} vs {exp}  (tol={tol})"

        # ── Choice string matching ──
        # eval_symbolic failed; try matching raw answer against choices.
        # e.g. answer="2√{3}" matches choice "2√{3}" → index D → compare label
        choices = prob.get("choices", [])
        if isinstance(v, str) and choices:
            v_n = _norm_choice_str(v)
            for i, c in enumerate(choices):
                if _norm_choice_str(str(c)) == v_n:
                    idx_label = chr(ord('A') + i)
                    if answer_label:
                        ok = idx_label == answer_label.upper()
                        return ok, f"choice match: {v!r} → {idx_label}" + \
                            (f" vs {answer_label}" if not ok else "")
                    # No label; try evaluating the choice itself.
                    try:
                        actual = eval_symbolic(str(c))
                        ok = abs(actual - float(exp)) <= tol
                        return ok, f"{round(actual, 4)} vs {exp}  (tol={tol}, via choice match)"
                    except Exception:
                        break

    # ── Compass bearing comparison ──
    if exp_raw:
        gt_deg = _compass_to_degrees(exp_raw)
        if gt_deg is not None:
            tol = prob.get("tolerance", 1.0)
            try:
                model_deg = float(v)
                ok = abs(model_deg - gt_deg) <= tol
                return ok, f"bearing: {model_deg} vs {gt_deg}  (tol={tol})"
            except (ValueError, TypeError):
                pass

    # ── Symbolic comparison against expected_raw ──
    # Try eval_symbolic on both model answer and cleaned GT
    if exp_raw:
        cleaned_gt = _clean_gt_value(exp_raw)
        try:
            model_val = eval_symbolic(str(v)) if isinstance(v, str) else float(v)
            gt_val = eval_symbolic(cleaned_gt)
            tol = prob.get("tolerance", 1.0)
            ok = abs(float(model_val) - float(gt_val)) <= tol
            return ok, f"{round(float(model_val), 4)} vs {round(float(gt_val), 4)}  (tol={tol}, via raw clean)"
        except Exception:
            pass

    # ── Equation/expression normalized comparison ──
    if exp_raw and isinstance(v, str) and '=' in str(v):
        def _norm_eq(s):
            s = s.replace('\x0c', '\\f')
            s = s.replace('$', '').strip()
            # \frac{A}{B} -> A/B
            s = re.sub(r'\\frac\s*\{\s*([^}]+?)\s*\}\s*\{\s*([^}]+?)\s*\}',
                        r'(\1)/(\2)', s)
            s = s.replace('\\left', '').replace('\\right', '')
            s = re.sub(r'[\s{}\\]', '', s)
            # N^2 -> N**2 for consistent comparison
            s = s.replace('^', '**')
            return s.lower()
        if _norm_eq(str(v)) == _norm_eq(exp_raw):
            return True, f"equation match: {v!r} ≡ {exp_raw!r}"

    # ── String fallback: compare raw GT against model output ──
    if exp_raw and isinstance(v, str):
        # Normalize both: strip whitespace, degree symbols, case, $ delimiters
        def _norm(s):
            s = s.replace('\x0c', '\\f')  # restore form-feed → \frac
            s = s.replace('\n', ',')       # newline → comma (multi-select)
            s = re.sub(r'[\s°^\\circ{}$]+', '', s).lower()
            return s
        # Also try comparing model output against cleaned GT
        ok = _norm(v) == _norm(exp_raw) or _norm(v) == _norm(_clean_gt_value(exp_raw))
        if ok:
            return True, f"{v!r} vs {exp_raw!r} (string)"

    # ── LLM judge fallback ──
    if judge_fn and exp_raw:
        try:
            ok = judge_fn(str(v), exp_raw)
            tag = "equivalent" if ok else "different"
            return ok, f"LLM judge: {str(v)!r} vs {exp_raw!r} → {tag}"
        except Exception:
            pass

    # ── Final: string fallback that returns False ──
    if exp_raw and isinstance(v, str):
        return False, f"{v!r} vs {exp_raw!r} (string)"

    return False, f"cannot evaluate {v!r}"


# ── Dataset loaders (refactored to eval/loaders/) ────────────────────────────
# Each loader is now in its own file under eval/loaders/.
# Import directly: from loaders import DATASET_LOADERS




# ═══════════════════════════════════════════════════════════════════════════════
#  MathCanvas Answer Judge (GPT-4.1 / proxy)
# ═══════════════════════════════════════════════════════════════════════════════

# Weights from MathCanvas official: exponentially increasing per sub-question
MATHCANVAS_SUB_WEIGHTS = {
    1: [1.0],
    2: [0.4348, 0.5652],
    3: [0.2506, 0.3258, 0.4236],
    4: [0.1616, 0.2101, 0.2732, 0.3551],
}

_MATHCANVAS_JUDGE_PROMPT = (
    Path(__file__).parent / "prompts" / "mathcanvas_judge.txt"
)


def judge_mathcanvas_answer(
    question: str,
    gt_answer: str,
    pred_answer: str,
    judge_model: str = "gpt-4.1",
) -> dict:
    """Judge a MathCanvas answer using LLM (GPT-4.1).

    Uses MathCanvas official evaluation protocol: parse GT and prediction
    into sub-answer lists, judge correctness per part, compute weighted score.

    Returns dict with keys:
        analysis, gt_answers, pred_answers, correctness,
        complete_score (0 or 1), weighted_score (0.0-1.0)
    """
    from openai import OpenAI
    from symbolic.utils.env_loader import load_env_file
    load_env_file()

    # Route: USE_JUDGE_PROXY or USE_PROXY → proxy; else → official OpenAI
    _proxy = (os.environ.get("USE_JUDGE_PROXY", "").strip() == "1"
              or os.environ.get("USE_PROXY", "").strip() == "1")
    if _proxy:
        client = OpenAI(
            api_key=os.environ["PROXY_API_KEY"],
            base_url=os.environ.get("PROXY_BASE_URL", "") + "/v1",
        )
    else:
        client = OpenAI()

    # Load prompt template
    prompt_path = _MATHCANVAS_JUDGE_PROMPT
    if prompt_path.exists():
        template = prompt_path.read_text()
    else:
        # Fallback inline prompt (simplified version of official)
        template = (
            "You are a math answer evaluator. Compare the predicted answer "
            "against the ground truth and output JSON with fields: "
            "analysis, gt_answers (list), pred_answers (list), correctness (list of bool).\n\n"
            "INPUT DATA:\n{input_data}\n\nOutput valid JSON only."
        )

    # Build input
    input_dict = {
        "question_text": question,
        "ground_truth_answer": gt_answer,
        "prediction_solution": pred_answer,
    }
    input_str = json.dumps(input_dict, ensure_ascii=False, indent=2)
    user_msg = template.format(input_data=input_str)

    for attempt in range(3):
        try:
            params = {
                "model": judge_model,
                "messages": [{"role": "user", "content": user_msg}],
                "max_completion_tokens": 4096,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            # gpt-4.1 doesn't support reasoning_effort
            if "gpt-4.1" not in judge_model:
                params["reasoning_effort"] = "low"

            resp = client.chat.completions.create(**params)
            raw = resp.choices[0].message.content

            # Parse JSON
            text = raw.split("```json")[-1].split("```")[0]
            text = text.replace("\u2018", "'").replace("\u2019", "'")
            text = text.replace("\u201c", '"').replace("\u201d", '"')
            result = json.loads(text)

            # Compute scores
            correctness = result.get("correctness", [])
            n = len(correctness)

            # Complete score: all correct
            complete = 1 if all(correctness) else 0

            # Weighted score
            if n in MATHCANVAS_SUB_WEIGHTS:
                weights = MATHCANVAS_SUB_WEIGHTS[n]
                weighted = sum(w for c, w in zip(correctness, weights) if c)
            elif n == 0:
                weighted = 0.0
            else:
                # Fallback: equal weights
                weighted = sum(correctness) / n

            result["complete_score"] = complete
            result["weighted_score"] = round(weighted, 4)
            return result

        except Exception as e:
            if attempt == 2:
                return {
                    "analysis": f"Judge failed after 3 attempts: {e}",
                    "gt_answers": [], "pred_answers": [],
                    "correctness": [],
                    "complete_score": 0, "weighted_score": 0.0,
                    "error": str(e),
                }
            continue
