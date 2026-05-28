#!/usr/bin/env python3
"""
Re-score existing *_result.json files using the current parse_answer + validate.

Usage:
  python eval/rescore_from_logs.py --dataset solidgeo --model gpt-5.4@high
  python eval/rescore_from_logs.py --dataset solidgeo --model 'gemini-3-flash-preview@high' --mode all

Only touches result.json files; never re-queries the model. The response text
is recovered from the paired *_log.txt file.
"""
import argparse, json, re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eval_common import parse_answer, validate
from loaders import DATASET_LOADERS


def _recover_response(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(errors='ignore')
    m = re.search(r'── Response ──\n(.*?)\n── Result ──', text, re.DOTALL)
    if m:
        resp = m.group(1).strip()
        if resp and resp != "(no response)":
            return resp
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASET_LOADERS))
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--model', required=True,
                    help='Model slug as used in result filenames, e.g. gpt-5.4@high')
    ap.add_argument('--mode', choices=['baseline', 'construct', 'all'], default='all')
    ap.add_argument('--dry-run', action='store_true', help='Show diffs without writing')
    ap.add_argument('--level', default=None, help='Optional Level filter (SolidGeo)')
    args = ap.parse_args()

    # Build problem map {qa_id: prob_dict} for validation
    loader = DATASET_LOADERS[args.dataset]
    kwargs = {'data_dir': args.data_dir, 'sample': None}
    if args.level:
        kwargs['level'] = args.level
    problems = {str(p['id']): p for p in loader(**kwargs)}

    # Scan result files
    eval_root = ROOT / args.dataset
    slug = args.model
    mode_suffixes = []
    if args.mode in ('baseline', 'all'):
        mode_suffixes.append(('baseline', f'{slug}_baseline_result.json', f'{slug}_baseline_log.txt'))
    if args.mode in ('construct', 'all'):
        mode_suffixes.append(('construct', f'{slug}_result.json', f'{slug}_log.txt'))

    for mode, res_name, log_name in mode_suffixes:
        updated = flipped_pass = flipped_fail = unchanged = missing = 0
        flipped_to_pass = []
        for result_path in eval_root.rglob(res_name):
            qid = result_path.parent.name.replace('prob_', '')
            prob = problems.get(qid)
            if not prob:
                continue
            d = json.loads(result_path.read_text())
            old_passed = bool(d.get('passed'))
            # Recover response from log
            log_path = result_path.parent / log_name
            resp = _recover_response(log_path)
            if not resp:
                missing += 1
                continue
            new_answer = parse_answer(resp)
            new_passed, new_detail = validate(new_answer, prob)
            if new_passed != old_passed or new_detail != (d.get('detail') or ''):
                updated += 1
                if new_passed and not old_passed:
                    flipped_pass += 1
                    flipped_to_pass.append(qid)
                elif old_passed and not new_passed:
                    flipped_fail += 1
                if not args.dry_run:
                    d['answer_raw'] = new_answer
                    d['passed'] = bool(new_passed)
                    d['detail'] = new_detail
                    result_path.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            else:
                unchanged += 1

        print(f'\n=== {slug} [{mode}] ===')
        print(f'  updated:       {updated}  (pass→ {flipped_pass},  fail→ {flipped_fail})')
        print(f'  unchanged:     {unchanged}')
        print(f'  missing logs:  {missing}')
        if flipped_to_pass:
            print(f'  flipped to PASS: {sorted(flipped_to_pass, key=lambda x: int(x) if x.isdigit() else 0)[:30]}{"..." if len(flipped_to_pass)>30 else ""}')


if __name__ == '__main__':
    main()
