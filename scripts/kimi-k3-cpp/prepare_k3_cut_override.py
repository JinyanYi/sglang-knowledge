#!/usr/bin/env python3
"""Generate json-model-override-args for Kimi-K3 layer truncation.

IMPORTANT: Must emit a FULL text_config copy. SGLang does config.update(override),
which replaces text_config wholesale — a partial dict drops num_attention_heads
and triggers AssertionError in get_hf_text_config.

Example:
  python3 prepare_k3_cut_override.py --keep 24 -o /tmp/k3_cut_l24.json
  # then: --json-model-override-args "$(cat /tmp/k3_cut_l24.json)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="/home/weights/Kimi-K3-W4A8/config.json",
        help="Full Kimi-K3 config.json",
    )
    ap.add_argument("--keep", type=int, required=True, help="Keep first N layers")
    ap.add_argument("-o", "--output", required=True, help="Output override JSON path")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    tc = cfg.get("text_config") or cfg
    if not isinstance(tc, dict):
        raise SystemExit("text_config is not a dict in config.json")
    n_full = int(tc["num_hidden_layers"])
    keep = args.keep
    if keep < 1 or keep > n_full:
        raise SystemExit(f"--keep must be in [1, {n_full}], got {keep}")

    block = int(tc.get("attn_res_block_size") or 12)
    if keep % block != 0 and keep != n_full:
        print(
            f"[warn] keep={keep} is not a multiple of attn_res_block_size={block}; "
            f"prefer keep in {{{', '.join(str(i) for i in range(block, n_full, block))}}}"
        )

    tc_out = dict(tc)
    tc_out["num_hidden_layers"] = keep
    lac = dict(tc.get("linear_attn_config") or {})
    lac["kda_layers"] = [i for i in lac.get("kda_layers", []) if i <= keep]
    lac["full_attn_layers"] = [i for i in lac.get("full_attn_layers", []) if i <= keep]
    tc_out["linear_attn_config"] = lac

    override = {"text_config": tc_out}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(override, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  num_hidden_layers: {n_full} -> {keep}")
    print(f"  num_attention_heads: {tc_out.get('num_attention_heads')}")
    print(f"  kda_layers: {len(lac['kda_layers'])}, full_attn: {lac['full_attn_layers']}")
    if keep % 2 == 0:
        half = keep // 2
        print(f"  suggested SGLANG_PP_LAYER_PARTITION={half},{half}")


if __name__ == "__main__":
    main()
