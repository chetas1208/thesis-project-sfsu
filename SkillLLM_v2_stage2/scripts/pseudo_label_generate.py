"""Experiment 2, step 1 (GPU1): run the anchor, QLoRA-loop v1 and v2 adapters over the unlabeled chunk pool.

One 4-bit NF4 base is loaded once (same quantisation/loader as MistralExtractor, i.e. the held-out eval path); the three LoRA adapters are
attached with peft multi-adapter and switched with set_adapter. Chunks are processed in shuffled blocks; inside a block every adapter runs, so
any finished prefix of blocks has all three votes. Greedy decoding, ORIGINAL prompt (all three adapters were trained with it), batched with
left padding, max_new_tokens 384 (the loop's dev/probe setting). Resumable: per (adapter) jsonl files are appended per batch and finished ids
are skipped. Private text is only read from / written to the artifacts dir (never reports/docs).

Env: PSEUDO_BLOCK (default 400), PSEUDO_BATCH (24), PSEUDO_MAX_CHUNKS (default all), PSEUDO_OUT.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

POOL_DIR = Path(os.environ.get("PSEUDO_OUT", "artifacts/skillllm_v2/qlora_loop_v4_distill/pseudo_pool"))
ADAPTERS = {
    "anchor": "artifacts/skillllm_v2/mistral_lora/model",
    "v1": "artifacts/skillllm_v2/qlora_loop/best",
    "v2": "artifacts/skillllm_v2/qlora_loop_v2/best",
}
MODEL = "models/mistral-small-3.1-24b-instruct-2503"


def read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def main() -> None:
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoTokenizer, BitsAndBytesConfig

    from skillllm_v2.extract.mistral_extractor import MistralExtractor, _ensure_chat_template, _load_model, _load_tokenizer, parse_skill_json

    chunks = [json.loads(l) for l in (POOL_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(7).shuffle(chunks)
    if os.environ.get("PSEUDO_MAX_CHUNKS"):
        chunks = chunks[: int(os.environ["PSEUDO_MAX_CHUNKS"])]
    block = int(os.environ.get("PSEUDO_BLOCK", "400"))
    batch_size = int(os.environ.get("PSEUDO_BATCH", "24"))
    helper = MistralExtractor(allow_fallback=True)  # prompt + payload->span alignment only

    tok = _load_tokenizer(AutoTokenizer, MODEL)
    _ensure_chat_template(tok, MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    t_load = time.perf_counter()
    base = _load_model(transformers, MODEL, {"quantization_config": quant, "device_map": {"": 0}, "trust_remote_code": True})
    model = None
    for name, path in ADAPTERS.items():
        if model is None:
            model = PeftModel.from_pretrained(base, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
    model.eval()
    print(f"loaded base + {len(ADAPTERS)} adapters in {time.perf_counter() - t_load:.0f}s; alloc={torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    out_files = {name: POOL_DIR / f"preds_{name}.jsonl" for name in ADAPTERS}
    timing = POOL_DIR / "gen_timing.jsonl"
    for b_start in range(0, len(chunks), block):
        blk = chunks[b_start : b_start + block]
        for name in ADAPTERS:
            done = read_done(out_files[name])
            todo = [c for c in blk if c["id"] not in done]
            if not todo:
                continue
            model.set_adapter(name)
            prompts = {c["id"]: tok.apply_chat_template(helper.build_messages(c["text"]), tokenize=False, add_generation_prompt=True) for c in todo}
            order = sorted(todo, key=lambda c: len(prompts[c["id"]]))
            t0 = time.perf_counter()
            parse_fail = 0
            with out_files[name].open("a", encoding="utf-8") as fh:
                for s in range(0, len(order), batch_size):
                    items = order[s : s + batch_size]
                    inputs = tok([prompts[c["id"]] for c in items], return_tensors="pt", padding=True).to("cuda:0")
                    with torch.no_grad():
                        gen = model.generate(**inputs, max_new_tokens=384, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
                    for j, c in enumerate(items):
                        raw = tok.decode(gen[j][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
                        try:
                            skills = [sk.to_dict() for sk in helper._skills_from_payload(c["text"], parse_skill_json(raw))]
                            ok = True
                        except Exception:
                            skills, ok = [], False
                            parse_fail += 1
                        fh.write(json.dumps({"id": c["id"], "parse_ok": ok, "predictions": skills}, ensure_ascii=False) + "\n")
                    fh.flush()
            secs = time.perf_counter() - t0
            with timing.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"adapter": name, "block_start": b_start, "docs": len(todo), "seconds": secs, "parse_failures": parse_fail, "batch_size": batch_size}) + "\n")
            print(f"block@{b_start} adapter={name} docs={len(todo)} {secs:.0f}s ({secs / len(todo):.2f}s/doc) parse_fail={parse_fail}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
