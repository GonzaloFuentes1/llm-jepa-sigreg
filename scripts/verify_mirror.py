"""Check that the ungated unsloth mirror matches meta-llama/Llama-3.2-1B-Instruct.

1. Weights: the LFS sha256 of model.safetensors, read from the public Hub API for both
   repos (the Meta files themselves are gated, their metadata is not).
2. Tokenizer: the Meta tokenizer files are fetched from a public copy whose git blob ids
   equal those listed for the Meta repo (blob ids are content hashes, so the files are
   byte-identical). Both tokenizers then encode every view of NL-RX-SYNTH and the ids
   are compared.

Usage: python scripts/verify_mirror.py --data-dir data --out results/mirror_check.json
"""

import argparse
import json
import urllib.request

from transformers import AutoTokenizer

from llm_jepa_sigreg.data import encode_example, load_split
from llm_jepa_sigreg.model import add_special_tokens, configure_padding

META = "meta-llama/Llama-3.2-1B-Instruct"
UNSLOTH = "unsloth/Llama-3.2-1B-Instruct"
UNSLOTH_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
META_COPY = "alpindale/Llama-3.2-1B-Instruct"
FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")


def _json(url: str):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def files(repo: str) -> dict:
    """Public file metadata: git blob id and, for LFS files, the sha256 of the content."""
    siblings = _json(f"https://huggingface.co/api/models/{repo}?blobs=true")["siblings"]
    return {s["rfilename"]: s for s in siblings}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results/mirror_check.json")
    args = p.parse_args()

    meta, unsloth, copy = files(META), files(UNSLOTH), files(META_COPY)
    report = {
        "weights_sha256": {
            "meta": meta["model.safetensors"]["lfs"]["sha256"],
            "unsloth": unsloth["model.safetensors"]["lfs"]["sha256"],
        },
        "meta_copy_blob_ids_match": {f: meta[f]["blobId"] == copy[f]["blobId"] for f in FILES},
        "unsloth_blob_ids_match_meta": {
            f: meta[f]["blobId"] == unsloth[f]["blobId"] for f in FILES
        },
    }
    report["weights_identical"] = len(set(report["weights_sha256"].values())) == 1
    configs = [
        _json(f"https://huggingface.co/{repo}/resolve/{rev}/config.json")
        for repo, rev in ((META_COPY, "main"), (UNSLOTH, UNSLOTH_REVISION))
    ]
    report["config_differences"] = {
        k: {"meta": configs[0].get(k), "unsloth": configs[1].get(k)}
        for k in sorted(set(configs[0]) | set(configs[1]))
        if configs[0].get(k) != configs[1].get(k)
    }

    tokenizers = {
        "meta": AutoTokenizer.from_pretrained(META_COPY),
        "unsloth": AutoTokenizer.from_pretrained(UNSLOTH, revision=UNSLOTH_REVISION),
    }
    report["tokenizer_settings"] = {
        name: {"pad_token": t.pad_token, "padding_side": t.padding_side, "eos": t.eos_token}
        for name, t in tokenizers.items()
    }
    for t in tokenizers.values():
        add_special_tokens(t)
        configure_padding(t, pad_as_eos=False)
    mismatches, total = 0, 0
    for split in ("synth_train", "synth_test"):
        for messages in load_split(split, args.data_dir):
            enc = [
                encode_example(t, messages, predictors=2, label_scope="all")
                for t in tokenizers.values()
            ]
            total += 1
            mismatches += enc[0] != enc[1]
    report["tokenization"] = {"examples": total, "mismatches": mismatches}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
