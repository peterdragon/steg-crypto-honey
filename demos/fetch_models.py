#!/usr/bin/env python3
"""Fetch the GPT-2 / DistilGPT-2 weights the optional LLM demos need.

The large model files live under ``demos/models/{gpt2,distilgpt2}/`` and are
git-ignored (see ``demos/.gitignore``), so a fresh clone has all the code but
none of the weights.  This script downloads *exactly* the files the demos load
-- they call ``transformers.*.from_pretrained()`` on the local directory -- from
the Hugging Face Hub, reproducing the on-disk layout the loaders expect.

Only the LLM-backed demos need these weights
(``honey_gpt2_cli.py``, ``gpt2_discop.py``, ``neural_dte.py``);
the pure-Python demos run without them.

Usage
-----
    python3 fetch_models.py                # download every missing model
    python3 fetch_models.py --force        # re-download even if present
    python3 fetch_models.py gpt2           # just one (gpt2 | distilgpt2)
    ./.venv/bin/python fetch_models.py     # from inside the demos venv

    # reuse weights you already have somewhere (no re-download): symlink a
    # directory that contains gpt2/ and/or distilgpt2/ subdirectories.
    python3 fetch_models.py --link ~/.cache/my-llms
    python3 fetch_models.py gpt2 --link /shared/models

Downloading requires ``huggingface_hub`` (installed transitively by
``transformers``; otherwise ``pip install -r requirements-llm.txt`` or
``pip install huggingface_hub``).  Linking needs nothing extra.
Set ``HF_TOKEN`` in the environment if you are behind an authenticated proxy;
these two models are public and normally need no token.
"""
from pathlib import Path
import argparse
import sys

# local directory name  ->  canonical Hugging Face repo id
REPOS = {
    "gpt2": "openai-community/gpt2",
    "distilgpt2": "distilbert/distilgpt2",
}

# The exact files the demos load.  Restricting to these keeps the download
# small and deterministic: the source repos also ship TensorFlow (tf_model.h5),
# PyTorch-pickle (pytorch_model.bin), ONNX, and rust weights we neither load nor
# want.  model.safetensors is the only large file (~500 MB gpt2, ~330 MB distil).
ALLOW = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

MODELS_DIR = Path(__file__).resolve().parent / "models"


def _present(dst: Path) -> bool:
    """True if the two load-critical files are already on disk."""
    return (dst / "model.safetensors").is_file() and (dst / "config.json").is_file()


def fetch(name: str, force: bool) -> bool:
    from huggingface_hub import snapshot_download

    dst = MODELS_DIR / name
    if _present(dst) and not force:
        print(f"[skip] {name}: already present at {dst}  (use --force to refetch)")
        return True

    repo = REPOS[name]
    print(f"[get ] {name}: downloading {len(ALLOW)} files from '{repo}' -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(dst), allow_patterns=ALLOW)

    missing = [f for f in ALLOW if not (dst / f).is_file()]
    if missing:
        print(f"[warn] {name}: expected files still missing: {missing}", file=sys.stderr)
        return False
    print(f"[ok  ] {name}: ready")
    return True


def link(name: str, src_root: str, force: bool) -> bool:
    """Symlink ``models/<name>`` to ``<src_root>/<name>`` for people who already
    have the weights (e.g. a shared model store), avoiding a re-download."""
    dst = MODELS_DIR / name
    src = (Path(src_root).expanduser() / name).resolve()

    if not src.is_dir() or not (src / "model.safetensors").is_file():
        print(f"[warn] {name}: no usable model at {src} "
              f"(need <src>/{name}/model.safetensors); skipping", file=sys.stderr)
        return False

    if dst.is_symlink() or dst.exists():
        if not force:
            print(f"[skip] {name}: {dst} already exists (use --force to relink)")
            return True
        if dst.is_symlink() or dst.is_file():
            dst.unlink()                    # safe: only removes a link/file
        else:
            print(f"[warn] {name}: {dst} is a real directory; move it aside first, "
                  f"then relink", file=sys.stderr)
            return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=True)
    print(f"[link] {name}: {dst} -> {src}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download GPT-2 / DistilGPT-2 weights for the LLM demos.")
    ap.add_argument("models", nargs="*", choices=list(REPOS),
                    help="which model(s) to fetch (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="re-download / relink even if the target already exists")
    ap.add_argument("--link", metavar="DIR",
                    help="symlink from an existing store (a directory holding "
                         "gpt2/ and/or distilgpt2/) instead of downloading")
    args = ap.parse_args()

    names = args.models or list(REPOS)

    if args.link:
        ok = all(link(name, args.link, args.force) for name in names)
        return 0 if ok else 2

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("error: huggingface_hub is not installed.\n"
              "  pip install -r requirements-llm.txt   (or: pip install huggingface_hub)",
              file=sys.stderr)
        return 1

    ok = all(fetch(name, args.force) for name in names)
    if ok:
        print("\ndone. sanity check:\n"
              "  python3 -c \"from transformers import AutoModelForCausalLM as M; "
              "M.from_pretrained('models/gpt2'); print('gpt2 loads')\"")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
