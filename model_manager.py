#!/usr/bin/env python3
"""Flash-MoE Model Manager — list, search, and download compatible models.

Compatible models: Qwen3.5 MoE with MLX quantization (model_type: qwen3_5_moe).
These include GatedDeltaNet linear attention + full attention layers with
switch_mlp expert routing.

Usage:
    python model_manager.py                    # List local + search remote
    python model_manager.py --local            # List local models only
    python model_manager.py --search           # Search HuggingFace for compatible models
    python model_manager.py --download <repo>  # Download a specific model
    python model_manager.py --check <path>     # Check if a local model is compatible
"""

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

HF_CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))
HF_API = "https://huggingface.co/api"

# Known compatible model types
COMPATIBLE_MODEL_TYPES = {"qwen3_5_moe"}

# Search queries — MLX-quantized Qwen3.5 models
SEARCH_QUERIES = [
    "mlx-community Qwen3.5",
    "mlx Qwen3.5 MoE",
    "lmstudio-community Qwen3.5 MLX",
]

# MoE model name patterns: "35B-A3B", "122B-A10B", "397B-A17B" etc.
# The "-A<N>B" suffix indicates active parameters = MoE architecture
import re
MOE_PATTERN = re.compile(r'\d+B-A\d+B')


def find_config_json(model_path: Path) -> Path | None:
    """Find config.json in a model directory, handling HF cache layout."""
    direct = model_path / "config.json"
    if direct.exists():
        return direct
    snapshots = model_path / "snapshots"
    if snapshots.exists():
        for snap in sorted(snapshots.iterdir(), reverse=True):
            candidate = snap / "config.json"
            if candidate.exists():
                return candidate
    return None


def check_compatibility(model_path: Path) -> dict:
    """Check if a local model is compatible with Flash-MoE.

    Returns a dict with:
        compatible: bool
        reason: str (if not compatible)
        info: dict (model details if compatible)
    """
    config_path = find_config_json(model_path)
    if not config_path:
        return {"compatible": False, "reason": "No config.json found"}

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "")
    if model_type not in COMPATIBLE_MODEL_TYPES:
        return {
            "compatible": False,
            "reason": f"Incompatible model_type: {model_type} (need: {', '.join(COMPATIBLE_MODEL_TYPES)})",
        }

    tc = config.get("text_config", {})
    if not tc:
        return {"compatible": False, "reason": "Missing text_config in config.json"}

    # Check for required fields
    required = [
        "hidden_size", "num_hidden_layers", "num_experts",
        "num_experts_per_tok", "moe_intermediate_size",
    ]
    missing = [k for k in required if k not in tc]
    if missing:
        return {"compatible": False, "reason": f"Missing fields: {', '.join(missing)}"}

    # Check quantization
    qc = config.get("quantization_config", config.get("quantization", {}))
    bits = qc.get("bits", "?")
    group_size = qc.get("group_size", "?")

    # Check for packed experts
    model_dir = config_path.parent
    has_packed = (model_dir / "packed_experts").exists() or any(
        (model_dir.parent / "packed_experts").exists()
        for _ in [None]
    )

    # Check for extracted weights
    # Look relative to cwd (where infer runs from)
    has_weights = Path("metal_infer/model_weights.bin").exists() or Path("model_weights.bin").exists()

    info = {
        "model_type": model_type,
        "hidden_size": tc.get("hidden_size"),
        "num_layers": tc.get("num_hidden_layers"),
        "num_experts": tc.get("num_experts"),
        "experts_per_tok": tc.get("num_experts_per_tok"),
        "moe_intermediate": tc.get("moe_intermediate_size"),
        "vocab_size": tc.get("vocab_size"),
        "bits": bits,
        "group_size": group_size,
        "has_packed_experts": has_packed,
        "has_extracted_weights": has_weights,
        "config_path": str(config_path),
    }

    # Estimate sizes
    ne = tc.get("num_experts", 0)
    nl = tc.get("num_hidden_layers", 0)
    mid = tc.get("moe_intermediate_size", 0)
    hid = tc.get("hidden_size", 0)
    if isinstance(bits, int) and bits > 0:
        vals_per_u32 = 32 // bits
        expert_bytes = 0
        # gate + up: [mid, hid]
        for _ in range(2):
            w = mid * ((hid + vals_per_u32 - 1) // vals_per_u32) * 4
            s = mid * ((hid + group_size - 1) // group_size) * 2
            expert_bytes += w + s + s  # weight + scales + biases
        # down: [hid, mid]
        w = hid * ((mid + vals_per_u32 - 1) // vals_per_u32) * 4
        s = hid * ((mid + group_size - 1) // group_size) * 2
        expert_bytes += w + s + s

        total_expert_gb = ne * nl * expert_bytes / (1024**3)
        active_per_token_mb = tc.get("num_experts_per_tok", 0) * expert_bytes / (1024**2)
        info["expert_size_bytes"] = expert_bytes
        info["total_expert_disk_gb"] = round(total_expert_gb, 1)
        info["active_per_token_mb"] = round(active_per_token_mb, 1)

    # Count total params (rough estimate)
    total_params_b = ne * nl * mid * hid * 3 * 2 / 1e9  # gate+up+down, *2 for bidir
    info["approx_total_params"] = f"~{total_params_b:.0f}B" if total_params_b > 1 else f"~{total_params_b*1000:.0f}M"

    return {"compatible": True, "info": info}


def list_local_models():
    """List locally cached HuggingFace models and check compatibility."""
    if not HF_CACHE.exists():
        print("No HuggingFace cache found at", HF_CACHE)
        return []

    models = []
    for entry in sorted(HF_CACHE.iterdir()):
        if not entry.name.startswith("models--"):
            continue
        # Convert models--org--name to org/name
        parts = entry.name.split("--", 2)
        if len(parts) >= 3:
            repo_id = f"{parts[1]}/{parts[2]}"
        else:
            repo_id = entry.name

        result = check_compatibility(entry)
        result["repo_id"] = repo_id
        result["path"] = str(entry)
        models.append(result)

    return models


def search_remote_models():
    """Search HuggingFace for compatible Qwen3.5 MoE models."""
    if not requests:
        print("Install 'requests' to search HuggingFace: pip install requests")
        return []

    seen = set()
    results = []

    for query in SEARCH_QUERIES:
        try:
            resp = requests.get(
                f"{HF_API}/models",
                params={
                    "search": query,
                    "limit": 30,
                    "sort": "downloads",
                    "direction": -1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            for model in resp.json():
                repo_id = model.get("id", "")
                if repo_id in seen:
                    continue
                seen.add(repo_id)

                # Filter: must have "qwen" and "moe" or "3.5" indicators
                lower = repo_id.lower()
                tags = [t.lower() for t in model.get("tags", [])]

                is_qwen35 = "qwen3.5" in lower or "qwen3_5" in lower
                is_mlx = "mlx" in lower or "mlx" in " ".join(tags)
                is_moe = bool(MOE_PATTERN.search(repo_id))

                # We need: Qwen3.5 + MLX quantized + MoE architecture
                if is_qwen35 and is_mlx and is_moe:
                    # Extract quant info from name
                    quant = ""
                    for q in ["3bit", "4bit", "6bit", "8bit"]:
                        if q in lower:
                            quant = q
                            break

                    results.append({
                        "repo_id": repo_id,
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                        "quant": quant,
                        "last_modified": model.get("lastModified", "")[:10],
                    })
        except Exception as e:
            print(f"Warning: search failed for '{query}': {e}", file=sys.stderr)

    return results


def download_model(repo_id: str):
    """Download a model from HuggingFace."""
    # Try huggingface-cli first
    hf_cli = None
    for cmd in ["huggingface-cli", "hf"]:
        try:
            subprocess.run([cmd, "--help"], capture_output=True, check=True)
            hf_cli = cmd
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    if hf_cli:
        print(f"Downloading {repo_id} via {hf_cli}...")
        subprocess.run([hf_cli, "download", repo_id], check=True)
    else:
        # Fall back to Python
        try:
            from huggingface_hub import snapshot_download
            print(f"Downloading {repo_id} via huggingface_hub...")
            path = snapshot_download(repo_id)
            print(f"Downloaded to: {path}")
        except ImportError:
            print("ERROR: No download tool available.")
            print("Install one of:")
            print("  pip install huggingface-hub     # Python library")
            print("  pip install huggingface-cli     # CLI tool")
            print(f"\nOr manually: git clone https://huggingface.co/{repo_id}")
            sys.exit(1)


def format_size(gb: float) -> str:
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{gb * 1024:.0f} MB"


def print_model_info(info: dict, indent: str = "  "):
    """Print formatted model info."""
    print(f"{indent}Architecture:  {info['num_layers']} layers, "
          f"hidden={info['hidden_size']}, "
          f"{info['num_experts']} experts (K={info['experts_per_tok']})")
    print(f"{indent}Quantization:  {info['bits']}-bit, group_size={info['group_size']}")
    if "total_expert_disk_gb" in info:
        print(f"{indent}Expert data:   {format_size(info['total_expert_disk_gb'])} on disk, "
              f"{info['active_per_token_mb']:.1f} MB active/token")
    if "approx_total_params" in info:
        print(f"{indent}Parameters:    {info['approx_total_params']} total")

    # Readiness indicators
    ready = True
    if not info.get("has_packed_experts"):
        print(f"{indent}Packed experts: NOT FOUND (run repack_experts.py)")
        ready = False
    else:
        print(f"{indent}Packed experts: OK")
    if not info.get("has_extracted_weights"):
        print(f"{indent}Weights file:   NOT FOUND (run extract_weights.py)")
        ready = False
    else:
        print(f"{indent}Weights file:   OK")

    if ready:
        print(f"{indent}Status:        READY TO RUN")
    else:
        print(f"{indent}Status:        NEEDS PREPARATION (see above)")


def main():
    parser = argparse.ArgumentParser(
        description="Flash-MoE Model Manager — list, search, and download compatible models"
    )
    parser.add_argument("--local", action="store_true", help="List local models only")
    parser.add_argument("--search", action="store_true", help="Search HuggingFace only")
    parser.add_argument("--download", type=str, metavar="REPO", help="Download a model (e.g. mlx-community/Qwen3.5-35B-A3B-4bit)")
    parser.add_argument("--check", type=str, metavar="PATH", help="Check if a local model is compatible")
    args = parser.parse_args()

    if args.check:
        path = Path(args.check).expanduser()
        result = check_compatibility(path)
        if result["compatible"]:
            print(f"COMPATIBLE: {path}")
            print_model_info(result["info"])
        else:
            print(f"NOT COMPATIBLE: {result['reason']}")
        return

    if args.download:
        download_model(args.download)
        # Check compatibility after download
        cache_name = "models--" + args.download.replace("/", "--")
        cache_path = HF_CACHE / cache_name
        if cache_path.exists():
            result = check_compatibility(cache_path)
            if result["compatible"]:
                print(f"\nModel is compatible with Flash-MoE!")
                print_model_info(result["info"])
                print(f"\nNext steps:")
                print(f"  1. python repack_experts.py --model {cache_path}")
                print(f"  2. python metal_infer/extract_weights.py --model {cache_path}")
                print(f"  3. ./metal_infer/infer --model {cache_path} --prompt 'Hello' --tokens 20")
        return

    # Default: show both local and remote
    show_local = not args.search
    show_remote = not args.local

    if show_local:
        print("=" * 60)
        print("LOCAL MODELS")
        print("=" * 60)
        models = list_local_models()
        if not models:
            print("  No models found in", HF_CACHE)
        else:
            compatible_count = 0
            for m in models:
                if m["compatible"]:
                    compatible_count += 1
                    print(f"\n  {m['repo_id']}")
                    print_model_info(m["info"], indent="    ")
                    print(f"    Path: {m['path']}")
                else:
                    print(f"\n  {m['repo_id']} (incompatible: {m.get('reason', 'unknown')})")
            print(f"\n  {compatible_count}/{len(models)} compatible models found")

    if show_remote:
        print()
        print("=" * 60)
        print("AVAILABLE ON HUGGINGFACE")
        print("=" * 60)
        remote = search_remote_models()
        if not remote:
            print("  No compatible models found (or search failed)")
        else:
            # Mark which ones are already local
            local_repos = set()
            if HF_CACHE.exists():
                for entry in HF_CACHE.iterdir():
                    if entry.name.startswith("models--"):
                        parts = entry.name.split("--", 2)
                        if len(parts) >= 3:
                            local_repos.add(f"{parts[1]}/{parts[2]}")

            for m in remote:
                local_tag = " [LOCAL]" if m["repo_id"] in local_repos else ""
                quant_tag = f" [{m['quant']}]" if m.get("quant") else ""
                print(f"\n  {m['repo_id']}{local_tag}{quant_tag}")
                print(f"    Downloads: {m['downloads']:,}  Likes: {m['likes']}")

            print(f"\n  {len(remote)} models found")
            print(f"\n  Download with: python model_manager.py --download <repo_id>")


if __name__ == "__main__":
    main()
