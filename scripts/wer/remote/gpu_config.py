from __future__ import annotations


# Per-GPU SM arch. Single-arch builds finish nvcc in ~2 min vs ~6 min for a
# fat 75;86;89 build. Switching --gpu rebuilds for the new arch the first time
# (then sits cached in the build Volume alongside any other arch you've used).
GPU_TO_ARCH = {
    "T4": "75",
    "L4": "89",
    "A10G": "86",
    "L40S": "89",
    "A100": "80",
    "A100-80GB": "80",
    "H100": "90",
    "H200": "90",
}


def build_dir(src_fp: str, gpu_id: str) -> str:
    """Build dir for (current source fingerprint, this GPU's SM arch).
    Co-keyed so two GPUs never collide and source changes always rebuild."""
    return f"/build/cuda-{src_fp}-sm{GPU_TO_ARCH[gpu_id]}"
