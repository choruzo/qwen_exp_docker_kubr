#!/usr/bin/env bash
# Rebuild the ROCm 7.2.4 BF16 Tensile kernels with the bounds fix from PR #8909.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_root/.cache/rocm-libraries-7.2.4"
overlay_dir="$repo_root/.cache/hipblaslt-gfx1201-bf16-overlay"
patch_file="$repo_root/patches/hipblaslt-rocm-7.2.4-pr8909-backport.patch"
image="${ROCM_TRAIN_IMAGE:-docker-k8s-finetune-train:rocm-e0fd6548}"
revision=dabb6df2b988f8eabed1e2fecefaaf4e818bc7ef

if [[ ! -d "$source_dir/.git" ]]; then
    git clone --depth 1 --filter=blob:none --sparse --branch rocm-7.2.4 \
        https://github.com/ROCm/rocm-libraries.git "$source_dir"
    git -C "$source_dir" sparse-checkout set projects/hipblaslt shared/origami
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]] || {
    echo "Unexpected rocm-libraries revision" >&2
    exit 1
}
if git -C "$source_dir" apply --check "$patch_file"; then
    git -C "$source_dir" apply "$patch_file"
elif ! git -C "$source_dir" apply --reverse --check "$patch_file"; then
    echo "The source checkout does not match the pinned backport" >&2
    exit 1
fi

docker run --rm --entrypoint bash -v "$source_dir:/src" \
    -w /src/projects/hipblaslt "$image" -lc '
set -euo pipefail
cmake --preset rocisa -S /src/projects/hipblaslt \
    -B /src/projects/hipblaslt/build-rocisa \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE=/opt/unsloth-venv/bin/python
cmake --build /src/projects/hipblaslt/build-rocisa --target rocisa -j4
/opt/unsloth-venv/bin/python -m pip install --no-cache-dir \
    --target /src/projects/hipblaslt/build-rocisa/python-deps msgpack==1.1.1
mkdir -p /src/projects/hipblaslt/build-bf16/logic
find /src/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1201 \
    -type f -name "*BBS*" \
    -exec cp -t /src/projects/hipblaslt/build-bf16/logic {} +
test "$(find /src/projects/hipblaslt/build-bf16/logic -type f | wc -l)" -eq 5
PYTHONPATH=/src/projects/hipblaslt/build-rocisa/python-deps:/src/projects/hipblaslt/build-rocisa/tensilelite/rocisa/lib:/src/projects/hipblaslt/tensilelite \
    python -m Tensile.TensileCreateLibrary \
    --architecture gfx1201 --assembler /opt/rocm/llvm/bin/clang++ \
    --cxx-compiler /opt/rocm/llvm/bin/clang++ --jobs 4 \
    --library-format msgpack --logic-format yaml \
    --disable-asm-comments --no-enumerate \
    /src/projects/hipblaslt/build-bf16/logic \
    /src/projects/hipblaslt/build-bf16/output HIP
'

mkdir -p "$overlay_dir"
docker run --rm --entrypoint bash -v "$source_dir:/src:ro" \
    -v "$overlay_dir:/overlay" "$image" -lc 'python - <<"PY"
from pathlib import Path
import shutil
stock = Path("/opt/unsloth-venv/lib/python3.12/site-packages/torch/lib/hipblaslt/library")
generated = Path("/src/projects/hipblaslt/build-bf16/output/library")
overlay = Path("/overlay")
objects = sorted(generated.glob("TensileLibrary_BB_BB*gfx1201.co"))
assert len(objects) == 4, f"expected 4 corrected BF16 objects, found {len(objects)}"
for item in stock.iterdir():
    dest = overlay / item.name
    if not dest.exists() and not dest.is_symlink():
        dest.symlink_to(item)
for obj in objects:
    for item in (obj, obj.with_suffix(".dat")):
        dest = overlay / item.name
        assert dest.is_symlink() or dest.is_file(), item.name
        dest.unlink()
        shutil.copy2(item, dest)
assert all((overlay / item.name).is_file() for item in objects)
print("Corrected BF16 objects:", len(objects))
print("Stock library links:", sum(item.is_symlink() for item in overlay.iterdir()))
PY'

sha256sum "$overlay_dir"/TensileLibrary_BB_BB*gfx1201.co
