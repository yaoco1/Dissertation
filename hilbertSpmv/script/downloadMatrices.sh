#!/bin/bash
# downloadMatrices.sh


BASE_URL="https://sparse.tamu.edu//MM"

# Format: "Group/name = destination path"
declare -A MATRICES=(
    ["Williams/cant"]="data/sciComputingMatrix/cant"
    ["Boeing/ct20stif"]="data/sciComputingMatrix/ct20stif"
    ["Schenk/nlpkkt80"]="data/randomSparseMatrix/nlpkkt80"
    ["Schenk_ISEI/barrier2-1"]="data/randomSparseMatrix/barrier2-1"
    ["SNAP/roadNet-CA"]="data/graphMatrix/roadNet-CA"
    ["SNAP/roadNet-TX"]="data/graphMatrix/roadNet-TX"
    ["SNAP/com-Youtube"]="data/largeScaleGraphMatrix/com-Youtube"
)

TMP_DIR=$(mktemp -d)
echo "Temporary directory: $TMP_DIR"
echo ""

for key in "${!MATRICES[@]}"; do
    dest="${MATRICES[$key]}"
    name="${key##*/}"
    url="${BASE_URL}/${key}.tar.gz"

    echo "─── Downloading $name ───"
    echo "  URL:  $url"
    echo "  Dest: $dest/"

    mkdir -p "$dest"

    # Download
    #if ! wget --show-progress -O "$TMP_DIR/${name}.tar.gz" "$url"; then
    if ! wget -O "$TMP_DIR/${name}.tar.gz" "$url"; then
        echo "  [ERROR] Download failed for $name."
        continue
    fi

    # Extract — SuiteSparse tarballs contain a subdirectory named after the matrix
    if ! tar -xzf "$TMP_DIR/${name}.tar.gz" -C "$TMP_DIR"; then
        echo "  [ERROR] Extraction failed for $name."
        continue
    fi

    # Move the .mtx file into dest/
    mtx_file=$(find "$TMP_DIR/$name" -name "*.mtx" | head -1)
    if [ -z "$mtx_file" ]; then
        echo "  [ERROR] No .mtx file found after extraction."
        continue
    fi

    cp "$mtx_file" "$dest/${name}.mtx"
    echo "  [OK] Saved to $dest/${name}.mtx"
    echo ""
done

# Clean|
rm -rf "$TMP_DIR"

echo "════════════════════════════════════════"
echo " Done. Verify with:"
echo "   find data/ -name '*.mtx' | sort"
find data/ -name '*.mtx' | sort
echo "════════════════════════════════════════"