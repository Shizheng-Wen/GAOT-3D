#!/bin/bash

# Set Hugging Face dataset owner and prefix
HF_OWNER="neashton"
HF_PREFIX="drivaerml"

# Set the local target directory where files will be downloaded
LOCAL_DIR="/cluster/work/math/camlab-data/drivaerml/surface"

# Create the local directory if it does not exist
echo "Creating local directory: $LOCAL_DIR"
mkdir -p $LOCAL_DIR

# Loop through specific run folders
# for i in $(seq 1 500); do

# You can modify this list to include only the runs you want to download
RUNS=(193 194 178 168 166 165 164 157 140 139 133 132 131)

for i in "${RUNS[@]}"; do
    RUN_DIR="run_$i"
    echo "Downloading boundary_$i.vtp to $LOCAL_DIR/boundary_$i.vtp"

    wget "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/$RUN_DIR/boundary_$i.vtp" -O "$LOCAL_DIR/boundary_$i.vtp"

    # Check if the download was successful and provide a warning if it failed
    if [ $? -ne 0 ]; then
        echo "Warning: Download of boundary_$i.vtp failed. Please note that some run files are currently missing from the dataset."
        echo "Known missing runs include: 167, 211, 218, 221, 248, 282, 291, 295, 316, 325, 329, 364, 370, 376, 403, 473."
    fi
done

echo "All specified boundary_i.vtp files download process has completed."