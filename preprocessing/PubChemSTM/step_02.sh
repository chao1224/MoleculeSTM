
for block_id in {0..325}; do
    echo "$block_id"
    python step_02_download_SDF.py --block_id="$block_id"
done
