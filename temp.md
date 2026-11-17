# 1. 停掉 v1 的深度訓練
pkill -f "baseline training"
pkill -f -- "--group (signal|image)"

# 2. 設定
cd /home/cliffyang/Project/SARS-CoV-2-Quantum
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/cliffyang/miniconda3/envs/qml_env/bin/python
DATA=Data/spike-lineage/dataset_v2.tsv
OUT=Results/lineage_baselines_v2

# 3. 傳統實驗（背景；先 ae_embedding 再 onehot，估計 3–4 小時）
nohup bash -c "
  for rep in ae_embedding onehot; do
    $PY -u -m Research.lineage_baselines.run --data $DATA --out $OUT --group classical --representation \$rep --seeds 0 1 2 3 4
  done; echo ALLDONE" > $OUT.classical.log 2>&1 &

# 4. 深度實驗（背景；估計約 50 小時，可隨時中斷續跑）
nohup systemd-inhibit --what=sleep --why="baseline training" bash -c "
  $PY -u -m Research.lineage_baselines.run --data $DATA --out $OUT --group signal --seeds 0 1 2
  for rep in rp gaf mtf; do
    $PY -u -m Research.lineage_baselines.run --data $DATA --out $OUT --group image --representation \$rep --seeds 0 1 2
  done; echo ALLDONE" > $OUT.deep.log 2>&1 &


怎麼補跑那 3 個 run

檢查點還在，停在發散前一輪（LSTM 的 seed0 在第 84 輪、seed1 在第 69 輪；mWDN 的 seed2 在第 79 輪）。續跑只需要訓練剩下的十幾到三十幾輪。
補跑要等目前的深度流程（影像組）結束，或另開終端機，再執行：

cd ~/Project/SARS-CoV-2-Quantum
/home/cliffyang/miniconda3/envs/qml_env/bin/python -u -m Research.lineage_baselines.run \
  --data Data/spike-lineage/dataset_v2.tsv --out Results/lineage_baselines_v2 \
  --group signal --model lstm3bi --seeds 0 1
/home/cliffyang/miniconda3/envs/qml_env/bin/python -u -m Research.lineage_baselines.run \
  --data Data/spike-lineage/dataset_v2.tsv --out Results/lineage_baselines_v2 \
  --model mwdn --seeds 2

---
# 與 cumsum 目錄相同設定：image 組、cnn3/cnn4 × rp/gaf/mtf、signal_mode=cumsum
# 建議種子可先 seed0，或補 0 1 2

OUT=Results/lineage_baselines_cumsum_v2
DATA=Data/spike-lineage/dataset_v2.tsv

for rep in rp gaf mtf; do
  python -u -m Research.lineage_baselines.run \
    --data "$DATA" \
    --out "$OUT" \
    --group image \
    --representation "$rep" \
    --signal-mode cumsum \
    --seeds 0
done

cd ~/Project/SARS-CoV-2-Quantum

---

OUT=Results/lineage_baselines_cumsum_v2
DATA=Data/spike-lineage/dataset_v2.tsv
LOG="$OUT/train.log"
mkdir -p "$OUT"

nohup bash -c '
  for rep in rp gaf mtf; do
    echo "===== $(date -Is) START $rep ====="
    python -u -m Research.lineage_baselines.run \
      --data "'"$DATA"'" \
      --out "'"$OUT"'" \
      --group image \
      --representation "$rep" \
      --signal-mode cumsum \
      --seeds 0
    echo "===== $(date -Is) DONE $rep ====="
  done
  echo "ALLDONE $(date -Is)"
' >> "$LOG" 2>&1 &

echo "PID=$!  log=$LOG"
disown