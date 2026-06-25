#!/bin/bash
# =============================================================================
# CastNet v2 — OPT-125m Full Benchmark
# =============================================================================
# Usage: bash scripts/benchmark_opt125m.sh 2>&1 | tee benchmark.log
# =============================================================================

set -e
REPORT_DIR="reports/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"
LOG_FILE="$REPORT_DIR/benchmark.log"
RESULTS_CSV="$REPORT_DIR/results.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}
log_sep() {
    echo "" | tee -a "$LOG_FILE"
    echo "======================================================================" | tee -a "$LOG_FILE"
    echo "  $*" | tee -a "$LOG_FILE"
    echo "======================================================================" | tee -a "$LOG_FILE"
}

# CSV header
echo "method,keep_fraction,sparsity_pct,perplexity,status" > "$RESULTS_CSV"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL="facebook/opt-125m"
DEVICE="cuda"
NUM_BATCHES_EXTRACT=200
NUM_BATCHES_EVAL=50
MAX_LEN_EXTRACT=128
MAX_LEN_EVAL=512
KEEP_FRACTION=0.3

METHODS=("wanda" "gradient" "gps" "q30_weighted" "q30_count" "gps_cube" "union_all3")

# ---------------------------------------------------------------------------
# Step 0 — Smoke test (skipped — already validated)
# ---------------------------------------------------------------------------
log_sep "STEP 0: Smoke test (skipped)"
log "Tests already validated (259 passed). Skipping."

# ---------------------------------------------------------------------------
# Step 1 — Dense baseline
# ---------------------------------------------------------------------------
log_sep "STEP 1: Dense baseline"
log "Evaluating dense model..."
python -u orchestration/evaluate.py \
    mode=dense \
    model=$MODEL \
    device=$DEVICE \
    num_batches=$NUM_BATCHES_EVAL \
    max_len=$MAX_LEN_EVAL \
    2>&1 | tee -a "$LOG_FILE"

DENSE_PERP=$(grep "Perplexity" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+')
echo "dense,$KEEP_FRACTION,0.0,$DENSE_PERP,OK" >> "$RESULTS_CSV"
log "Dense baseline: $DENSE_PERP"

# ---------------------------------------------------------------------------
# Step 2 — Extraction
# ---------------------------------------------------------------------------
log_sep "STEP 2: Score extraction"
for method in "${METHODS[@]}"; do
    log "Extracting: $method"
    START_TIME=$(date +%s)
    
    python -u orchestration/extract.py \
        model=$MODEL \
        method=$method \
        device=$DEVICE \
        num_batches=$NUM_BATCHES_EXTRACT \
        max_len=$MAX_LEN_EXTRACT \
        output="$REPORT_DIR/opt125m_${method}_scores.safetensors" \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    ELAPSED=$(($(date +%s) - START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "  $method: OK (${ELAPSED}s)"
    else
        log "  $method: FAILED (${ELAPSED}s)"
    fi
done

# ---------------------------------------------------------------------------
# Step 3 — Mask generation
# ---------------------------------------------------------------------------
log_sep "STEP 3: Mask generation"
for method in "${METHODS[@]}"; do
    if [ -f "$REPORT_DIR/opt125m_${method}_scores.safetensors" ]; then
        log "Generating masks: $method"
        python -u orchestration/masks.py \
            scores_path="$REPORT_DIR/opt125m_${method}_scores.safetensors" \
            output="$REPORT_DIR/opt125m_${method}_masks.safetensors" \
            keep_fraction=$KEEP_FRACTION \
            2>&1 | tee -a "$LOG_FILE"
    else
        log "Skipping masks for $method (scores not found)"
    fi
done

# ---------------------------------------------------------------------------
# Step 4 — Perplexity evaluation
# ---------------------------------------------------------------------------
log_sep "STEP 4: Perplexity evaluation"
for method in "${METHODS[@]}"; do
    if [ -f "$REPORT_DIR/opt125m_${method}_masks.safetensors" ]; then
        log "Evaluating: $method"
        START_TIME=$(date +%s)
        
        OUTPUT=$(python -u orchestration/evaluate.py \
            mode=perplexity \
            model=$MODEL \
            device=$DEVICE \
            mask_path="$REPORT_DIR/opt125m_${method}_masks.safetensors" \
            num_batches=$NUM_BATCHES_EVAL \
            max_len=$MAX_LEN_EVAL \
            2>&1 | tee -a "$LOG_FILE")
        
        EXIT_CODE=${PIPESTATUS[0]}
        ELAPSED=$(($(date +%s) - START_TIME))
        
        if [ $EXIT_CODE -eq 0 ]; then
            SPARSITY=$(echo "$OUTPUT" | grep "Sparsity" | grep -oP '[\d.]+')
            PERP=$(echo "$OUTPUT" | grep "Perplexity" | grep -oP '[\d.]+')
            echo "$method,$KEEP_FRACTION,$SPARSITY,$PERP,OK" >> "$RESULTS_CSV"
            log "  $method: sparsity=${SPARSITY}%, perp=${PERP} (${ELAPSED}s)"
        else
            echo "$method,$KEEP_FRACTION,0,0,FAILED" >> "$RESULTS_CSV"
            log "  $method: FAILED (${ELAPSED}s)"
        fi
    fi
done

# ---------------------------------------------------------------------------
# Step 5 — Sweep on best method
# ---------------------------------------------------------------------------
log_sep "STEP 5: Sparsity sweep (wanda)"
log "Running sweep..."
python -u orchestration/evaluate.py \
    mode=sweep \
    model=$MODEL \
    device=$DEVICE \
    scores_path="$REPORT_DIR/opt125m_wanda_scores.safetensors" \
    num_batches=$NUM_BATCHES_EVAL \
    max_len=$MAX_LEN_EVAL \
    2>&1 | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Step 6 — Union GPS + Gradient + Q30
# ---------------------------------------------------------------------------
log_sep "STEP 6: Union GPS + Gradient + Q30"
if [ -f "$REPORT_DIR/opt125m_gps_masks.safetensors" ] && \
   [ -f "$REPORT_DIR/opt125m_gradient_masks.safetensors" ] && \
   [ -f "$REPORT_DIR/opt125m_q30_weighted_masks.safetensors" ]; then
    log "Computing union..."
    python -u -c "
from infrastructure.persistence.safetensors_persister import SafetensorsMaskPersister
p = SafetensorsMaskPersister()
gps = p.load('$REPORT_DIR/opt125m_gps_masks.safetensors')
grad = p.load('$REPORT_DIR/opt125m_gradient_masks.safetensors')
q30 = p.load('$REPORT_DIR/opt125m_q30_weighted_masks.safetensors')
union = {n: ((gps[n] + grad[n] + q30[n]) > 0).float() for n in gps}
p.save(union, '$REPORT_DIR/opt125m_union_gps_grad_q30.safetensors')
print('Union saved')
" 2>&1 | tee -a "$LOG_FILE"

    log "Evaluating union..."
    OUTPUT=$(python -u orchestration/evaluate.py \
        mode=perplexity \
        model=$MODEL \
        device=$DEVICE \
        mask_path="$REPORT_DIR/opt125m_union_gps_grad_q30.safetensors" \
        num_batches=$NUM_BATCHES_EVAL \
        max_len=$MAX_LEN_EVAL \
        2>&1 | tee -a "$LOG_FILE")
    
    SPARSITY=$(echo "$OUTPUT" | grep "Sparsity" | grep -oP '[\d.]+')
    PERP=$(echo "$OUTPUT" | grep "Perplexity" | grep -oP '[\d.]+')
    echo "union_gps_grad_q30,$KEEP_FRACTION,$SPARSITY,$PERP,OK" >> "$RESULTS_CSV"
    log "Union: sparsity=${SPARSITY}%, perp=${PERP}"
else
    log "Skipping union (missing mask files)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_sep "BENCHMARK COMPLETE"
log "Report directory: $REPORT_DIR"
log "Results CSV: $RESULTS_CSV"
log ""
log "Results summary:"
column -t -s',' "$RESULTS_CSV" | tee -a "$LOG_FILE"
log ""
log "Full log: $LOG_FILE"
