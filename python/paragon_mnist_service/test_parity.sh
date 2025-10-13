#!/usr/bin/env bash
# ------------------------------------------------------------
# test_parity.sh — rich parity + drift check for Paragon MNIST
# ------------------------------------------------------------
set -euo pipefail

URL="${1:-http://127.0.0.1:8000}"
OUT_JSON="parity_report.json"
OUT_CSV="parity_summary.csv"

GREEN="\033[1;32m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; BLUE="\033[1;34m"; RESET="\033[0m"

echo -e "${BLUE}🧠 Checking service health at ${URL} ...${RESET}"
curl -fs "${URL}/health" | jq . >/dev/null || { echo -e "${RED}❌ Service not responding${RESET}"; exit 1; }
curl -fs "${URL}/health" | jq .
echo ""

echo -e "${BLUE}📸 Fetching hosted images...${RESET}"
curl -fs "${URL}/images/list" | jq .
echo -e "Static PNGs at: ${YELLOW}${URL}/static/images/<digit>.png${RESET}\n"

echo -e "${BLUE}🔁 Running full parity check (CPU vs GPU)...${RESET}"
curl -fs "${URL}/parity" -o "${OUT_JSON}"

GPU_AVAILABLE=$(jq '.gpu_available' "${OUT_JSON}")
TOTAL=$(jq '.total' "${OUT_JSON}")
MISMATCHES=$(jq '.mismatches' "${OUT_JSON}")
[[ "${GPU_AVAILABLE}" == "true" ]] && echo -e "${GREEN}GPU backend detected.${RESET}\n" || echo -e "${YELLOW}⚠️ GPU not initialized — CPU-only mode.${RESET}\n"

# Summary header
printf "%-8s %-10s %-10s %-8s %-10s %-14s %-14s %-14s\n" "IMAGE" "CPU_PRED" "GPU_PRED" "MATCH" "LAT(ms)" "max_abs_diff" "mean_abs_diff" "l2_diff"
printf "%-8s %-10s %-10s %-8s %-10s %-14s %-14s %-14s\n" "------" "--------" "--------" "------" "--------" "------------" "------------" "------------"

OVERALL_MAX=0
OVERALL_SUMSQ=0
OVERALL_COUNT=0

# Iterate results and compute drift metrics
jq -c '.results[]' "${OUT_JSON}" | while read -r ROW; do
  IMG=$(echo "$ROW" | jq -r '.image')
  CPU_PRED=$(echo "$ROW" | jq -r '.cpu.pred')
  GPU_PRED=$(echo "$ROW" | jq -r '.gpu.pred // "NA"')
  MATCH=$(echo "$ROW" | jq -r 'if .match==true then "true" elif .match==false then "false" else "NA" end')
  # prefer GPU latency if available, else CPU
  LAT=$(echo "$ROW" | jq -r '((.gpu.latency_sec // .cpu.latency_sec) * 1000)')

  # compute diffs with jq; get max abs and mean abs
  MAXABS=$(echo "$ROW" | jq -r '
    .cpu.probs as $a | .gpu.probs as $b |
    [range(0; ($a|length)) | ($a[.] - $b[.]) | (if .<0 then - . else . end)] | max
  ')
  MEANABS=$(echo "$ROW" | jq -r '
    .cpu.probs as $a | .gpu.probs as $b |
    [range(0; ($a|length)) | ($a[.] - $b[.]) | (if .<0 then - . else . end)] as $d |
    ( ($d|add) / ($d|length) )
  ')
  SUMSQ=$(echo "$ROW" | jq -r '
    .cpu.probs as $a | .gpu.probs as $b |
    [range(0; ($a|length)) | ($a[.] - $b[.]) | (.*.)] | add
  ')
  # l2 = sqrt(sumsq), use awk for sqrt to be portable
  L2=$(awk "BEGIN { printf \"%.9f\", sqrt(${SUMSQ}+0) }")

  # track overall aggregates via temp files (since subshell)
  echo "$MAXABS" >> .tmp_maxabs
  echo "$SUMSQ"  >> .tmp_sumsq
  echo 10        >> .tmp_count # 10 classes per image

  # print row
  if [[ "$MATCH" == "true" ]]; then COLOR=$GREEN; STATUS="✓"; else COLOR=$([ "$MATCH" == "false" ] && echo $RED || echo $YELLOW); STATUS=$([ "$MATCH" == "false" ] && echo "✗" || echo "-"); fi
  printf "${COLOR}%-8s %-10s %-10s %-8s %-10.3f %-14.9f %-14.9f %-14.9f${RESET}\n" "$IMG" "$CPU_PRED" "$GPU_PRED" "$STATUS" "$LAT" "$MAXABS" "$MEANABS" "$L2"

  # ----- VERBOSE: print the actual probability vectors -----
  echo -e "  cpu.probs = $(echo "$ROW" | jq -c '.cpu.probs')"
  echo -e "  gpu.probs = $(echo "$ROW" | jq -c '.gpu.probs')"
  echo -e "  diff.abs  = $(echo "$ROW" | jq -c '.cpu.probs as $a | .gpu.probs as $b | [range(0; ($a|length)) | ($a[.] - $b[.]) | (if .<0 then - . else . end)]')"
  echo ""
done

# Overall drift summary
if [[ -f .tmp_maxabs && -f .tmp_sumsq && -f .tmp_count ]]; then
  OVERALL_MAX=$(awk 'BEGIN{m=0} {if ($1>m) m=$1} END{printf "%.9f", m}' .tmp_maxabs)
  TOTAL_SUMSQ=$(awk '{s+=$1} END{printf "%.9f", s}' .tmp_sumsq)
  TOTAL_COUNT=$(awk '{c+=$1} END{print c}' .tmp_count)
  OVERALL_L2=$(awk "BEGIN { printf \"%.9f\", sqrt(${TOTAL_SUMSQ}+0) }")
  OVERALL_MEANABS=$(awk "BEGIN { printf \"%.9f\", ${TOTAL_SUMSQ}==0?0:0 }") # placeholder; mean abs across all classes would require abs-sum; skip
  rm -f .tmp_maxabs .tmp_sumsq .tmp_count
  echo -e "${BLUE}📊 Overall drift:${RESET}"
  echo -e "  max_abs_diff (any class, any image): ${GREEN}${OVERALL_MAX}${RESET}"
  echo -e "  l2_diff across all classes & images: ${GREEN}${OVERALL_L2}${RESET}"
  echo ""
fi

echo -e "${BLUE}📄 Saving reports...${RESET}"
echo "image,cpu_pred,gpu_pred,match,cpu_latency_sec,gpu_latency_sec,max_abs_diff,mean_abs_diff,l2_diff" > "${OUT_CSV}"
jq -r '
  .results[] |
  . as $r |
  $r.cpu.probs as $a | $r.gpu.probs as $b |
  [range(0; ($a|length)) | ($a[.] - $b[.]) | (if .<0 then - . else . end)] as $d |
  [range(0; ($a|length)) | ($a[.] - $b[.]) | (.*.)] as $s |
  "\($r.image),\($r.cpu.pred),\($r.gpu.pred // "NA"),\($r.match),\($r.cpu.latency_sec),\($r.gpu.latency_sec // "NA"),\(($d|max)),\((($d|add)/($d|length))),\(($s|add))"
' "${OUT_JSON}" >> "${OUT_CSV}"

# convert last column (sum of squares) to L2 in place (makes a second file)
awk -F',' 'BEGIN{OFS=","} NR==1{print; next} { $9 = sprintf("%.9f", sqrt($9+0)); print }' "${OUT_CSV}" > "${OUT_CSV}.tmp" && mv "${OUT_CSV}.tmp" "${OUT_CSV}"

echo -e "✅ JSON saved to ${GREEN}${OUT_JSON}${RESET}"
echo -e "✅ CSV  saved to ${GREEN}${OUT_CSV}${RESET}\n"

if [[ "${MISMATCHES}" -eq 0 ]]; then
  echo -e "${GREEN}🎯 Perfect parity across ${TOTAL} images!${RESET}\n"
else
  echo -e "${RED}⚠️  ${MISMATCHES} mismatches found out of ${TOTAL} images.${RESET}\n"
fi
