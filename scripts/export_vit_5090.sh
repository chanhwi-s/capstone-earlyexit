#!/usr/bin/env bash
# ============================================================
#  export_vit_5090.sh  —  ViT ONNX 세그먼트 Export (RTX 5090)
#
#  SelectiveExitViT (2-exit / 3-exit) + PlainViT 를 세그먼트별 ONNX로 추출.
#  학습 완료 후, Orin으로 전송하기 전에 5090에서 실행.
#
#  사전 조건:
#    - 학습 완료된 체크포인트가 experiments/ 에 존재
#    - 가상환경 활성화
#    - timm, onnx 설치
#
#  사용법:
#    bash scripts/export_vit_5090.sh            # all (plain+2exit+3exit)
#    bash scripts/export_vit_5090.sh plain      # PlainViT만
#    bash scripts/export_vit_5090.sh 2exit      # 2-exit만
#    bash scripts/export_vit_5090.sh 3exit      # 3-exit만
#
#  출력:
#    {EXP_DIR}/onnx/plain_vit/plain_vit.onnx
#    {EXP_DIR}/onnx/ee_vit_2exit/seg1.onnx  seg2.onnx
#    {EXP_DIR}/onnx/ee_vit_3exit/seg1.onnx  seg2.onnx  seg3.onnx
#
#  Orin 전송:
#    scp -r experiments/exp_YYYYMMDD_HHMMSS \
#        cap6@<orin_ip>:capstone-earlyexit/experiments/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

MODEL="${1:-all}"   # 인자: all | plain | 2exit | 3exit

# 서버 환경 설정 (5090 공유 서버)
# 시스템 레벨 HF_HOME 이 /home/shared 등으로 설정되어 있을 수 있으므로 강제 override
export HF_HOME="/home/cap10/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/cap10/.cache/huggingface/hub"

echo "============================================"
echo "  ViT ONNX Export (RTX 5090)"
echo "  Model   : $MODEL"
echo "  SRC_DIR : $SRC_DIR"
echo "============================================"

cd "$SRC_DIR"

python export/export_onnx_vit_selective.py \
    --model "$MODEL" \
    ${EXIT_BLOCKS_2:+--exit-blocks-2 $EXIT_BLOCKS_2} \
    ${EXIT_BLOCKS_3:+--exit-blocks-3 $EXIT_BLOCKS_3}

echo ""
echo "============================================"
echo "  ONNX export 완료"
echo ""
echo "  다음 단계: Orin으로 전송"
echo "    scp -r experiments/exp_* \\"
echo "        cap6@<orin_ip>:capstone-earlyexit/experiments/"
echo ""
echo "  전송 후 Orin에서:"
echo "    bash scripts/orin_vit_pipeline.sh"
echo "============================================"
