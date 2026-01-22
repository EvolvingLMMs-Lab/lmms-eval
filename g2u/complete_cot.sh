#!/bin/bash
# UniWorld Visual CoT - Complete Evaluation
# Runs all Visual CoT tasks with HF upload

HF_REPO="caes0r/uniworld-results"

bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0,1" "uni_mmmu_jigsaw100_visual_cot" "./logs/jigsaw_cot" "LanguageBind/UniWorld-V1" "29700" "${HF_REPO}"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0,1" "uni_mmmu_maze100_visual_cot" "./logs/maze_cot" "LanguageBind/UniWorld-V1" "29701" "${HF_REPO}"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0,1" "uni_mmmu_sliding54_visual_cot" "./logs/sliding_cot" "LanguageBind/UniWorld-V1" "29702" "${HF_REPO}"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0" "chartqa100_visual_cot" "./logs/chartqa_cot" "LanguageBind/UniWorld-V1" "29703" "${HF_REPO}"
bash /home/aiscuser/lmms-eval/g2u/uniworld_cot.sh "0" "illusionbench_arshia_logo_scene_visual_cot" "./logs/illusionbench_cot" "LanguageBind/UniWorld-V1" "29704" "${HF_REPO}"

echo ""
echo "======================================"
echo "All Visual CoT evaluations completed!"
echo "Results uploaded to: ${HF_REPO}"
echo "======================================"

