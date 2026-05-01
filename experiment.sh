#!/bin/bash
# ============================================================
# Go2 多方向步态训练 + 评测完整流程
# ============================================================
set -e

# ============================================================
# 全局配置 — 每次重新实验只需修改这里
# ============================================================
REPO=/data/users/zheng/projects/courses/EEC289/EEC289A_Robotics-Homework
CONFIG=configs/colab_runtime_config.json
GPU=2
RUN_NAME=run_v9          # ← 改这里开新实验 (e.g. run_v9, run_v10)

# 训练模式:
#   "both"     — 从零开始完整训练 Stage 1 + Stage 2（约 2~3 小时）
#   "stage_2"  — 跳过 Stage 1，直接从已有 checkpoint 继续 Stage 2（约 1~2 小时）
TRAIN_MODE=both

# 仅当 TRAIN_MODE=stage_2 时生效：指定复用的 Stage 1 checkpoint 路径
# 留空则由脚本自动查找上一个 run 的最新 Stage 1 checkpoint
STAGE1_CKPT=

# ============================================================

cd "$REPO"

# 路径均从 RUN_NAME 自动派生，无需手动修改
ARTIFACTS=artifacts/$RUN_NAME
CHECKPOINT=$ARTIFACTS/best_checkpoint
DEMO_DIR=artifacts/demo_${RUN_NAME}
EVAL_DIR=artifacts/eval_${RUN_NAME}
FIG_DIR=figures
LIVE_PLOT_INTERVAL=20

TRAIN_OVERVIEW_METRICS=eval/episode_reward,eval/avg_episode_length,eval/episode_reward_std,training/sps
TRAIN_MAIN_TERMS=eval/episode_reward/tracking_lin_vel,eval/episode_reward/tracking_ang_vel,eval/episode_reward/energy,eval/episode_reward/feet_slip,eval/episode_reward/action_rate,eval/episode_reward/termination
TRAIN_AUX_TERMS=eval/episode_reward/lin_vel_z,eval/episode_reward/ang_vel_xy,eval/episode_reward/orientation,eval/episode_reward/pose,eval/episode_reward/feet_air_time,eval/episode_reward/feet_height,eval/episode_reward/feet_clearance,eval/episode_reward/stand_still,eval/episode_reward/torques,eval/episode_reward/dof_pos_limits

LIVE_PLOT_PID=

plot_curves() {
    local metrics="$1"
    local out_base="$2"
    local formats="$3"
    local row_height_mm="${4:-42}"

    python3 scripts/plot_train_curves.py \
        --run-dir "$ARTIFACTS" \
        --metrics "$metrics" \
        --out "$out_base" \
        --formats "$formats" \
        --ema-span 2 \
        --row-height-mm "$row_height_mm"
}

start_live_plots() {
    mkdir -p "$FIG_DIR"
    (
        while true; do
            plot_curves "$TRAIN_OVERVIEW_METRICS" "$FIG_DIR/${RUN_NAME}_train_overview_live" png 34 >/dev/null 2>&1 || true
            plot_curves "$TRAIN_MAIN_TERMS" "$FIG_DIR/${RUN_NAME}_train_terms_live" png 28 >/dev/null 2>&1 || true
            sleep "$LIVE_PLOT_INTERVAL"
        done
    ) &
    LIVE_PLOT_PID=$!
    echo "[train] 后台实时曲线已启动 (PID=$LIVE_PLOT_PID)"
    echo "[train] 实时图: $FIG_DIR/${RUN_NAME}_train_overview_live.png"
    echo "[train] 实时图: $FIG_DIR/${RUN_NAME}_train_terms_live.png"
}

stop_live_plots() {
    if [ -n "${LIVE_PLOT_PID:-}" ] && kill -0 "$LIVE_PLOT_PID" 2>/dev/null; then
        kill "$LIVE_PLOT_PID" 2>/dev/null || true
        wait "$LIVE_PLOT_PID" 2>/dev/null || true
    fi
    LIVE_PLOT_PID=
}

export_final_plots() {
    mkdir -p "$FIG_DIR"
    echo "[plot] 导出完整训练曲线..."
    plot_curves "$TRAIN_OVERVIEW_METRICS" "$FIG_DIR/${RUN_NAME}_train_overview" png,pdf 34
    plot_curves "$TRAIN_MAIN_TERMS" "$FIG_DIR/${RUN_NAME}_train_terms_main" png,pdf 28
    plot_curves "$TRAIN_AUX_TERMS" "$FIG_DIR/${RUN_NAME}_train_terms_aux" png,pdf 24
}

trap 'stop_live_plots' EXIT

# ============================================================
# Step 1: 训练
# ============================================================
# Stage 1 (10M steps): 单轴多方向课程（vx / vy / yaw 的正负方向）
# Stage 2 (5M steps): 在 Stage 1 基础上学习纯单项 + 与 yaw 的组合
#   不再训练 vx + vy 的混合命令，避免前进时伴随左右平移偏移
#   vx ∈ [-0.8, 0.8] m/s    (前进 / 后退)
#   vy ∈ [-0.5, 0.5] m/s    (左平移 / 右平移)
#   yaw ∈ [-1.0, 1.0] rad/s (逆时针 / 顺时针转向)
# 输出: $ARTIFACTS/best_checkpoint/

if [ "$TRAIN_MODE" = "both" ]; then
    echo "[train] 模式: 从零完整训练 Stage 1 + Stage 2"
    start_live_plots
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --config $CONFIG \
        --stage both \
        --output-dir $ARTIFACTS \
        # --wandb --wandb-project go2-locomotion --wandb-run-name $RUN_NAME

elif [ "$TRAIN_MODE" = "stage_2" ]; then
    if [ -z "$STAGE1_CKPT" ]; then
        echo "[train] 未指定 STAGE1_CKPT，自动使用 Stage 1 最新 checkpoint"
        STAGE1_CKPT=$(ls -d artifacts/run_*/stage_1/checkpoints/[0-9]* 2>/dev/null | sort | tail -1)
        if [ -z "$STAGE1_CKPT" ]; then
            echo "[错误] 找不到任何 Stage 1 checkpoint，请先运行 TRAIN_MODE=both 或手动指定 STAGE1_CKPT"
            exit 1
        fi
    fi
    echo "[train] 模式: 仅训练 Stage 2，复用 Stage 1 checkpoint: $STAGE1_CKPT"
    start_live_plots
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --config $CONFIG \
        --stage stage_2 \
        --restore-checkpoint-dir $STAGE1_CKPT \
        --output-dir $ARTIFACTS \
        --wandb --wandb-project go2-locomotion --wandb-run-name $RUN_NAME

else
    echo "[错误] TRAIN_MODE 必须为 'both' 或 'stage_2'，当前值: $TRAIN_MODE"
    exit 1
fi

stop_live_plots
export_final_plots

# ============================================================
# Step 2: 生成 Demo 视频
# ============================================================
# 按 colab_runtime_config.json 中的 demo_rollout 段展示各方向运动能力:
#   静止 → 前进(0.5/0.7/0.9) → 后退(-0.5/-0.7/-0.9)
#   → 左平移(0.25/0.4/0.5) → 右平移(-0.25/-0.4/-0.5)
#   → 左转(0.6/0.8/1.0) → 右转(-0.6/-0.8/-1.0)
#   全部为纯命令段，不再展示前进 + 侧移的混合命令
# 每段 5 秒，共 25 段，总时长 125 秒
# 输出: $DEMO_DIR/demo.mp4
echo "[demo] 生成 Demo 视频..."
CUDA_VISIBLE_DEVICES=$GPU python test_policy.py \
    --config $CONFIG \
    --checkpoint-dir $CHECKPOINT \
    --stage-name stage_2 \
    --output-dir $DEMO_DIR

# ============================================================
# Step 3: 生成公开评测 Rollout 数据
# ============================================================
# 在标准评测指令序列下运行策略，记录完整轨迹数据
# 输出: $EVAL_DIR/rollout_public_eval.npz  (用于评分)
#       $EVAL_DIR/episode_0.mp4             (第一个 episode 视频)
echo "[eval] 生成评测 Rollout..."
CUDA_VISIBLE_DEVICES=$GPU python generate_public_rollout.py \
    --config $CONFIG \
    --checkpoint-dir $CHECKPOINT \
    --stage-name stage_2 \
    --output-dir $EVAL_DIR \
    --num-episodes 4 \
    --render-first-episode

# ============================================================
# Step 4: 计算评测分数
# ============================================================
# 从 rollout 数据计算 5 项指标 (越低越好):
#   velocity_tracking_error (权重 0.35) — 线速度跟踪误差
#   yaw_tracking_error      (权重 0.20) — 偏航速度跟踪误差
#   fall_rate               (权重 0.20) — 跌倒率
#   energy_proxy            (权重 0.15) — 能耗估算
#   foot_slip_proxy         (权重 0.10) — 脚底滑动量
# 输出: $EVAL_DIR/public_eval.json
echo "[eval] 计算评测分数..."
python public_eval.py \
    --config $CONFIG \
    --rollout-npz $EVAL_DIR/rollout_public_eval.npz \
    --output-json $EVAL_DIR/public_eval.json

echo ""
echo "========================================"
echo "实验完成: $RUN_NAME  (模式: $TRAIN_MODE)"
echo "========================================"
echo "  训练总览图: $FIG_DIR/${RUN_NAME}_train_overview.png"
echo "  训练主奖励项: $FIG_DIR/${RUN_NAME}_train_terms_main.png"
echo "  训练辅助奖励项: $FIG_DIR/${RUN_NAME}_train_terms_aux.png"
echo "  Demo 视频: $DEMO_DIR/demo.mp4"
echo "  评测分数:  $EVAL_DIR/public_eval.json"
echo "========================================"