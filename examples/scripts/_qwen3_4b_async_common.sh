#!/usr/bin/env bash
# Shared full async-GRPO launch profile. Scenario scripts only select data and recipe.
set -euo pipefail

run_qwen3_4b_async() {
  local train_file=$1 val_file=$2 flow=$3 flow_config=$4 project=$5 experiment=$6
  shift 6
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
  export VLLM_USE_V1=1
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  local model=${MODEL:-Qwen/Qwen3-4B}
  python3 -m claw_r1.async_main \
    algorithm.adv_estimator=grpo data.train_files="$train_file" data.val_files="$val_file" \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-128} data.max_prompt_length=${MAX_PROMPT_LENGTH:-4096} data.max_response_length=${MAX_RESPONSE_LENGTH:-4096} \
    data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True \
    actor_rollout_ref.model.path="$model" actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-6} actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32} actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE:-8} \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-1} actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_MEMORY_FRACTION:-0.8} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-8} actor_rollout_ref.rollout.n=${ROLLOUT_N:-5} \
    actor_rollout_ref.rollout.agent.default_agent_flow="$flow" actor_rollout_ref.rollout.agent.agent_flow_config_path="$flow_config" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BATCH_SIZE:-8} actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False trainer.critic_warmup=0 trainer.logger='["console","swanlab"]' trainer.project_name="$project" trainer.experiment_name="$experiment" \
    trainer.n_gpus_per_node=${TRAIN_GPUS_PER_NODE:-4} trainer.nnodes=${TRAIN_NNODES:-1} trainer.save_freq=${SAVE_FREQ:-20} trainer.test_freq=${TEST_FREQ:-20} trainer.total_epochs=${TOTAL_EPOCHS:-15} \
    rollout.n_gpus_per_node=${ROLLOUT_GPUS_PER_NODE:-2} rollout.nnodes=${ROLLOUT_NNODES:-1} \
    async_training.trigger_parameter_sync_step=${SYNC_INTERVAL:-1} async_training.use_rollout_log_probs=true async_training.max_queue_size=null \
    "$@"
}
