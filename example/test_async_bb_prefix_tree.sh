# ── Environment ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6}
export VLLM_USE_V1=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_HOME=/usr/local/cuda

# ── Model ────────────────────────────────────────────────────────────────────
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}

# ── Data paths ───────────────────────────────────────────────────────────────
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k/test.parquet}

python3 -m claw_r1.async_main \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.agent.default_agent_flow=blackbox_gsm8k_agent \
    actor_rollout_ref.rollout.agent.agent_flow_config_path=claw_r1/blackbox_agent/agent_flow_config.yaml \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='test_async_bb_prefix_tree' \
    trainer.experiment_name='qwen2_5_3b_bb_gsm8k_prefix_tree' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    rollout.n_gpus_per_node=1 \
    rollout.nnodes=1 \
    async_training.trigger_parameter_sync_step=1 \
    async_training.use_rollout_log_probs=true \
    async_training.max_queue_size=null \
    async_training.enable_prefix_tree_merge=true \
    "$@"
