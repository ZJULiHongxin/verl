set -x
ENGINE=${1:-vllm}
#export VLLM_ATTENTION_BACKEND=XFORMERS

INDEX=0

GPU_TYPE=(A800 L40S)

PPO_MINI_BATCH_SIZE_OPTIONS=(64 64)
PPO_MICRO_BATCH_SIZE_PER_GPU_OPTIONS=(2 2)
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU_OPTIONS=(2 2)
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU_OPTIONS=(4 4)

PARAM_OFFLOAD_OPTIONS=(False True)
OPTIMIZER_OFFLOAD_OPTIONS=(False True)

ROOT=/mnt/jfs/copilot/lhx/ui_data/AndroidControl/0414_AW

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$ROOT/train.parquet \
    data.val_files=$ROOT/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=18000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE_OPTIONS[$INDEX]} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU_OPTIONS[$INDEX]} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD_OPTIONS[$INDEX]} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD_OPTIONS[$INDEX]} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU_OPTIONS[$INDEX]} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU_OPTIONS[$INDEX]} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD_OPTIONS[$INDEX]} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="[console]" \
    trainer.log_val_generations=1 \
    trainer.project_name='verl_grpo_androidworld' \
    trainer.experiment_name='test' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    custom_reward_function.path=recipe/agent_r1/reward_score.py \
    custom_reward_function.name=compute_score