seeds=5
envs=("halfcheetah-medium-v2" "walker2d-medium-v2" "hopper-medium-v2" "halfcheetah-medium-replay-v2" "walker2d-medium-replay-v2", "hopper-medium-replay-v2" "halfcheetah-medium-expert-v2" "walker2d-medium-expert-v2" "hopper-medium-expert-v2")
exp_config='gym_pl'
tjn_ckpt_dir=''

for seed in $(seq 0 $((seeds-1))); do
    for env_name in "${envs[@]}"; do
        python -m gcpc.train seed=$seed env_name=$env_name model=policynet exp=$exp_config model.model_config.tjn_ckpt_path=$tjn_ckpt_dir/$env_name.ckpt
    done
done