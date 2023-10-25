seeds=5
envs=("kitchen-mixed-v0" "kitchen-partial-v0")
exp_config='kitchen_pl'
tjn_ckpt_dir=''

for seed in $(seq 0 $((seeds-1))); do
    for env_name in "${envs[@]}"; do
        python -m gcpc.train seed=$seed env_name=$env_name model=policynet exp=$exp_config model.model_config.tjn_ckpt_path=$tjn_ckpt_dir/$env_name.ckpt
    done
done