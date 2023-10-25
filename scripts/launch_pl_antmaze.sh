seeds=5
envs=("antmaze-umaze-v2" "antmaze-umaze-diverse-v2" "antmaze-medium-diverse-v2" "antmaze-medium-play-v2" "antmaze-large-diverse-v2" "antmaze-large-play-v2" "antmaze-ultra-diverse-v0" "antmaze-ultra-play-v0")
exp_config='antmaze_pl'
tjn_ckpt_dir=''

for seed in $(seq 0 $((seeds-1))); do
    for env_name in "${envs[@]}"; do
        python -m gcpc.train seed=$seed env_name=$env_name model=policynet exp=$exp_config model.model_config.tjn_ckpt_path=$tjn_ckpt_dir/$env_name.ckpt
    done
done