



python_file=android_control_low




base_model_path="/model/UI-TARS-2B-SFT"

exp_dir='./exp'


mkdir -p ${exp_dir}



cp ${python_file}.py ${exp_dir}



python -m vllm.entrypoints.openai.api_server --served-model-name ui-tars --model ${base_model_path} --gpu_memory_utilization 0.6 -tp 1  --trust-remote-code &


sleep 200


python ${python_file}.py \
    --exp_dir ${exp_dir}  --base_model_path ${base_model_path}  >>  ${exp_dir}/infer_androidcontrol.log 2>&1

