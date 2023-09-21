from transformers import AutoModelForCausalLM
import sys
import torch
sys.path.append('/home/arnav.chavan/NIPS23/peft/src') 
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf") 
peft_model_id = "/l/users/arnav.chavan/checkpoints_glora_llama2_BS=32_R=4" 
model = PeftModel.from_pretrained(base_model, peft_model_id)
config_path = '/l/users/arnav.chavan/evolution_checkpoints_avg5%_llama2/glora4_BS=32/checkpoint-5.pth.tar'
if config_path:
    i = 0
    ckpt = torch.load(config_path)
    config = ckpt['keep_top_k'][50][0]
    for name, l in model.model.named_modules():
        if any(layer_name in name for layer_name in ["q_proj","k_proj","v_proj"]):
            l.eval_config = config[i]
            i+=1
    print(f'Setup config for {i} layers')
model = model.merge_and_unload()
model.save_pretrained("/l/users/arnav.chavan/merge_glora_llama2")