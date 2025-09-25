from diffsynth import FluxImagePipeline, ModelManager, load_state_dict
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.pipelines.flux_image import lets_dance_flux
from dataset import LoraDataset
from merger import LoraPatcher
from utils import load_lora
import torch, os
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
import pandas as pd


device = 'cuda'
model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device)
model_manager.load_models([
    "FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "FLUX/FLUX.1-dev/text_encoder_2",
    "FLUX/FLUX.1-dev/ae.safetensors",
    "FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager)
pipe.enable_auto_lora()

patcher_path = ''
lora_patcher = LoraPatcher(merger_type='no_base').to(dtype=torch.bfloat16, device=device)
lora_patcher.load_state_dict(load_state_dict(patcher_path1))

dataset = pd.read_csv('dataset.csv')

text = []
model_urls = []
case_num = []
for seed in range(100):
    batch = dataset.iloc[seed]
    num_lora = torch.randint(3, len(batch), (1,))[0]
    lora_state_dicts = [
        FluxLoRAConverter().align_to_all_format(load_state_dict(batch[i]["model_file"],torch_dtype=torch.bfloat16, device=device)) for i in range(num_lora)
    ]
    urls = []
    for i in range(num_lora):
        model_file = batch[i]["model_file"]
        name = model_file.split("/")[-2]
        ower = model_file.split("/")[-3]
        urls.append(f'https://www.modelscope.cn/models/{ower}/{name}')
    urls = ",".join(urls)

    image = pipe(
        prompt=batch[0]["text"],
        seed=seed,
    )
    image.save(f"{save_path}/image_{seed}_nolora.jpg")
    for i in range(num_lora):
        image = pipe(
            prompt=batch[0]["text"],
            lora_state_dicts=[lora_state_dicts[i]], 
            lora_patcher=None,
            seed=seed,
        )
        image.save(f"{save_path}/image_{seed}_{i}.jpg")

    image = pipe(
        prompt=batch[0]["text"],
        lora_state_dicts=lora_state_dicts, 
        lora_patcher=lora_patcher,
        seed=seed,
    )
    image.save(f"{save_path}/image_{seed}_merger1.jpg")

    image = pipe(
        prompt=batch[0]["text"],
        lora_state_dicts=lora_state_dicts, 
        lora_patcher=lora_patcher2,
        seed=seed,
    )
    image.save(f"{save_path}/image_{seed}_merger2.jpg")

    image = pipe(
        prompt=batch[0]["text"],
        lora_state_dicts=lora_state_dicts, 
        lora_patcher=lora_patcher3,
        seed=seed,
    )
    image.save(f"{save_path}/image_{seed}_merger3.jpg")


    text.append(batch[0]["text"])
    model_urls.append(urls)
    case_num.append(seed)

    df_new = pd.DataFrame({"case_num":case_num, "model_urls": model_urls,"text": text})
    df_new.to_csv(f"{save_path}/metadata.csv", index=False)
    
