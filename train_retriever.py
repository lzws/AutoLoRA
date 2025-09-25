from diffsynth import FluxImagePipeline, ModelManager, load_state_dict
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.pipelines.flux_image import lets_dance_flux

from retriever import TextEncoder, LoRAEncoder
from utils import load_lora
import torch, os
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel
import pandas as pd
import random
from PIL import Image
from torchvision.transforms import v2
from diffsynth.data.video import crop_and_resize

class LoraDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch=1000, loras_per_item=1):
        self.base_path = base_path
        data_df = pd.read_csv(metadata_path)
        self.model_file = data_df["model_file"].tolist()
        self.image_file = data_df["image_file"].tolist()
        self.text = data_df["text"].tolist()
        self.text_short = data_df["text_short"].tolist()
        self.text_medium = data_df["text_medium"].tolist()

        self.max_resolution = 1920 * 1080
        self.steps_per_epoch = steps_per_epoch
        self.loras_per_item = loras_per_item
        
        
    def read_image(self, image_file):
        image = Image.open(image_file).convert("RGB")
        width, height = image.size
        if width * height > self.max_resolution:
            scale = (width * height / self.max_resolution) ** 0.5
            image = image.resize((int(width / scale), int(height / scale)))
            width, height = image.size
        if width % 16 != 0 or height % 16 != 0:
            image = crop_and_resize(image, height // 16 * 16, width // 16 * 16)
        image = v2.functional.to_image(image)
        image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        image = v2.functional.normalize(image, [0.5], [0.5])
        return image
    
    
    def get_data(self, data_id):


        model_file = self.model_file[data_id]

        texts = [self.text[data_id], self.text_short[data_id], self.text_medium[data_id]]
        text_id = torch.randint(0, len(texts), (1,))[0]
        text = texts[text_id]
        data = {
            "model_file": model_file,
            "image": self.read_image(self.image_file[data_id]),
            "text": text
        }
        return data


    def __getitem__(self, index):
        data = []
        while len(data) < self.loras_per_item:
            data_id = torch.randint(0, len(self.model_file), (1,))[0]
            data_id = (data_id + index) % len(self.model_file) # For fixed seed.
            data.append(self.get_data(data_id))
        return data

    def __len__(self):
        return self.steps_per_epoch


class LoRARetrieverTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.text_encoder = TextEncoder().to(torch.bfloat16)
        state_dict = load_state_dict("models/FLUX/FLUX.1-dev/text_encoder/model.safetensors")
        self.text_encoder.load_state_dict(TextEncoder.state_dict_converter().from_civitai(state_dict))
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        self.lora_encoder = LoRAEncoder().to(torch.bfloat16)
        
        self.tokenizer = CLIPTokenizer.from_pretrained("diffsynth/tokenizer_configs/stable_diffusion_3/tokenizer_1")
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1. / 0.07)))
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self


    def forward(self, batch):
        text = [data["text"] for data in batch]

        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids.to(self.device)
        text_emb = self.text_encoder(input_ids)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        lora_emb = []
        for data in batch:
            # print(data["model_file"])
            lora = FluxLoRAConverter().align_to_all_format(load_state_dict(data["model_file"], torch_dtype=torch.bfloat16, device=self.device))
            
            lora_emb.append(self.lora_encoder(lora))
        lora_emb = torch.concat(lora_emb)
        lora_emb = lora_emb / lora_emb.norm(dim=-1, keepdim=True)
        
        # logit_scale = self.logit_scale.exp()

        similarity = text_emb @ lora_emb.T
        print(similarity)
        loss = -torch.log(torch.softmax(similarity, dim=0).diag()) - torch.log(torch.softmax(similarity, dim=1).diag())
        loss = 10 * loss.mean()
        return loss
    
    
    def trainable_modules(self):
        return self.lora_encoder.parameters()


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        
    
    def on_step_end(self, loss):
        pass
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).lora_encoder.state_dict()
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-lora_encoder-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


if __name__ == '__main__':
    model = LoRARetrieverTrainingModel()
    metadata_path = 'dataset/train_metadata.csv'

    dataset = LoraDataset("data/lora/models/", metadata_path, steps_per_epoch=100, loras_per_item=32)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1, collate_fn=lambda x: x[0])
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=1e-4)
    model_logger = ModelLogger("models/lora_retriever/", remove_prefix_in_ckpt="lora_encoder.")
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    for epoch_id in range(20):
        for data in tqdm(dataloader):

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                print(loss)
        model_logger.on_epoch_end(accelerator, model, epoch_id)

#  nohup accelerate launch --num_processes 4 --gpu_ids 4,5,6,7 --main_process_port=29502 train_retriever.py > zlog/train_retriever_new/train_retriever_newnorm_Good_and_Civitai_Enhancer_filter_2.log 2>&1 &

