import torch
import pandas as pd
from PIL import Image
import lightning as pl
from diffsynth import ModelManager, FluxImagePipeline, download_models, load_state_dict
from diffsynth.models.lora import LoRAFromCivitai, FluxLoRAConverter, FluxLoRAFromCivitai
from diffsynth.data.video import crop_and_resize
from diffsynth.pipelines.flux_image import lets_dance_flux
from torchvision.transforms import v2
from lightning.pytorch.strategies import DDPStrategy
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from merger import LoraPatcher5
from utils import LoRA_State_dicts_Decomposition



class LoraMergerNoCross(torch.nn.Module):
    def __init__(self, dim, w=0.512):
        super().__init__()

        self.weight_lora = torch.nn.Parameter(torch.randn((dim,)))
        self.weight_out = torch.nn.Parameter(torch.ones((dim,)) * w)
        self.bias = torch.nn.Parameter(torch.randn((dim,)))
        self.activation = torch.nn.Sigmoid()
        self.norm_base = torch.nn.LayerNorm(dim, eps=1e-5)
        self.norm_lora = torch.nn.LayerNorm(dim, eps=1e-5)
        
    def forward(self, base_output, lora_outputs):
        norm_base_output = self.norm_base(base_output)
        norm_lora_outputs = self.norm_lora(lora_outputs)
        gate = self.activation(
            norm_lora_outputs * self.weight_lora \
            + self.bias
        )
        
        output = base_output + (self.weight_out * gate * lora_outputs).sum(dim=0)
        return output

class LoraMergerNobase(torch.nn.Module):
    def __init__(self, dim, w=0.512):
        super().__init__()


        self.weight_lora = torch.nn.Parameter(torch.randn((dim,)))
        self.weight_cross = torch.nn.Parameter(torch.randn((dim,)))

        self.weight_out = torch.nn.Parameter(torch.ones((dim,)) * w)
        self.bias = torch.nn.Parameter(torch.randn((dim,)))
        self.activation = torch.nn.Sigmoid()
        self.norm_base = torch.nn.LayerNorm(dim, eps=1e-5)
        self.norm_lora = torch.nn.LayerNorm(dim, eps=1e-5)
        
    def forward(self, base_output, lora_outputs):
        norm_base_output = self.norm_base(base_output)
        norm_lora_outputs = self.norm_lora(lora_outputs)
        gate = self.activation(
            + norm_lora_outputs * self.weight_lora \
            + (norm_base_output * norm_lora_outputs) * self.weight_cross + self.bias
        )
        
        output = base_output + (self.weight_out * gate * lora_outputs).sum(dim=0)
        return output

class LoraMergerNoLoRA(torch.nn.Module):
    def __init__(self, dim, w=0.512):
        super().__init__()
        self.weight_base = torch.nn.Parameter(torch.randn((dim,)))

        self.weight_cross = torch.nn.Parameter(torch.randn((dim,)))
        self.weight_out = torch.nn.Parameter(torch.ones((dim,)) * w)
        self.bias = torch.nn.Parameter(torch.randn((dim,)))
        self.activation = torch.nn.Sigmoid()
        self.norm_base = torch.nn.LayerNorm(dim, eps=1e-5)
        self.norm_lora = torch.nn.LayerNorm(dim, eps=1e-5)
        
    def forward(self, base_output, lora_outputs):
        norm_base_output = self.norm_base(base_output)
        norm_lora_outputs = self.norm_lora(lora_outputs)
        gate = self.activation(
            norm_base_output * self.weight_base \
            # + norm_lora_outputs * self.weight_lora \
            + (norm_base_output * norm_lora_outputs) * self.weight_cross + self.bias
        )
        
        output = base_output + (self.weight_out * gate * lora_outputs).sum(dim=0)
        return output



merger_class= {
    'no_base': LoraMergerNobase,
    'no_cross': LoraMergerNoCross,

}


class LoraPatcher(torch.nn.Module):
    def __init__(self, lora_patterns=None, merger_type='all',w=0.512):
        super().__init__()
        if lora_patterns is None:
            lora_patterns = self.default_lora_patterns()
        model_dict = {}
        for lora_pattern in lora_patterns:
            name, dim = lora_pattern["name"], lora_pattern["dim"]
            model_dict[name.replace(".", "___")] = merger_class[merger_type](dim,w)
        self.model_dict = torch.nn.ModuleDict(model_dict)
        
    def default_lora_patterns(self):
        lora_patterns = []
        lora_dict = {
            "attn.a_to_qkv": 9216, "attn.a_to_out": 3072, "ff_a.0": 12288, "ff_a.2": 3072, "norm1_a.linear": 18432,
            "attn.b_to_qkv": 9216, "attn.b_to_out": 3072, "ff_b.0": 12288, "ff_b.2": 3072, "norm1_b.linear": 18432,
        }
        for i in range(19):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix]
                })
        lora_dict = {"to_qkv_mlp": 21504, "proj_out": 3072, "norm.linear": 9216}
        for i in range(38):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"single_blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix]
                })
        return lora_patterns
        
    def forward(self, base_output, lora_outputs, name):
        return self.model_dict[name.replace(".", "___")](base_output, lora_outputs)


class LoraDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, steps_per_epoch=1000,loras_per_item=1):
        data_df = pd.read_csv(metadata_path)
        self.model_file = data_df["model_file"].tolist()
        self.image_file = data_df["image_file"].tolist()
        self.text = data_df["text"].tolist()
        if 'text_short' in data_df.columns:
            self.text_short = data_df["text_short"].tolist()
            self.text_medium = data_df["text_medium"].tolist()
        else:
            self.text_short = data_df["text"].tolist()
            self.text_medium = data_df["text"].tolist()
        if 'description_en' in data_df.columns:
            self.description_en = data_df["description_en"].tolist()
        else:
            self.description_en = ['missing'] * len(self.text)

        self.max_resolution = 1920 * 1080
        self.steps_per_epoch = steps_per_epoch
        self.loras_per_item=loras_per_item
        
        
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


        texts = [self.text[data_id], self.text_short[data_id], self.text_medium[data_id]]
        text_id = torch.randint(0, len(texts), (1,))[0]
        text = texts[text_id]
        if self.description_en[data_id] != 'missing':
            if torch.rand(1).item() < 0.5:
                text_id = torch.randint(1, len(texts), (1,))[0]
                text = texts[text_id]
            else:
                text = self.description_en[data_id] + ' ' + texts[1]
            
        data = {
            "model_file": self.model_file[data_id],
            "image": self.read_image(self.image_file[data_id]),
            "text": text
        }
        return data


    def __getitem__(self, index):
        data = []
        models = []
        while len(data) < self.loras_per_item:
            data_id = torch.randint(0, len(self.model_file), (1,))[0]
            data_id = (data_id + index) % len(self.model_file) # For fixed seed.
            data_w = self.get_data(data_id)
            if data_w['model_file'] not in models:
                models.append(data_w['model_file'])
                data.append(data_w)
        return data


    def __len__(self):
        return self.steps_per_epoch


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-4,
        use_gradient_checkpointing=True,
        state_dict_converter=FluxLoRAConverter.align_to_diffsynth_format,
        merger_type='all',
        patcher_path=None,
        rank = 4,
        wei=0.512,
        cfg=0.3

    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device=self.device)
        model_manager.load_models([
            "FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            "FLUX/FLUX.1-dev/text_encoder_2",
            "FLUX/FLUX.1-dev/ae.safetensors",
            "FLUX/FLUX.1-dev/flux1-dev.safetensors"
        ])
        self.pipe = FluxImagePipeline.from_model_manager(model_manager)
        self.lora_patcher = LoraPatcher(merger_type=merger_type,w=wei)
        if patcher_path is not None:
            print(f"Load patcher from {patcher_path}")
            self.lora_patcher.load_state_dict(load_state_dict(patcher_path, torch_dtype=torch.bfloat16, device=self.device))
        self.lora_patcher.train()
        self.pipe.enable_auto_lora()
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        # Set parameters
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.state_dict_converter = FluxLoRAFromCivitai()
        self.rank = rank
        self.cfg=cfg


    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()


    def training_step(self, batch, batch_idx):
        # Data
        batch = batch[0]
        text = batch[0]["text"]
        if torch.rand(1).item() < self.cfg:
            text = ['']
        image = batch[0]["image"].unsqueeze(0)

        text_2 = batch[1]['text']
        image_2 = batch[1]['image'].unsqueeze(0)


        num_lora = 2
        lora_state_dicts = [
            FluxLoRAConverter().align_to_all_format(load_state_dict(batch[i]["model_file"], torch_dtype=torch.bfloat16, device=self.device)) for i in range(num_lora)
        ]

        new_state_dict = LoRA_State_dicts_Decomposition(lora_state_dicts,self.rank)
        lora_state_dicts.append(new_state_dict)


        if lora_state_dicts[0] == {}:
            print(f"+++++++++++++++++ {batch['model_file']}  No lora state dict. ++++++++++++++++++++++++")
        if lora_state_dicts[1] == {}:
            print(f"+++++++++++++++++ {batch['model_file_extra']}  No lora state dict. ++++++++++++++++++++++++")
        lora_alpahs = [1, 1, 1, 1]

        # Prepare input parameters
        self.pipe.device = self.device
        prompt_emb = self.pipe.encode_prompt(text, positive=True)
        prompt_emb_2 = self.pipe.encode_prompt(text_2, positive=True)
        p_embs = [prompt_emb, prompt_emb_2]
        if "latents" in batch:
            latents = batch["latents"].to(dtype=self.pipe.torch_dtype, device=self.device)
        else:
            latents = self.pipe.vae_encoder(image.to(dtype=self.pipe.torch_dtype, device=self.device))

            all_latents = [latents]

        

        #compute single lora output 
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        noise = torch.randn_like(all_latents[0])
        extra_input = self.pipe.prepare_extra_input(all_latents[0])
        noisy_latents = self.pipe.scheduler.add_noise(all_latents[0], noise, timestep)
        training_target = self.pipe.scheduler.training_target(all_latents[0], noise, timestep)

        # Compute lora 1 loss
        noise_pred = lets_dance_flux(
            self.pipe.dit,
            hidden_states=noisy_latents, timestep=timestep, **p_embs[0], **extra_input,
            lora_state_dicts=lora_state_dicts, lora_alpahs=lora_alpahs, lora_patcher=self.lora_patcher,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        loss_1 = torch.nn.functional.mse_loss(training_target.float(), noise_pred.float())

        # Compute diff loss
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        noise = torch.randn_like(all_latents[0])
        extra_input = self.pipe.prepare_extra_input(all_latents[0])
        noisy_latents = self.pipe.scheduler.add_noise(all_latents[0], noise, timestep)
        training_target = self.pipe.scheduler.training_target(all_latents[0], noise, timestep)

        noise_pred = lets_dance_flux(
            self.pipe.dit,
            hidden_states=noisy_latents, timestep=timestep, **p_embs[1], **extra_input,
            lora_state_dicts=lora_state_dicts, lora_alpahs=lora_alpahs, lora_patcher=self.lora_patcher,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        loss_2 = torch.nn.functional.mse_loss(training_target.float(), noise_pred.float())

        loss = loss_1 + max(0, 0.5 - loss_2)


        print(f"loss: {loss}")
        
        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.lora_patcher.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'interval': 'epoch'  
        }
           

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        checkpoint.update(self.lora_patcher.state_dict())




import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set_name", type=str, help="")
    parser.add_argument("--merger_type", type=str, help="")
    parser.add_argument("--devices", nargs='+', type=int, help="--devices 0 1 2")
    parser.add_argument("--rank", type=int, help="")
    parser.add_argument("--wei", type=float, default=0.5, help="")
    parser.add_argument("--cfg", type=float, default=0.3, help="")
    args = parser.parse_args()


    train_set_name = args.train_set_name

    merger_type = args.merger_type
    devices = args.devices
    rank = args.rank
    wei = args.wei
    cfg = args.cfg
    print(f"train_set_name: {train_set_name}, merger_type: {merger_type}, devices: {devices}")
    model = LightningModel(learning_rate=1e-4, merger_type=merger_type, patcher_path=patcher_path, rank=rank, wei=wei, cfg=cfg)
    dataset = LoraDataset(f"dataset/{train_set_name}.csv", steps_per_epoch=400,loras_per_item=2)


    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1, collate_fn=lambda x: x)
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=devices,
        precision="bf16",

        strategy="auto",
        default_root_dir=f"./models/lora_merger/{train_set_name}",
        accumulate_grad_batches=1,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1,every_n_epochs=2)],
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


# nohup python3 train_merger_new_decomposition.py --train_set_name "training_dataset" --merger_type "no_base" --devices 2 3 4 5 --rank 2 --wei 0.512 --cfg 0.3 > log.log 2>&1 &
