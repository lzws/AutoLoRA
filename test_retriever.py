from diffsynth import FluxImagePipeline, ModelManager, load_state_dict
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.pipelines.flux_image import lets_dance_flux
from dataset import LoraDataset
from retriever import TextEncoder, LoRAEncoder
from merger import LoraPatcher
# from utils import load_lora
import torch, os
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel
import pandas as pd



class LoRARetrieverTrainingModel(torch.nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        
        self.text_encoder = TextEncoder().to(torch.bfloat16)
        state_dict = load_state_dict("models/FLUX/FLUX.1-dev/text_encoder/model.safetensors")
        self.text_encoder.load_state_dict(TextEncoder.state_dict_converter().from_civitai(state_dict))
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        self.lora_encoder = LoRAEncoder().to(torch.bfloat16)
        state_dict = load_state_dict(pretrained_path)
        self.lora_encoder.load_state_dict(state_dict)
        
        self.tokenizer = CLIPTokenizer.from_pretrained("diffsynth/tokenizer_configs/stable_diffusion_3/tokenizer_1")
        
        
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
        text_emb = text_emb / text_emb.norm()
        
        lora_emb = []
        for data in batch:
            lora = FluxLoRAConverter().align_to_all_format(load_lora(data["model_file"],torch_dtype=torch.bfloat16, device=self.device))
            lora_emb.append(self.lora_encoder(lora))
        lora_emb = torch.concat(lora_emb)
        lora_emb = lora_emb / lora_emb.norm()
        
        similarity = text_emb @ lora_emb.T
        print(similarity)
        loss = -torch.log(torch.softmax(similarity, dim=0).diag()) - torch.log(torch.softmax(similarity, dim=1).diag())
        loss = 10 * loss.mean()
        return loss
    
    
    def trainable_modules(self):
        return self.lora_encoder.parameters()
    
    @torch.no_grad()
    def process_lora_list(self, lora_list,all_loras_embedding=None):
        if all_loras_embedding is not None:
            all_loras_embedding = torch.load(all_loras_embedding,weights_only=True)
            lora_name_list = list(all_loras_embedding.keys())
            lora_embedding_list = list(all_loras_embedding.values())
            lora_embeddings = torch.concat(lora_embedding_list)
            lora_embeddings = torch.nn.functional.normalize(lora_embeddings, dim=-1)

            self.lora_emb = lora_embeddings
            self.lora_list = lora_name_list
        else:
            lora_emb = []
            all_loras_embedding = {}
            for lora_path in tqdm(lora_list):
                lora = FluxLoRAConverter().align_to_all_format(load_state_dict(lora_path, torch_dtype=torch.bfloat16, device=self.device))
                lemb = self.lora_encoder(lora)
                lora_emb.append(lemb)
                all_loras_embedding[lora_path] = lemb
            lora_emb = torch.concat(lora_emb)
            lora_emb = lora_emb / lora_emb.norm()
            torch.save(all_loras_embedding, f'loras_{len(lora_list)}_embeddings.ckpt')
            
            self.lora_emb = lora_emb
            self.lora_list = lora_list
    
    @torch.no_grad()
    def retrieve(self, text, k=1):
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids.to(self.device)
        text_emb = self.text_encoder(input_ids)
        text_emb = text_emb / text_emb.norm()
        
        similarity = text_emb @ self.lora_emb.T
        topk = torch.topk(similarity, k, dim=1).indices[0]
        
        lora_list = []
        model_url_list = []
        for lora_id in topk:
            print(self.lora_list[lora_id])
            lora = FluxLoRAConverter().align_to_all_format(load_state_dict(self.lora_list[lora_id],torch_dtype=torch.bfloat16, device=self.device))
            lora_list.append(lora)
            model_id = self.lora_list[lora_id].split("/")[3:5]
            model_url_list.append(f"https://www.modelscope.cn/models/{model_id[0]}/{model_id[1]}")
        return lora_list, model_url_list
    
    @torch.no_grad()
    def retrieve_rank(self, text,target_name,k=1):

        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).input_ids.to(self.device)
        text_emb = self.text_encoder(input_ids)
        text_emb = text_emb / text_emb.norm()
        
        score = text_emb @ self.lora_emb.T

        zipped_list = zip(score[0], self.lora_list)
        sorted_list = sorted(zipped_list, key=lambda x: x[0], reverse=True)
        sorted_score_list, sorted_name_list = zip(*sorted_list)
        sorted_score_list = list(sorted_score_list)
        sorted_name_list = list(sorted_name_list)
        # print(f"sorted_name_list: {len(sorted_name_list)}")

        target_name_rank = sorted_name_list.index(target_name)

        return target_name_rank, sorted_score_list, sorted_name_list




retriever = LoRARetrieverTrainingModel("models/lora_retriever/epoch-19.safetensors").to(dtype=torch.bfloat16, device="cuda")

retriever.process_lora_list(list(set(i for i in pd.read_csv("dataset/metadata.csv")["model_file"])),all_loras_embedding="loras_embeddings.ckpt")



dataset = pd.read_csv('dataset/test.csv')

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

text_list = []
model_url_list = []
for seed in range(100):
    text = dataset.iloc[seed]["text"]
    print(text)
    loras, urls = retriever.retrieve(text, k=3)
    print(urls)
    image = pipe(
        prompt=text,
        seed=seed,
    )
    image.save(f"data/lora_outputs/image_{seed}_top0.jpg")
    for i in range(2, 3):
        image = pipe(
            prompt=text,
            lora_state_dicts=loras[:i+1],
            lora_patcher=lora_patcher,
            seed=seed,
        )
        image.save(f"data/lora_outputs/image_{seed}_top{i+1}.jpg")
        
    text_list.append(text)
    model_url_list.append(urls)
    df = pd.DataFrame()
    df["text"] = text_list
    df["models"] = [",".join(i) for i in model_url_list]
    df.to_csv("data/lora_outputs/metadata.csv", index=False, encoding="utf-8-sig")


