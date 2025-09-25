import torch, copy
from ..models.utils import init_weights_on_device


def cast_to(weight, dtype, device):
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight)
    return r


class AutoWrappedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device):
        super().__init__()
        self.module = module.to(dtype=offload_dtype, device=offload_device)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            module = self.module
        else:
            module = copy.deepcopy(self.module).to(dtype=self.computation_dtype, device=self.computation_device)
        return module(*args, **kwargs)
    

class AutoWrappedLinear(torch.nn.Linear):
    def __init__(self, module: torch.nn.Linear, offload_dtype, offload_device, onload_dtype, onload_device, computation_dtype, computation_device):
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None, dtype=offload_dtype, device=offload_device)
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, x, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            weight, bias = self.weight, self.bias
        else:
            weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
            bias = None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
        return torch.nn.functional.linear(x, weight, bias)



class AutoLoRAWraper(torch.nn.Module):
    def __init__(self,module: torch.nn.Linear, name='', device=None, dtype=None):
        super().__init__()
        self.module = module
        self.name = name
        self.h = torch.nn.Linear(1, 1, bias=False)
        self.f = torch.nn.Linear(module.out_features,1, bias=False)
        self.g = torch.nn.Linear(module.out_features,1, bias=False)
    def forward(self, x, lora_state_dicts=[], lora_alpahs=[], **kwargs):
        out = self.module(x) # (1, seq_len, hidden_size) or (1, hidden_size)
        lora_a_name = f'{self.name}.lora_A.weight'
        lora_b_name = f'{self.name}.lora_B.weight'
        
        out_ = out / out.norm(dim=-1, keepdim=True)
        for _, lora_state_dict in enumerate(lora_state_dicts):
            if lora_a_name in lora_state_dict and lora_b_name in lora_state_dict:
                # print(f'lora_a_name: {lora_a_name}')
                lora_A = lora_state_dict[lora_a_name]
                lora_B = lora_state_dict[lora_b_name]
                out_lora = x @ lora_A.T @ lora_B.T

                out_lora_ = out_lora / out_lora.norm(dim=-1, keepdim=True)
                a = self.f(out_.to(self.f.weight.dtype))
                b = self.g(out_lora_.to(self.g.weight.dtype))
                alpha = self.h(a + b).to(out_lora.dtype) # (1, seq_len, 1) or (1, 1)

                out = out + out_lora * alpha
        return out

class AutoLoRALinear(torch.nn.Linear):
    def __init__(self, name='', in_features=1, out_features=2, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.name = name

    def forward(self, x, lora_state_dicts=[], lora_alpahs=[1.0,1.0], lora_patcher=None, t=1, **kwargs):
        
        flag = False
        if len(lora_state_dicts) > 0:
            for k,v in lora_state_dicts[0].items():
                if 'lora_A.' not in k and 'lora_B.' not in k:
                    flag = True
                    break
        weight_name = f'{self.name}.weight'
        if flag and weight_name in lora_state_dicts[0]:
            print(f'{self.name}: {weight_name}')
            lora_weight = lora_state_dicts[0][weight_name]
            out = torch.nn.functional.linear(x, self.weight+lora_weight, self.bias)
            return out

        out = torch.nn.functional.linear(x, self.weight, self.bias)

        lora_a_name = f'{self.name}.lora_A.weight'
        lora_b_name = f'{self.name}.lora_B.weight'


        lora_output = []

        for i, lora_state_dict in enumerate(lora_state_dicts):

            if lora_a_name in lora_state_dict and lora_b_name in lora_state_dict:
                lora_A = lora_state_dict[lora_a_name]
                lora_B = lora_state_dict[lora_b_name]
                out_lora = x @ lora_A.T @ lora_B.T
                if lora_patcher is None:
                    out = out + out_lora * lora_alpahs[i]
                lora_output.append(out_lora * lora_alpahs[i])

        if len(lora_output) > 0 and lora_patcher is not None:    
            lora_output = torch.stack(lora_output)
            out = lora_patcher(out, lora_output, self.name, t)
        return out


class AutoRetrievalLoRALinear(torch.nn.Linear):
    def __init__(self, name='', in_features=1, out_features=2, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.name = name

    def forward(self, x, lora_state_dicts=[], lora_alpahs=[1.0,1.0], lora_patcher=None, target_layer_name=['blocks.0.norm1_b.linear'], t=1, target_step=[1], **kwargs):
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_a_name = f'{self.name}.lora_A.weight'
        lora_b_name = f'{self.name}.lora_B.weight'

        # print(f't: {t},target_step: {target_step}')

        if self.name in target_layer_name and t in target_step:
            print(f'start retrieve... {self.name}')
            _ = lora_patcher(out, x, self.name, t)
        return out


class KLoRALinear(torch.nn.Linear):
    def __init__(self, name='', in_features=1, out_features=2, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.name = name
        self.pattern = "s*"

    def get_klora_weight(self, timestep=1, lora_state_dicts=[],sum_timesteps=13832,average_ratio=1,alpha=1.5,beta=1.275):
        # sum_timesteps = self.sum_timesteps
        
        # alpha = self.alpha
        # beta = self.beta
        gamma = average_ratio

        lora_a_name = f'{self.name}.lora_A.weight'
        lora_b_name = f'{self.name}.lora_B.weight'


        matrix1 = None
        matrix2 = None
        if lora_a_name in lora_state_dicts[0]:
            weight_1_a = lora_state_dicts[0][lora_b_name]
            weight_1_b = lora_state_dicts[0][lora_a_name]
            matrix1 = weight_1_a @ weight_1_b
            
        if lora_a_name in lora_state_dicts[1]:
            weight_2_a = lora_state_dicts[1][lora_b_name]
            weight_2_b = lora_state_dicts[1][lora_a_name]
            matrix2 = weight_2_a @ weight_2_b


        if matrix1 == None:
            return matrix2

        if matrix2 == None:
            return matrix1
        
        k = weight_1_a.shape[1] * weight_2_a.shape[1]

        
        # compute the sum of top k values
        time_ratio = timestep % sum_timesteps

        abs_matrix = torch.abs(matrix1)
        top_k_values, _ = torch.topk(abs_matrix.flatten(), k)
        top_k_sum1 = top_k_values.sum()

        abs_matrix = torch.abs(matrix2)
        top_k_values, _ = torch.topk(abs_matrix.flatten(), k)
        top_k_sum2 = top_k_values.sum()
        
        
        scale = alpha * time_ratio / sum_timesteps + beta
        if self.pattern == "s*":
            scale = scale % alpha   
        # apply scaling factor to the sum of top k values
        top_k_sum1 = top_k_sum1 / gamma
        top_k_sum2 = top_k_sum2 * scale
        
        temp_ratio = top_k_sum1 / top_k_sum2
        if temp_ratio > 1:
            return matrix1
        else:
            return matrix2

    def forward(self, x, lora_state_dicts=[{},{}], lora_alpahs=[1.0,1.0], lora_patcher=None, sum_timesteps=13832,average_ratio=1,alpha=1.5,beta=1.275,t=1):
        out = torch.nn.functional.linear(x, self.weight, self.bias)

        lora_a_name = f'{self.name}.lora_A.weight'
        if lora_a_name in lora_state_dicts[0] or lora_a_name in lora_state_dicts[1]:
            weight = self.get_klora_weight(timestep=t, lora_state_dicts=lora_state_dicts, sum_timesteps=sum_timesteps,average_ratio=average_ratio,alpha=alpha,beta=beta)
            if weight is not None:
                hidden_states = torch.nn.functional.linear(x, weight=weight)
                out = out + hidden_states
        return out


class DARELoRALinear(torch.nn.Linear):
    def __init__(self, name='', in_features=1, out_features=2, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.name = name
        self.p=0.8

    def forward(self, x, lora_state_dicts=[], lora_alpahs=[1.0,1.0], lora_patcher=None, **kwargs):
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_a_name = f'{self.name}.lora_A.weight'
        lora_b_name = f'{self.name}.lora_B.weight'

        # print(f'{self.name}: {lora_a_name} or {lora_b_name}')

        lora_output = []
        for i, lora_state_dict in enumerate(lora_state_dicts):

            if lora_a_name in lora_state_dict and lora_b_name in lora_state_dict:
                lora_A = lora_state_dict[lora_a_name]
                lora_B = lora_state_dict[lora_b_name]
                matrix1 = lora_B @ lora_A
                m_t = torch.bernoulli(torch.full_like(matrix1, self.p))
                matrix1 = (1 - m_t) * matrix1
                matrix1 = matrix1 / (1 - self.p)
                out_lora = x @ matrix1.T
                out = out + out_lora * lora_alpahs[i]

        return out



def inject_lora_wraper(model:torch.nn.Module, module_map: dict, name_prefix=''):
    targets = list(module_map.keys())
    for name, module in model.named_children():
        if name_prefix != '':
            full_name = name_prefix + '.' + name
        else:
            full_name = name
        if isinstance(module, targets[1]):
            new_module = AutoLoRAWraper(module=module, name=full_name)
            setattr(model, name, new_module)
        elif isinstance(module, targets[0]):
            pass
        else:
            inject_lora_wraper(module, module_map, full_name)
    

def enable_auto_lora(model:torch.nn.Module, module_map: dict, name_prefix=''):
    targets = list(module_map.keys())
    for name, module in model.named_children():
        if name_prefix != '':
            full_name = name_prefix + '.' + name
        else:
            full_name = name
        if isinstance(module,targets[1]):
            # ToDo: replace the linear to the AutoLoRALinear 
            new_module = AutoLoRALinear(
                name=full_name, 
                in_features=module.in_features, 
                out_features=module.out_features, 
                bias=module.bias is not None, 
                device=module.weight.device, 
                dtype=module.weight.dtype)
            new_module.weight.data.copy_(module.weight.data)
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
        elif isinstance(module, targets[0]):
            pass
        else:
            enable_auto_lora(module, module_map, full_name)

def enable_k_lora(model:torch.nn.Module, module_map: dict, name_prefix=''):
    targets = list(module_map.keys())
    for name, module in model.named_children():
        if name_prefix != '':
            full_name = name_prefix + '.' + name
        else:
            full_name = name
        if isinstance(module,targets[1]):

            new_module = KLoRALinear(
                name=full_name, 
                in_features=module.in_features, 
                out_features=module.out_features, 
                bias=module.bias is not None, 
                device=module.weight.device, 
                dtype=module.weight.dtype)
            new_module.weight.data.copy_(module.weight.data)
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
        elif isinstance(module, targets[0]):
            pass
        else:
            enable_k_lora(module, module_map, full_name)

def enable_DARE_lora(model:torch.nn.Module, module_map: dict, name_prefix=''):
    targets = list(module_map.keys())
    for name, module in model.named_children():
        if name_prefix != '':
            full_name = name_prefix + '.' + name
        else:
            full_name = name
        if isinstance(module,targets[1]):

            # ToDo: replace the linear to the AutoLoRALinear 
            new_module = DARELoRALinear(
                name=full_name, 
                in_features=module.in_features, 
                out_features=module.out_features, 
                bias=module.bias is not None, 
                device=module.weight.device, 
                dtype=module.weight.dtype)
            new_module.weight.data.copy_(module.weight.data)
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
        elif isinstance(module, targets[0]):
            pass
        else:
            enable_DARE_lora(module, module_map, full_name)

def enable_retrieve_lora(model:torch.nn.Module, module_map: dict, name_prefix=''):
    targets = list(module_map.keys())
    for name, module in model.named_children():
        if name_prefix != '':
            full_name = name_prefix + '.' + name
        else:
            full_name = name
        if isinstance(module,targets[1]):
            # print(full_name)
            # print(module)
            # ToDo: replace the linear to the AutoLoRALinear 
            new_module = AutoRetrievalLoRALinear(
                name=full_name, 
                in_features=module.in_features, 
                out_features=module.out_features, 
                bias=module.bias is not None, 
                device=module.weight.device, 
                dtype=module.weight.dtype)
            new_module.weight.data.copy_(module.weight.data)
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)
        elif isinstance(module, targets[0]):
            pass
        else:
            enable_retrieve_lora(module, module_map, full_name)


def enable_vram_management_recursively(model: torch.nn.Module, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None, total_num_param=0):
    for name, module in model.named_children():
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                if max_num_param is not None and total_num_param + num_param > max_num_param:
                    module_config_ = overflow_module_config
                else:
                    module_config_ = module_config
                module_ = target_module(module, **module_config_)
                setattr(model, name, module_)
                total_num_param += num_param
                break
        else:
            total_num_param = enable_vram_management_recursively(module, module_map, module_config, max_num_param, overflow_module_config, total_num_param)
    return total_num_param


def enable_vram_management(model: torch.nn.Module, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None):
    enable_vram_management_recursively(model, module_map, module_config, max_num_param, overflow_module_config, total_num_param=0)
    model.vram_management_enabled = True

