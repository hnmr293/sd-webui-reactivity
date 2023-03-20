# =======================================================================================
from scripts.reactivitylib.utils import ensure_install
ensure_install('plotly')
# =======================================================================================

import sys
import math
import traceback
from dataclasses import dataclass
import heapq
from typing import List, Union, Dict, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import open_clip
import tqdm
import gradio as gr

from modules import script_callbacks
from modules import shared
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWordsBase as CLIP
from modules.ui import refresh_symbol
from modules.ui_components import ToolButton

from scripts.reactivitylib.sdhook import each_unet_attn_layers
from scripts.reactivitylib.lora import reload_loras, available_loras, load_lora
from scripts.reactivitylib.utils import each_slice


NAME = 'Reactivity'

class ClipWrapper:
    def __init__(self, te: CLIP):
        self.te = te
        self.v1 = hasattr(te.wrapped, 'tokenizer')
        self.t = (
            te.wrapped.tokenizer if self.v1
            else open_clip.tokenizer._tokenizer
        )
    
    def token_to_id(self, token: str) -> int:
        if self.v1:
            return self.t._convert_token_to_id(token) # type: ignore
        else:
            return self.t.encoder[token]
    
    def id_to_token(self, id: int) -> str:
        if self.v1:
            return self.t.convert_ids_to_tokens(id) # type: ignore
        else:
            return self.t.decoder[id]
    
    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        if self.v1:
            return self.t.convert_ids_to_tokens(ids) # type: ignore
        else:
            return [self.t.decoder[id] for id in ids]
    
    @property
    def tokenizer(self):
        return self.t


@dataclass
class Item:
    clip_norm: float
    token_index: int
    token: str
    norm_repr: float
    
    @classmethod
    def create_from(cls, this: 'Item'):
        return cls(this.clip_norm, this.token_index, this.token, this.norm_repr)

@dataclass
class ItemAsc(Item):
    # ascending
    def __lt__(self, other: Item):
        return self.norm_repr < other.norm_repr

@dataclass
class ItemDes(Item):
    # descending
    def __lt__(self, other: Item):
        return self.norm_repr > other.norm_repr

class Queue:
    
    def __init__(self, size: int):
        self.max_size = size
        self.q_min: List[ItemAsc] = []
        self.q_max: List[ItemDes] = []
    
    def queue(self, item: Item):
        self.queue_min(item)
        self.queue_max(item)
    
    def queue_min(self, item: Item):
        item = ItemAsc.create_from(item)
        self._q(item, self.q_min)
    
    def queue_max(self, item: Item):
        item = ItemDes.create_from(item)
        self._q(item, self.q_max)
    
    def _q(self, item, q):
        if len(q) < self.max_size:
            heapq.heappush(q, item)
        else:
            heapq.heappushpop(q, item)

def compute_kv_model(unet: nn.Module, context: Tensor):
    kvs: List[float] = []
    
    for name, layer in each_unet_attn_layers(unet):
        if 'xattn' not in name:
            continue
        
        to_k = layer.to_k.to('cuda')
        to_v = layer.to_v.to('cuda')
        kv = compute_kv(to_k, to_v, context)
        kvs.append(kv)
    
    return kvs

def compute_kv(
    to_k: nn.Module,
    to_v: nn.Module,
    context: Tensor,
) -> float:
    k = to_k(context)
    v = to_v(context)
    
    kv = torch.matmul(k.unsqueeze(1), v.unsqueeze(0))
    return torch.linalg.matrix_norm(kv).item()

def compute_kv_loras(lora_mod, context: Tensor):
    kvs: List[float] = []
    
    attn2_keys = [k for k in lora_mod.modules.keys() if '_attn2_to_k' in k]
    for to_k_key in attn2_keys:
        to_v_key = to_k_key.replace('_to_k', '_to_v')
        if to_k_key not in lora_mod.modules or to_v_key not in lora_mod.modules:
            continue
        
        to_k_mod = lora_mod.modules[to_k_key]
        to_v_mod = lora_mod.modules[to_v_key]
        
        kv = compute_kv_lora(to_k_mod, to_v_mod, context)
        kvs.append(kv)

    return kvs

def compute_kv_lora(
    to_k, #: LoraUpDownModule,
    to_v, #: LoraUpDownModule,
    context: Tensor,
) -> float:
    k = apply_lora(to_k.up, to_k.down, to_k.alpha, context)
    v = apply_lora(to_v.up, to_v.down, to_v.alpha, context)
    
    kv = torch.matmul(k.unsqueeze(1), v.unsqueeze(0))
    return torch.linalg.matrix_norm(kv).item()
    
def apply_lora(
    up: nn.Module,
    down: nn.Module,
    alpha: Union[float,None],
    context: Tensor,
):
    d = up.weight.shape[1] # type: ignore
    a = (
        alpha / d if alpha is not None else
        1.0
    )
    result = up(down(context)) * a
    return result


state = 'stop'

def stop_running():
    global state
    if state == 'running':
        state = 'stop'

def run(
    num: Union[int,float],
    batch_size: Union[int,float],
    lora: Union[str,None],
):
    global state
    state = 'running'
    
    num = int(num)
    bs = int(batch_size)
    if lora is None:
        lora = ''
    
    model = shared.sd_model
    if model is None:
        raise ValueError('model is not loaded.')
    
    te = model.cond_stage_model
    clip = ClipWrapper(te)
    
    lora_mod = (
        None if len(lora) == 0 else
        load_lora(lora)
    )
    
    vocab: Dict[str,int] = clip.tokenizer.get_vocab() # type: ignore
    
    q = Queue(num)
    
    #import pdb; pdb.set_trace()
    for xs in tqdm.tqdm(each_slice(vocab.items(), bs), total=math.ceil(len(vocab)/bs)):
        
        if state == 'stop':
            return None, None
        
        batch = []
        words = []
        for word, idx in xs:
            tokens = [te.id_start, idx, te.id_end] + [te.id_end] * 74
            batch.append(tokens)
            words.append((idx, word))
        
        result = te.encode_with_transformers(torch.IntTensor(batch).to(te.wrapped.device))[:,1,:] # (bs,768)
        
        clip_norms = torch.linalg.vector_norm(result, dim=-1) # (bs,)
        
        def item(index: int):
            clip_norm = clip_norms[index].item()
            vector = result[index,:]
            tidx, word = words[index]
            
            vector = vector.to('cuda')
            
            if lora_mod is None:
                kvs = compute_kv_model(model.model.diffusion_model, vector)
            else:
                kvs = compute_kv_loras(lora_mod, vector)
            
            return Item(clip_norm, tidx, word, np.max(kvs))

        #import pdb; pdb.set_trace()
    
        indices = torch.argsort(clip_norms)
        for idx in indices:
            
            if state == 'stop':
                return None, None
            
            item_ = item(int(idx.item()))
            q.queue(item_)
        
    asc = []
    des = []
    
    #import pdb; pdb.set_trace()
    
    q_des = reversed(sorted(q.q_min))
    q_asc = reversed(sorted(q.q_max))
    for a, d in zip(q_asc, q_des):
        asc.append([a.token, a.token_index, a.norm_repr])
        des.append([d.token, d.token_index, d.norm_repr])
    
    return des, asc
    

def add_tab():
    def wrap(fn, values: int = 1):
        def f(*args, **kwargs):
            v, e = None, ''
            try:
                with torch.inference_mode():
                    v = fn(*args, **kwargs)
            except Exception:
                ex = traceback.format_exc()
                print(ex, file=sys.stderr)
                e = str(ex).replace('\n', '<br/>')
            if 1 < values:
                if v is None:
                    v = [None] * values
                return [*v, e]
            else:
                return [v, e]
        return f
    
    with gr.Blocks(analytics_enabled=False) as ui:
        error = gr.HTML(elem_id=f'{NAME.lower()}-error')
        
        with gr.Row():
            #model = gr.Dropdown(sd_models.checkpoint_tiles(), label='Model')
            lora = gr.Dropdown(choices=available_loras(), label='LoRA module')
            refresh_lora = ToolButton(value=refresh_symbol)
        
        num = gr.Slider(minimum=1, maximum=500, value=10, step=1, label='Num. of output words.')
        batch_size = gr.Slider(minimum=1, maximum=1024, value=256, step=1, label='Batch size')
        
        with gr.Row():
            button = gr.Button(variant='primary')
            stop = gr.Button(value='Interrupt')
            close = gr.Button(value='Close')
        
        result1 = gr.Dataframe(headers=['Token', 'Token Index', 'Score'], datatype=['str', 'number', 'number'], type='array', label='Maximum', interactive=False)
        result2 = gr.Dataframe(headers=['Token', 'Token Index', 'Score'], datatype=['str', 'number', 'number'], type='array', label='Minimum', interactive=False)
    
        def close_fn():
            return None, None
        
        refresh_lora.click(fn=reload_loras, inputs=[], outputs=[lora])
        button.click(fn=wrap(run, 2), inputs=[num, batch_size, lora], outputs=[result1, result2, error])
        stop.click(fn=stop_running, inputs=[], outputs=[])
        close.click(fn=close_fn, inputs=[], outputs=[result1, result2])
    
    return [(ui, NAME, NAME.lower())]

script_callbacks.on_ui_tabs(add_tab)
