# Custom transformer used for the modular addition grokking experiments

from transformers import PreTrainedModel, PretrainedConfig
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    
    def give_name(self, name):
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        return x

class TransformerConfig(PretrainedConfig):
    model_type = "custom_transformer"
    
    def __init__(
        self,
        d_vocab=114,
        d_model=128,
        n_ctx=5,
        num_layers=1,
        num_heads=4,
        d_mlp=512,
        act_type="ReLU",
        use_ln=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.n_ctx = n_ctx
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_mlp = d_mlp
        self.act_type = act_type
        self.use_ln = use_ln
        self.d_head = d_model // num_heads

class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_model))
    
    def forward(self, x):
        return t.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_vocab))
    
    def forward(self, x):
        return (x @ self.W_U)

class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model)/np.sqrt(d_model))
    
    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]

class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-4, use_ln=False):
        super().__init__()
        self.use_ln = use_ln
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon
    
    def forward(self, x):
        if self.use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
        return x
    
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx):
        super().__init__()
        # 0.64/d_model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))

        self.d_head = d_head
        self.num_heads = num_heads
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def apply_causal_mask(self, attn_scores):
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, -1e5)
        return attn_scores
    
    def forward(self, x):    
        k = self.hook_k(t.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(t.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(t.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked / np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(t.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = t.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type):
        super().__init__()
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.act_type = act_type
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']
        
    def forward(self, x):
        x = self.hook_pre(t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_ln):
        super().__init__()
        self.ln1 = LayerNorm(d_model, use_ln=use_ln)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx)
        self.ln2 = LayerNorm(d_model, use_ln=use_ln)
        self.mlp = MLP(d_model, d_mlp, act_type)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    
    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.ln1(self.hook_resid_pre(x)))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp(self.ln2(x))))
        return x

class CustomTransformer(PreTrainedModel):
    config_class = TransformerConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embed = Embed(d_vocab=config.d_vocab, d_model=config.d_model)
        self.pos_embed = PosEmbed(max_ctx=config.n_ctx, d_model=config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                d_mlp=config.d_mlp,
                d_head=config.d_head,
                num_heads=config.num_heads,
                n_ctx=config.n_ctx,
                act_type=config.act_type,
                use_ln=config.use_ln
            ) for _ in range(config.num_layers)
        ])
        self.unembed = Unembed(d_vocab=config.d_vocab, d_model=config.d_model)
        
        # Initialize weights and apply final processing
        # self.post_init()
        
        # Name all hook points
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.give_name(name)

    def get_input_embeddings(self):
        return self.embed
    
    def set_input_embeddings(self, value):
        self.embed = value
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        x = self.pos_embed(x)

        for block in self.blocks:
            x = block(x)
        
        logits = self.unembed(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.d_vocab), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )
    
    def hook_points(self):
        return [module for name, module in self.named_modules() if isinstance(module, HookPoint)]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('both')
    
    def cache_all(self, cache, incl_bwd=False):
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')