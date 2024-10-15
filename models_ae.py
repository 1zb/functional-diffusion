from functools import wraps

import numpy as np

import math

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from torch_cluster import fps

from timm.models.layers import DropPath

def cdist2(x, y):
    # |x_i - y_j|_2^2 = <x_i - y_j, x_i - y_j> = <x_i, x_i> + <y_j, y_j> - 2*<x_i, y_j>
    x_sq_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_sq_norm = y.pow(2).sum(dim=-1)
    x_dot_y = x @ y.transpose(-1,-2)
    sq_dist = x_sq_norm + y_sq_norm.unsqueeze(dim=-2) - 2*x_dot_y
    # For numerical issues
    sq_dist.clamp_(min=0.0)
    return torch.sqrt(sq_dist)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None, modulated=False):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

        self.modulated = modulated
        if self.modulated:
            self.gamma = nn.Linear(dim, dim, bias=False)
            self.beta = nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.modulated:
            label = kwargs.pop('label')
            gamma = self.gamma(label) # b 1 c
            beta = self.beta(label) # b 1 c
            # print('layernorm', x.shape, beta.shape)
            x = gamma * x + beta
            # print('layernorm', x.shape, beta.shape)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None, attn_mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        # print(q.shape, k.shape, v.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        # print(q.shape, k.shape, v.shape)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        if exists(attn_mask):
            # attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            attn_mask = repeat(attn_mask, 'i j -> (b h) i j', b=x.shape[0], h = h)
            # print(attn_mask)
            sim.masked_fill_(~attn_mask, -torch.finfo(sim.dtype).max)
            

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        # print(x.shape, self.freqs.shape)
        # x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.einsum('..., n->... n', x, 2 * np.pi * self.freqs)
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x

class Network(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        depth=4,
        heads=8, 
        dim_head=64,
        function_dim=1,
    ):
        super().__init__()

        heads = dim // dim_head
            
        self.map_noise = FourierEmbedding(num_channels=dim)

        self.layers = nn.ModuleList([])
        
        get_latent_attn = lambda: PreNorm(dim, Attention(dim, dim, heads = heads, dim_head = dim_head), context_dim = dim)
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.depth = depth

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim, heads = heads, dim_head = dim_head), context_dim = dim),
                PreNorm(dim, FeedForward(dim)),
                nn.ModuleList([
                    nn.ModuleList([
                        PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head), modulated=True),
                        PreNorm(dim, FeedForward(dim), modulated=True),
                    ]) for _ in range(1)
                ])
            ]))


        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            None
        ])        
        
        self.to_output = PreNorm(dim, nn.Linear(dim, 1), modulated=True)


        self.latent = nn.Embedding(512, dim)

        self.point_embed = PointEmbed(dim=dim)

        self.value_embed = nn.Linear(function_dim, dim)

        self.cond_token = nn.Embedding(1, dim)
    
    def forward_features(self, context_points, context_values, alpha_embeddings, cond):
        # context_points: b n 3
        # context_values: b n 1
        # alpha: b

        context = self.point_embed(context_points) + self.value_embed(context_values)#.squeeze(-1))

        context = torch.split(context, context.shape[1]//self.depth, dim=1)
                
        x = repeat(self.latent.weight, 'n c -> b n c', b=alpha_embeddings.shape[0])
                
        for i, (cross_attn, cross_ff, layers) in enumerate(self.layers):
            
            if cond is None:
                c = context[i]
            else:
                c = torch.cat([context[i], cond + self.cond_token.weight[None]], dim=1)
            
            x = cross_attn(x, context=c) + x
            x = cross_ff(x) + x


            for self_attn, self_ff in layers:
                x = self_attn(x, label=alpha_embeddings) + x
                x = self_ff(x, label=alpha_embeddings) + x
        return x
    
    def decode(self, queries, x, alpha_embeddings):
        queries = self.point_embed(queries)

        ####
        cross_attn, cross_ff = self.cross_attend_blocks
                
        o = cross_attn(queries, context = x, mask = None)# + queries_embeddings
        o = self.to_output(o, label=alpha_embeddings)#.squeeze(-1)
        return o
                
    def forward(self, context_points, context_values, queries, alpha, cond):

        alpha_embeddings = self.map_noise(alpha)[:, None]

        x = self.forward_features(context_points, context_values, alpha_embeddings, cond)
        return self.decode(queries, x, alpha_embeddings)


class Diffusion(nn.Module):
    def __init__(self, N=0):
        super().__init__()

        self.model = Network(depth=24, dim=768)
        self.condition = nn.Embedding(55, 768)

        self.logvar_fourier = FourierEmbedding(num_channels=768)
        self.logvar_linear = nn.Linear(768, 1)
    
    def forward(self, context_points, context_values, query_points, query_sdf, pc, categories):
        B, _, _, device = *context_points.shape, context_points.device

        cond = self.condition(categories)[:, None]

        rnd_normal = torch.randn([pc.shape[0]], device=pc.device)
        t = (rnd_normal * 1.2 - 1.2).exp()

        M = 2048
        x_i = torch.rand(B, M, 3).to(device) * 2 - 1

        s_i = torch.randn(B, 1, 64, 64, 64).to(device)


        f_t = context_values + t[:, None] * self.init(x_i, s_i, context_points, blocks=4096)

        query_points = torch.cat([pc, query_points], dim=1)
        query_sdf = torch.cat([torch.zeros_like(pc[:, :, 0]), query_sdf], dim=1)

        denominator = torch.sqrt(1 + t**2)

        d = self.model(context_points, f_t[:, :, None] / denominator[:, None, None], query_points, t.log() / 4, cond).squeeze(-1)


        logvar = self.logvar_linear(self.logvar_fourier(t / denominator))

        loss_recon = (d - query_sdf)**2

        loss = 1/ logvar.exp() * loss_recon + logvar 
        loss = torch.sum(loss) / d.shape[0]

        return loss

    @torch.no_grad()
    def init(self, x_i, grid, queries, blocks=1):

        return F.grid_sample(grid, queries[:, :, None, None], align_corners=False).squeeze(-1).squeeze(-1).squeeze(1)
    
    @torch.no_grad()
    def sample(self, categories, query_points, n_steps=64):
        
        if categories is not None:
            cond = self.condition(categories)[:, None]
        else:
            cond = None

        B, device = query_points.shape[0], query_points.device

        sigma_max, sigma_min, rho = 80, 0.002, 7

        step_indices = torch.arange(n_steps, dtype=torch.float32, device=device)

        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (n_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        t_steps = torch.as_tensor(sigma_steps)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        t_next = t_steps[0]

        context_points = torch.rand(B, 1024*48, 3, device=device) * 2 - 1

        M = 2048
        x_i = torch.rand(B, M, 3).to(device) * 2 - 1
        s_i = torch.randn(B, 1, 64, 64, 64).to(device)

        context_sdf = self.init(x_i, s_i, context_points, blocks=2048) * t_steps[0]

        query_sdf = self.init(x_i, s_i, query_points, blocks=2048) * t_steps[0]

        query_sdfs = [query_sdf.clone()]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1

            alpha_embeddings = self.model.map_noise(t_cur[None].log() / 4)[:, None].expand(B, -1, -1)

            x = self.model.forward_features(context_points, context_sdf[:, :, None] / (1 + t_cur**2).sqrt(), alpha_embeddings, cond)

            d = self.model.decode(context_points, x, alpha_embeddings).squeeze(-1)

            context_sdf = context_sdf + (context_sdf - d) * (t_next - t_cur) / t_cur

            # print(i, t_next, t_cur, t_next / t_cur)
            # print(context_sdf.max().item(), context_sdf.min().item(), context_sdf.mean().item(), context_sdf.std().item())

        d = self.model.decode(query_points, x, alpha_embeddings).squeeze(-1)

        query_sdf = d
        return query_sdf
    