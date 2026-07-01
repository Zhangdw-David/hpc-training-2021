"""
Minimal latest-style diffusion model implementation using a multimodal
transformer as the denoising backbone. This file provides PyTorch
components: a simple multimodal transformer encoder-decoder used as a
denoiser and a Gaussian diffusion training/sampling wrapper.

Notes:
- This is a compact, self-contained example for experimentation and
  educational use. It omits many production features (efficient schedulers,
  distributed training helpers, weight initialization schemes, etc.).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
	def __init__(self, dim: int, max_len: int = 5000):
		super().__init__()
		pe = torch.zeros(max_len, dim)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
		pe[:, 0::2] = torch.sin(position * div_term)
		if dim % 2 == 1:
			# last column stays zero for odd dims
			pe[:, 1::2] = torch.cos(position * div_term[:-1])
		else:
			pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe.unsqueeze(0))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, T, C)
		return x + self.pe[:, : x.size(1), :]


class MultimodalTransformerDenoiser(nn.Module):
	"""A compact multimodal transformer used as the diffusion denoiser.

	Inputs expected during forward:
	- noisy_image_tokens: (B, N_img, C)
	- text_tokens: (B, N_txt, C) or None
	- time_emb: (B, C)
	Returns reconstructed image tokens (B, N_img, C)
	"""

	def __init__(self, dim: int = 512, n_heads: int = 8, depth: int = 6, mlp_dim: int = 2048,
				 img_tokens: int = 256, txt_tokens: int = 128, dropout: float = 0.1):
		super().__init__()
		self.dim = dim
		self.img_tokens = img_tokens
		self.txt_tokens = txt_tokens

		# input projections
		self.img_proj = nn.Linear(3, dim)  # image pixels -> token dim (expect flattened RGB)
		self.text_proj = nn.Embedding(30522, dim)  # vocabulary projection (can be replaced)

		# positional encodings for both modalities
		self.pos_img = PositionalEncoding(dim, max_len=img_tokens)
		self.pos_txt = PositionalEncoding(dim, max_len=txt_tokens)

		# time embedding
		self.time_mlp = nn.Sequential(
			nn.Linear(dim, dim * 4),
			nn.SiLU(),
			nn.Linear(dim * 4, dim),
		)

		# Transformer encoder that consumes concatenated multimodal tokens
		encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=mlp_dim,
												   dropout=dropout, activation='gelu')
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

		# output projection back to RGB tokens
		self.out_proj = nn.Linear(dim, 3)

		# small classifier-free guidance support: dropout of text conditioning
		self.register_buffer('_null_text_token', torch.zeros(1, 1, dim))

	def forward(self, noisy_img: torch.Tensor, text_ids: Optional[torch.Tensor], t: torch.Tensor,
				text_drop_prob: float = 0.0) -> torch.Tensor:
		# noisy_img: (B, N_img, 3)
		# text_ids: (B, N_txt) or None
		B = noisy_img.size(0)

		img_tok = self.img_proj(noisy_img)  # (B, N_img, dim)
		img_tok = self.pos_img(img_tok)

		if text_ids is None:
			txt_tok = self._null_text_token.expand(B, self.txt_tokens, -1)
		else:
			txt_tok = self.text_proj(text_ids)  # (B, N_txt, dim)
			txt_tok = self.pos_txt(txt_tok)
			# optionally drop text for classifier-free guidance
			if text_drop_prob > 0.0 and self.training:
				mask = (torch.rand(B, 1, device=txt_tok.device) < text_drop_prob).float()
				txt_tok = txt_tok * (1.0 - mask) + self._null_text_token * mask

		# time embedding broadcast to tokens
		time_emb = self.time_mlp(t)  # (B, dim)
		time_tok = time_emb.unsqueeze(1).expand(-1, self.img_tokens + self.txt_tokens, -1)

		# concatenate modalities: [text, image]
		tokens = torch.cat([txt_tok, img_tok], dim=1)  # (B, N_txt+N_img, dim)
		tokens = tokens + time_tok

		# transformer expects (S, B, C)
		tokens = tokens.permute(1, 0, 2)
		out = self.transformer(tokens)
		out = out.permute(1, 0, 2)

		# take the image portion and project back to RGB
		img_out = out[:, -self.img_tokens :, :]
		rgb = self.out_proj(img_out)
		return rgb


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
	return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
	"""Simple Gaussian diffusion wrapper supporting training loss and sampling."""

	def __init__(self, denoiser: nn.Module, timesteps: int = 1000):
		super().__init__()
		self.denoiser = denoiser
		self.timesteps = timesteps
		betas = linear_beta_schedule(timesteps)
		alphas = 1.0 - betas
		alphas_cumprod = torch.cumprod(alphas, dim=0)

		self.register_buffer('betas', betas)
		self.register_buffer('alphas', alphas)
		self.register_buffer('alphas_cumprod', alphas_cumprod)
		self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
		self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

	def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
		# x_start: (B, N, 3)
		if noise is None:
			noise = torch.randn_like(x_start)
		sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
		sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
		return sqrt_acp * x_start + sqrt_omacp * noise

	def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, text_ids: Optional[torch.Tensor] = None,
				 noise: Optional[torch.Tensor] = None, text_drop_prob: float = 0.0):
		if noise is None:
			noise = torch.randn_like(x_start)
		x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
		# time embed as sinusoidal
		time_emb = self._time_embedding(t, self.denoiser.dim)
		pred = self.denoiser(x_noisy, text_ids, time_emb, text_drop_prob=text_drop_prob)
		return F.mse_loss(pred, noise)

	def _time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
		# sinusoidal embedding similar to Transformer
		half = dim // 2
		emb = math.log(10000) / (half - 1)
		emb = torch.exp(torch.arange(half, device=t.device) * -emb)
		emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
		emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
		if dim % 2 == 1:
			emb = F.pad(emb, (0, 1))
		return emb

	@torch.no_grad()
	def sample(self, shape: Tuple[int, int, int], text_ids: Optional[torch.Tensor] = None,
			   guidance_scale: float = 1.0) -> torch.Tensor:
		# shape: (B, N_img, 3)
		device = next(self.denoiser.parameters()).device
		B = shape[0]
		img = torch.randn(shape, device=device)
		for i in reversed(range(self.timesteps)):
			t = torch.full((B,), i, device=device, dtype=torch.long)
			time_emb = self._time_embedding(t, self.denoiser.dim)
			# classifier-free guidance: predict with and without text
			if guidance_scale != 1.0 and text_ids is not None:
				eps_cond = self.denoiser(img, text_ids, time_emb, text_drop_prob=0.0)
				eps_uncond = self.denoiser(img, None, time_emb, text_drop_prob=0.0)
				eps = eps_uncond + (eps_cond - eps_uncond) * guidance_scale
			else:
				eps = self.denoiser(img, text_ids, time_emb, text_drop_prob=0.0)

			beta = self.betas[i]
			alpha = self.alphas[i]
			alpha_cum = self.alphas_cumprod[i]
			sqrt_recip_alpha = (1.0 / math.sqrt(alpha))
			# predict x0 and compute posterior mean (simplified single-step update)
			pred_x0 = (img - math.sqrt(1 - alpha_cum) * eps) / math.sqrt(alpha_cum)
			if i > 0:
				noise = torch.randn_like(img)
				sigma = math.sqrt(beta)
				img = pred_x0 * math.sqrt(alpha_cum / alpha) + sigma * noise
			else:
				img = pred_x0
		return img


if __name__ == '__main__':
	# quick smoke test
	denoiser = MultimodalTransformerDenoiser(dim=256, n_heads=8, depth=4, img_tokens=64, txt_tokens=16)
	model = GaussianDiffusion(denoiser, timesteps=100)
	B = 2
	N_img = 64
	x = torch.randn(B, N_img, 3)
	t = torch.randint(0, 100, (B,))
	text = torch.randint(0, 30522, (B, 16))
	loss = model.p_losses(x, t, text)
	print('loss', loss.item())

