import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
import math

from .common import LayerNorm2d


class FrequencyEncoding(nn.Module):  
    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

    def get_out_dim(self) -> int:
        out_dim = self.in_dim * self.num_frequencies * 2  
        if self.include_input:
            out_dim += self.in_dim 
        return out_dim

    def forward(self, in_tensor, apply_perturbation=False):
        """
        :param in_tensor: 
        :param apply_perturbation: 
        :return: 
        """
        scaled_in_tensor = 2 * math.pi * in_tensor

      
        freqs = torch.logspace(0.0, math.log2(self.num_frequencies), steps=self.num_frequencies, base=2).to(in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  

  
        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + math.pi / 2.0], dim=-1))

       
        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)

       
        if apply_perturbation:
            perturbation = torch.randn_like(encoded_inputs) * 0.01
            encoded_inputs = encoded_inputs + perturbation

        return encoded_inputs

class MaskDecoder_prompt_large(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        coord_normalized_min: float = -1.0,
        coord_normalized_max: float = 1.0,
        w0: int = 10,
        features: int = 256,
        top_k_ratio: float = 0.125,
        cls_channel: int = 1
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 8),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 16),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 16, transformer_dim // 16, kernel_size=2, stride=2),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 16, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

       
        # self.freq_enc = FrequencyEncoding(
        #     in_dim=2, num_frequencies=w0, min_freq_exp=0.0, max_freq_exp=w0, include_input=True
        # )
  
        self.freq_enc = FrequencyEncoding(
            in_dim=2, num_frequencies=w0, include_input=True
        )
    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    
        b, c, h, w = image_pe.shape
        coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)), dim=-1).to(
            image_pe.device)
        coords = coords.unsqueeze(0).expand(b, -1, -1, -1)  
       
        encoded_coords = self.freq_enc(coords, apply_perturbation=False)
        perturbed_coords = self.freq_enc(coords, apply_perturbation=True)

      
        encoded_coords = encoded_coords.permute(0, 3, 1, 2)  
        perturbed_coords = perturbed_coords.permute(0, 3, 1, 2)

       
        if encoded_coords.shape[1] != image_pe.shape[1]:
            padding_size = image_pe.shape[1] - encoded_coords.shape[1]
            if padding_size > 0:
               
                encoded_coords = F.pad(encoded_coords, (0, 0, 0, 0, 0, padding_size))
                perturbed_coords = F.pad(perturbed_coords, (0, 0, 0, 0, 0, padding_size))
            elif padding_size < 0:
              
                encoded_coords = encoded_coords[:, :image_pe.shape[1], :, :]
                perturbed_coords = perturbed_coords[:, :image_pe.shape[1], :, :]

       
        image_pe_combined = torch.cat([image_pe, encoded_coords, perturbed_coords], dim=1)

        
        reduction_layer = nn.Conv2d(image_pe_combined.shape[1], self.transformer_dim, kernel_size=1).to(
            image_pe_combined.device)
        image_pe_combined = reduction_layer(image_pe_combined)

        # print("image_pe shape:", image_pe.shape)
        # print("encoded_coords shape:", encoded_coords.shape)
        # print("perturbed_coords shape:", perturbed_coords.shape)
        # print("image_pe_combined shape:", image_pe_combined.shape)

        masks, iou_pred, dense_features = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe_combined,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

       
        consistency_loss = F.mse_loss(encoded_coords, perturbed_coords)  

        return masks, iou_pred, dense_features,consistency_loss

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, 1, dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, 1, dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        dense_features = upscaled_embedding
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred, dense_features


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
