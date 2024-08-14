import torch.nn as nn
import torch.nn.functional as F
import torch
import clip
import numpy as np
import alpha_clip

class GraspModel_CLIP(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel_CLIP, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc, text, alpha = None):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, text, alpha)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc, text):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc, text)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }

class TextEncoder(nn.Module):
    def __init__(self, out_channel=128 ,device='cuda'):
        super(TextEncoder, self).__init__()
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Linear(512, out_channel)
               
    def forward(self, text):
        embed = self.model.encode_text(text).unsqueeze(0)
        embed = F.normalize(embed.mean(dim=1), dim=-1).float()
        embed = F.sigmoid(self.fc(embed))
        return embed

class PromptGDModel(GraspModel_CLIP):
    def __init__(self, clip_model='ViT-B/32', alpha = False, alpha_vision_ckpt_pth=None, robot_checkpoint=None):
        super(PromptGDModel, self).__init__()
        self.alpha = alpha
        if not alpha:
            clip_model, _ = clip.load(clip_model, device='cuda')
            self.clip_model = clip_model.float()
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model, _ = alpha_clip.load(clip_model, alpha_vision_ckpt_pth=alpha_vision_ckpt_pth, device=device)  # change to your own ckpt path
            if robot_checkpoint:
                clip_model.load_state_dict(torch.load(robot_checkpoint))
            self.clip_model = clip_model.float()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        clip_output_dim = self.clip_model.visual.output_dim
        num_patch = self.clip_model.visual.positional_embedding.shape[0]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.linear = nn.Linear(num_patch, 56*56)
        self.ln_mask = nn.LayerNorm(clip_output_dim)   # LayerNorm for mask
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'),
                                nn.Conv2d(clip_output_dim, 512, 3, padding=1),
                                nn.BatchNorm2d(512),
                                nn.GELU()
                                )
                                
        ## Depthwise and pointwise
        self.pos_output = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, groups=512),
            nn.Conv2d(512, 1, 1)
        )
        self.cos_output = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, groups=512),
            nn.Conv2d(512, 1, 1)
        )
        self.sin_output = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, groups=512),
            nn.Conv2d(512, 1, 1)
        )
        self.width_output = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, groups=512),
            nn.Conv2d(512, 1, 1)
        )
        
    def forward(self, x, text, alpha = None):
        # Encode image
        x = self.clip_model.visual.conv1(x)
        
        if self.alpha:
            x = x + self.clip_model.visual.conv1_alpha(alpha)
            
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)                     # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 
                        1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.clip_model.visual.ln_post(x)
        x = x @ self.clip_model.visual.proj  # B, L, C
            
        
        ls = []
        for a in text:
            a = self.clip_model.encode_text(a.squeeze(0))
            ls.append(a)
            
        assert len(ls) == text.shape[0]
        text_embed = torch.stack(ls) # B, T, C
        
        similarity_matrix = self.logit_scale*(x @ text_embed.permute(0,2,1)) / (torch.norm(x, dim=-1).T @ torch.norm(text_embed, dim=-1)) # B, L, T
        mask = F.sigmoid(similarity_matrix).mean(dim=-1) # B, L
        x = x + x*mask.unsqueeze(-1)
        x = self.ln_mask(x)
        
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        B, C, _ = x.shape
        x = x.reshape(B, C, 56, 56)
        x = self.up(x)
        
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
    
        
        
        
        
        
            
        
        
