import torch
import torch.nn as nn
import os
from typing import Union, List, OrderedDict
import torch
import numpy as np
from torch.nn import functional as F
from . import vv_open_clip as open_clip
# from utils.loss import FocalLoss, BinaryDiceLoss
import cv2
from matplotlib import pyplot as plt
from .. import prompt as prompting


class PromptLearner_normal(nn.Module):
    def __init__(self, classnames, status, clip_model, tokenizer, dim, n_ctx, device):
        super().__init__()
        vis_dim = dim
        ctx_dim = dim

        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )

        classnames = [name.replace("_", " ") for name in classnames]

        self.tokenized_prompts = {}
        embedding = {}

        self.token_prefix = {}
        self.token_suffix = {}

        for class_name in classnames:

            class_name = prompting.cls_map[class_name]

            p = [
                prompt + " " + status_i.format(class_name) + "." for status_i in status
            ]

            self.tokenized_prompts[class_name] = tokenizer(p)
            self.tokenized_prompts[class_name].requires_grad = False

            with torch.no_grad():
                embedding[class_name] = clip_model.token_embedding(
                    self.tokenized_prompts[class_name]
                )

            self.token_prefix[class_name] = embedding[class_name][:, :1, :].to(device)
            self.token_prefix[class_name].requires_grad = False

            self.token_suffix[class_name] = embedding[class_name][:, 1 + n_ctx :, :].to(device)
            self.token_suffix[class_name].requires_grad = False

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, class_name):

        ctx = self.ctx
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        prefix = self.token_prefix[class_name]
        suffix = self.token_suffix[class_name]

        n_cls = prefix.shape[0]

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)

        return prompts


class PromptLearner_abnormal(nn.Module):
    def __init__(
        self,
        classnames,
        status,
        clip_model,
        tokenizer,
        dim,
        n_ctx,
        device,
        positions=None,
    ):
        super().__init__()
        vis_dim = dim
        ctx_dim = dim

        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )
        if positions == None:
            self.positions = [
                "top left",
                "top",
                "top right",
                "left",
                "center",
                "right",
                "bottom left",
                "bottom",
                "bottom right",
            ]
        else:
            self.positions = positions

        classnames = [name.replace("_", " ") for name in classnames]

        self.tokenized_prompts = {}
        embedding = {}

        self.token_prefix = {}
        self.token_suffix = {}

        for origin_class_name in classnames:

            class_name = prompting.cls_map[origin_class_name]

            p = [
                prompt + " " + status_i.format(class_name) + " at " + position + "."
                for status_i in status[origin_class_name]
                for position in self.positions
            ]

            # print(p)

            self.tokenized_prompts[class_name] = tokenizer(p)
            with torch.no_grad():
                embedding[class_name] = clip_model.token_embedding(
                    self.tokenized_prompts[class_name]
                )

            self.token_prefix[class_name] = embedding[class_name][:, :1, :].to(device)
            self.token_prefix[class_name].requires_grad = False

            self.token_suffix[class_name] = embedding[class_name][:, 1 + n_ctx :, :].to(device)
            self.token_suffix[class_name].requires_grad = False

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, class_name):

        ctx = self.ctx
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        prefix = self.token_prefix[class_name]
        suffix = self.token_suffix[class_name]

        n_cls = prefix.shape[0]

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)
        return prompts


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)


class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                assert 0 == 1  # error
        return tokens

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=2):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.SiLU(),
        )

    def forward(self, x):
        y = self.fc(x) + x
        return y


class CovLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer, self).__init__()
        self.fc_33 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_11 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_77 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_55 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_51 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=(5, 1), padding="same")
                for _ in range(k)
            ]
        )
        self.fc_15 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=(1, 5), padding="same")
                for _ in range(k)
            ]
        )

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                x = tokens[i][:, 1:, :]
                x = x.view(
                    x.shape[0],
                    int(np.sqrt(x.shape[1])),
                    int(np.sqrt(x.shape[1])),
                    x.shape[2],
                )
                # print(x.shape)
                x_temp = (
                    self.fc_11[i](x.permute(0, 3, 1, 2))
                    + self.fc_33[i](x.permute(0, 3, 1, 2))
                    + self.fc_55[i](x.permute(0, 3, 1, 2))
                    + self.fc_77[i](x.permute(0, 3, 1, 2))
                    + self.fc_15[i](x.permute(0, 3, 1, 2))
                    + self.fc_51[i](x.permute(0, 3, 1, 2))
                )
                tokens[i] = x_temp
                tokens[i] = (
                    tokens[i]
                    .permute(0, 2, 3, 1)
                    .view(tokens[i].shape[0], -1, tokens[i].shape[1])
                )
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](
                    tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous()
                )
        return tokens


class FiLo(nn.Module):
    def __init__(self, obj_list, args, device) -> None:
        super().__init__()

        self.args = args

        self.device = device

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            args.clip_model, args.image_size, pretrained=args.clip_pretrained
        )
        self.clip_model.eval()

        self.tokenizer = open_clip.get_tokenizer(args.clip_model)

        self.decoder_cov = CovLayer(1024, 768, 3)
        self.decoder_linear =  LinearLayer(1024, 768, 4)
        self.text_encoder = TextEncoder(self.clip_model)
        self.text_encoder.eval()

        self.normal_prompt_learner = PromptLearner_normal(
            obj_list,
            prompting.status_normal,
            self.clip_model,
            self.tokenizer,
            768,
            args.n_ctx,
            args.device
        )

        self.abnormal_prompt_learner = PromptLearner_abnormal(
            obj_list,
            prompting.status_abnormal,
            self.clip_model,
            self.tokenizer,
            768,
            args.n_ctx,
            args.device
        )

        self.adapter = Adapter(768)

    def forward(self, items, with_adapter=False, only_train_adapter=False, positions=None):
        
        image = items["img"].to(self.device)
        cls_name = items["cls_name"]


        with torch.no_grad():
            image_features, patch_tokens = self.clip_model.encode_image(
                image, self.args.features_list
            )
        

        if with_adapter:
            image_features = self.adapter(image_features)
        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)



        if only_train_adapter:
            with torch.no_grad():
                normal_prompts = self.normal_prompt_learner(image_features, prompting.cls_map[cls_name])
                normal_tokenized_prompts = self.normal_prompt_learner.tokenized_prompts[
                    prompting.cls_map[cls_name]
                ]

                abnormal_prompts = self.abnormal_prompt_learner(image_features, prompting.cls_map[cls_name])
                abnormal_tokenized_prompts = self.abnormal_prompt_learner.tokenized_prompts[
                    prompting.cls_map[cls_name]
                ]

                normal_text_features = self.text_encoder(
                    normal_prompts[0], normal_tokenized_prompts
                )
                normal_text_features = normal_text_features / normal_text_features.norm(
                    dim=-1, keepdim=True
                )

                abnormal_text_features = self.text_encoder(
                    abnormal_prompts[0], abnormal_tokenized_prompts
                )
                abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(
                    dim=-1, keepdim=True
                )

                normal_text_features = normal_text_features.mean(dim=0, keepdim=True)
                normal_text_features = normal_text_features / normal_text_features.norm()
                normal_text_features = normal_text_features.unsqueeze(1)

        else:

            normal_prompts = self.normal_prompt_learner(image_features, prompting.cls_map[cls_name])
            normal_tokenized_prompts = self.normal_prompt_learner.tokenized_prompts[
                prompting.cls_map[cls_name]
            ]

            abnormal_prompts = self.abnormal_prompt_learner(image_features, prompting.cls_map[cls_name])
            abnormal_tokenized_prompts = self.abnormal_prompt_learner.tokenized_prompts[
               prompting.cls_map[cls_name]
            ]

            normal_text_features = self.text_encoder(
                normal_prompts[0], normal_tokenized_prompts
            )
            normal_text_features = normal_text_features / normal_text_features.norm(
                dim=-1, keepdim=True
            )

            abnormal_text_features = self.text_encoder(
                abnormal_prompts[0], abnormal_tokenized_prompts
            )
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(
                dim=-1, keepdim=True
            )

            normal_text_features = normal_text_features.mean(dim=0, keepdim=True)
            normal_text_features = normal_text_features / normal_text_features.norm()
            normal_text_features = normal_text_features.unsqueeze(1)


        ab_position = []
        if positions != None:
            ab_position = positions


        if len(ab_position) > 0:
            tmp_abnormal_text_features = []
            for ab_p in ab_position:
                position_idx = prompting.positions_list.index(ab_p)
                tmp_abnormal_text_features.append(abnormal_text_features[position_idx::9])
        
            abnormal_text_features = torch.cat(tmp_abnormal_text_features,dim=0)
        

        abnormal_text_features = abnormal_text_features.mean(dim=0, keepdim=True)
        abnormal_text_features = abnormal_text_features / abnormal_text_features.norm()
        abnormal_text_features = abnormal_text_features.unsqueeze(1)

        text_features = torch.cat([normal_text_features, abnormal_text_features], dim=1)

        text_features = text_features / text_features.norm()

        text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)

        anomaly_maps = None

        if not only_train_adapter:
            
            patch_tokens_qkv = self.decoder_linear(patch_tokens[::2])
            patch_tokens_vv = self.decoder_cov(patch_tokens[1::2])
            
            anomaly_maps = []
            for layer in range(len(patch_tokens_qkv)):
                patch_tokens_qkv[layer] = patch_tokens_qkv[layer] / patch_tokens_qkv[
                    layer
                ].norm(dim=-1, keepdim=True)

                anomaly_map = (
                    100.0 * patch_tokens_qkv[layer] @ text_features.transpose(-2, -1)
                )

                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(
                    anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                    size=self.args.image_size,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map)

            for layer in range(len(patch_tokens_vv)):
                patch_tokens_vv[layer] = patch_tokens_vv[layer] / patch_tokens_vv[
                    layer
                ].norm(dim=-1, keepdim=True)

                anomaly_map = (
                    100.0 * patch_tokens_vv[layer] @ text_features.transpose(-2, -1)
                )

                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(
                    anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                    size=self.args.image_size,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map)

        return text_probs, anomaly_maps
