import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from inference.models.clip_fusion.cross_attention import CrossTransformer
from inference.models.clip_fusion.embedder import get_embedder

from inference.models.ragt.mobile_vit import get_model
from inference.models.ragt.Anchor import *

from inference.models.grasp_model import LanguageGraspModel


class CLIPFusion(LanguageGraspModel):
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0, grasp_dim: int=5, width: int=512, layers: int=6, heads: int=8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.to(torch.float32)   # In float16, clip will output nan.
        self.cross_attn = CrossTransformer(width=width, layers=layers, heads=heads).to(self.device)

        self.grasp_embbedding = nn.Sequential(
                                nn.Linear(grasp_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                ).to(self.device)

        self.pos_projection, pos_proj_dim = get_embedder(multires=5, input_dim=4)

        self.bbox_pos_embbedding = nn.Sequential(
                                nn.Linear(pos_proj_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                ).to(self.device)

        # self.text_embedding = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.GELU(),
        #     nn.Linear(1024, 2048),
        #     nn.GELU(),
        #     nn.Linear(2048, width),
        #     nn.GELU(),
        #     nn.Linear(width, width)
        # ).to(self.device)

        # Reshaped cosine layers
        self.conv_upsampling = nn.Sequential(
            nn.Linear(width, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 3136)
        )

        # Reshaped linear layers
        # self.upsampling_layer = nn.Sequential(
        #     nn.Linear(1, 64),
        #     nn.GELU(),
        #     nn.Linear(64, 128),
        #     nn.GELU()
        # )


        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        self._setup_pretrained_model()

    def _setup_pretrained_model(self):
        _default_weight_path = "weights/RAGT-3-3.pth"

        self.ragt_model = get_model()
        self.ragt_model.load_state_dict(torch.load(_default_weight_path))
        self.ragt_model.eval()
        for param in self.ragt_model.parameters():
            param.requires_grad = False

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode_bbox(self, x):
        with torch.no_grad():
            batch_size, num_bboxes, channel, h, w = x.shape
            x = x.reshape(batch_size * num_bboxes, channel, h, w)
            x = self.clip_model.encode_image(x.to(self.device))
            x = x.reshape(batch_size, num_bboxes, -1)
        return x

    def encode_text(self, x):
        with torch.no_grad():
            tokens = clip.tokenize(x).to(self.device)
            text_feat = self.clip_model.encode_text(tokens)
            text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_mask

    def encode_grasp(self, x):
        grasp_emb = self.grasp_embbedding(x.to(self.device)) # shape = [N, L', D]
        return grasp_emb

    def encode_bbox_pos(self, x):
        bbox_pos_emb = self.bbox_pos_embbedding(x.to(self.device)) # shape = [N, L', D]
        return bbox_pos_emb

    def mult_fusion(self, bbox_feat, text_feat):
        text_feat = text_feat.unsqueeze(-2)
        text_feat = text_feat.repeat(1, bbox_feat.shape[1], 1)
        fusion_feat = bbox_feat * text_feat
        return fusion_feat

    def forward(self, bboxes, pos_bboxes, text, predicted_grasp):
        # encode bbox
        bbox_feat = self.encode_bbox(bboxes) # shape = [N, L, D] D=512
        # encode text
        text_feat, _ = self.encode_text(text) # shape = [N, D]
        # fusion
        fusion_feat = self.mult_fusion(bbox_feat, text_feat) # shape = [N, L, D]

        # # normalized features
        # bbox_feat_normlized = bbox_feat / bbox_feat.norm(dim=-1, keepdim=True)
        # text_feat_normlized = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # # cosine similarity as logits
        # logit_scale = self.clip_model.logit_scale.exp()
        # logits_per_image = logit_scale * bbox_feat_normlized @ text_feat_normlized.t()

        # # logits_per_text = logit_scale * text_feat_normlized @ bbox_feat_normlized.t()
        # probs = logits_per_image.softmax(dim=-2).reshape(logits_per_image.shape[0], -1)

        # encode grasp
        grasp_feat = self.encode_grasp(predicted_grasp) # shape = [N, L', D]
        grasp_feat = grasp_feat.permute(1, 0, 2)  # NL'D -> L'ND
        fusion_feat = fusion_feat.float().permute(1, 0, 2)  # NLD -> LND

        # encode bbox positions
        pos_bboxes = self.pos_projection(pos_bboxes)
        bbox_pos_feat = self.encode_bbox_pos(pos_bboxes) # shape = [N, L, D]
        # bbox_pos_feat = bbox_pos_feat.permute(1, 0, 2) # NLD -> LND

        # add fusion
        bbox_compound_feat = bbox_pos_feat + bbox_feat
        # concat fusion
        # bbox_compound_feat = torch.cat((bbox_feat, bbox_pos_feat), dim=-1)
        # bbox_compound_feat = self.pos_vision_fusion(bbox_compound_feat)
        bbox_compound_feat = bbox_compound_feat.permute(1, 0, 2)

        # cross attention
        cross_feat, attn_weights = self.cross_attn(q=grasp_feat, k=bbox_compound_feat, v=fusion_feat) # shape = [N, L', D]

        cross_feat = self.conv_upsampling(cross_feat)
        cross_feat = cross_feat.view(-1, 1, 56, 56).repeat(1, 128, 1, 1)
        x = cross_feat

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def _get_index_and_bias(self, output, confidence_threshold):
        N, C, H, W = output.shape
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(N, H, W, num_anchors, -1)
        mask_obj = (F.sigmoid(output[..., 0]) >= confidence_threshold)
        index = mask_obj.nonzero()
        bias = output[mask_obj]
        return index, bias

    def _get_coordinate(self, index, bias):
        confidence = torch.sigmoid(bias[:, 0])
        cx = (index[:, 2] + torch.sigmoid(bias[:, 1])) * field_of_grid_cell
        cy = (index[:, 1] + torch.sigmoid(bias[:, 2])) * field_of_grid_cell
        w = anchor_w * torch.exp(bias[:, 3])
        h = anchor_h * torch.exp(bias[:, 4])
        theta = (index[:, 3] + torch.sigmoid(bias[:, 5])) * theta_margin
        return confidence, cx, cy, w, h, theta

    def _predict_grasp_pose(self, x):
        output = self.ragt_model(x)
        index, bias = self._get_index_and_bias(output, 0.95)
        confidence, cx, cy, w, h, theta = self._get_coordinate(index, bias)
        return torch.cat([cx.unsqueeze(1), cy.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1), theta.unsqueeze(1)], dim=1)

    def compute_loss(self, xc, yc, prompt, query, bboxes, pos_bboxes):
        y_pos, y_cos, y_sin, y_width = yc
        l = []
        for _xc in xc:
            _xc = _xc.unsqueeze(0)
            predicted_grasps = self._predict_grasp_pose(_xc)
            if predicted_grasps.data.shape[0] > 0:
                l.append(predicted_grasps[0].unsqueeze(0).unsqueeze(0))
            else:
                l.append(torch.zeros(1, 1, 5).cuda())
        predicted_grasps = torch.cat(l, dim=0)
        batch_size, num_grasps, _ = predicted_grasps.data.shape

        pos_pred, cos_pred, sin_pred, width_pred = self(bboxes, pos_bboxes, query, predicted_grasps)
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
