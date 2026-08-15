
import torch
import torch.nn as nn
from annotator.uniformer.mmcv.cnn import ConvModule, normal_init
from annotator.uniformer.mmcv.ops import point_sample

from annotator.uniformer.mmseg.models.builder import HEADS
from annotator.uniformer.mmseg.ops import resize
from ..losses import accuracy
from .cascade_decode_head import BaseCascadeDecodeHead

def calculate_uncertainty(seg_logits):
    
    top2_scores = torch.topk(seg_logits, k=2, dim=1)[0]
    return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)

@HEADS.register_module()
class PointHead(BaseCascadeDecodeHead):
    
    def __init__(self,
                 num_fcs=3,
                 coarse_pred_each_layer=True,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=False),
                 **kwargs):
        super(PointHead, self).__init__(
            input_transform='multiple_select',
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.num_fcs = num_fcs
        self.coarse_pred_each_layer = coarse_pred_each_layer

        fc_in_channels = sum(self.in_channels) + self.num_classes
        fc_channels = self.channels
        self.fcs = nn.ModuleList()
        for k in range(num_fcs):
            fc = ConvModule(
                fc_in_channels,
                fc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += self.num_classes if self.coarse_pred_each_layer                else 0
        self.fc_seg = nn.Conv1d(
            fc_in_channels,
            self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0)
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)
        delattr(self, 'conv_seg')

    def init_weights(self):
        
        normal_init(self.fc_seg, std=0.001)

    def cls_seg(self, feat):
        
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.fc_seg(feat)
        return output

    def forward(self, fine_grained_point_feats, coarse_point_feats):
        x = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
        for fc in self.fcs:
            x = fc(x)
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_point_feats), dim=1)
        return self.cls_seg(x)

    def _get_fine_grained_point_feats(self, x, points):
        
        fine_grained_feats_list = [
            point_sample(_, points, align_corners=self.align_corners)
            for _ in x
        ]
        if len(fine_grained_feats_list) > 1:
            fine_grained_feats = torch.cat(fine_grained_feats_list, dim=1)
        else:
            fine_grained_feats = fine_grained_feats_list[0]

        return fine_grained_feats

    def _get_coarse_point_feats(self, prev_output, points):
        
        coarse_feats = point_sample(
            prev_output, points, align_corners=self.align_corners)

        return coarse_feats

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg,
                      train_cfg):
        
        x = self._transform_inputs(inputs)
        with torch.no_grad():
            points = self.get_points_train(
                prev_output, calculate_uncertainty, cfg=train_cfg)
        fine_grained_point_feats = self._get_fine_grained_point_feats(
            x, points)
        coarse_point_feats = self._get_coarse_point_feats(prev_output, points)
        point_logits = self.forward(fine_grained_point_feats,
                                    coarse_point_feats)
        point_label = point_sample(
            gt_semantic_seg.float(),
            points,
            mode='nearest',
            align_corners=self.align_corners)
        point_label = point_label.squeeze(1).long()

        losses = self.losses(point_logits, point_label)

        return losses

    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        
        x = self._transform_inputs(inputs)
        refined_seg_logits = prev_output.clone()
        for _ in range(test_cfg.subdivision_steps):
            refined_seg_logits = resize(
                refined_seg_logits,
                scale_factor=test_cfg.scale_factor,
                mode='bilinear',
                align_corners=self.align_corners)
            batch_size, channels, height, width = refined_seg_logits.shape
            point_indices, points = self.get_points_test(
                refined_seg_logits, calculate_uncertainty, cfg=test_cfg)
            fine_grained_point_feats = self._get_fine_grained_point_feats(
                x, points)
            coarse_point_feats = self._get_coarse_point_feats(
                prev_output, points)
            point_logits = self.forward(fine_grained_point_feats,
                                        coarse_point_feats)

            point_indices = point_indices.unsqueeze(1).expand(-1, channels, -1)
            refined_seg_logits = refined_seg_logits.reshape(
                batch_size, channels, height * width)
            refined_seg_logits = refined_seg_logits.scatter_(
                2, point_indices, point_logits)
            refined_seg_logits = refined_seg_logits.view(
                batch_size, channels, height, width)

        return refined_seg_logits

    def losses(self, point_logits, point_label):
        
        loss = dict()
        loss['loss_point'] = self.loss_decode(
            point_logits, point_label, ignore_index=self.ignore_index)
        loss['acc_point'] = accuracy(point_logits, point_label)
        return loss

    def get_points_train(self, seg_logits, uncertainty_func, cfg):
        
        num_points = cfg.num_points
        oversample_ratio = cfg.oversample_ratio
        importance_sample_ratio = cfg.importance_sample_ratio
        assert oversample_ratio >= 1
        assert 0 <= importance_sample_ratio <= 1
        batch_size = seg_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        point_coords = torch.rand(
            batch_size, num_sampled, 2, device=seg_logits.device)
        point_logits = point_sample(seg_logits, point_coords)
        
        point_uncertainties = uncertainty_func(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(
            point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(
            batch_size, dtype=torch.long, device=seg_logits.device)
        idx += shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
            batch_size, num_uncertain_points, 2)
        if num_random_points > 0:
            rand_point_coords = torch.rand(
                batch_size, num_random_points, 2, device=seg_logits.device)
            point_coords = torch.cat((point_coords, rand_point_coords), dim=1)
        return point_coords

    def get_points_test(self, seg_logits, uncertainty_func, cfg):
        
        num_points = cfg.subdivision_num_points
        uncertainty_map = uncertainty_func(seg_logits)
        batch_size, _, height, width = uncertainty_map.shape
        h_step = 1.0 / height
        w_step = 1.0 / width

        uncertainty_map = uncertainty_map.view(batch_size, height * width)
        num_points = min(height * width, num_points)
        point_indices = uncertainty_map.topk(num_points, dim=1)[1]
        point_coords = torch.zeros(
            batch_size,
            num_points,
            2,
            dtype=torch.float,
            device=seg_logits.device)
        point_coords[:, :, 0] = w_step / 2.0 + (point_indices %
                                                width).float() * w_step
        point_coords[:, :, 1] = h_step / 2.0 + (point_indices //
                                                width).float() * h_step
        return point_indices, point_coords
