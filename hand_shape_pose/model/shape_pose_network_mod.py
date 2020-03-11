# Copyright (c) Liuhao Ge. All Rights Reserved.
# Some of the code for Graph ConvNet is forked from https://github.com/xbresson/spectral_graph_convnets
r"""
Networks for mesh generation and pose estimation using Spectral Graph ConvNet
"Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering",
M Defferrard, X Bresson, P Vandergheynst, NPIS 2016
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hand_shape_pose.model.net_hg_mod import Net_HM_HG
from hand_shape_pose.model.net_hm_feat_mesh import Net_HM_Feat_Mesh
from hand_shape_pose.model.net_mesh_pose import Graph_CNN_Mesh_Pose
from hand_shape_pose.util.net_util import load_net_model
from hand_shape_pose.util.graph_util import build_hand_graph
from hand_shape_pose.util.image_util import BHWC_to_BCHW, normalize_image, uvd2xyz
from hand_shape_pose.util.heatmap_util import compute_uv_from_heatmaps


class ShapePoseNetwork(nn.Module):
	"""
	Main class for 3D hand shape and pose inference network.
	It consists of three main parts:
	- heat-map estimation network
	- shape estimation network
	- pose estimation network
	"""
	def __init__(self, cfg, output_dir):
		super(ShapePoseNetwork, self).__init__()

		# 1. Build graph for Hand Mesh
		self.graph_L, self.graph_mask, self.graph_perm_reverse, self.hand_tri = \
			build_hand_graph(cfg.GRAPH.TEMPLATE_PATH, output_dir)

		# 2. Create model
		num_joints = cfg.MODEL.NUM_JOINTS
		self.net_hm = Net_HM_HG(num_joints,
								num_stages=cfg.MODEL.HOURGLASS.NUM_STAGES,
								num_modules=cfg.MODEL.HOURGLASS.NUM_MODULES,
								num_feats=cfg.MODEL.HOURGLASS.NUM_FEAT_CHANNELS)

		num_heatmap_chan = self.net_hm.numOutput
		num_feat_chan = self.net_hm.nFeats
		num_mesh_output_chan = 3
		num_pose_output_chan = (num_joints - 1)# * 3

		self.net_feat_mesh = Net_HM_Feat_Mesh(num_heatmap_chan, num_feat_chan,
											  num_mesh_output_chan, self.graph_L)
		self.net_mesh_pose = Graph_CNN_Mesh_Pose(num_mesh_output_chan,
												 num_pose_output_chan, self.graph_L)

	def load_model(self, cfg):
		load_net_model(cfg.MODEL.PRETRAIN_WEIGHT.HM_NET_PATH, self.net_hm)
		load_net_model(cfg.MODEL.PRETRAIN_WEIGHT.MESH_NET_PATH, self.net_feat_mesh)
		load_net_model(cfg.MODEL.PRETRAIN_WEIGHT.POSE_NET_PATH, self.net_mesh_pose)

	def to(self, *args, **kwargs):
		super(ShapePoseNetwork, self).to(*args, **kwargs)
		self.graph_L = [l.to(*args, **kwargs) for l in self.graph_L]
		self.net_feat_mesh.mesh_net.graph_L = self.graph_L
		self.net_mesh_pose.graph_L = self.graph_L
		self.graph_mask = self.graph_mask.to(*args, **kwargs)

	def forward(self, images):
		"""
		:param images: B x H x W x C
		:param cam_param: B x 4, [fx, fy, u0, v0]
		:param bbox: B x 4, bounding box in the original image, [x, y, w, h]
		:param pose_root: B x 3
		:param pose_scale: B
		:return:
		"""
		num_sample = images.shape[0]
		images = BHWC_to_BCHW(images)  # B x C x H x W
		images = normalize_image(images)

		# 1. Heat-map estimation
		est_hm_list= self.net_hm(images)


		# combine heat-map estimation results to compute pose xyz in camera coordiante system
		est_pose_uv = compute_uv_from_heatmaps(est_hm_list[-1], images.shape[2:4])  # B x K x 3

		return est_pose_uv[:, :, :2]
