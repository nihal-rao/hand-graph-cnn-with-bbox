# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Basic evaluation script for PyTorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os.path as osp
import torch
import cv2
import numpy as np
import time

from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network_mod import ShapePoseNetwork
from hand_shape_pose.data.build import build_dataset
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import draw_2d_skeleton
from hand_shape_pose.util import renderer



def main():
	parser = argparse.ArgumentParser(description="3D Hand Shape and Pose Inference")
	parser.add_argument(
		"--config-file",
		default="configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"opts",
		help="Modify config options using the command-line",
		default=None,
		nargs=argparse.REMAINDER,
	)

	args = parser.parse_args()

	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.freeze()

	# 1. Load network model
	path = 'webcam_output'
	mkdir(path)
	model = ShapePoseNetwork(cfg, path)
	device = cfg.MODEL.DEVICE
	model.to(device)
	model.load_model(cfg)

	mesh_renderer = renderer.MeshRenderer(model.hand_tri.astype('uint32'))

	# 3. Inference
	model.eval()
	cpu_device = torch.device("cpu")
	cap = cv2.VideoCapture(0)
	i=0
	start = time.time()
	end = 0
	while(True):
		ret, images = cap.read()

		images = cv2.resize(images, (256, 256),  interpolation = cv2.INTER_AREA) 
		images = torch.tensor(images)
		images = torch.unsqueeze(images,0)
		images = images.to(device)
		with torch.no_grad():
			est_pose_uv = model(images)
			est_pose_uv = est_pose_uv.to(cpu_device)
		images = images.to(cpu_device)
		images=torch.squeeze(images,0)
		est_pose_uv = np.asarray(est_pose_uv)
		images=images.numpy()
		est_pose_uv=est_pose_uv[0]
		skeleton_overlay = draw_2d_skeleton(images, est_pose_uv)
		cv2.imshow('result',skeleton_overlay)
		name=str(i)+'.jpg'
		cv2.imwrite(osp.join(path , name), skeleton_overlay)
		i=i+1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			end = time.time()
			break

# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	print('FPS = {}'.format(i/(end-start)))



if __name__ == "__main__":
	main()
