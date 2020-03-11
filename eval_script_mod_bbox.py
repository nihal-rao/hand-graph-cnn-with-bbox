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

from hand_detection import test
import pyautogui as pyag

def main():
	pyag.FAILSAFE = False

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
	path = 'webcam_output_bbox'
	mkdir(path)
	model = ShapePoseNetwork(cfg, path)
	device = cfg.MODEL.DEVICE
	model.to(device)
	model.load_model(cfg)

	mesh_renderer = renderer.MeshRenderer(model.hand_tri.astype('uint32'))

	#2. Load hand_detector
	hand = test.hand_detector()
	
	# 3. Inference
	model.eval()
	cpu_device = torch.device("cpu")
	cap = cv2.VideoCapture(0)
	i=0
	start = time.time()
	end = 0
	config_dict= { 'y_up': 1.0,'y_low':0.5,'x_up':0.8,'x_low':0.2,'std_threshold':2e6,'buffer_size':10}
	"""
	y_up and y_low are the height thresholds within which the keypoints are mapped to the screen.
	ie., keypoints are mapped from a space which is (y_max-y_low)*img_height. If keypoints are outside this region cursor position does not change.
	std_threshold is the threshold for std deviation. If std_deviation is above the threshold (implies hand is moving),
	the cursor position is the location as detected in the previous image. If stddeviation is below the threshold (hand is static),
	the cursor position is the moving average of buffer_size previous images.
	"""
	prev_cursor_x =[]
	prev_cursor_y =[]
	prev_cursor = None
	screen_size = np.asarray(pyag.size())
	while(True):
		ret, frame = cap.read()
		img_h,img_w,_ = frame.shape
		bbox = hand.detect_hand(frame)
		bbox = bbox.astype(int)
		if bbox.size == 0:
			continue #If no bbox detected skip keypoint detection
		images = frame[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]]
		bb_h,bb_w,_ = images.shape
		images = cv2.resize(images, (256, 256),  interpolation = cv2.INTER_AREA) 
		images = torch.tensor(images)
		images = torch.unsqueeze(images,0)
		images = images.to(device)
		with torch.no_grad():
			est_pose_uv = model(images)
			est_pose_uv = est_pose_uv.to(cpu_device)
		est_pose_uv = np.asarray(est_pose_uv)
		est_pose_uv=est_pose_uv[0]
		est_pose_uv[:,0] = est_pose_uv[:,0]*bb_w/256
		est_pose_uv[:,1] = est_pose_uv[:,1]*bb_h/256

		est_pose_uv[:,0]+=bbox[0][0]
		est_pose_uv[:,1]+=bbox[0][1]
		if ((est_pose_uv[0,1] > (img_h*config_dict['y_up'])) or (est_pose_uv[0,1] < (img_h*config_dict['y_low']))) or ((est_pose_uv[0,0]>(img_w*config_dict['x_up'])) or (est_pose_uv[0,0] < (img_w*config_dict['x_low']))): 
			continue
		cursor_x = int((est_pose_uv[0,0]-(img_w*config_dict['x_low']))*1920./(img_w*(config_dict['x_up']-config_dict['x_low'])))
		cursor_y = int((est_pose_uv[0,1]-(img_h*config_dict['y_low']))*1080./(img_h*(config_dict['y_up']-config_dict['y_low'])))
		if len(prev_cursor_x) <= config_dict['buffer_size']:
			prev_cursor_x.append(cursor_x)
			prev_cursor_y.append(cursor_y)
		elif len(prev_cursor_x) > config_dict['buffer_size']:
			prev_cursor_x.append(cursor_x)
			prev_cursor_y.append(cursor_y)
			_ = prev_cursor_x.pop(0)
			_ = prev_cursor_y.pop(0)
		prev_cursor = np.column_stack((prev_cursor_x,prev_cursor_y))
		mean = np.mean(prev_cursor,0)
		var_dist=np.var(np.sum(((prev_cursor-mean)/screen_size)**2,1))
		if var_dist > config_dict['std_threshold']:
			pyag.moveTo(cursor_x,cursor_y)	
		else :
			pyag.moveTo(int(mean[0]),int(mean[1])) 
		skeleton_overlay = draw_2d_skeleton(frame, est_pose_uv)
		cv2.imshow('result',skeleton_overlay)
		#name=str(i)+'.jpg'
		#cv2.imwrite(osp.join(path , name), frame)
		i=i+1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			end = time.time()
			break

# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	print('no.of frames ={}'.format(i))
	print('FPS = {}'.format(i/(end-start)))



if __name__ == "__main__":
	main()
