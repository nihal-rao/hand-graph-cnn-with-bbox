from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
import cv2
from .data import cfg 
from .layers.functions.prior_box import PriorBox
from .utils.nms_wrapper import nms
from .models.faceboxes import FaceBoxes
from .utils.box_utils import decode

def load_model(model, pretrained_path):
	print('Loading pretrained model from {}'.format(pretrained_path))
	device = torch.cuda.current_device()
	pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
	if "state_dict" in pretrained_dict.keys():
		pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
	else:
		pretrained_dict = remove_prefix(pretrained_dict, 'module.')
	check_keys(model, pretrained_dict)
	model.load_state_dict(pretrained_dict, strict=False)
	return model

def check_keys(model, pretrained_state_dict):
	ckpt_keys = set(pretrained_state_dict.keys())
	model_keys = set(model.state_dict().keys())
	used_pretrained_keys = model_keys & ckpt_keys
	unused_pretrained_keys = ckpt_keys - model_keys
	missing_keys = model_keys - ckpt_keys
	print('Missing keys:{}'.format(len(missing_keys)))
	print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
	print('Used keys:{}'.format(len(used_pretrained_keys)))
	return True

def remove_prefix(state_dict, prefix):
	print('remove prefix \'{}\''.format(prefix))
	f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
	return {f(key): value for key, value in state_dict.items()}

class hand_detector():
	def __init__(self):
		self.train_path = '/home/mancmanomyst/hand-graph-cnn-master/hand_detection/weights/Final_HandBoxes.pth'
		self.cpu = False
		self.confidence_threshold = 0.2
		self.top_k = 5000
		self.nms_threshold = 0.2
		self.keep_top_k = 750
		net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
		net = load_model(net, self.train_path)
		device = torch.device("cuda")
		self.hnet = net.to(device)
		cudnn.benchmark = True

	def detect_hand(self,frame):
		#Inputs - image, H*W*C
		#Outputs - dets , B*4 (B=1 because only the bbox with highest score is returned.)
		torch.set_grad_enabled(False)
		self.hnet.eval()    
		device = torch.device("cuda")
		resize = 1
		scale_hand_y = 1.3
		scale_hand_x = 1.2
		to_show = frame
		img = np.float32(to_show)
		im_height, im_width, _ = img.shape
		scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
		img -= (104, 117, 123)
		img = img.transpose(2, 0, 1)
		img = torch.from_numpy(img).unsqueeze(0)
		img = img.to(device)
		scale = scale.to(device)

				 
		out = self.hnet(img)  # forward pass
		priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
		priors = priorbox.forward()
		priors = priors.to(device)
		loc, conf, _ = out
		prior_data = priors.data
		boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
		boxes = boxes * scale / resize
		boxes = boxes.cpu().numpy()
		scores = conf.data.cpu().numpy()[:, 1]

					# ignore low scores
		inds = np.where(scores > self.confidence_threshold)[0]
		boxes = boxes[inds]
		scores = scores[inds]

					# keep top-K before NMS
		order = scores.argsort()[::-1][:self.top_k]
		boxes = boxes[order]
		scores = scores[order]

					# do NMS
		dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
					# keep = py_cpu_nms(dets, self.nms_threshold)
		keep = nms(dets, self.nms_threshold, force_cpu=self.cpu)
		dets = dets[keep, :]

					# keep top-K faster NMS
		dets = dets[:self.keep_top_k, :]
		# dets [0:2] = x,y corrdinate of bbox top left corner
		# dets [2:4] = x,y corrdinate of bbox bottom right corner 
		for i in range(dets.shape[0]):
			curr_ht = dets[0][3]-dets[0][1]
			curr_w = dets[0][2]-dets[0][0]
			dets[0][1] = dets[0][3]-(scale_hand_y*curr_ht) 
			dets[0][0] = dets[0][2]-(scale_hand_x*curr_w)
			dets[0][2] = dets[0][2]+((scale_hand_x-1)*curr_w)
			if dets[0][0]<0:
				dets[0][0]=0
			if dets[0][1]<0:
				dets[0][1]=0	
			#cv2.rectangle(to_show, (dets[0][0], dets[0][1]), (dets[0][2], dets[0][3]), [0, 0, 255], 3)
			break
			
		#to_show = to_show[int(dets[0][1]):int(dets[0][3]),int(dets[0][0]):int(dets[0][2])]
		#cv2.imshow('image', to_show)
		return dets
























