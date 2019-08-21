import os
import numpy as np
import json
import cv2
from statistics import mean

HOR_THRESH = 100
VER_THRESH = 100

class LANES:
	"""docstring for ClassName"""
	def __init__(self):
		self.lane = {}
		for i in range(160, 720, 10):
			self.lane[i] = -2

	def show(self):
		for i in range(160, 720, 10):
			print(self.lane[i])


count = 0
for img_name in os.listdir('images/'):
	if('out' not in img_name):
		continue
	img = cv2.imread('images/' + img_name, 0)
	pred = {}
	count += 1
	# size = np.size(img)
	# skel = np.zeros(img.shape,np.uint8)
	 
	# ret,img = cv2.threshold(img,127,255,0)
	# element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	# done = False
	 
	# while( not done):
	#     eroded = cv2.erode(img,element)
	#     temp = cv2.dilate(eroded,element)
	#     temp = cv2.subtract(img,temp)
	#     skel = cv2.bitwise_or(skel,temp)
	#     img = eroded.copy()
	 
	#     zeros = size - cv2.countNonZero(img)
	#     if zeros==size:
	#         done = True

	# # cv2.imshow(img_name, skel)
	# # cv2.waitKey(0)
	# img = skel
	print(img_name)
	centres = []
	lanes = []
	for row in range(710, 150, -10):
		cont_points = []
		flag = 0
		curr_points = []
		for col in range(0, img.shape[1]):
			if(img[row][col] > 127):
				curr_points.append(col)
				flag = 1
			else:
				if(flag):
					flag = 0
					cont_points.append(curr_points)
					curr_points = []
		
		for i in range(0, len(cont_points)):
			cont_points[i] = int(mean(cont_points[i]))

		if(len(cont_points) == 0):
			continue

		if(len(centres) == 0):
			centres = cont_points
			for i in range(0, len(centres)):
				l = LANES()
				l.lane[row] = centres[i]
				lanes.append(l)
			continue

		# print centres
		centres_new = centres
		for point in cont_points:
			min_dist = 1000
			min_idx = -2
			for i, ct in enumerate(centres):
				if(min_dist > abs(ct - point)):
					min_dist = abs(ct - point)
					min_idx = i
			if(min_dist <= VER_THRESH):
				lanes[min_idx].lane[row] = point
				centres_new[min_idx] = (point + centres_new[min_idx])/2
			else:
				l = LANES()
				l.lane[row] = point
				lanes.append(l)
				centres_new.append(point)
			centres = centres_new

	final_lanes = []
	for l in lanes:
		temp = l.lane.values()
		c = 0
		for i in range(len(temp)):
			if(temp[i] != -2):
				c += 1
		if(c <= 3):
			continue
		this_lane = []
		for i in range(160, 720, 10):
			this_lane.append(l.lane[i])
		final_lanes.append(this_lane)
		# print this_lane
	# print centres
	print(len(final_lanes))
	# break
	pred['lanes'] = final_lanes
	pred['raw_file'] = 'images/' + img_name
	pred['run_time'] = 100
	if(count == 1):
		json_data = json.dumps(pred)
	else:
		json_data += '\n'
		json_data += json.dumps(pred)


file = open('pred_json.json', 'wb')
file.write(json_data)
