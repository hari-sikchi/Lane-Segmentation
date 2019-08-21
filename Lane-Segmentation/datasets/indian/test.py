import cv2
import numpy as np
 

cap = cv2.VideoCapture('D2_mute.mp4')
 
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
mean_arr = np.zeros((3))
count = 0

while(cap.isOpened()):
	ret, frame = cap.read()

	# Mean
	if ret == True:
		count += 1
		a = np.mean(frame, axis=0)
		a = np.mean(a, axis = 0)
		mean_arr += a
		print(count)
	if ret == False:
		break


# Mean
print(mean_arr/count)
print(count)
 
cap.release()
 
cv2.destroyAllWindows()