import cv2
import time
import matplotlib.pyplot as plt

# Chargez l'image
img = cv2.imread('Michelangelo/frag_eroded/frag_eroded_4.png',0)

#SIFT
sift = cv2.SIFT_create()
start_time = time.time()
keypoints, descriptors = sift.detectAndCompute(img, None)
img_keypoints_shift = cv2.drawKeypoints(img, keypoints, outImage=None)
end_time = time.time()
print(" Shift Time taken:", end_time - start_time, "seconds","Number of keypoints:", len(keypoints))


#ORB
orb = cv2.ORB_create()
start_time = time.time()
keypoints, descriptors = orb.detectAndCompute(img, None)
img_keypoints_orb = cv2.drawKeypoints(img, keypoints, outImage=None)
end_time = time.time()
print("ORB Time taken:", end_time - start_time, "seconds","Number of keypoints:", len(keypoints))



#FAST
fast = cv2.FastFeatureDetector_create()
start_time = time.time()
keypoints = fast.detect(img, None)
img_keypoints_fast = cv2.drawKeypoints(img, keypoints, outImage=None)
end_time = time.time()
print("Fast taken:", end_time - start_time, "seconds","Number of keypoints:", len(keypoints))




#BRIEF
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
start_time = time.time()
keypoints, descriptors = brief.compute(img, keypoints)
img_keypoints_brief = cv2.drawKeypoints(img, keypoints, outImage=None)
end_time = time.time()
print("Brief taken:", end_time - start_time, "seconds","Number of keypoints:", len(keypoints))


#BRISK
brisk = cv2.BRISK_create()
start_time = time.time()
keypoints, descriptors = brisk.detectAndCompute(img, None)
img_keypoints_briks = cv2.drawKeypoints(img, keypoints, outImage=None)
end_time = time.time()
print("Brisk taken:", end_time - start_time, "seconds","Number of keypoints:", len(keypoints))


#Freak
freak = cv2.xfeatures2d.FREAK_create()
start_time = time.time()
keypoints, descriptors = freak.compute(img, keypoints)
img_keypoints_freak = cv2.drawKeypoints(img, keypoints, outImage=None)
end_time = time.time()
print("Freak taken:", end_time - start_time, "seconds","Number of keypoints:", len(keypoints))



#print all the the images
# plt.subplot(231),plt.imshow(img, cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(img_keypoints_shift, cmap = 'gray')
plt.title('SIFT'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(img_keypoints_orb, cmap = 'gray')
plt.title('ORB'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(img_keypoints_fast, cmap = 'gray')
plt.title('FAST'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(img_keypoints_brief, cmap = 'gray')
plt.title('BRIEF'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(img_keypoints_briks, cmap = 'gray')
plt.title('BRISK'), plt.xticks([]), plt.yticks([])
plt.subplot(231),plt.imshow(img_keypoints_freak, cmap = 'gray')
plt.title('FREAK'), plt.xticks([]), plt.yticks([])
plt.show()







#find and draw the keypoints with surf in the image
# surf = cv2.xfeatures2d.SURF_create()
# keypoints, descriptors = surf.detectAndCompute(img, None)
# img_keypoints = cv2.drawKeypoints(img, keypoints, outImage=None)
# plt.imshow(img_keypoints)
# plt.show()









