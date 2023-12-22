import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import os


img = cv2.imread('Michelangelo/frag_eroded/frag_eroded_4.png',0)
fresque = cv2.imread('Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg',0)


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
plt.subplot(231),plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
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


# #test the matching with orb
keypoints, descriptors = orb.detectAndCompute(img, None)
fresquekeypoints,fresquedescriptors = orb.detectAndCompute(fresque, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors,fresquedescriptors)
matches = sorted(matches, key = lambda x:x.distance)
img_matches_orb = cv2.drawMatches(img,keypoints,fresque,fresquekeypoints,matches[:10],None, flags=2)


# #test the matching with sift
keypoints, descriptors = sift.detectAndCompute(img, None)
fresquekeypoints,fresquedescriptors = sift.detectAndCompute(fresque, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors,fresquedescriptors, k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
img_matches_sift = cv2.drawMatchesKnn(img,keypoints,fresque,fresquekeypoints,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)






#test the matching with brisk
keypoints, descriptors = brisk.detectAndCompute(img, None)
fresquekeypoints,fresquedescriptors = brisk.detectAndCompute(fresque, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors,fresquedescriptors)
matches = sorted(matches, key = lambda x:x.distance)
img_matches_brisk  = cv2.drawMatches(img,keypoints,fresque,fresquekeypoints,matches[:10],None, flags=2)

#test the matching with freak
keypoints, descriptors = freak.compute(img, keypoints)
fresquekeypoints,fresquedescriptors = freak.compute(fresque, fresquekeypoints)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors,fresquedescriptors)
matches = sorted(matches, key = lambda x:x.distance)
img_matches_freak  = cv2.drawMatches(img,keypoints,fresque,fresquekeypoints,matches[:10],None, flags=2)

#test the matching with brief
keypoints, descriptors = brief.compute(img, keypoints)
fresquekeypoints,fresquedescriptors = brief.compute(fresque, fresquekeypoints)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors,fresquedescriptors)
matches = sorted(matches, key = lambda x:x.distance)
img_matches_brief  = cv2.drawMatches(img,keypoints,fresque,fresquekeypoints,matches[:10],None, flags=2)

#test the matching with fast
keypoints = fast.detect(img, None)
fresquekeypoints = fast.detect(fresque, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors,fresquedescriptors, k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
img_matches_fast = cv2.drawMatchesKnn(img,keypoints,fresque,fresquekeypoints,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)





#show the two matches sift and orb
plt.subplot(231),plt.imshow(img_matches_orb),plt.title('ORB')
plt.subplot(232),plt.imshow(img_matches_sift),plt.title('SIFT')
plt.subplot(233),plt.imshow(img_matches_brisk),plt.title('BRISK')
plt.subplot(234),plt.imshow(img_matches_freak),plt.title('FREAK')
plt.subplot(235),plt.imshow(img_matches_brief),plt.title('BRIEF')
plt.subplot(236),plt.imshow(img_matches_fast),plt.title('FAST')
plt.show()








    
















    
























