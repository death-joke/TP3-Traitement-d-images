import cv2
import matplotlib.pyplot as plt

# Chargez l'image
img = cv2.imread('Michelangelo/frag_eroded/frag_eroded_4.png',0)

# Créez un objet SIFT
# sift = cv2.xfeatures2d.SIFT_create()
# keypoints, descriptors = sift.detectAndCompute(img, None)
# img_keypoints = cv2.drawKeypoints(img, keypoints, outImage=None)
# plt.imshow(img_keypoints)
# plt.show()

#find and draw the keypoints with surf in the image
surf = cv2.xfeatures2d.SURF_create()
keypoints, descriptors = surf.detectAndCompute(img, None)
img_keypoints = cv2.drawKeypoints(img, keypoints, outImage=None)
plt.imshow(img_keypoints)
plt.show()
