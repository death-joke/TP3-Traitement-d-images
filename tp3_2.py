import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def ransac(fragment_keypoints, fresco_keypoints, matches, threshold=10, num_iterations=100):
    best_params = None
    best_inliers = 0
    

    for _ in range(num_iterations):
        
        random_match_indices = np.random.choice(len(matches), 3, replace=False)
        random_matches = [matches[i][0] for i in random_match_indices]

        fragment_pts = np.float32([fragment_keypoints[m.queryIdx].pt for m in random_matches]).reshape(-1, 1, 2)
        fresco_pts = np.float32([fresco_keypoints[m.trainIdx].pt for m in random_matches]).reshape(-1, 1, 2)

        # Correct shape of pts
        fragment_pts = np.squeeze(fragment_pts)
        fresco_pts = np.squeeze(fresco_pts)


        # Estimate affine partial transformation (rotation + translation)
        M, _ = cv2.estimateAffinePartial2D(fragment_pts, fresco_pts)

        # #print x,y,theta
        # print("x:", M[0, 2])
        # print("y:", M[1, 2])
        # print("theta:", np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi)
        # print('##################################################')


      

        fragment_pts = np.float32([fragment_keypoints[m.queryIdx].pt for m in random_matches]).reshape(-1, 1, 2)
        # Apply the estimated transformation to all fragment points   
        
        if M is None:
            transformed_pts = fragment_pts

        try:
            transformed_pts = cv2.transform(fragment_pts, M)
        except Exception as e:
            print(e)
            print("M:",M)
            print("fragment_pts:",fragment_pts)
            return None


        transformed_pts = np.squeeze(transformed_pts)

        # Calculate the Euclidean distance between transformed points and actual fresco keypoints
        distances = np.sqrt(np.sum((transformed_pts - fresco_pts) ** 2, axis=1))
       

        # Count inliers (points within the threshold)
        inliers = np.sum(distances < threshold)
        


        # Update the best parameters if the current model has more inliers
        if inliers > best_inliers:
            best_inliers = inliers
            best_params = M
            
    return best_params



print("quelle fresque reconstruite ?")
print("1- Michelangelo_ThecreationofAdam_1707x775.jpg")
print("2- Domenichino_Virgin-and-unicorn.jpg")
user_choice = input("choix : ")

if user_choice == "1":
    
    ###############with all 
    dir_path = 'Michelangelo/frag_eroded/'  
    filenames = os.listdir(dir_path)
    filenames.sort()
    fragments = [cv2.imread(os.path.join(dir_path, filename)) for filename in filenames]
    fresque = cv2.imread('Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg',0)

    sift =cv2.SIFT_create()
    bf = cv2.BFMatcher()
    fresquekeypoints,fresquedescriptors = sift.detectAndCompute(fresque, None)

    transformations = []
    #run ransac for each fragment
    for j in range(len(fragments)):
        
        img = cv2.imread('Michelangelo/frag_eroded/frag_eroded_'+str(j)+'.png',0)
        keypoints, descriptors = sift.detectAndCompute(img, None)
    


        matches = bf.knnMatch(descriptors,fresquedescriptors, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        
        if len(good) > 2:

            # Calculate the center of the fragment image
            center_x = img.shape[1] / 2
            center_y = img.shape[0] / 2

            # Change the origin of the keypoints to the center of the fragment image
            for kp in keypoints:
                kp.pt = (kp.pt[0] - center_x, kp.pt[1] - center_y)


            # Round the coordinates of the matching points
            for i in range(len(good)):
                x = round(fresquekeypoints[good[i][0].trainIdx].pt[0])
                y = round(fresquekeypoints[good[i][0].trainIdx].pt[1])
                fresquekeypoints[good[i][0].trainIdx].pt = (x, y)

                x = round(keypoints[good[i][0].queryIdx].pt[0])
                y = round(keypoints[good[i][0].queryIdx].pt[1])
                keypoints[good[i][0].queryIdx].pt = (x, y)

            best_transformation=ransac(keypoints, fresquekeypoints, good, threshold=0.1, num_iterations=1000)
            if best_transformation is not None:
                x = round(best_transformation[0, 2])
                y = round(best_transformation[1, 2])
                theta = np.arctan2(best_transformation[1, 0], best_transformation[0, 0]) * 180 / np.pi
                #si theta est positif on le met en négatif
                if theta > 0:
                    theta = -theta
                if x>0 and y>0:                
                    transformations.append((j, x, y, theta))
        print(j/len(fragments)*100,"%")
                
               
    
    with open('michelangelo_solution.txt', 'w') as f:
        for i, x, y, theta in transformations:
            f.write(f"{i} {x} {y} {theta}\n")

if user_choice == "2":
    
    ###############with all 
    dir_path = 'Domenichino_Virgin-and-unicorn/frag_eroded/'  
    filenames = os.listdir(dir_path)
    filenames.sort()
    fragments = [cv2.imread(os.path.join(dir_path, filename)) for filename in filenames]
    fresque = cv2.imread('Domenichino_Virgin-and-unicorn/Domenichino_Virgin-and-unicorn.jpg',0)

    sift =cv2.SIFT_create()
    bf = cv2.BFMatcher()
    fresquekeypoints,fresquedescriptors = sift.detectAndCompute(fresque, None)

    transformations = []
    #run ransac for each fragment
    for j in range(len(fragments)):
        if os.path.exists('Domenichino_Virgin-and-unicorn/frag_eroded/frag_eroded_'+str(j)+'.png'):
        
            img = cv2.imread('Domenichino_Virgin-and-unicorn/frag_eroded/frag_eroded_'+str(j)+'.png',0)
            keypoints, descriptors = sift.detectAndCompute(img, None)
        


            matches = bf.knnMatch(descriptors,fresquedescriptors, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            
            if len(good) > 2:

                # Calculate the center of the fragment image
                center_x = img.shape[1] / 2
                center_y = img.shape[0] / 2

                # Change the origin of the keypoints to the center of the fragment image
                for kp in keypoints:
                    kp.pt = (kp.pt[0] - center_x, kp.pt[1] - center_y)


                # Round the coordinates of the matching points
                for i in range(len(good)):
                    x = round(fresquekeypoints[good[i][0].trainIdx].pt[0])
                    y = round(fresquekeypoints[good[i][0].trainIdx].pt[1])
                    fresquekeypoints[good[i][0].trainIdx].pt = (x, y)

                    x = round(keypoints[good[i][0].queryIdx].pt[0])
                    y = round(keypoints[good[i][0].queryIdx].pt[1])
                    keypoints[good[i][0].queryIdx].pt = (x, y)

                best_transformation=ransac(keypoints, fresquekeypoints, good, threshold=0.1, num_iterations=1000)
                if best_transformation is not None:
                    x = round(best_transformation[0, 2])
                    y = round(best_transformation[1, 2])
                    theta = np.arctan2(best_transformation[1, 0], best_transformation[0, 0]) * 180 / np.pi
                    #si theta est positif on le met en négatif
                    if theta > 0:
                        theta = -theta
                    if x>0 and y>0:                
                        transformations.append((j, x, y, theta))
            print(j/len(fragments)*100,"%")
           
                
               
    
    with open('domenichino_solution.txt', 'w') as f:
        for i, x, y, theta in transformations:
            f.write(f"{i} {x} {y} {theta}\n")








