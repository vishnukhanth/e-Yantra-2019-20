###############################################################################
## Author: Team Supply Bot
## Edition: eYRC 2019-20
## Instructions: Do Not modify the basic skeletal structure of given APIs!!!
###############################################################################


######################
## Essential libraries
######################
import cv2
import numpy as np
import os
import math
import csv
import copy






############################################
## Build your algorithm in this function
## ip_image: is the array of the input image
## imshow helps you view that you have loaded
## the corresponding image
############################################
def process(ip_image):
    ###########################
    ## Your Code goes here
    ###########################
    img = ip_image
    img_o = img.copy()

    # applying bilateral filter to remove noise with edge preservation.
    b_filter = cv2.bilateralFilter(img,9,75,75)

    # defining the hsv ranges for green, red and white colour.
    low_green = np.array([25, 52, 72])
    high_green = np.array([55, 255, 255])

    low_red = np.array([170,150,100])
    high_red = np.array([180, 255, 255])

    white_lower = np.array([0,0,168])
    white_upper = np.array([179,100,255])

    # changing the colour space of filtered image.
    hsv = cv2.cvtColor(b_filter, cv2.COLOR_BGR2HSV)

    # gray scale of filtered image.
    gray = cv2.cvtColor(b_filter, cv2.COLOR_BGR2GRAY)

    # applying adaptive gaussian threshold to adapt different light illuminations.
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,17,8)

    # defining kernel for morphological modification of image.
    kernel = np.ones((3,3),np.uint8)

    # function for morphologicalmodification(opening, closing and dilation).
    def morphological(image, flag):
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        # if second argument is 1 then dilation will be done.
        if flag == 1:
            dilation = cv2.dilate(closing,kernel,iterations = 1)
            return dilation
        return closing

    # function for computing center of a contour.
    def center(contour):
        M = cv2.moments(contour)
        c1 = int(M["m10"] / M["m00"])
        c2 = int(M["m01"] / M["m00"])
        return c1,c2

    # function for detecting red and green coins.
    # detects multiple coins in the inner black ring.
    def coin_det(cnts):
        new = []
        distance = []
        for i in range(len(cnts)):
            m = cv2.moments(cnts[i])
            x = int(m["m10"] / m["m00"])
            y = int(m["m01"] / m["m00"])
            # finding distance between center of white circle and the center of coloured contour
            dist = math.sqrt(((x2-x)**2)+((y2-y)**2))
            # since, inside state all red and green unwanted noise are inside inner black ring
            #(contd..) the red and green contour which is far away from center is definitely a coin.
            distance.append(dist)
        maximum = max(distance)
        for i in range(len(cnts)):
            # checking for other coins with same distance as first detected coin with
            # (contd..) error(distance) range of is 10 pixels
            if abs(distance[i] - maximum) <= 10:
                new.append(cnts[i])    
        return new

    # function for drawing contours.
    def draw_coins(contours):
        coordinates = []
        for i in contours:
            (x,y),radius = cv2.minEnclosingCircle(i)
            center = (int(x),int(y))
            radius = int(radius)
            coordinates.append((x,y))
            cv2.circle(img_o, center, radius,(255,0,0), 2)
        return coordinates

    # applying morphological threshold to gaussian threshold to remove noise.
    morph = morphological(th, 1)

    # finding contours and sorting them according to area(descending order).
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    # isolating only the state ie) the outer black circle.
    # this done by picking the contour with largest area which is a circle.
    # for checking shape contour approximation is used.
    for i in range(len(cnts)) :
        epsilon = 0.01*cv2.arcLength(cnts[i], True)
        approx = cv2.approxPolyDP(cnts[i], epsilon, True)
        # if len(approx) >= 12 then it will be a circle mostly.
        if len(approx) >= 12:
            j = i
            break
        
    # creating dummy image 
    dummy = np.zeros(img.shape[0:2], np.uint8)
    # creating a mask for state and its done by filling the biggest circular contour with white pixels in dummy image.
    state = cv2.drawContours(dummy, cnts, j, (255,255,255), -1)

    # bitwise_and to get only the state.
    res = cv2.bitwise_and(b_filter ,b_filter, mask = state)
    
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

    # masking for white, green and red colours in the state.
    mask = cv2.inRange(hsv, white_lower, white_upper)
    mask1 = cv2.inRange(hsv, low_green, high_green)
    mask2 = cv2.inRange(hsv, low_red, high_red)

    # denoising of masks
    mask3 = morphological(mask, 0)
    mask4 = morphological(mask1, 0)
    mask5 = morphological(mask2, 0)

    # finding center of the state.
    c1, c2 = center(cnts[0])

    # finding contours in denoised white mask. 
    contours2, hierarchy2 = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    distance = []
    new = []
    # finding distance between center of the state and all the contours in mask3.
    for i in range(len(contours2)):
        area = cv2.contourArea(contours2[i])
        hull = cv2.convexHull(contours2[i])
        hullarea = cv2.contourArea(hull)
        solidity = area/float(hullarea)
        if solidity >= 0.9 :
            new.append(contours2[i])
            point = contours2[i][0]
            dist = math.sqrt(((c1-point[0][0])**2)+((c2-point[0][1])**2))
            distance.append(dist)

    # the contour which is near to the center of the state is the center white circle.
    a = distance.copy()
    a.sort()
    i = distance.index(a[0])

    center_white = new[i]

    # finding center of center white circle.
    M = cv2.moments(center_white)
    x2 = int(M["m10"] / M["m00"])
    y2 = int(M["m01"] / M["m00"])

    # finding contours in denoised red mask.
    contours, hierarchy = cv2.findContours(mask5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # detecting red coins.
    red_coins = coin_det(contours)
    # finding contours in denoised green mask.
    contours1, hierarchy1 = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # detecting green coins.
    green_coins = coin_det(contours1)

    # drawing red coins and getting their centers.
    red_coin_centers = draw_coins(red_coins)
    
    # drawing green coins and getting their centers.
    green_coin_centers = draw_coins(green_coins)

    # finding angle between first detected red and green coin.
    coordinate1 = np.array([red_coin_centers[0][0],red_coin_centers[0][1]])
    coordinate2 = np.array([x2,y2])
    coordinate3 = np.array([green_coin_centers[0][0],green_coin_centers[0][1]])
    vector1 = coordinate1 - coordinate2
    vector2 = coordinate3 - coordinate2
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    angle = round(float(angle), 2)

    angle = "Angle : "+str(angle)

    # writing angle in the image.
    cv2.putText(img_o, angle, (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # displaying op_image.
    cv2.imshow("op_image", img_o)
    
    op_image = img_o
    
    return op_image

    
####################################################################
## The main program which provides read in input of one image at a
## time to process function in which you will code your generalized
## output computing code
## Modify the image name as per instruction
####################################################################
def main():
    ################################################################
    ## variable declarations
    ################################################################
    i = 1
    ## reading in video 
    cap = cv2.VideoCapture(1) #if you have a webcam on your system, then change 0 to 1
    ## getting the frames per second value of input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    ## setting the video counter to frame sequence
    cap.set(3, 640)
    cap.set(4, 480)
    ## reading in the frame
    ret, frame = cap.read()
    ## verifying frame has content
    print(frame.shape)

    while(ret):
        i = i + 1
        ret, frame = cap.read()
        ## display to see if the frame is correct
        cv2.imshow("window", frame)
        cv2.waitKey(int(1000/fps));
        ## calling the algorithm function
        op_image = process(frame)
        cv2.imwrite("SB#64_task3I.jpg",op_image)
        if i == 500 :
            cv2.destroyAllWindows()
            break


    

############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main()
