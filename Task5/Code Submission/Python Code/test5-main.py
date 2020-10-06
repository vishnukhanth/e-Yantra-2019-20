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
from aruco_lib import *
import serial

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
    ret_flag = 0
    city_flag = 0
    
    img = ip_image
    img_o = img.copy()
    cpy = img.copy()

    
    # aruco detection..
    id_list = []
    id_list = detect_Aruco(ip_image)
    img1 = mark_Aruco(img.copy(), id_list)
    cv2.imshow("Aruco",img1)
    

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

    # function for hiding aruco in the image
    def hide_aruco(ids, mask_img):
        keys = list(id_list.keys())[0]
        point1 = tuple((id_list[keys])[0])
        point2 = tuple((id_list[keys])[2])
        cv2.rectangle(mask_img, point1, point2, 0, 50)
        return mask_img

    # function for isolating only the highways...
    def Highway_isolation(contours, x, y):
        distance_list = []
        filtered = filter(lambda a : cv2.contourArea(a) >= 5000, contours)
        min_dist = 9999999
        j = -1
        for i in filtered:
            dist = cv2.pointPolygonTest(i,(x,y),True)
            dist = abs(dist)
            if dist <= min_dist:
                min_dist = dist
                j = i
        return j

    # function for clockwise numbering of cities...
    def clockwise_numbering(contour, x, y):
        coordinates = []
        coordinates.append(tuple([int(x), int(y)]))
        for i in contour:
            epsilon = 0.01*cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, epsilon, True)
            p = (approx[0])[0]
            point = tuple([int(p[0]), int(p[1])])
            coordinates.append(point)
        coordinates.sort(key=lambda k: k[0])
        clockwise = []
        r1 = coordinates[0][1]
        r2 = coordinates[-1][1]
        temp = []
        ordered = coordinates.copy()
        for i in range(len(coordinates)):
            if ((coordinates[i])[1]) in range(0,max(r1,r2)):
                clockwise.append(coordinates[i])
            else:
                temp.append(coordinates[i])
        temp.reverse()
        for i in temp:
            clockwise.append(i)
        index = clockwise.index((x, y))
        for i in range(len(clockwise)):
            ordered[i-index] = clockwise[i]
            
        return ordered

    # function for getting coin locations...
    def coin_locations(cities, red_coin, green_coin):
        loc = []
        def location(coins):
            output = ""
            temp1 = []
            cnt = 0
            for i in coins:
                minimum = 9999999
                for j in range(len(cities)):
                    dist = math.sqrt((((cities[j])[0]-i[0])**2)+(((cities[j])[1]-i[1])**2))
                    if dist < minimum :
                        minimum = dist
                        k = j+1
                if flag == 0:
                    temp = "Node No.\tType of Relief Aid"+"\n"+str(k)+"\t\t"+"Medical Aid"
                    loc.append(k)
                    return temp
                else :
                    temp = "\n"+str(k)+"\t\t"+"Food Supply"
                    output = output + temp
                    loc.append(k)
                    cnt = cnt + 1
                    if cnt == 2 :
                        return loc, output

        flag = 0
        string = location(red_coin)
        flag = 1
        location, string1 = location(green_coin)
        string = string+string1
        
        return loc, string


    # function for calculating angle...
    def calculate_angle(point1, point2, point3):
        # finding angle between first detected red and green coin.
        a = ''
        coordinate1 = np.array([point1[0], point1[1]])
        coordinate2 = np.array([point2[0], point2[1]])
        coordinate3 = np.array([point3[0], point3[1]])
        v1 = coordinate1 - coordinate2
        v2 = coordinate3 - coordinate2
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        angle = round(float(angle), 2)
        direction = math.degrees(math.atan2(v1[0]*v2[1]-v1[1]*v2[0], v1[0]*v2[0]+v1[1]*v2[1]))
        direction = round(float(direction), 2)
        if direction < 0 :
            a = 'a'
        else :
            a = 'c'

        return angle, a

    '''
    # function for calculating angle...
    def calculate_angle(point1, point2, point3):
        # finding angle between first detected red and green coin.
        coordinate1 = np.array([point1[0], point1[1]])
        coordinate2 = np.array([point2[0], point2[1]])
        coordinate3 = np.array([point3[0], point3[1]])
        v1 = coordinate1 - coordinate2
        v2 = coordinate3 - coordinate2
        angle = math.degrees(math.atan2(v1[0]*v2[1]-v1[1]*v2[0], v1[0]*v2[0]+v1[1]*v2[1]))
        angle = round(float(angle), 2)

        return angle
    '''
    
    # path planning for bot traversal
    def path_planning(aruco, center, red_coin, green_coin, locations):
        path = []
        Direction = []
        path.append(locations[0])
        ang, a = calculate_angle(aruco, center, red_coin[0])
        Direction.append(a)
        ang1, a1 = calculate_angle(red_coin[0], center, green_coin[0])
        ang2, a2 = calculate_angle(red_coin[0], center, green_coin[1])
        
        if ang1 < ang2 :
            path.append(locations[1])
            Direction.append(a1)
            ang3, a3 = calculate_angle(green_coin[0], center, green_coin[1])
            path.append(locations[2])
            Direction.append(a3)
            ang4, a4 = calculate_angle(green_coin[1], center, aruco)
            path.append(1)
            Direction.append(a4)
            
        if ang2 < ang1 :
            path.append(locations[2])
            Direction.append(a2)
            ang3, a3 = calculate_angle(green_coin[1], center, green_coin[0])
            path.append(locations[1])
            Direction.append(a3)
            ang4, a4 = calculate_angle(green_coin[0], center, aruco)
            path.append(1)
            Direction.append(a4)

        return path, Direction
        
    # applying morphological threshold to gaussian threshold to remove noise.
    morph = morphological(th, 1)

    # finding contours and sorting them according to area(descending order).
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull_a = []
    
    for i in contours:
        hull = cv2.convexHull(i,returnPoints = True)
        hull_a.append(cv2.contourArea(hull))

    cnts = []
    for i in range(len(hull_a)):
        maximum = max(hull_a)
        j = hull_a.index(maximum)
        cnts.append(contours[j])
        hull_a[j] = -1

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
    hull = cv2.convexHull(cnts[j],returnPoints = True)

    dummy = np.zeros(img.shape[0:2], np.uint8)
    # creating a mask for state and its done by filling the biggest circular contour with white pixels in dummy image.
    state = cv2.drawContours(dummy, [hull], 0, (255,255,255), -1)

    # bitwise_and to get only the state.
    res = cv2.bitwise_and(b_filter ,b_filter, mask = state)
    cv2.imshow("result", res)
    
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

    # masking for white, green and red colours in the state.
    mask = cv2.inRange(hsv, white_lower, white_upper)
    cv2.imshow("wh", mask)

    mask1 = cv2.inRange(hsv, low_green, high_green)
    mask2 = cv2.inRange(hsv, low_red, high_red)

    # denoising of masks
    mask3 = morphological(mask, 0)

    cv2.imshow("White", mask3)
    mask4 = morphological(mask1, 0)
    mask5 = morphological(mask2, 0)

    cv2.imshow("greeen", mask4)
    cv2.imshow("red", mask5)

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

    # print("Green coins detected :",len(green_coins))

    # drawing red coins and getting their centers.
    red_coin_centers = draw_coins(red_coins)
    
    # drawing green coins and getting their centers.
    green_coin_centers = draw_coins(green_coins)

    # print("Number of green coins ... :",len(green_coin_centers))


    #angle = calculate_angle(red_coin_centers[0], [x2, y2], green_coin_centers[0])
    #angle = "Angle : "+str(angle)

    # writing angle in the image.
    # cv2.putText(img_o, angle, (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # displaying op_image.
    cv2.imshow("op_image", img_o)
 
    op_image = img_o

    # numbering of cities

    # hiding the aruco so that it don't interfere in city detection...
    aruco_hided = hide_aruco(id_list, mask)
    cv2.imshow("Aruco hided",aruco_hided)
    dilation = cv2.dilate(mask, kernel ,iterations = 2)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("dilation", dilation)

    # edges = cv2.Canny(dilation,100,200)
    # cv2.imshow("edges", edges)

    key = list(id_list.keys())[0]
    point1 = (id_list[key])[0]
    point2 = (id_list[key])[1]
    point3 = (id_list[key])[2]
    point4 = (id_list[key])[3]

    # getting the center of aruco...
    center_x = int((point1[0]+point2[0]+point3[0]+point4[0])/4)
    center_y = int((point1[1]+point2[1]+point3[1]+point4[1])/4)

    # mid point of first and second point of aruco
    orient_centre_x = int((point1[0]+point2[0])/2)
    orient_centre_y = int((point1[1]+point2[1])/2)
    orient_center = [orient_centre_x, orient_centre_y]

    
    # isolating the highway                    
    Highway = Highway_isolation(contours, center_x, center_y)

    dummy1 = np.zeros(img.shape[0:2], np.uint8)
    cv2.drawContours(dummy1, [Highway], 0, (255,255,255), -1)

    cv2.imshow("Highway", dummy1)
    hull = cv2.convexHull(Highway,returnPoints = True)
    cv2.drawContours(dummy1, [hull], 0, (255,255,255), 3)

    cv2.imshow("Convexity defects", dummy1)

    # Isolating cities...
    # The logic is simple...since the cities have more area when compared to highway, continous erosion
    # (contd...) until the number of contours gets to 9
    # since cities are more dense that remaining contours will surely be a city.

    erosion = cv2.erode(dummy1, kernel ,iterations = 2)

    erosion1 = erosion
    kernel = np.ones((2,2),np.uint8)

    while 1:
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode(erosion, kernel ,iterations = 1)
        cn, h = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cn) ==8 or len(cn) == 9:
            erosion = cv2.dilate(erosion, kernel, iterations=2)
            city_flag = 1
            break
        elif len(cn) == 0 :
            # print("No cities found")
            break

    cv2.imshow("erosion", erosion)

     #if len(cn) == 8:
       # ret_flag = 1
    coord = clockwise_numbering(cn, center_x, center_y)

    count = 1
    for i in coord :
        cv2.putText(cpy, str(count), (i[0], i[1]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        count = count + 1

    cv2.circle(cpy, (center_x, center_y), 1,(255,0,0), 3)    

    # green_coin_centers = []

    # actual
    loc, output = coin_locations(coord, red_coin_centers, green_coin_centers)

    '''
    print("Number of green coins :",len(green_coin_centers))
    print("Number of red coins :",len(red_coin_centers))

    print("Green coins :", green_coin_centers)
    print("Red coins :",red_coin_centers)
    '''

    # loc, output = coin_locations(coord, green_coin_centers, red_coin_centers)

    print("Locations are :",loc)
    print("Result :\n",output)

    # angle1 = calculate_angle(red_coin_centers[0], [x2, y2], [center_x, center_y])
    # angle1 = "Angle : "+str(angle1)
    

    # writing angle in the image.
    # cv2.putText(cpy, angle1, (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Cities", cpy)
    # print("location of red coin is :",loc)

    # Actual
    path, direction = path_planning([center_x, center_y], [x2, y2], red_coin_centers, green_coin_centers, loc)

    # path, direction = path_planning([center_x, center_y], [x2, y2], green_coin_centers, red_coin_centers, loc)


    print("Planned path is :", path)
    print("Planned direction is :",direction)

    c_path = []

    if direction[0] == 'a':
        c_path.append(9-path[0]+1)
    else:
        c_path.append(path[0] - 1)

    for i in range(1,4):
        if direction[i] == 'a':
            if path[i-1] > path[i] :
                c_path.append(path[i-1]-path[i])
            else:
                c_path.append((9-path[i])+path[i-1])
        else :
            if path[i-1] < path[i] :
                c_path.append(path[i]-path[i-1])
            else :
                c_path.append((9-path[i-1])+path[i])

    print("Final path is", c_path)
    flag2 = -1
    flag3 = -1
    val = -1
    # if ret_flag == 1 :

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
        
    # return coord, output, loc

    # else:
         #return flag2, flag3, output, val

    return c_path, direction, city_flag

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
    # print(frame.shape)

    path_f = []
    direct = []
    # capturing the frame which is stable
    while(ret):
        i = i + 1
        ret, frame = cap.read()
        ## display to see if the frame is correct
        cv2.imshow("window", frame)
        cv2.waitKey(int(1000/fps));
        ## calling the algorithm function
        # coordinates,output, loc = process(frame)
        path_f, direct, flag = process(frame)
        # if path_f != -1 and direct != -1 :
            # cv2.destroyAllWindows()
            # print(output)
            # break
        
        if i == 2000 or flag == 1:
            break

    cv2.waitKey(0)
    ser = serial.Serial("COM5", 9600, timeout=2.00)

    def send(data):
        flag = 0
        while(1):
            if flag == 1:
                break
            ser.write(data.encode('utf-8'))
            if (ser.isOpen()):
                while (1):
                    a = ser.read(1)
                    print(a)
                    if a == b'y' :
                        print("Request received")
                        flag = 1
                        break
                    
    for i in path_f :
        send(str(i))
    for i in direct :
        send(str(i))
        

    
############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main()
