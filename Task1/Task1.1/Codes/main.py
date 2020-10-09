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




########################################################################
## using os to generalise Input-Output
########################################################################
codes_folder_path = os.path.abspath('.')
images_folder_path = os.path.abspath(os.path.join('..', 'Images'))
generated_folder_path = os.path.abspath(os.path.join('..', 'Generated'))




############################################
## Build your algorithm in this function
## ip_image: is the array of the input image
## imshow helps you view that you have loaded
## the corresponding image
############################################
def process(ip_image):
    ###########################
    ## Your Code goes here
    angle = 0.00
    ## Your Code goes here
    ###########################
    cv2.imshow("window", ip_image)
    cv2.waitKey(0);
    cv2.destroyAllWindows()

    hsv = cv2.cvtColor(ip_image,cv2.COLOR_BGR2HSV)
    # upper and lower hsv values for green and red
    green_low = np.array([50,100,100])
    green_high = np.array([70,255,255])
    red_low = np.array([0,255,255])
    red_high = np.array([10,255,255])
    
    gray = cv2.cvtColor(ip_image, cv2.COLOR_BGR2GRAY)
    # mask for green and red
    mask1 = cv2.inRange(hsv, green_low, green_high)
    mask2 = cv2.inRange(hsv, red_low, red_high)

    # calculating moments for mask1
    moment1 = cv2.moments(mask1)
    x1 = int(moment1["m10"] / moment1["m00"])
    y1 = int(moment1["m01"] / moment1["m00"])

    # calculating moments for mask2
    moment3 = cv2.moments(mask2)
    x3 = int(moment3["m10"] / moment3["m00"])
    y3 = int(moment3["m01"] / moment3["m00"])

    # finding center
    ret,thresh = cv2.threshold(gray, 127,255, 0)
    moment2 = cv2.moments(thresh)
    x2 = int(moment2["m10"] / moment2["m00"])
    y2 = int(moment2["m01"] / moment2["m00"])

    # finding angle
    coordinate1 = np.array([x1,y1])
    coordinate2 = np.array([x2,y2])
    coordinate3 = np.array([x3,y3])
    vector1 = coordinate1 - coordinate2
    vector2 = coordinate3 - coordinate2
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    angle = round(float(angle), 2)
    
    return angle




    
####################################################################
## The main program which provides read in input of one image at a
## time to process function in which you will code your generalized
## output computing code
## Do not modify this code!!!
####################################################################
def main():
    ################################################################
    ## variable declarations
    ################################################################
    i = 1
    line = []
    ## Reading 1 image at a time from the Images folder
    for image_name in os.listdir(images_folder_path):
        ## verifying name of image
        print(image_name)
        ## reading in image 
        ip_image = cv2.imread(images_folder_path+"/"+image_name)
        ## verifying image has content
        print(ip_image.shape)
        ## passing read in image to process function
        A = process(ip_image)
        ## saving the output in  a list variable
        line.append([str(i), image_name , str(A)])
        ## incrementing counter variable
        i+=1
    ## verifying all data
    print(line)
    ## writing to angles.csv in Generated folder without spaces
    with open(generated_folder_path+"/"+'angles.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(line)
    ## closing csv file    
    writeFile.close()



    

############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main()
