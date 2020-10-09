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
import cv2.aruco as aruco
from aruco_lib import *
import copy




########################################################################
## using os to generalise Input-Output
########################################################################
codes_folder_path = os.path.abspath('.')
images_folder_path = os.path.abspath(os.path.join('..', 'Videos'))
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
    ###########################
    id_list = []
    #contrast improvement
    enhanced_bright = np.zeros(ip_image.shape, ip_image.dtype)
    enhanced_bright[:, :, :] = (1.5*ip_image[:, :, :]) + 40

    b_g_r = []
    k = 1.5e-2
    # splitting the channels
    b_g_r = cv2.split(enhanced_bright)
    # kernel design
    impulse_response = np.zeros((20, 20))
    impulse_response[:, (impulse_response.shape[1]-1) // 2] = 1
    impulse_response = impulse_response / np.sum(impulse_response)
    deblurred_channels = []
    for i in range(3):
        temp = np.copy(b_g_r[i])
        img_transform = np.fft.fft2(temp)
        impulse_transform = np.fft.fft2(impulse_response, s = b_g_r[i].shape)
        deconv = np.conj(impulse_transform)/ ((np.abs(impulse_transform))**2 + k)
        img_estimate = deconv * img_transform
        # estimate of original image
        img_deblurred = np.abs(np.fft.ifft2(img_estimate))
        img_deblurred = img_deblurred.astype('uint8')
        deblurred_channels.append(img_deblurred)

    # merging the channels
    ip_image = cv2.merge((deblurred_channels[0], deblurred_channels[1], deblurred_channels[2]))
    # detecting aruco
    id_list = detect_Aruco(ip_image)
    # marking aruco 
    ip_image = mark_Aruco(ip_image, id_list)
    # calculating robot state to get id and the angle
    id_list = calculate_Robot_State(ip_image, id_list)
    ids = list(id_list.keys())
    id_list = id_list[ids[0]]
    os.chdir(generated_folder_path)
    cv2.imwrite("aruco_with_id.png", ip_image)

    return ip_image, id_list


    
####################################################################
## The main program which provides read in input of one image at a
## time to process function in which you will code your generalized
## output computing code
## Do not modify this code!!!
####################################################################
def main(val):
    ################################################################
    ## variable declarations
    ################################################################
    i = 1
    ## reading in video 
    cap = cv2.VideoCapture(images_folder_path+"/"+"aruco_bot.mp4")
    ## getting the frames per second value of input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    ## getting the frame sequence
    frame_seq = int(val)*fps
    ## setting the video counter to frame sequence
    cap.set(1, frame_seq)
    ## reading in the frame
    ret, frame = cap.read()
    ## verifying frame has content
    print(frame.shape)
    ## display to see if the frame is correct
    cv2.imshow("window", frame)
    cv2.waitKey(0);
    ## calling the algorithm function
    op_image, aruco_info = process(frame)
    ## saving the output in  a list variable
    line = [str(i), "Aruco_bot.jpg", str(aruco_info[0]), str(aruco_info[3])]
    ## incrementing counter variable
    i+=1
    ## verifying all data
    print(line)
    ## writing to angles.csv in Generated folder without spaces
    with open(generated_folder_path+"/"+'output.csv', 'w') as writeFile:
        print("About to write csv")
        writer = csv.writer(writeFile)
        writer.writerow(line)
    ## closing csv file    
    writeFile.close()




    

############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main(input("time value in seconds:"))
