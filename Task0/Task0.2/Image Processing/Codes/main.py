import cv2
import numpy as np
import os


images = []
os.chdir('../Images')
names = os.listdir()
# storing all the images in a list.
for i in names:
    images.append(cv2.imread(i, 1))
os.chdir('../Generated')

def partA():
    f = open('stats.csv', 'w')
    j = 0
    for i in images:
        # string to store row data of csv file.
        string = '{0},{1},{2},{3},{4},{5},{6}\n'.format(names[j], str(i.shape[0]), str(i.shape[1]), str(len(i.shape)),
                                                        str(i[(i.shape[0] // 2), (i.shape[1] // 2), 0]),
                                                        str(i[(i.shape[0] // 2), (i.shape[1] // 2), 1]),
                                                        str(i[(i.shape[0] // 2), (i.shape[1] // 2), 2]))
        j = j+1
        f.write(string)
    f.close()
    
def partB():
    img = images[1]
    # changing Blue channel to 0.
    img[:, :, 0] = 0
    # changing green channel to 0.
    img[:, :, 1] = 0
    cv2.imwrite('cat_red.jpg', img)    

def partC():
    img = images[2]
    img = img.astype(float)
    img /= 255.0
    # creating dummy array with 4 channels.
    new_img = np.zeros((img.shape[0], img.shape[1], 4), dtype = np.float32)
    new_img[:, :, 0] = img[:, :, 0]
    new_img[:, :, 1] = img[:, :, 1]
    new_img[:, :, 2] = img[:, :, 2]
    # setting  alpha channel value to 0.5.
    new_img[:, :, 3] = 0.5
    new_img *= 255.0
    cv2.imwrite('flowers_alpha.png', new_img)
    
def partD():
    img = images[3]
    # creating dummy array to store the gray scale image.
    new_img = np.zeros((img.shape[0], img.shape[1]), dtype = np.float32)
    # slicing numpy array and applying the given formula to obtain gray scale image.
    new_img[:, :] = (0.11*img[:, :, 0])+(0.59*img[:, :, 1])+(0.3*img[:, :, 2])
    cv2.imwrite('horse_gray.jpg', new_img)
    
partA()
partB()
partC()
partD()
