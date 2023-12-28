import cv2 as cv2
import numpy as np
import sys
from IPython.display import display, Image
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from pyzbar.pyzbar import decode

def findRect(contr):
    al = []
    ap = []
    for i in contr:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        
        if len(approx) == 4 and area>224500.0:
            al.append(i)
            ap.append(approx)
    return al[:4],ap[:4]
def roundClose(img, i, n, s):
    return abs(img[i].shape[s] - (img[i].shape[s] // n) * n) 

def split(img,num_sections):
    rows = np.vsplit(img,num_sections)
    return rows

def print_img(img):
    plt.imshow(img,cmap='gray')
    plt.show()

def rowSplit(name):
    image = cv2.imread(name)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edged = cv2.Canny(blurred, 10,30)
    
    qr_codes = decode(gray)
    qr = qr_codes[0][0].decode('utf-8')
    

    countours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    countours = sorted(countours, key=lambda c: cv2.boundingRect(c)[0])

    img_contr = image.copy()
    img_gray = edged.copy()
    cv2.drawContours(img_contr,countours,-1,(0,255,0),5)

    cc,bb = findRect(countours)

    img_contr = image.copy()
    cv2.drawContours(img_contr,cc,-1,(0,255,0),10)

    paper = []
    for i in bb:
        paper.append(four_point_transform(img_gray, i.reshape(4, 2)))

    thresh = []
    for i in paper:
        thresh.append(cv2.threshold(i, 50, 500,cv2.THRESH_BINARY_INV)[1])

    cropped_image = []
    cropped_image.append(thresh[0][60:, 83:])
    cropped_image.append(thresh[1][60:, 83:])
    cropped_image.append(thresh[2][60:, 83:])
    cropped_image.append(thresh[3][60:, 83:])

    h = []
    w = []
    for i in range(0,3):
        h.append(roundClose(cropped_image, i, 18, 0) )
        w.append(roundClose(cropped_image, i, 4, 1) )
    h.append(roundClose(cropped_image, 3, 16, 0) )
    w.append(roundClose(cropped_image, 3, 4, 1) ) 

    for i in range(0,4):
        cropped_image[i] = cropped_image[i][h[i]:,w[i]:]
        cropped_image[i][:, -6:] = 255

    split_row = []
    k = 0
    for i in cropped_image:
        if k==3:
            row = np.vsplit(i,16)
            split_row.append(row)
        else:
            k = k+1
            row = np.vsplit(i,18)
            split_row.append(row)

    sorted_split_row = []
    for i in range(0,4):
        for j in split_row[i]: 
            sorted_split_row.append(j)
    return qr,sorted_split_row

def findAns(row):
    kk = []
    for j in range(0,70):
        col = np.hsplit(row[j],4)
        ss = []
        for i in col:
            center = (25, 23)  
            radius = 16           
            mask = i.copy()
            cv2.circle(mask, center, radius, (0, 255, 0), thickness=cv2.FILLED)
            roi = cv2.bitwise_and(i, cv2.bitwise_not(mask))
            roi = cv2.Canny(roi, 50, 150)
            ss.append(np.count_nonzero(roi))
        k = sorted(ss)
        if k[3]-k[0]<50 and k[3]-k[2]<50:
            kk.append(-1)
        elif k[1]-k[0]<50 and k[2]-k[1]>50:
            kk.append(-2)
        else:
            kk.append(ss.index(min(ss))+1)

    return kk
