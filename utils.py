import cv2 as cv2
import numpy as np
import sys, csv, os, ast
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
        
        if len(approx) == 4 and area>224000:
            al.append(i)
            ap.append(approx)
    return al[:4],ap[:4]

def findtopRect(contr):
    al = []
    ap = []
    for i in contr:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area>3000 :
            al.append(i)
            ap.append(approx)
    return al[:2],ap[:2]

def roundClose(img, i, n, s):
    return abs(img[i].shape[s] - (img[i].shape[s] // n) * n) 

def split(img,num_sections):
    rows = np.vsplit(img,num_sections)
    return rows

def print_img(img):
    plt.imshow(img,cmap='gray')
    plt.show()

##
#add cropiing in code for both top and bottom if need 
##

##
## Split omr for multiple choise
##

def rowSplit(name):
    image = cv2.imread(name)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edged = cv2.Canny(blurred, 10,30)
    
    qr_codes = decode(image)
    #print(qr_codes)
    qr = qr_codes[0][0].decode('utf-8')
    

    countours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    countours = sorted(countours, key=lambda c: cv2.boundingRect(c)[0])

    #img_contr = image.copy()
    img_gray = edged.copy()
    #cv2.drawContours(img_contr,countours,-1,(0,255,0),5)
    cc,bb = findRect(countours)


    paper = []
    for i in bb:
        paper.append(four_point_transform(img_gray, i.reshape(4, 2)))

    thresh = []
    for i in paper:
        thresh.append(cv2.threshold(i, 50, 500,cv2.THRESH_BINARY_INV)[1])

    cropped_image = []
    cropped_image.append(thresh[0][50:, 83:])
    cropped_image.append(thresh[1][50:, 83:])
    cropped_image.append(thresh[2][50:, 83:])
    cropped_image.append(thresh[3][50:, 83:])

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
        #till this the logic is ok 
        #but here to select which option is 
        t1,t2 = 50,25
        if (k[3]-k[0]<t1 and k[3]-k[2]<t1):
            kk.append(-1)
        elif k[1]-k[0]<t2 and k[3]-k[0]>t1 :
            kk.append(-2)
        else:
            kk.append(ss.index(min(ss))+1)
        
        #plt.imshow(row[j],cmap='gray'),plt.title(j+1),plt.axis('off'),plt.text(0.5, 0.05, kk[-1], color='red', fontsize=20,ha='center', va='center', transform=plt.gca().transAxes),plt.show(),print(ss)
        
    return kk

##
## Split omr for top sheet
##

def findTop(name):
    image = cv2.imread(name)

    qr_codes = decode(image)
    #print(qr_codes)
    qr = qr_codes[0][0].decode('utf-8')

    image = image[400:-300,200:]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edged = cv2.Canny(blurred, 10,30)

    countours,_ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    countours = sorted(countours, key=lambda c: cv2.boundingRect(c)[0])

    contour_image = image.copy()

    cc,bb = findtopRect(countours)
    cv2.drawContours(contour_image,cc,-1,(0,255,0),10)

    x, y, w, h = cv2.boundingRect(bb[1])
    sett = edged[y:y+h, x:x+w].copy()
    sett = sett[137:,30:-30]
    height, width = sett.shape
    top_left,top_right,bottom_right,bottom_left = (0, 0),(width - 1, 0),(width - 1, height - 1),(0, height - 1)
    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    #corners = corners.reshape((-1, 1, 2))


    paper = []
    paper.append(four_point_transform(edged.copy(), bb[0].reshape(4, 2)))
    paper.append(four_point_transform(sett.copy(), corners.reshape(4, 2)))

    paper[0] = paper[0][125:,:]

    thresh = []
    for i in paper:
        thresh.append(cv2.threshold(i, 50, 500,cv2.THRESH_BINARY_INV)[1])

    h = []
    w = []
    h.append(roundClose(thresh, 0, 10, 0) )
    w.append(roundClose(thresh, 0, 9, 1) )

    h.append(roundClose(thresh, 1, 4, 0) )
    w.append(roundClose(thresh, 1, 1, 1) ) 

    for i in range(0,2):
        thresh[i] = thresh[i][h[i]:,w[i]:]

    split_row_roll = np.hsplit(thresh[0],9)

    roll = extractRollNumber(split_row_roll)
    ansSet = extractSetNumber(thresh[1])
    
    return qr, roll, ansSet

def extractRollNumber(row):
    roll = ""
    for i in row:
        k = np.vsplit(i,10)
        kk,ss = [],[]
        for j in k:
            center = (25, 23)  
            radius = 16           
            mask = j.copy()
            cv2.circle(mask, center, radius, (0, 255, 0), thickness=cv2.FILLED)
            roi = cv2.bitwise_and(j, cv2.bitwise_not(mask))
            roi = cv2.Canny(roi, 50, 150)
            ss.append(np.count_nonzero(roi))
            kk.append(ss.index(min(ss)))
        roll = roll + str(kk[-1])
    return roll

def extractSetNumber(row):
    k = np.vsplit(row,4)
    kk,ss = [],[]
    for j in k:
        center = (25, 23)  
        radius = 16           
        mask = j.copy()
        cv2.circle(mask, center, radius, (0, 255, 0), thickness=cv2.FILLED)
        roi = cv2.bitwise_and(j, cv2.bitwise_not(mask))
        roi = cv2.Canny(roi, 50, 150)
        ss.append(np.count_nonzero(roi))
        kk.append(ss.index(min(ss))+1)
    return kk[-1]




###
### write in csv and get data from csv
###
def omrTocsv(omr_files,csv_output):
    for name in omr_files:
        qr,row = rowSplit(name)
        ans = findAns(row)
        with open(csv_output, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([qr,ans])
            
def omrTopTocsv(omr_top_files, csv_setcode):
    for name in omr_top_files:
        qr, roll, s = findTop(name)
        with open(csv_setcode, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([qr,roll,s])
            
def getMarkList(csv_output):         
    with open(csv_output, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        data = [row for row in reader]
        return data
    
def getqrRollSet(csv_setcode):
    a = {}
    with open(csv_setcode, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            a[row[0]] = [row[1],row[2]]    
    return a

def getAnskey(csv_anskey):
    a = {}
    with open(csv_anskey, 'r', newline='') as key:
        reader = csv.reader(key)
        for row in reader:
            a[row[0]] = row[1]
    return a

def generateResultSheet(csv_mark, data, qrWithSet, qrSetandres):
    with open(csv_mark, 'w', newline='') as marks:
        csv_writer = csv.writer(marks)

        for row in data:
            mark = 0
            if row[0] in qrWithSet:
                anskey = qrWithSet[row[0]][1]
                l1 = ast.literal_eval(row[1])
                l2 = ast.literal_eval(qrSetandres[str(anskey)])

                for i in range(0,70):
                    if l1[i]==l2[i]:
                        mark = mark+1

                csv_writer.writerow([qrWithSet[row[0]][0],mark])     
