from imutils.video import VideoStream
from pyzbar import pyzbar
import argparse
import datetime
import imutils
import time
import numpy as np
import cv2


def contrast_stretch(im):

    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out


def NDVI_Calc():

    camera = True

    labelsPath = "./mask-rcnn-coco/object_detection_classes_coco.txt" 
    LABELS = open(labelsPath).read().strip().split("\n")

    colorsPath = "./mask-rcnn-coco/colors.txt" 
    COLORS = open(colorsPath).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

    weightsPath = "./mask-rcnn-coco/frozen_inference_graph.pb" 
    configPath  = "./mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt" 

    threshold = 0.5
    confidence_thr = 0.57

    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    if camera:

        print("[INFO] starting video stream...")
        vs = VideoStream(src="nvarguscamerasrc wbmode = 1 ! video/x-raw(memory:NVMM), " \
            "width=(int)1920, height=(int)1080,format=(string)NV12, " \
            "framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, " \
            "format=(string)BGRx ! videoconvert ! video/x-raw, " \
            "format=(string)BGR ! appsink").start()
            #"nvcamerasrc fpsRange=(string)15 15, auto-exposure=1, exposure-time=.03, wbmode=9, wbManualMode=3").start()
        time.sleep(3.0)

        csv = open("Codes.txt", "w")
        found = set()

        while True:

            frame = vs.read()
            frame = imutils.resize(frame, width=500)
            # find the barcodes in the frame and decode each of the barcodes
            barcodes = pyzbar.decode(frame)
            
            for barcode in barcodes:

                (x, y, w, h) = barcode.rect 
                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type

                if barcodeData not in found:
                    csv.write("{},{}\n".format(datetime.datetime.now(),
                        barcodeData))
                    csv.flush()
                    found.add(barcodeData)            

            key = cv2.waitKey(1) & 0xFF            
            if len(barcodes):
                cv2.imwrite('./source.png',frame)
                break

            if key == ord("q"):
                break

        
        print("[INFO] cleaning up...")
        csv.close()
        time.sleep(2.0)
        cv2.destroyAllWindows()
        vs.stream.release()
        vs.stop()

        clone = cv2.imread('./source.png')
        image = clone.copy()

        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
        end = time.time()

        print(end - start)

        plant_masks = []
        vase_masks = []

        vase_or = np.zeros_like(clone)
        plant_or = np.zeros_like(clone)

        # loop over the number of detected objects
        for i in range(0, boxes.shape[2]):
        
            classID = int(boxes[0, 0, i, 1])    
            confidence = boxes[0, 0, i, 2]

            #Only Potted Plants 
            if classID == 63:
                if confidence > 0:       
                    box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    boxW = endX - startX
                    boxH = endY - startY
                    
                    mask = masks[i, classID]

                    mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_NEAREST)
                    mask = (mask > threshold)

                    roi = clone[startY:endY, startX:endX]

                    visMask = (mask * 255).astype("uint8")
                    instance = cv2.bitwise_and(roi, roi, mask=visMask)

                    roi = roi[mask]
                    clone[startY:endY, startX:endX][mask] = roi

                    blank_plant = np.zeros_like(clone)
                    blank_plant[startY:endY, startX:endX][mask] = roi

                    plant_masks.append(blank_plant)

                    plant_or = cv2.bitwise_or(plant_or,blank_plant)
                    #cv2.imwrite("plant.jpg", plant_or) 

            #Only vase
            if classID == 85:
                if confidence > 0:       
                    box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    boxW = endX - startX
                    boxH = endY - startY
                    
                    mask = masks[i, classID]

                    mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_NEAREST)
                    mask = (mask > threshold)

                    roi = clone[startY:endY, startX:endX]

                    visMask = (mask * 255).astype("uint8")
                    instance = cv2.bitwise_and(roi, roi, mask=visMask)

                    roi = (roi[mask]).astype("uint8")            
                    clone[startY:endY, startX:endX][mask] = roi

                    blank_vase = np.zeros_like(clone)
                    blank_vase[startY:endY, startX:endX][mask] = roi

                    vase_masks.append(blank_vase)

                    vase_or = cv2.bitwise_or(vase_or,blank_vase)
                    #cv2.imwrite("vase.jpg", vase_or) 
            

        img_bwx = cv2.bitwise_xor(plant_or,vase_or)

        hsv = cv2.cvtColor(img_bwx, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (145, 25, 25), (162, 255,255))

        ## slice the green
        imask = mask > 0
        green = np.zeros_like(img_bwx, np.uint8)
        green[imask] = img_bwx[imask]

        # Get the individual colour components of the image    
        b, g, r = cv2.split(image)
        # Calculate the NDVI

        bottom = (r.astype(float) + b.astype(float))
        bottom[bottom == 0] = 0.01  # Make sure we don't divide by zero!

        ndvi = (r.astype(float) - b) / bottom             
        ndvi = contrast_stretch(ndvi) 
        ndvi = ndvi.astype(np.uint8)    

        ndvi = cv2.applyColorMap(ndvi, cv2.COLORMAP_JET)        

        final = np.zeros_like(ndvi, np.uint8)
        final[imask] = ndvi[imask]

        inter = final/255
        index = str(round(np.average((inter[inter > 0.2])*2 - 1),2))

        out = {barcodeData: np.mean(index)}

    return out




