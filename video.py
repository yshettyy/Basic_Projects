import cv2
import numpy as np
def videorecord():
    #Capture the video frame
    cap = cv2.VideoCapture(0)
    #Fourcc code to write the output
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
    #Converting to 1080p resolution
    def make_1080p():
        cap.set(3, 1280)
        cap.set(4, 720)

    #Converting to 480p resolution
    def make_480p():
        cap.set(3, 640)
        cap.set(4, 480)

    def change_res(width, height):
        cap.set(3, width)
        cap.set(4, height)

    def nothing(x):
        pass

    #Trackbar to determine the HSV value
    #cv2.namedWindow("Tracking")
    #cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
    #cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
    #cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
    #cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
    #cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
    #cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

    #to rescale the frame
    def rescale_frame(frame, percent=75):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            #cv2.imshow("Original", frame) to see the original frame
            blur_frame=cv2.GaussianBlur(frame,(7,7),0)  #to blur the image
            #cv2.imshow("Blur frame", blur_frame)
            hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV) #Converting to HSV model
            #To determine the HSV values
            #l_h = cv2.getTrackbarPos("LH", "Tracking")
            #l_s = cv2.getTrackbarPos("LS", "Tracking")
            #l_v = cv2.getTrackbarPos("LV", "Tracking")
            #u_h = cv2.getTrackbarPos("UH", "Tracking")
            #u_s = cv2.getTrackbarPos("US", "Tracking")
            #u_v = cv2.getTrackbarPos("UV", "Tracking")

            kernel = np.ones((7, 7), np.uint8)

            #HSV value for red
            lower_red=np.array([136,87,111],np.uint8)
            upper_red=np.array([180,255,255],np.uint8)
            mask_red=cv2.inRange(hsv,lower_red,upper_red)
            rescale_mask_red=rescale_frame(mask_red,percent=30)
            #cv2.imshow("Mask red",rescale_mask_red)

            #HSV value for orange
            lower_orange = np.array([10, 100, 20],np.uint8)
            upper_orange = np.array([25, 255, 255],np.uint8)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            rescale_mask_orange = rescale_frame(mask_orange,percent=30)
            #cv2.imshow("Mask orange",rescale_mask_orange)

            #HSV value for blue
            lower_blue = np.array([99,115,150],np.uint8)
            upper_blue = np.array([110,255,255],np.uint8)
            mask_blue=cv2.inRange(hsv, lower_blue, upper_blue)
            rescale_mask_blue = rescale_frame(mask_blue,percent=30)
            #cv2.imshow("Mask blue",rescale_mask_blue)

            #HSV value for yellow
            lower_yellow=np.array([11,109,0],np.uint8)
            upper_yellow=np.array([60,255,255],np.uint8)
            mask_yellow=cv2.inRange(hsv,lower_yellow,upper_yellow)
            rescale_mask_yellow=rescale_frame(mask_yellow,percent=30)
            #cv2.imshow("Mask yellow",rescale_mask_yellow)

            #HSV value for green
            lower_green=np.array([33,80,40],np.uint8)
            upper_green=np.array([102,255,255],np.uint8)
            mask_green=cv2.inRange(hsv,lower_green,upper_green)
            rescale_mask_green=rescale_frame(mask_green,percent=30)
            #cv2.imshow("Mask green",rescale_mask_green)

            #To remove unnecessary noise
            mask_red_opening = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            mask_red_closing=cv2.morphologyEx(mask_red_opening, cv2.MORPH_CLOSE, kernel)
            res_red = cv2.bitwise_and(frame, frame, mask= mask_red_closing)
            res_frame_red=rescale_frame(res_red, percent=30)
            #cv2.imshow("red",res_frame_red)

            mask_orange_opening = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
            mask_orange_closing=cv2.morphologyEx(mask_orange_opening, cv2.MORPH_CLOSE, kernel)
            res_orange = cv2.bitwise_and(frame, frame, mask=mask_orange_closing)
            res_frame_orange=rescale_frame(res_orange, percent=30)
            #cv2.imshow("orange",res_frame_orange)

            mask_blue_opening = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
            mask_blue_closing=cv2.morphologyEx(mask_blue_opening, cv2.MORPH_CLOSE, kernel)
            res_blue = cv2.bitwise_and(frame, frame, mask=mask_blue_closing)
            res_frame_blue=rescale_frame(res_blue, percent=30)
            #cv2.imshow("blue",res_frame_blue)

            mask_yellow_opening = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
            mask_yellow_closing=cv2.morphologyEx(mask_yellow_opening, cv2.MORPH_CLOSE, kernel)
            res_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow_closing)
            res_frame_yellow=rescale_frame(res_yellow, percent=30)
            #cv2.imshow("yellow",res_frame_yellow)

            mask_green_opening = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
            mask_green_closing=cv2.morphologyEx(mask_green_opening, cv2.MORPH_CLOSE, kernel)
            res_green = cv2.bitwise_and(frame, frame, mask=mask_green_closing)
            res_frame_green=rescale_frame(res_green, percent=30)
            #cv2.imshow("green",res_frame_green)



            #Tracking the Red Color
            (contours,hierarchy)=cv2.findContours( mask_red_closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>2000):
                    x,y,w,h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                    cv2.putText(frame,"Red color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))

            #Tracking the Blue Color
            (contours,hierarchy)=cv2.findContours(mask_blue_closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>2000):
                    x,y,w,h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.putText(frame,"Blue color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))

            #Tracking the Orange Color
            (contours,hierarchy)=cv2.findContours(mask_orange_closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>2000):
                    x,y,w,h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,165,255),2)
                    cv2.putText(frame,"Orange color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255))

            #Tracking the Green Color
            (contours,hierarchy)=cv2.findContours(mask_green_closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>2000):
                    x,y,w,h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(frame,"Green color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))

            #Tracking the Yellow Color
            (contours,hierarchy)=cv2.findContours(mask_yellow_closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>2000):
                    x,y,w,h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                    cv2.putText(frame,"Yellow color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255))


            cv2.imshow("Color Tracking",frame)  #Show the frames
            #Write the image of the detected objects
            cv2.imwrite("image_red.png",res_red)
            cv2.imwrite("image_orange.png",res_orange)
            cv2.imwrite("image_blue.png",res_blue)
            cv2.imwrite("image_yellow.png",res_yellow)
            cv2.imwrite("image_green.png",res_green)
            out.write(frame)    #Write the frames

            if cv2.waitKey(1) & 0xFF == ord('q'):  #key is used to interrupt the program
                break
        else:
            break

    #to release the objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    videorecord()
