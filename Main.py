import cv2
import numpy as np
import utlis  

webCamFeed = True  # Use webcam feed (True) or an image file (False)
pathImage = "1.jpg"  # Path to the image file if not using webcam
cap = cv2.VideoCapture(0)  # Capturing video from the default camera (0)
cap.set(10, 160)  # Setting property 10 (brightness) to a value of 160
heightImg = 640  # Setting the height of the image
widthImg = 480  # Setting the width of the image

# Initializing trackbars
utlis.initializeTrackbars()

count = 0  # Initializing a counter variable

# Looping indefinitely
while True:
    # Capturing an image either from webcam or file
    if webCamFeed:
        success, img = cap.read()  # Capture a frame from the webcam
    else:
        img = cv2.imread(pathImage)  # Read an image from a file

    img = cv2.resize(img, (widthImg, heightImg))  # Resize the image to specified dimensions

    # Creating a blank image
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    # Converting the image to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian blur to the grayscale image
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    # Getting threshold values from trackbars
    thres = utlis.valTrackbars()

    # Applying Canny edge detection
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])

    # Creating a kernel for morphological operations
    kernel = np.ones((5, 5))

    # Dilating the thresholded image
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)

    # Eroding the dilated image
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # Creating a copy of the image for drawing contours
    imgContours = img.copy()
    imgBigContour = img.copy()

    # Finding contours in the thresholded image
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Drawing all contours on the image
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # Finding the biggest contour and its area
    biggest, maxArea = utlis.biggestContour(contours)

    if biggest.size != 0:
        biggest = utlis.reorder(biggest)

        # Drawing the biggest contour on a separate image
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)

        # Performing perspective transformation
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # Cropping the transformed image
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # Converting the warped image to grayscale
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

        # Applying adaptive thresholding
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)

        # Applying median blur
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # Organizing images in an array for display
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

    else:
        # If no biggest contour found, display blank images
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # Labeling the images for display
    lables = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]]

    # Stacking images for display
    stackedImage = utlis.stackImages(imageArray, 0.75, lables)

    # Displaying the result
    cv2.imshow("Result", stackedImage)

    # Saving the scanned image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved",
                    (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1  # Incrementing the counter
