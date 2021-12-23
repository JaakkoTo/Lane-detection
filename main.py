import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Packages imported")


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters # get average slope and y-intercept
    # define the height of the lines
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    # calculating the x coordinates
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    # outputs start and endpoint coordinates
    return np.array([x1, y1, x2, y2])


# def average_slope_intercept(image, lines):
#     left = []
#     right = []
#     for line in lines:
#         print(line)
#         x1, y1, x2, y2 = line.reshape(4)
#         parameters = np.polyfit((x1, x2), (y1, y2), 1)
#         slope = parameters[0]
#         y_int = parameters[1]
#         if slope < 0:
#             left.append((slope, y_int))
#         else:
#             right.append((slope, y_int))
#
#     right_avg = np.average(right, axis=0)
#     left_avg = np.average(left, axis=0)
#     left_line = make_coordinates(image, left_avg)
#     right_line = make_coordinates(image, right_avg)
#     return np.array([left_line, right_line])


global_left_fit_average = []
global_right_fit_average = []


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    global global_left_fit_average
    global global_right_fit_average

    if lines is not None:
        for line in lines: # loop through all the lines
            x1, y1, x2, y2 = line.reshape(4)  # Extract x,y values from each line
            # Determine the slope and y-intercept of each line segment
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            # negative slopes to left lines and positive slopes to right lines
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # In some frames all the slopes are > 0 hence left_fit list is empty resulting in error.
    # Solve this by using left fit average from previous frame
    if len(left_fit) == 0:
        left_fit_average = global_left_fit_average
    else:
        left_fit_average = np.average(left_fit, axis=0)
        global_left_fit_average = left_fit_average

    right_fit_average = np.average(right_fit, axis=0)
    global_right_fit_average = right_fit_average
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # converting rgb image to gray image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # applying gaussian blur to reduce noise on the grayscale image
    canny = cv2.Canny(blur, 50, 150)  # computes the gradient and recognizes the strongest gradient
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image) # create blacked out image
    if lines is not None: # making sure the lists line points aren't empty
        for x1, y1, x2, y2 in lines: # loop through all the lines
            # extract 2 pairs of coordinates, color blue, thickness 10
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            # output the lines
        return line_image


def region_of_interest(image):  # defining a triangle like region where we are finding the lanes
    height = image.shape[0]
    polygons = np.array([
        [(150, height), (1150, height), (550, 250)]
    ])  # matrix of the desired polygon area
    mask = np.zeros_like(image) # black image
    cv2.fillPoly(mask, polygons, 255) # combining the white polygon in the region of interest with the black picture
    masked_image = cv2.bitwise_and(image, mask) # isolating edges with the lane lines using binary numbers
    return masked_image

    # image = cv2.imread("Resources/test_image.jpg")  # read the image from path
    # lane_image = np.copy(image)
    # canny_image = canny(lane_image)
    # cropped_image = region_of_interest(canny_image)
    # lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # averaged_lines = average_slope_increase(lane_image, lines)
    # line_image = display_lines(lane_image, averaged_lines)
    # blend_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    # #Using hough transform to detect lines
    # cv2.imshow("Output", blend_image)  # display the image
    # cv2.waitKey(0)


cap = cv2.VideoCapture("Resources/test_video.mp4") # path to video
while cap.isOpened(): # making the v ideo run
    _, frame = cap.read() # read video capture
    canny_image = canny(frame) # display canny image
    cropped_image = region_of_interest(canny_image) # display region of interest from canny image
    # apply hough transformation
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines) # average out the lines to make one line
    line_image = display_lines(frame, averaged_lines) # display the lines
    blend_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # make the image more clcear
    cv2.imshow("Output", blend_image)  # display the image
    if cv2.waitKey(1) == ord('q'): # stop video with key 'q'
        break
cap.release()
cv2.destroyAllWindows()
