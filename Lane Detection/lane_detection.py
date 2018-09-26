import cv2
import numpy as np


def canny(image):
    # Grayscale the line_image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur the image
    blur = cv2.GaussianBlur(gray_image, (5,5), 0)
    # find the canny edge
    canny = cv2.Canny(blur, 50, 150)
    return canny

# cut the area needed to detect and then fill the area on an mask image
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    # Create a black mask image
    mask = np.zeros_like(image)
    # Fill the triangle region on the mask image
    cv2.fillPoly(mask, polygons, 255)
    # Combine the mask and canny by the binary value
    combine = cv2.bitwise_and(image, mask)
    return combine

# display lines in blank image
def display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # reshape each line to one dimensional array with 4 elements
            x1, y1, x2, y2 = line.reshape(4)
            # draw each line to blank image
            cv2.line(line_image, (x1,y1), (x2,y2), (0, 255, 0), 8)
    return line_image

def coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        #print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    #print(left_fit)
    #print(right_fit)
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    #print(left_fit_average, 'left')
    #print(right_fit_average, 'right')
    left_line = coordinates(image, left_fit_average)
    right_line = coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

#image = cv2.imread('test_image.jpg')
#lane_image = np.copy(image)
#canny_edge = canny(lane_image)
# combine the blank image with cropped area
#combined_image = region_of_interest(canny_edge)
# HoughLineP Function
#lines = cv2.HoughLinesP(combined_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#average_lines = average_slope_intercept(lane_image, lines)
#line_image = display_line(lane_image, average_lines)
# combine the line_image with the real image
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

#cv2.imshow('Lane Detection', combo_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cap = cv2.VideoCapture('test_video.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    canny_edge = canny(frame)
    # combine the blank image with cropped area
    combined_image = region_of_interest(canny_edge)
    # HoughLineP Function
    lines = cv2.HoughLinesP(combined_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_lines = average_slope_intercept(frame, lines)
    line_image = display_line(frame, average_lines)
    # combine the line_image with the real image
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow('Lane Detection', combo_image)
    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
