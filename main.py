import numpy as np
from PIL import ImageGrab
import cv2

def get_angle(contour, clock):
    [_, _, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    return calculate_angle(x - clock[0], clock[1] - y) % 360
def calculate_angle(vx, vy):
    if vx == 0 and vy > 0:
        return 90
    elif vx == 0 and vy < 0:
        return 270
    elif vx > 0 and vy > 0:
        return (np.arctan(vy/vx)/3.1415*180)[0]
    elif vx > 0 and vy < 0:
        return (np.arctan(vy/vx)/3.1415*180)[0]
    elif vx < 0 and vy < 0:
        return 180 + (np.arctan(vy/vx)/3.1415*180)[0]
    elif vx < 0 and vy > 0:
        return 180 + (np.arctan(vy/vx)/3.1415*180)[0]
    elif vy == 0 and vx > 0:
        return 0
    elif vy == 0 and vx < 0:
        return 180
def get_hour(angle):
    return int(((360-angle)//30+3)%12)
def get_minute(angle):
    return int(((360-angle)//6+15)%60)
def get_second(angle):
    return int(((360-angle)//6+15)%60)

if __name__ == '__main__':
    name = "Analog Clock"
    x_coord = 0
    y_coord = 0
    width = 1980
    height = 1050
    box = (x_coord, y_coord, x_coord + width, y_coord + height)
    second_color = [111, 127, 36]
    minute_color = [61, 96, 27]
    hour_color = [19, 68, 30]
    while True:
        # Grab a screenshot and put it into a numpy object
        img_raw = ImageGrab.grab(bbox=box)
        img = np.array(img_raw.getdata(), dtype='uint8')\
            .reshape((img_raw.size[1], img_raw.size[0], 3))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.1, 100)[0]
        circles = sorted(circles, key=lambda circ: circ[2], reverse=True)
        clock = circles[0]

        if not(clock[0] == 0) and not(clock[1] == 0):
            # Create the masks for each hand
            second_mask = cv2.inRange(img, np.array(second_color), np.array(second_color))
            minute_mask = cv2.inRange(img, np.array(minute_color), np.array(minute_color))
            hour_mask = cv2.inRange(img, np.array(hour_color), np.array(hour_color))

            # Mask the image so only each hand shows up
            second_img = cv2.bitwise_and(img, img, mask=second_mask)
            minute_img = cv2.bitwise_and(img, img, mask=minute_mask)
            hour_img = cv2.bitwise_and(img, img, mask=hour_mask)

            # Run Canny edge detection
            second_edges = cv2.Canny(second_img, 100, 200)
            minute_edges = cv2.Canny(minute_img, 100, 200)
            hour_edges = cv2.Canny(hour_img, 100, 200)

            # Capture features of each image (for the hands of the clock)
            second_contours, _ = cv2.findContours(second_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            minute_contours, _ = cv2.findContours(minute_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hour_contours, _ = cv2.findContours(hour_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Sort the contours in terms of area to get the respective hands
            second_hand = sorted(second_contours, key=cv2.contourArea, reverse=True)
            minute_hand = sorted(minute_contours, key=cv2.contourArea, reverse=True)
            hour_hand = sorted(hour_contours, key=cv2.contourArea, reverse=True)

            if len(second_hand) > 0 and len(minute_hand) > 0 and len(hour_hand) > 0:
                # Calculate the angles of each of the hands
                second_angle = get_angle(second_hand[0], clock)
                minute_angle = get_angle(minute_hand[0], clock)
                hour_angle = get_angle(hour_hand[0], clock)

                # Display the time
                print("{0}:{1}:{2}".format(get_hour(hour_angle), get_minute(minute_angle), get_second(second_angle)))
