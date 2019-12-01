import numpy as np
import cv2
from matplotlib import pyplot as plt

# Import templates in grayscale
fourfour_template = cv2.imread('Templates/44.png', 0)
treble_template = cv2.imread('Templates/treble.png', 0)
qnote_template = cv2.imread('Templates/qnote.png', 0)
hnote_template = cv2.imread('Templates/hnote.png', 0)
qrest_template = cv2.imread('Templates/qrest.png', 0)
staff_template = cv2.imread('Templates/staff3.png', 0)

# RGB Colors
BLACK = (0, 0, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


def template_match(image_bin, image_rgb, template, threshold, rec_color):
    image_rect = image_rgb.copy()
    w_temp, h_temp = template.shape[::-1]  # Dimensions of template
    res = cv2.matchTemplate(image_bin, template, cv2.TM_CCOEFF_NORMED)  # Template match

    # Only keep locations >= threshold (symbol likely matched), loc_unf in the format: [yaxis_vector], [xaxis_vector]
    loc_unf = np.where(res >= threshold)

    # Remove duplicates by tracking the x-axis values
    loc = loc_unf
    index = loc_unf[1][0]  # Assume first element is unique
    counter = 0
    for x in loc_unf[1][1:]:
        if abs(x - index) > 5:  # Detect duplicate if it is within a certain range
            index = x  # Unique element detected
            counter = counter + 1
        else:  # Delete duplicate
            loc = np.delete(loc, counter + 1, 1)

    # Create rectangles in image to showcase symbols found
    for pt in zip(*loc[::-1]):  # Loop through each x,y point in locations matrix
        print("X: {0} Y:{1}".format(pt[0], pt[1]))  # Print location found
        cv2.rectangle(image_rect, pt, (pt[0] + w_temp, pt[1] + h_temp), rec_color, 1)  # Create rectangle around symbol

    return loc, image_rect


img_rgb = cv2.imread('Test1.png')  # Import test image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
ret, img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # Create binary image

img_rect = img_rgb.copy()
# Find all the symbols and create a rectangle around them
print("Finding quarter notes:")
qnotes_loc, img_rect = template_match(img, img_rect, qnote_template, 0.9, RED)

print("Finding half notes:")
hnotes_loc, img_rect = template_match(img, img_rect, hnote_template, 0.9, GREEN)

print("Finding quarter rests:")
qrest_loc, img_rect = template_match(img, img_rect, qrest_template, 0.9, BLUE)

print("Finding a 4 4 time signature:")
fourfour_loc, img_rect = template_match(img, img_rect, fourfour_template, 0.9, BLACK)

print("Finding a treble:")
treble_loc, img_rect = template_match(img, img_rect, treble_template, 0.9, BLACK)

print("Finding a staff:")
staff_loc, img_rect = template_match(img, img_rect, staff_template, 0.9, BLACK)

cv2.imwrite('symbols_found.png', img_rect)  # Keep file with rectangles

# Go through each matched feature and calculate the note value

print("")
print("Detecting notes")
staffy = staff_loc[0]  # Recall X and Y are flipped
print("staffy: {0}".format(staffy))


for ypt in qnotes_loc[0][:]:
    diff = ypt - staffy
    print("diff: {0}".format(diff))


def note_detector(v, ref):
    note = ''
    for x in v:
        if (((x[1] + 15) - ref) > 89) & (((x[1] + 15) - ref) < 123):
            note = 'f5'
            print(note)
        elif (((x[1] + 15) - ref) > 106) & (((x[1] + 15) - ref) < 140):
            note = 'e5'
            print(note)
        elif (((x[1] + 15) - ref) > 123) & (((x[1] + 15) - ref) < 157):
            note = 'd5'
            print(note)
        elif (((x[1] + 15) - ref) > 140) & (((x[1] + 15) - ref) < 174):
            note = 'c5'
            print(note)
        elif (((x[1] + 15) - ref) > 157) & (((x[1] + 15) - ref) < 191):
            note = 'b4'
            print(note)
        elif (((x[1] + 15) - ref) > 174) & (((x[1] + 15) - ref) < 208):
            note = 'a4'
            print(note)
        elif (((x[1] + 15) - ref) > 191) & (((x[1] + 15) - ref) < 225):
            note = 'g4'
            print(note)
        elif (((x[1] + 15) - ref) > 208) & (((x[1] + 15) - ref) < 242):
            note = 'f4'
            print(note)
        elif (((x[1] + 15) - ref) > 225) & (((x[1] + 15) - ref) < 259):
            note = 'e4'
            print(note)
