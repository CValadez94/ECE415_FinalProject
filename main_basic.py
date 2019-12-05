import numpy as np
# import imutils
import cv2

# Import templates in grayscale
fourfour_template = cv2.imread('Templates/44.png', 0)
# threefour_template = cv2.imread('Templates/33.png', 0)
treble_template = cv2.imread('Templates/treble.png', 0)
# qnote_template = cv2.imread('Templates/qn2.png', 0)
qnote_template = cv2.imread('Templates/qnote.png', 0)
hnote_template = cv2.imread('Templates/hnote.png', 0)
qrest_template = cv2.imread('Templates/qrest.png', 0)
# staff_template = cv2.imread('Templates/st.png', 0)
staff_template = cv2.imread('Templates/staff.png', 0)

# RGB Colors
BLACK = (0, 0, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

notes_matrix = None


def template_match(image_bin, image_rgb, template, threshold, rec_color):
    image_rect = image_rgb.copy()
    w_temp, h_temp = template.shape[::-1]  # Dimensions of template
    res = cv2.matchTemplate(image_bin, template, cv2.TM_CCOEFF_NORMED)  # Template match

    # Only keep locations >= threshold (symbol likely matched), loc_unf in the format: [yaxis_vector], [xaxis_vector]
    loc_unf = np.where(res >= threshold)
    if loc_unf[0].size == 0:  # Check if we even got any hits first
        # print("   No hits")
        no_hits = True
        loc = 0
    else:
        no_hits = False
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
            # print("   X: {0} Y:{1}".format(pt[0], pt[1]))  # Print location found
            cv2.rectangle(image_rect, pt, (pt[0] + w_temp, pt[1] + h_temp), rec_color, 1)  # Create rectangle around
        # print("   {0} hits".format(len(loc[0])))
    return loc, image_rect, no_hits


def note_detector(v):
    note = ''
    octave = 0
    if ((v + 15) > 89) & ((v + 15) < 123):
        note = 'A'
        octave = 5
    elif ((v + 15) > 106) & ((v + 15) < 140):
        note = 'E'
        octave = 5
    elif ((v + 15) > 123) & ((v + 15) < 157):
        note = 'D'
        octave = 5
    elif ((v + 15) > 140) & ((v + 15) < 174):
        note = 'C'
        octave = 5
    elif ((v + 15) > 157) & ((v + 15) < 191):
        note = 'B'
        octave = 4
    elif ((v + 15) > 174) & ((v + 15) < 208):
        note = 'A'
        octave = 4
    elif ((v + 15) > 191) & ((v + 15) < 225):
        note = 'G'
        octave = 4
    elif ((v + 15) > 208) & ((v + 15) < 242):
        note = 'F'
        octave = 4
    elif ((v + 15) > 225) & ((v + 15) < 259):
        note = 'E'
        octave = 4
    return note, octave


img_rgb = cv2.imread('Example_Scores/Test1.png')  # Import test image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
ret, img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # Create binary image

img_rect = img_rgb.copy()

# Find all the symbols and create a rectangle around them
# Each column in array notes_matrix is a unique beat
#   row1=note row2=octave row3=duration row4=yaxis row4=xaxis
print("Finding quarter notes:")
qnotes_loc, img_rect, no_hits = template_match(img, img_rect, qnote_template, 0.9, RED)
if no_hits is False:
    lenv = len(qnotes_loc[0])
    qnotes_loc = np.vstack((np.full(lenv, 'A'), np.full(lenv, 4), np.full(lenv, 0.25), qnotes_loc))
    if notes_matrix is None:
        notes_matrix = qnotes_loc
    else:
        notes_matrix = np.concatenate((notes_matrix, qnotes_loc), axis=1)
print("Finding half notes:")
hnotes_loc, img_rect, no_hits = template_match(img, img_rect, hnote_template, 0.9, GREEN)
if no_hits is False:
    lenv = len(hnotes_loc[0])
    hnotes_loc = np.vstack((np.full(lenv, 'A'), np.full(lenv, 4), np.full(lenv, 0.5), hnotes_loc))
    if notes_matrix is None:
        notes_matrix = hnotes_loc
    else:
        notes_matrix = np.concatenate((notes_matrix, hnotes_loc), axis=1)
print("Finding quarter rests:")
qrest_loc, img_rect, no_hits = template_match(img, img_rect, qrest_template, 0.9, BLUE)
if no_hits is False:
    lenv = len(qrest_loc[0])
    qrest_loc = np.vstack((np.full(lenv, 'R'), np.full(lenv, 4), np.full(lenv, 0.25), qrest_loc))
    if notes_matrix is None:
        notes_matrix = qrest_loc
    else:
        notes_matrix = np.concatenate((notes_matrix, qrest_loc), axis=1)
print("Finding a 4 4 time signature:")
fourfour_loc, img_rect, no_hits = template_match(img, img_rect, fourfour_template, 0.9, BLACK)
print("Finding a treble:")
treble_loc, img_rect, no_hits = template_match(img, img_rect, treble_template, 0.9, BLACK)
print("Finding a staff:")
staff_loc, img_rect, no_hits = template_match(img, img_rect, staff_template, 0.9, BLACK)

cv2.imwrite('symbols_found.png', img_rect)  # Keep img file with rectangles

# Sort notes_matrix so they are in correct horizontal order
notes_matrix = notes_matrix[:, notes_matrix[4, :].astype(int).argsort()]

# Calculate mean of staff y axis location. This is reference point for note detection
staffy = (np.rint(np.median(staff_loc[0])))
print("Staff y coordinate: ".format(staffy))

# Go through each matched feature and update to the correct note (including octave)
print("\nDetecting notes")










for i in range(len(notes_matrix[0])):
    if notes_matrix[0][i] != 'R':       # Skip rest notes, those are good to go
        notes_matrix[0][i], notes_matrix[1][i] = \
            note_detector(notes_matrix[3][i].astype(int) - staffy)

print(notes_matrix)