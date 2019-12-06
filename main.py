import numpy as np
from operator import itemgetter
from EasyMIDI import EasyMIDI, Track, Note, Chord
# import imutils
import cv2
from matplotlib import pyplot as plt

# Import templates in grayscale
fourfour_template = cv2.imread('Templates/44.png', 0)
threefour_template = cv2.imread('Templates/34.png', 0)
treble_template = cv2.imread('Templates/tr.png', 0)
qnote_template = cv2.imread('Templates/qn2.png', 0)
# qnote_template = cv2.imread('Templates/qnote.png', 0)
hnote_template = cv2.imread('Templates/hnote.png', 0)
qrest_template = cv2.imread('Templates/qrest.png', 0)
staff_template = cv2.imread('Templates/st.png', 0)
sharp_template = cv2.imread('Templates/sharp.png', 0)

# RGB Colors
BLACK = (0, 0, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

# Note definitions; element one is Note, element two is octave
note_defs = {
    0: ('G', 6),
    1: ('F', 6),
    2: ('E', 6),
    3: ('D', 6),
    4: ('C', 6),
    5: ('B', 5),
    6: ('A', 5),
    7: ('G', 5),
    8: ('F', 5),
    9: ('E', 5),
    10: ('D', 5),
    11: ('C', 5),
    12: ('B', 4),
    13: ('A', 4),
    14: ('G', 4),
    15: ('F', 4),
    16: ('E', 4),
    17: ('D', 4),
    18: ('C', 4),
    19: ('B', 3),
    20: ('A', 3),
    21: ('G', 3),
    22: ('F', 3),
    23: ('E', 3),
    24: ('D', 3)
}


def template_match(image_bin, image_rgb, template, threshold, rec_color):
    image_rect = image_rgb.copy()
    res = cv2.matchTemplate(image_bin, template, cv2.TM_CCOEFF_NORMED)  # Template match
    w_temp, h_temp = template.shape[::-1]  # Dimensions of template

    # Only keep locations >= threshold (symbol likely matched), loc_unf in the format: [yaxis_vector], [xaxis_vector]
    loc_unf = np.where(res >= threshold)
    if loc_unf[0].size == 0:  # Check if we even got any hits first
        no_hits = True
        loc = []
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

        for pt in zip(*loc[::-1]):
            cv2.rectangle(image_rect, pt, (pt[0] + w_temp, pt[1] + h_temp), rec_color, 1)
    return loc, image_rect, no_hits


def duplicate_remover(m):
    unique_x = m[1][0].astype(int)  # Assume first element is unique
    m_filt = m
    counter = 0
    # for pt in zip(*m[::-1]):
    for pt in m[1][1:]:
        # print("   X: {0} Y:{1}".format(pt[0], pt[1]))  # Print location found
        # Detect duplicate if it is within a certain range
        if abs(pt - unique_x) > 5:
            unique_x = pt  # Unique element detected
            counter = counter + 1
        else:  # Delete duplicate
            m_filt = np.delete(m_filt, counter + 1, 1)
    return m_filt


def draw_rect(m, image_rect, w_temp, h_temp, rec_color):
    # Create rectangles in image to showcase symbols found
    for pt in zip(*m[::-1]):  # Loop through each x,y point in locations matrix
    # for pt in m:
        print("0:{0} 1:{1}".format(pt[0], pt[1]))
        cv2.rectangle(image_rect, pt, (pt[0] + w_temp, pt[1] + h_temp), rec_color, 1)  # Create rectangle around
    return img_rect


def note_detector2(v, offset):
    staff_separation = 14
    b = np.round((v+offset)/staff_separation)
    if b > 24:
        print("Could not detect a note: b={0}".format(b))
        note = ''
        octave = ''
    else:
        note = note_defs[b][0]
        octave = note_defs[b][1]
    return note, octave


def create_midi(m):
    # Convert to MIDI
    print("Writing MIDI file")
    easyMIDI = EasyMIDI()
    track1 = Track("acoustic grand piano")
    unique_x = notes_matrix[4][0].astype(int)  # Keep track of x coordinate
    chord_counter = 0
    chord_notes = []
    for i in range(len(notes_matrix[0]) - 1):
        next_x = notes_matrix[4][i + 1].astype(int)
        if abs(unique_x - next_x) < 5:  # Duplicate or possible chord found
            chord_counter += 1  # Increment counter
        else:
            unique_x = next_x
            if chord_counter > 0:
                # if False:
                for j in range(chord_counter):
                    N = Note(notes_matrix[0][i - j], notes_matrix[1][i - j].astype(int),
                             notes_matrix[2][i - j].astype(float))
                    chord_notes.append(N)

                chord = Chord(chord_notes)
                track1.addNote(chord)
                chord_notes = []
                chord_counter = 0
            else:
                N = Note(notes_matrix[0][i], notes_matrix[1][i].astype(int), notes_matrix[2][i].astype(float))
                track1.addNote(N)

    easyMIDI.addTrack(track1)
    easyMIDI.writeMIDI("output.mid")


img_rgb = cv2.imread('Example_Scores/LOZ_SOS_Complete_Row1.png')  # Import test image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
ret, img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # Create binary image

img_rect = img_rgb.copy()

# Find all the symbols and create a rectangle around them
# Each column in array notes_matrix is a unique beat
#   row1=note row2=octave row3=duration row4=yaxis row4=xaxis
notes_matrix = None
print("Finding quarter notes:")
qnotes_loc, img_rect, no_hits = template_match(img, img_rect, qnote_template, 0.9, RED)
if no_hits is False:
    print("   {0} hits".format(len(qnotes_loc[0])))
    lenv = len(qnotes_loc[0])
    qnotes_loc = np.vstack((np.full(lenv, 'A'), np.full(lenv, 4), np.full(lenv, 0.25), qnotes_loc))
    if notes_matrix is None:
        notes_matrix = qnotes_loc
    else:
        notes_matrix = np.concatenate((notes_matrix, qnotes_loc), axis=1)
else:
    print("   No hits")

print("Finding half notes:")
hnotes_loc, img_rect, no_hits = template_match(img, img_rect, hnote_template, 0.9, GREEN)
if no_hits is False:
    print("   {0} hits".format(len(hnotes_loc[0])))
    lenv = len(hnotes_loc[0])
    hnotes_loc = np.vstack((np.full(lenv, 'A'), np.full(lenv, 4), np.full(lenv, 0.5), hnotes_loc))
    if notes_matrix is None:
        notes_matrix = hnotes_loc
    else:
        notes_matrix = np.concatenate((notes_matrix, hnotes_loc), axis=1)
else:
    print("   No hits")

print("Finding quarter rests:")
qrest_loc, img_rect, no_hits = template_match(img, img_rect, qrest_template, 0.9, BLUE)
if no_hits is False:
    print("   {0} hits".format(len(qrest_loc[0])))
    lenv = len(qrest_loc[0])
    qrest_loc = np.vstack((np.full(lenv, 'R'), np.full(lenv, 4), np.full(lenv, 0.25), qrest_loc))
    if notes_matrix is None:
        notes_matrix = qrest_loc
    else:
        notes_matrix = np.concatenate((notes_matrix, qrest_loc), axis=1)
else:
    print("   No hits")

print("Finding sharps:")
sharp_loc, img_rect, no_hits = template_match(img, img_rect, sharp_template, 0.75, BLUE)
sort_ind = sharp_loc[1][:].astype(int).argsort()                # X axis sorting
sharp_loc = [sharp_loc[0][sort_ind], sharp_loc[1][sort_ind]]    # X axis sorting
sharp_loc = duplicate_remover(sharp_loc)                        # Remove duplicates along x axis

if no_hits is False:
    print("   {0} hits".format(len(sharp_loc[0])))
else:
    print("   No hits")

print("Finding a time signature:")
fourfour_loc, img_rect, no_hits = template_match(img, img_rect, fourfour_template, 0.9, BLACK)
if no_hits is True:
    threefour_loc, img_rect, no_hits = template_match(img, img_rect, threefour_template, 0.9, BLACK)
    if no_hits is True:
        print("   Could not detect time signature")
    else:
        print("   3/4 time signature detected")
else:
    print("   4/4 time signature detected")

print("Finding a treble:")
treble_loc, img_rect, no_hits = template_match(img, img_rect, treble_template, 0.9, BLACK)
if no_hits is False:
    print("   Treble detected")
else:
    print("   No hits")

print("Finding a staff:")
staff_loc, img_rect, no_hits = template_match(img, img_rect, staff_template, 0.9, BLACK)
if no_hits is False:
    print("   {0} hits".format(len(staff_loc[0])))
else:
    print("   No hits")
# Calculate mean of staff y axis location. This is reference point for note detection
staffy = (np.rint(np.median(staff_loc[0])))

cv2.imwrite('symbols_found.png', img_rect)  # Keep img file with rectangles


# Go through each matched feature and update to the correct note (including octave)
# Sort notes_matrix so they are in correct horizontal order
notes_matrix = notes_matrix[:, notes_matrix[4, :].astype(int).argsort()]
for i in range(len(notes_matrix[0])):
    if notes_matrix[0][i] != 'R':       # Skip rest notes, those are good to go
        notes_matrix[0][i], notes_matrix[1][i] = \
            note_detector2(notes_matrix[3][i].astype(int) - staffy, 12)

# Adjust for sharps
print("Adjusting pitches based on accidental sharps")
for i in range(len(sharp_loc)):
    for j in range(len(notes_matrix[0])):
        if abs(sharp_loc[1][i].astype(int) - notes_matrix[4][j].astype(int)) < 40:
            print("Need a sharp at j={0} for note {1}".format(j, notes_matrix[0][j]))
            j = len(notes_matrix[0])


# print("\nNotes detected:")
print(notes_matrix)

# create_midi(notes_matrix)
