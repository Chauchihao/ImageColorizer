from fileinput import filename
from pickle import TRUE
from tkinter.constants import NONE
import numpy as np
import cv2
import PySimpleGUI as sg
import os.path

prototxt = r'model/colorization_deploy_v2.prototxt'
model = r'model/colorization_release_v2.caffemodel'
points = r'model/pts_in_hull.npy'
points = os.path.join(os.path.dirname(__file__), points)
prototxt = os.path.join(os.path.dirname(__file__), prototxt)
model = os.path.join(os.path.dirname(__file__), model)
if not os.path.isfile(model):
    sg.popup_scrolled('Missing model file', 'You are missing the file "colorization_release_v2.caffemodel"',
                      'Download it and place into your "model" folder', 'You can download this file from this location:\n', r'https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1')
    exit()
net = cv2.dnn.readNetFromCaffe(prototxt, model)     #load model from disk
pts = np.load(points)

# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_filename=None, cv2_frame=None):
    """
    Where all the magic happens.  Colorizes the image provided. Can colorize either
    a filename OR a cv2 frame (read from a web cam most likely)
    :param image_filename: (str) full filename to colorize
    :param cv2_frame: (cv2 frame)
    :return: Tuple[cv2 frame, cv2 frame] both non-colorized and colorized images in cv2 format as a tuple
    """
    # load the input image from disk, scale the pixel intensities to the range [0, 1], and then convert the image from the BGR to Lab color space
    image = cv2.imread(image_filename) if image_filename else cv2_frame
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # resize the Lab image to 224x224 (the dimensions the colorization network accepts), split channels, extract the 'L' channel, and then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
    'print("[INFO] colorizing image...")'
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize the predicted 'ab' volume to the same dimensions as our input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # grab the 'L' channel from the *original* input image (not the resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # the current colorized image is represented as a floating point data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    return colorized

def convert_to_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert webcam frame to grayscale
    gray_3_channels = np.zeros_like(frame)  # Convert grayscale frame (single channel) to 3 channels
    gray_3_channels[:, :, 0] = gray
    gray_3_channels[:, :, 1] = gray
    gray_3_channels[:, :, 2] = gray
    return gray_3_channels

def histograms(frame):
    h, w, _ = frame.shape
    #Separate the source image in its three R,G and B planes.c
    bgr_planes = cv2.split(frame)
    histSize = 256

    histRange = (0, 256) # the upper boundary is exclusive

    accumulate = False

    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

    hist_w = w
    hist_h = h
    bin_w = int(round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

    for i in range(1, histSize):
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(b_hist[i]) ),
                ( 255, 0, 0), thickness=2)
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(g_hist[i]) ),
                ( 0, 255, 0), thickness=2)
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(r_hist[i]) ),
                ( 0, 0, 255), thickness=2)

    return histImage

def show_file_list(folder):
    img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
    # get list of files in folder
    file_list = os.listdir(folder)
    filenames = [file for file in file_list if os.path.isfile(
        os.path.join(folder, file)) and file.lower().endswith(img_types)]
    window['-FILE LIST-'].update(filenames)
    return filenames

def convert_image(original):
    gray_3_channels = convert_to_grayscale(original)
    colorized = colorize_image(cv2_frame=gray_3_channels)

    return gray_3_channels, colorized

def save_file_list(filenames, folder):
    temp = False
    for file in filenames:
        opath = os.path.join(folder, file)
        gpath = os.path.join(r'images/gray/', file)
        cpath = os.path.join(r'images/colorized/', file)
        if not os.path.isfile(gpath) and not os.path.isfile(cpath):
            original = cv2.imread(opath)
            gray_3_channels, colorized = convert_image(original)
            cv2.imwrite(gpath ,gray_3_channels)
            cv2.imwrite(cpath ,colorized)
            temp = True
    if (temp == False):
        sg.popup_quick_message('No new pictures to convert!', background_color='red', text_color='white', auto_close_duration=7, font='Any 16')
    else:
        sg.popup_quick_message('Convert and save complete!', background_color='red', text_color='white', auto_close_duration=7, font='Any 16')
    

# --------------------------------- The GUI ---------------------------------
sg.theme('Black')
windoww, windowh = sg.Window.get_screen_size()

# The image layout...3 columns

original_col = [[sg.Text('Original')],[sg.Image(filename='', key='-IN-')],[sg.Image(filename='', key='-HIST IN-')]]
gray_col = [[sg.Text('Gray')],[sg.Image(filename='', key='-OUTG-')],[sg.Image(filename='', key='-HIST OUTG-')]]
colorized_col = [[sg.Text('Colorized')],[sg.Image(filename='', key='-OUTC-')],[sg.Image(filename='', key='-HIST OUTC-')]]

# The window layout...2 columns

left_col = [[sg.Text('Folder'), sg.In(size=(35,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse(key='-FILEBROWSE-')],
            [sg.Listbox(values=[], enable_events=True, size=(50,40),key='-FILE LIST-')], [sg.Button('Exit')]]

images_col = [[sg.Column(original_col), sg.Column(gray_col), sg.Column(colorized_col)]]
# ----- Full layout -----
layout = [[sg.Column(left_col, size=(windoww * 0.25, windowh - 65)), sg.VSeperator(), sg.Column(images_col, vertical_alignment="t", size=(windoww * 0.85, windowh - 65))]]

# ----- Make the window -----
window = sg.Window('Photo Colorizer', layout, grab_anywhere=True, size=(windoww, windowh - 65),location=(-8,0))

# ----- Run the Event Loop ----- 
w = int(windoww*0.2)
h = int(windowh*0.35)

while True:
    event, values = window.read()
    
    if event in (None, 'Exit'):
        break
    if event == '-FOLDER-':         # Folder name was filled in, make a list of files in the folder
        folder = values['-FOLDER-']
        try:
            filenames = show_file_list(folder)
            save_file_list(filenames, folder)
        except:
            continue
    elif event == '-FILE LIST-':    # A file was chosen from the listbox
        try: 
            opath = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            gpath = os.path.join(r'images/gray/', values['-FILE LIST-'][0])
            cpath = os.path.join(r'images/colorized/', values['-FILE LIST-'][0])

            original = cv2.imread(opath)
            gray_3_channels = cv2.imread(gpath)
            colorized = cv2.imread(cpath)

            histograms_original = histograms(original)
            histograms_grayscale = histograms(gray_3_channels)
            histograms_colorized = histograms(colorized)

            window['-IN-'].update(data=cv2.imencode('.png', cv2.resize(original, (w, h)))[1].tobytes())
            window['-OUTG-'].update(data=cv2.imencode('.png', cv2.resize(gray_3_channels, (w, h)))[1].tobytes())
            window['-OUTC-'].update(data=cv2.imencode('.png', cv2.resize(colorized, (w, h)))[1].tobytes())
            window['-HIST IN-'].update(data=cv2.imencode('.png', cv2.resize(histograms_original, (w, h)))[1].tobytes())
            window['-HIST OUTG-'].update(data=cv2.imencode('.png', cv2.resize(histograms_grayscale, (w, h)))[1].tobytes())
            window['-HIST OUTC-'].update(data=cv2.imencode('.png', cv2.resize(histograms_colorized, (w, h)))[1].tobytes())
        except:
            sg.popup_quick_message('ERROR - File NOT found!!!', background_color='red', text_color='white', auto_close_duration=7, font='Any 16')
# ----- Exit program -----
window.close()