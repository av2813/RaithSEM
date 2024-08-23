'''
The provided code consists of various functions for image preprocessing, analysis, and visualization. Here's a summary of the key functions and their purposes:

    preprocess_image(path, lower_cutoff, pixel_scale_nm): This function loads and preprocesses the input image. It applies Gaussian blur and Otsu's thresholding to the image, and then returns the original image, binary image, and scales.

    plot_original(image): This function plots the original grayscale image.
    plot_binary_image(binary_image): This function plots the binary image with a grayscale colormap.

    detect_contours(binary_image): This function finds and returns the contours in the input binary image using the RETR_EXTERNAL retrieval mode and CHAIN_APPROX_SIMPLE method.

    plot_contours(image, contours): This function overlays the detected contours on the original image and returns the resulting plot.

    min_area_bounding_rectangles(image, contours, epsilon, conditions): This function calculates the minimum area bounding rectangles for the input contours and returns various properties of the rectangles.

    plot_histograms(lengths_nm, widths_nm, angles): This function plots histograms of the lengths, widths, and angles of the detected objects.

    plot_scatter(image, central_positions_x_nm, central_positions_y_nm, lengths_nm, widths_nm, x_scale, y_scale): This function plots scatter plots of the detected object properties on the input image.

    statistical_summary(...): This function calculates and prints statistical summaries of the detected object properties.

    width_mod_split(...): This function categorizes the detected objects into "thin" and "wide" bars based on a width cutoff, and then calculates and prints statistical summaries for each category.

    save_text(filename, lengths_nm, widths_nm, angles, central_positions_x_nm, central_positions_y_nm, key_params): This function saves the detected object properties to a text file along with any additional key parameters.
'''

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2
import math
import PIL
import matplotlib.patches as patches
import struct
#from PIL import ImageDraw, ImageFont

#import numpy as np
from PIL import Image, ImageDraw, ImageFont

def read_bmp(file_path, image_shape):
    with open(file_path, 'rb') as f:
        # Read the BMP file header (14 bytes)
        bmp_header = f.read(14)
        if bmp_header[:2] != b'BM':
            raise ValueError("Not a BMP file")
        #print(bmp_header)
        # Read the DIB header (40 bytes)
        dib_header = f.read(40)
        #print(struct.unpack('ii',dib_header))
        # Extract width and height from the DIB header
        width, height = struct.unpack('ii', dib_header[18:26])
        #print(width,height)
        # Read the pixel data offset
        pixel_data_offset = struct.unpack('I', bmp_header[10:14])[0]

        # Move the file pointer to the pixel data
        f.seek(pixel_data_offset)

        # Calculate the number of bytes per pixel
        # For 24-bit BMP, each pixel is represented by 3 bytes (BGR)
        bytes_per_pixel = 3

        # Read the pixel data
        pixel_data = f.read()

        # Create a NumPy array from the pixel data
        # The pixel data is stored in BGR format
        image = np.frombuffer(pixel_data, dtype=np.uint8)

        # Reshape the array to the correct dimensions
        image = image.reshape(image_shape)

        # Convert BGR to RGB format
        image = image[:, :, ::-1]

        return image

def read_raithfile(folder, file_im, file_meta, image_shape = (4000,4000,1)):
    '''
    
    '''
    path_im = os.path.join(folder, file_im)
    path_meta = os.path.join(folder, file_meta)
    print(path_im)
    print(path_meta)
    
    # Read the image file
    image = read_bmp(path_im,image_shape)
    
    # Read the metadata file
    with open(path_meta, 'r') as file:
        metadata = file.read()
    
    return image, metadata

def meta2dict(metadata_string):
    # Initialize an empty dictionary
    metadata_dict = {}

    # Split the string into sections
    sections = metadata_string.strip().split('\n\n')

    for section in sections:
        # Split each section into lines
        lines = section.strip().split('\n')
        # Get the section name (e.g., SLOWSCAN, METADATA)
        section_name = lines[0].strip('[]')
        # Initialize a sub-dictionary for this section
        metadata_dict[section_name] = {}
        
        # Process each line in the section
        for line in lines[1:]:
            if '=' in line:
                key, value = line.split('=', 1)
                metadata_dict[section_name][key.strip()] = value.strip()
    return(metadata_dict)

def find_scale_bar(image, scale_bar_position_bl=[750, 200], scale_bar_size=[100, 30], units='1um'):
    """
    Automatically locates and measures the scale bar in the input image.

    Parameters:
    - image (np.ndarray): Input image.
    - scale_bar_position_bl (list): Bottom-left position of the scale bar.
    - scale_bar_size (list): Size of the scale bar region.
    - units (str): Units of the scale bar.

    Returns:
    - len_scale (float): Length of the scale bar in pixels.
    - px_2_units (float): Conversion factor from pixels to specified units.
    """
    try:
        # Extract the region of interest (ROI) containing the scale bar
        roi = image[scale_bar_position_bl[0]:scale_bar_position_bl[0] + scale_bar_size[0],
              scale_bar_position_bl[1]:scale_bar_position_bl[1] + scale_bar_size[1]]

        # Sum the pixel values along the columns to create a 1D profile
        profile = np.sum(roi, axis=0)

        # Find the region where the pixel values are below a threshold
        s_bar_indices = np.where(profile < np.mean(profile) * 0.5)[0]

        # Calculate the length of the scale bar in pixels
        len_scale = float(s_bar_indices[-1] - s_bar_indices[0])

        # Calculate the conversion factor from pixels to specified units
        if 'um' in units:
            px_2_units = (float(units.replace('um', '')) * 1000) / len_scale
        elif 'nm' in units:
            px_2_units = float(units.replace('nm', '')) / len_scale
        else:
            raise ValueError("Invalid units. Please use 'um' or 'nm'.")

        return len_scale, px_2_units

    except Exception as e:
        print(f"Error in finding scale bar: {e}")
        return None, None

def add_scale_bar(image, area_rectangle, units, scale, position, font_size=10):
    """
    Adds a scale bar to the image based on the specified parameters.

    Parameters:
    - image (np.ndarray): Input image as a NumPy array.
    - area_rectangle (tuple): Rectangle defining the area to keep (x_start, y_start, x_end, y_end).
    - units (str): Units of the scale bar.
    - scale (float): Scale factor for conversion from pixels to the specified units.
    - position (tuple): Position of the scale bar (x_position, y_position).
    - font_size (int): Font size for the scale text.

    Returns:
    - image_with_scale (np.ndarray): Image with the added scale bar.
    """
    try:
        # Create a copy of the image to avoid modifying the original
        image_with_scale = np.copy(image)

        # Draw the scale bar directly on the NumPy array
        scale_length_pixels = int(scale)
        image_with_scale[position[1]:position[1] + 10, position[0]:position[0] + scale_length_pixels] = 255

        # Draw the scale text
        scale_text = f"{units}"
        font = ImageFont.truetype("arial.ttf", font_size)
        pil_image = Image.fromarray(image_with_scale[area_rectangle[1]:area_rectangle[3],
                                 area_rectangle[0]:area_rectangle[2]])
        draw = ImageDraw.Draw(pil_image)
        draw.text((position[0] - int(font_size), position[1] + 15), scale_text, fill="white", font=font)

        return np.array(pil_image)

    except Exception as e:
        print(f"Error in adding scale bar: {e}")
        return None
    
def find_scale_bar_auto(image, scale_length_mm):
    # Convert image to grayscale
    gray = image
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    _, binary = cv2.threshold(gray, 0, 255, 240)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and aspect ratio to find the scale bar
    scale_bar_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area > 100 and perimeter > 10:
            scale_bar_contour = contour
            break

    if scale_bar_contour is not None:
        # Calculate length of scale bar in pixels
        scale_length_pixels = cv2.arcLength(scale_bar_contour, True)

        # Calculate conversion factor from pixels to mm
        if scale_length_pixels > 0:
            px_to_mm = scale_length_mm / scale_length_pixels
            return px_to_mm

    return None

# Load image
image = cv2.imread('path_to_your_image.jpg')

# Specify the known length of the scale bar in mm
scale_length_mm = 10  # Example: 10 mm

def preprocess_image(image_path, lower_cutoff=700, pixel_scale_nm=1000/109):
    """
    Preprocesses the input SEM image.

    Parameters:
    - image_path (str): Path to the input image.
    - lower_cutoff (int): Lower cutoff for image cropping.
    - pixel_scale_nm (float): Pixel scale in nanometers.

    Returns:
    - img_gray (np.ndarray): Preprocessed grayscale image.
    - binary_image (np.ndarray): Binary image after thresholding.
    - x_scale (np.ndarray): X-axis scale in nanometers.
    - y_scale (np.ndarray): Y-axis scale in nanometers.
    """
    try:
        # Load the image
        # Read the image
        #image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #print(image)
        image = np.array(Image.open(image_path))
        if lower_cutoff == None:
            image_cropped = image
        else:
            # Crop and convert to grayscale
            image_cropped = image[:lower_cutoff, :]
        img_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

        # Gaussian blur and thresholding
        blurred = cv2.GaussianBlur(img_gray, (21, 21), 10)
        _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Generate pixel scales
        x_len, y_len = image_cropped.shape[:2]
        y_scale, x_scale = np.arange(0, x_len) * pixel_scale_nm, np.arange(0, y_len) * pixel_scale_nm

        return img_gray, binary_image, x_scale, y_scale

    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None, None, None, None
    
def preprocess_image2(image_raw, lower_cutoff=700, pixel_scale_nm=1000/109):
    """
    Preprocesses the input SEM image.

    Parameters:
    - image_path (str): Path to the input image.
    - lower_cutoff (int): Lower cutoff for image cropping.
    - pixel_scale_nm (float): Pixel scale in nanometers.

    Returns:
    - img_gray (np.ndarray): Preprocessed grayscale image.
    - binary_image (np.ndarray): Binary image after thresholding.
    - x_scale (np.ndarray): X-axis scale in nanometers.
    - y_scale (np.ndarray): Y-axis scale in nanometers.
    """
    try:
        # Load the image
        # Read the image
        #image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #print(image)
        image = image_raw
        if lower_cutoff == None:
            image_cropped = image
        else:
            # Crop and convert to grayscale
            image_cropped = image[:lower_cutoff, :]
        img_gray = image_cropped#cv2.cvtColor(, )#image_cropped#

        # Gaussian blur and thresholding
        blurred = cv2.GaussianBlur(img_gray, (21, 21), 10)
        _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Generate pixel scales
        x_len, y_len = image_cropped.shape[:2]
        y_scale, x_scale = np.arange(0, x_len) * pixel_scale_nm, np.arange(0, y_len) * pixel_scale_nm

        return img_gray, binary_image, x_scale, y_scale

    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None, None, None, None

def plot_original(image,px_size=1):
     # Plot the original image
    fig,ax = plt.subplots(figsize=(6, 6))
    plt.imshow(image, origin='lower', cmap='gray', extent=[0, image.shape[0]*px_size, 0, image.shape[1]*px_size])
    plt.title('Original image')
    return(fig)
    #plt.show()
    
def plot_binary_image(binary_image,px_size=1):
    # Plot the binary image
    fig, ax = plt.subplots(figsize=(6, 6))  
    plt.imshow(binary_image, origin='lower', cmap='gray', extent=[0, binary_image.shape[0]*px_size, 0, binary_image.shape[1]*px_size])  # Display the binary image with a grayscale colormap
    plt.title('Binary image')
    #plt.show()  # Uncomment this line to display the plot immediately
    return fig  # Return the figure object

def detect_contours(binary_image):
    # Find contours in the binary image using the RETR_EXTERNAL retrieval mode and CHAIN_APPROX_SIMPLE method
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Return the detected contours
    return contours

def grayscale_to_color(image, cmap='grey'):
    """
    Convert a grayscale image array to a color image.
    
    Parameters:
    - image: numpy array of shape (nd, nd) or (nd, nd, 1)
    - cmap: colormap to use (default is 'viridis')
    
    Returns:
    - color_image: numpy array of shape (nd, nd, 3) representing an RGB image
    """
    # Ensure the image is 2D
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:,:,0]
    
    # Normalize the image
    image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Apply colormap
    cmap = plt.get_cmap(cmap)
    color_image = cmap(image_normalized)
    
    # Convert to RGB (remove alpha channel if present)
    color_image = color_image[:,:,:3]
    
    # Ensure the values are in the range [0, 255] and the type is uint8
    color_image = (color_image * 255).astype(np.uint8)
    
    return color_image

def plot_contours(image, contours, px_size=1):
    image_with_contours = grayscale_to_color(image)
    cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 10)
    fig,ax = plt.subplots(figsize=(6, 6))
    plt.imshow(image_with_contours, origin = 'lower', extent=[0, image_with_contours.shape[0]*px_size, 0, image_with_contours.shape[1]*px_size])
    plt.title('Contours')
    #plt.show()
    return(fig)

def min_area_bounding_shapes(image, contours, shape_type='rectangle', epsilon=0.001, conditions=None, px_2_nm=1, show=False):
    """
    Calculate minimum area bounding shapes (circles, ellipses, or rectangles) for detected contours.

    Parameters:
    - image (np.ndarray): Input image as a NumPy array.
    - contours (list): List of detected contours.
    - shape_type (str): Type of shape to detect ('circle', 'ellipse', or 'rectangle').
    - epsilon (float): Approximation accuracy parameter.
    - conditions (list): List of conditions for selecting shapes.

    Returns:
    - all_shapes (list): All detected shapes.
    - select_shapes (list): Selected shapes based on conditions.
    - lengths (list): Lengths (radii for circles) of shapes.
    - widths (list): Widths of shapes (None for circles and ellipses).
    - angles (list): Orientations (angles) of shapes (None for circles and ellipses).
    - central_positions_x (list): X-coordinate positions of shape centers.
    - central_positions_y (list): Y-coordinate positions of shape centers.
    - centers (list): Centers of shapes.
    - fig (matplotlib.figure.Figure): Matplotlib figure object.
    """
    try:
        all_shapes = []
        select_shapes = []
        lengths = []
        widths = []
        angles = []
        central_positions_x = []
        central_positions_y = []
        centers = []

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, origin='lower')
        ax.set_title(f'Bounding {shape_type}s')

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            all_shapes.append(approx)

        for shape in all_shapes:
            if shape_type == 'circle':
                rect = cv2.minEnclosingCircle(shape)
                center, radius = rect
                lengths.append(radius)
                widths.append(None)  # No width for circles
                angles.append(None)  # No angle for circles
                rect_patch = patches.Circle(xy=center, radius=radius, linewidth=1, edgecolor='r', facecolor='none')
            elif shape_type == 'ellipse':
                ellipse = cv2.fitEllipse(shape)
                (center, axes, angle) = ellipse
                lengths.append(max(axes) / 2)
                widths.append(min(axes) / 2)
                angles.append(angle)
                rect_patch = patches.Ellipse(xy=center, width=axes[0], height=axes[1], angle=angle, linewidth=1, edgecolor='r', facecolor='none')
            elif shape_type == 'rectangle':
                rect = cv2.minAreaRect(shape)
                center, (width, length), angle = rect
                if angle < -45:
                    angle += 90
                lengths.append(length)
                widths.append(width)
                angles.append(angle)
                rect_points = cv2.boxPoints(rect)
                rect_patch = patches.Polygon(rect_points, closed=True, linewidth=1, edgecolor='r', facecolor='none')

            center_int = (int(center[0]), int(center[1]))
            central_positions_x.append(center[0])
            central_positions_y.append(center[1])
            centers.append(center)

            if conditions is not None:
                if all(condition(lengths[-1], widths[-1], angles[-1], center) for condition in conditions):
                    select_shapes.append(shape)
                    ax.add_patch(rect_patch)
                else:
                    del lengths[-1]
                    del widths[-1]
                    del angles[-1]
                    del centers[-1]
                    del central_positions_x[-1]
                    del central_positions_y[-1]
            else:
                select_shapes.append(shape)

        if show:
            plt.show()
        return all_shapes, select_shapes, lengths, widths, angles, central_positions_x, central_positions_y, centers, fig

    except Exception as e:
        print(f"Error in calculating bounding {shape_type}s: {e}")
        return [], [], [], [], [], [], [], None

def min_area_bounding_circles(image, contours, epsilion = 0.001, conditions=None):
    all_bars = []
    select_bars = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilion * perimeter, True)
        all_bars.append(approx)
    fig, ax = plt.subplots(1,1,figsize=(6, 6))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), origin='lower')
    ax.set_title('Bounding boxes')
    lengths = []
    widths = []
    angles = []
    central_positions_x = []
    central_positions_y = []
    centers = []
    for bar in all_bars:
        #(x,y),radius = cv2.minEnclosingCircle(bar)
        rect = cv2.minEnclosingCircle(bar)

        center, radius = rect
        #if angle < -45:
        #    angle += 90
        #box = cv2.boxPoints(((center), (width, length), angle))
        #box = np.int0(box)
        #rect_points = cv2.boxPoints(rect)  # Get the corner points of the rotated rectangle
        x,y = center
        center_int = (int(x),int(y))
        radius_int = int(radius)
        print(center, radius,)
        cv2.circle(image,center_int,radius_int,(0,255,0),2)
        #if length < width:
        #    length, width = width, length
        if conditions is not None:
            if all(condition(radius, center) for condition in conditions):
                select_bars.append(bar)
            else:
                continue
        else:
            select_bars.append(bar)
        rect_patch = patches.Circle(xy=(x,y),radius=radius, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect_patch)
        length = radius
        lengths.append(length)
        central_positions_x.append(center[0])
        central_positions_y.append(center[1])
        centers.append(center)
    #plt.show()
    return all_bars, select_bars, lengths, widths, angles, central_positions_x, central_positions_y, centers, fig

def min_area_bounding_rectangles(image, contours, epsilion = 0.001, conditions=None,px_2_nm=1):
    all_bars = []
    select_bars = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilion * perimeter, True)
        all_bars.append(approx)
    fig, ax = plt.subplots(1,1,figsize=(6, 6))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), origin='lower')
    ax.set_title('Bounding boxes')
    lengths = []
    widths = []
    angles = []
    central_positions_x = []
    central_positions_y = []
    centers = []
    for bar in all_bars:
        rect = cv2.minAreaRect(bar)

        center, (width, length), angle = rect
        if angle < -45:
            angle += 90
        box = cv2.boxPoints(((center), (width, length), angle))
        box = np.int0(box)
        rect_points = cv2.boxPoints(rect)  # Get the corner points of the rotated rectangle
        if length < width:
            length, width = width, length
        if conditions is not None:
            if all(condition(length/px_2_nm, width/px_2_nm, angle, center) for condition in conditions):
                select_bars.append(bar)
            else:
                continue
        else:
            select_bars.append(bar)
        rect_patch = patches.Polygon(rect_points, closed=True, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect_patch)
        lengths.append(length)
        widths.append(width)
        angles.append(angle)
        central_positions_x.append(center[0])
        central_positions_y.append(center[1])
        centers.append(center)
    #plt.show()
    return all_bars, select_bars, lengths, widths, angles, central_positions_x, central_positions_y, centers, fig

def convert_to_nm(dimensions, pixel_scale_nm):
    return np.array(dimensions) * pixel_scale_nm

# Function to calculate distance between two points
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
dist_list = []

# Function to calculate average lattice spacing
def average_lattice_spacing(coordinates):
    sorted_coords = sorted(coordinates, key=lambda coord: (coord[0], coord[1]))  # Sort coordinates based on x values
    total_distance = 0
    n = len(sorted_coords)
    for i in range(n - 1):
        x1, y1 = sorted_coords[i]
        x2, y2 = sorted_coords[i + 1]
        dist_list.append(distance(x1, y1, x2, y2))
        total_distance += distance(x1, y1, x2, y2)
    average_spacing = total_distance / (n - 1)
    return average_spacing, dist_list

# Data Analysis and Visualization Functions:
def plot_histograms(lengths_nm, widths_nm, angles):
    fig,ax = plt.subplots(1,3,figsize=(12, 4))
    #plt.subplot(131)
    ax[0].hist(lengths_nm, bins=5*int(np.log2(len(lengths_nm))+1), color='skyblue', edgecolor='black')
    ax[0].set_title('Length Histogram')
    ax[0].set_xlabel('Length (nm)')
    ax[0].set_ylabel('Frequency')

    #plt.subplot(132)
    ax[1].hist(widths_nm, bins=5*int(np.log2(len(widths_nm))+1), color='lightgreen', edgecolor='black')
    ax[1].set_title('Width Histogram')
    ax[1].set_xlabel('Width (nm)')
    ax[1].set_ylabel('Frequency')

    #plt.subplot(133)
    ax[2].hist(angles, bins=5*int(np.log2(len(angles))+1), color='lightcoral', edgecolor='black')
    #plt.hist(central_positions_y_nm, bins=int(np.sqrt(len(central_positions_y_nm))), color='lightcoral', edgecolor='black', alpha=0.5)
    ax[2].set_title('Angle Histogram')
    ax[2].set_xlabel('Angle (deg)')
    ax[2].set_ylabel('Frequency')
    #plt.show()
    return(fig)

def plot_histograms_circ(radius_nm):
    fig,ax = plt.subplots(1,1,figsize=(4, 4))
    #plt.subplot(131)
    ax.hist(radius_nm, bins=5*int(np.log2(len(radius_nm))+1), color='skyblue', edgecolor='black')
    ax.set_title('Radius Histogram')
    ax.set_xlabel('Radius (nm)')
    ax.set_ylabel('Frequency')
    return(fig)


def plot_scatter(image,central_positions_x_nm,central_positions_y_nm,lengths_nm, widths_nm,x_scale,y_scale):
    fig,ax = plt.subplots(2,1, figsize = (4,6))
    ax[0].imshow(image, cmap='gray',origin='lower',extent=(0,max(x_scale),0,max(y_scale)))
    scat_len = ax[0].scatter(central_positions_x_nm,central_positions_y_nm, c = lengths_nm)
    ax[0].set_title('Lengths')
    plt.colorbar(scat_len, ax=ax[0])
    ax[1].imshow(image, cmap='gray',origin='lower',extent=(0,max(x_scale),0,max(y_scale)))
    scat_wid = ax[1].scatter(central_positions_x_nm,central_positions_y_nm,c = widths_nm)
    ax[1].set_title('Widths')
    plt.colorbar(scat_wid, ax=ax[1])
    #plt.show()
    return(fig)

def plot_scatter_circ(image,central_positions_x_nm,central_positions_y_nm,lengths_nm, x_scale,y_scale):
    fig,ax = plt.subplots(1,1, figsize = (4,6))
    ax.imshow(image, cmap='gray',origin='lower',extent=(0,max(x_scale),0,max(y_scale)))
    scat_len = ax.scatter(central_positions_x_nm,central_positions_y_nm, c = lengths_nm)
    ax.set_title('Radius')
    plt.colorbar(scat_len, ax=ax)
    #plt.show()
    return(fig)


def statistical_summary(lengths_nm, widths_nm, angles,central_positions_x_nm,central_positions_y_nm, x_scale,y_scale):
    # Calculate the average and standard deviation for "thin" bars
    average_length= np.mean(lengths_nm)
    std_dev_length = np.std(lengths_nm)
    average_width= np.mean(widths_nm)
    std_dev_width = np.std(widths_nm)
    average_angle = np.mean(angles)
    std_dev_angle = np.std(angles)

    print("All Bars - Average Length:", average_length)
    print("All Bars - Standard Deviation Length:", std_dev_length)
    print("All Bars - Average Width:", average_width)
    print("All Bars - Standard Deviation Width:", std_dev_width)
    print("All Bars - Average Angle:", average_angle)
    print("All Bars - Standard Deviation Angle:", std_dev_angle)


def statistical_summary_circ(lengths_nm,central_positions_x_nm,central_positions_y_nm, x_scale,y_scale):
    # Calculate the average and standard deviation for "thin" bars
    average_length= np.mean(lengths_nm)
    std_dev_length = np.std(lengths_nm)

    print("All Bars - Average Length:", average_length)
    print("All Bars - Standard Deviation Length:", std_dev_length)
    return(average_length,std_dev_length)
    
def width_mod_split(image,lengths_nm, widths_nm, angles,central_positions_x_nm,central_positions_y_nm, x_scale,y_scale,width_cutoff=180):
    thin_bars_mask = widths_nm < width_cutoff
    wide_bars_mask = widths_nm >= width_cutoff

    # Filter the properties based on the categorization
    thin_lengths = lengths_nm[thin_bars_mask]
    thin_widths = widths_nm[thin_bars_mask]
    thin_angles = np.array(angles)[thin_bars_mask]
    thin_central_positions_x_nm = np.array(central_positions_x_nm)[thin_bars_mask]
    thin_central_positions_y_nm = np.array(central_positions_y_nm)[thin_bars_mask]

    wide_lengths = lengths_nm[wide_bars_mask]
    wide_widths = widths_nm[wide_bars_mask]
    wide_angles = np.array(angles)[wide_bars_mask]
    wide_central_positions_x_nm = np.array(central_positions_x_nm)[wide_bars_mask]
    wide_central_positions_y_nm = np.array(central_positions_y_nm)[wide_bars_mask]

    # Calculate the average and standard deviation for "thin" bars
    average_length_thin = np.mean(thin_lengths)
    std_dev_length_thin = np.std(thin_lengths)
    average_width_thin = np.mean(thin_widths)
    std_dev_width_thin = np.std(thin_widths)
    average_angle_thin = np.mean(thin_angles)
    std_dev_angle_thin = np.std(thin_angles)

    # Calculate the average and standard deviation for "wide" bars
    average_length_wide = np.mean(wide_lengths)
    std_dev_length_wide = np.std(wide_lengths)
    average_width_wide = np.mean(wide_widths)
    std_dev_width_wide = np.std(wide_widths)
    average_angle_wide = np.mean(wide_angles)
    std_dev_angle_wide = np.std(wide_angles)

    print("Thin Bars - Average Length:", average_length_thin)
    print("Thin Bars - Standard Deviation Length:", std_dev_length_thin)
    print("Thin Bars - Average Width:", average_width_thin)
    print("Thin Bars - Standard Deviation Width:", std_dev_width_thin)
    print("Thin Bars - Average Angle:", average_angle_thin)
    print("Thin Bars - Standard Deviation Angle:", std_dev_angle_thin)

    print("Wide Bars - Average Length:", average_length_wide)
    print("Wide Bars - Standard Deviation Length:", std_dev_length_wide)
    print("Wide Bars - Average Width:", average_width_wide)
    print("Wide Bars - Standard Deviation Width:", std_dev_width_wide)
    print("Wide Bars - Average Angle:", average_angle_wide)
    print("Wide Bars - Standard Deviation Angle:", std_dev_angle_wide)


    # Calculate the gap between the centers of the thin bars
    gap_thin_bars = np.mean(np.diff(central_positions_x_nm[thin_bars_mask]))

    # Calculate the gap between the centers of each adjacent wide bar
    gap_wide_bars = np.mean(np.diff(central_positions_x_nm[wide_bars_mask]))

    #print("Gap between the centers of the thin bars:", gap_thin_bars)
    #print("Gap between the centers of each adjacent wide bar:", gap_wide_bars)

    # Calculate the gap between the centers of the thin bars
    gap_thin_bars = np.mean(np.diff(central_positions_y_nm[thin_bars_mask]))

    # Calculate the gap between the centers of each adjacent wide bar
    gap_wide_bars = np.mean(np.diff(central_positions_y_nm[wide_bars_mask]))

    #print("Gap between the centers of the thin bars:", gap_thin_bars)
    #print("Gap between the centers of each adjacent wide bar:", gap_wide_bars)
    plot_scatter(image,thin_central_positions_x_nm,thin_central_positions_y_nm,thin_lengths, thin_widths,x_scale,y_scale)
    plot_scatter(image,wide_central_positions_x_nm,wide_central_positions_y_nm,wide_lengths, wide_widths,x_scale,y_scale)
    
def save_text(filename, lengths_nm, widths_nm, angles, central_positions_x_nm, central_positions_y_nm, key_params={}):
    header = f"\n\nlengths_nm,widths_nm,angles,central_positions_x_nm,central_positions_y_nm"
    comments_p1 = '# key_params: \n# '
    comments_p2 = '\n# '.join([f"{key}: {value}" for key, value in key_params.items()])
    comments = comments_p1+comments_p2+'\n'
    data = np.array([lengths_nm, widths_nm, angles, central_positions_x_nm, central_positions_y_nm], dtype=float)
    with open(filename, 'w') as file:
        file.write(comments)
        file.write(header+'\n')
        np.savetxt(file, data.T, delimiter=',')

def save_text_circ(filename, lengths_nm, central_positions_x_nm, central_positions_y_nm, key_params={}):
    header = f"\n\nlengths_nm,central_positions_x_nm,central_positions_y_nm"
    comments_p1 = '# key_params: \n# '
    comments_p2 = '\n# '.join([f"{key}: {value}" for key, value in key_params.items()])
    comments = comments_p1+comments_p2+'\n'
    data = np.array([lengths_nm, central_positions_x_nm, central_positions_y_nm], dtype=float)
    with open(filename, 'w') as file:
        file.write(comments)
        file.write(header+'\n')
        np.savetxt(file, data.T, delimiter=',')