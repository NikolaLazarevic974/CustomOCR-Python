
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def load_and_convert_to_grayscale(image_path):
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    return original_rgb, grayscale_image

def denoise_image(image, kernel_size=(5, 5)):
    denoised_image = cv2.GaussianBlur(image, kernel_size, 0)
    return denoised_image

def convert_to_binary(grayscale_image):
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #binary_image = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary_image
def display_conversion_steps(original, grayscale, denoised, binary):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(grayscale, cmap='gray')
    axes[1].set_title('Grayscale Image')
    axes[1].axis('off')
    
    axes[2].imshow(denoised, cmap='gray')
    axes[2].set_title('Denoised Image')
    axes[2].axis('off')

    axes[3].imshow(binary, cmap='gray')
    axes[3].set_title('Binary Image')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

def find_characters(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        255 - binary_image, connectivity=8
    )
    
    # stats columns: [x, y, width, height, area]
    # Skip label 0 (background)
    character_data = []
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        
        character_info = {
            'label': i,
            'bbox': (x, y, w, h),
            'area': area
        }
        character_data.append(character_info)
    
    return character_data, labels

def filter_characters(character_data, min_area_threshold, max_area_threshold):
    """
    Filters a list of character data based on a minimum and maximum area threshold.
    """
    filtered_characters = [
        char for char in character_data 
        if char['area'] > min_area_threshold and char['area'] < max_area_threshold
    ]
    return filtered_characters

def sort_characters_left_to_right(character_data):
    """
    Sort characters from left to right based on their x-position.
    
    Args:
        character_data: List of character dictionaries with 'bbox' key
        
    Returns:
        Sorted list of characters in left-to-right order
    """
    if len(character_data) == 0:
        return character_data
    
    # Sort by x position (first element of bbox tuple)
    sorted_characters = sorted(character_data, key=lambda char: char['bbox'][0])
    
    return sorted_characters



def visualize_character_detection(original_image, character_data, filtered_data, title="Character Detection Results"):
    # ---------------------------------------------
    # Helper to prepare the image (handles grayscale -> RGB)
    # ---------------------------------------------
    def get_rgb_base_image(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img.copy()

    # ---------------------------------------------
    # Helper to generate colors (reused from original)
    # ---------------------------------------------
    def generate_colors(data_list):
        colors = []
        for i in range(len(data_list)):
            hue = int((i * 180 / len(data_list)) % 180)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]
            colors.append((int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])))
        return colors

    # ---------------------------------------------
    # Helper to draw boxes on a single image
    # ---------------------------------------------
    def draw_boxes(img, data, colors):
        img_with_boxes = get_rgb_base_image(img)
        
        for i, char in enumerate(data):
            x, y, w, h = char['bbox']
            color = colors[i]
            
            # Draw rectangle
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)
            
            # Draw character index
            cv2.putText(img_with_boxes, str(i), (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        # Add a black outline
        outline_thickness = 5
        img_with_boxes = cv2.copyMakeBorder(img_with_boxes,
                                            outline_thickness, outline_thickness,
                                            outline_thickness, outline_thickness,
                                            cv2.BORDER_CONSTANT,
                                            value=[0, 0, 0])
        return img_with_boxes

    # =========================================================
    # 1. Create the 'All Characters' Image (Left)
    # =========================================================
    all_colors = generate_colors(character_data)
    img_all_chars = draw_boxes(original_image, character_data, all_colors)

    # =========================================================
    # 2. Create the 'Filtered Characters' Image (Right)
    # =========================================================
    filtered_colors = generate_colors(filtered_data) # Use new colors for filtered set or reuse all_colors
    img_filtered_chars = draw_boxes(original_image, filtered_data, filtered_colors)
    
    # ---------------------------------------------
    # Display Side by Side using Matplotlib
    # ---------------------------------------------
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9)) # Adjust figure size for two images
    fig.suptitle(title, fontsize=16)

    # Left Subplot: All Characters
    axes[0].imshow(img_all_chars)
    axes[0].set_title(f"All Candidates ({len(character_data)} found)")
    axes[0].axis('off')

    # Right Subplot: Filtered Characters
    axes[1].imshow(img_filtered_chars)
    axes[1].set_title(f"Filtered Results ({len(filtered_data)} kept)")
    axes[1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.show()


def pad_to_square(image, background_color=255, margin_percent=10):
    """
    Pads a grayscale image with a background color to make it square.
    (This function remains unchanged)
    """
    h, w = image.shape
    
    # Determine the base size of the square
    base_size = max(h, w)
    
    # Calculate margin in pixels
    margin = int(base_size * (margin_percent / 100))
    
    # New size includes the content plus margin on all sides
    new_size = base_size + 2 * margin
    
    # Create a new square image filled with the background color
    padded_image = np.full((new_size, new_size), background_color, dtype=image.dtype)
    
    # Calculate the position to paste the original image (centered)
    x_offset = (new_size - w) // 2
    y_offset = (new_size - h) // 2
    
    # Paste the original image into the center of the new square image
    padded_image[y_offset:y_offset + h, x_offset:x_offset + w] = image
    
    return padded_image


    
def save_extracted_characters(extracted_characters, output_dir="extracted_chars"):
    """
    Saves a list of character images to a local directory.

    Args:
        extracted_characters (list): A list of dictionaries, each containing an 'image' key.
        output_dir (str): The name of the directory where images will be saved.
    """
    # Create the output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)

    # Iterate through the list of extracted images and save each one
    for i, char_info in enumerate(extracted_characters):
        image_to_save = char_info['image']
        
        # Define the filename. You can customize this.
        filename = f"char_{i}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save the image using cv2.imwrite()
        # The image data type is converted to a format suitable for saving.
        cv2.imwrite(filepath, image_to_save)