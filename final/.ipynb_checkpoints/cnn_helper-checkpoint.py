import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def class_to_char(class_idx, alpha):
    if alpha:
        class_idx += 10
        
    mapping = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
        19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
        28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
        36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
        45: 'r', 46: 't'
    }
    
    if class_idx in mapping:
        return mapping[class_idx]
    else:
        return "?"



def display_array_as_image(array, title="Preprocessed Image"):
    display_array = array.copy()
    
    # Handle different input shapes
    if display_array.ndim == 4:
        # Shape: (batch_size, height, width, channels)
        # Take first image from batch
        display_array = display_array[0]
    
    if display_array.ndim == 3:
        # Shape: (height, width, channels)
        # Squeeze out the channel dimension
        display_array = np.squeeze(display_array)
    
    # Now should be 2D: (height, width)
    if display_array.ndim != 2:
        print(f"Warning: Unexpected array shape {array.shape}")
        return
    plt.figure(figsize=(6, 6))
    plt.imshow(display_array, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def preprocess_image_for_cnn(img_array, emnist):

    try:
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        
        img = Image.fromarray(img_array, mode='L')
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.LANCZOS)
        
        # Apply EMNIST transformations if needed
        if emnist:
            img = img.rotate(90, expand=False)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Convert back to array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Invert pixel values (white on black background)
        img_array = 1.0 - img_array
        
        # Reshape for CNN input: (batch_size, height, width, channels)
        cnn_array = img_array.reshape(1, 28, 28, 1)
        
        return cnn_array
        
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None