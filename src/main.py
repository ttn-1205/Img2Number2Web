import tensorflow as tf
import os
import cv2
import numpy as np
import tkinter as tk
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Constants for file paths:
# MODEL_PATH: Location of the pre-trained Keras model.
# OUTPUT_FOLDER: Directory for the raw canvas image capture.
# PROCESSED_DIGITS_FOLDER: Directory for individual digit images after contour detection. The model will process each digit image in this directory to return an answer, submitting to the Zetamac website.
MODEL_PATH = "D:\\Project 1 - Mouse to Number to Web\\Finetuned_MNIST.keras"
OUTPUT_FOLDER = "D:\\OneDrive\\Desktop\\Input"
PROCESSED_DIGITS_FOLDER = "D:\\OneDrive\\Desktop\\Output"

# Ensure that the necessary output directories exist. If they don't exist, create them.
model = tf.keras.models.load_model(MODEL_PATH)

# Ensure directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_DIGITS_FOLDER, exist_ok=True)
print(f"Canvas output directory ensured: {OUTPUT_FOLDER}")
print(f"Processed digits output directory ensured: {PROCESSED_DIGITS_FOLDER}")

# Initialize the Selenium WebDriver for Chrome.
# ChromeDriverManager automatically downloads and manages the correct ChromeDriver version.
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://arithmetic.zetamac.com/")

# Initialize WebDriverWait for explicit waits. This is crucial for web automation on dynamic pages, ensuring elements are loaded and interactive before attempting to interact with them, preventing errors.
wait = WebDriverWait(driver, 3) # Maximum 3-second wait time

# Attempt to locate the primary input box on the Zetamac website early.
# This helps confirm that the web page has loaded sufficiently for interaction.
# The input box is identified by its class name "answer".
try:
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "answer")))
    print("Zetamac input box is ready.")
except Exception as e:
    print(f"Warning: Could not find the input box at startup: {e}")

# -- Setup Tkinter canvas --
# Dimensions of the drawing canvas
CANVAS_DRAWING_WIDTH = 400
CANVAS_DRAWING_HEIGHT = 300
# Variables for tracking mouse position during drawing
last_x, last_y = None, None
# Boolean variable to control drawing activity. Drawing only occurs when enabled.
drawing_enabled = False

def while_drawing(event):
    """
    Callback function for mouse motion events on the canvas.
    Draws a line segment if drawing is enabled and previous coordinates exist.
    The line is black, 2 pixels wide, with round caps and smoothing.
    """
    global last_x, last_y
    if drawing_enabled:
        if last_x is not None and last_y is not None:
            canvas.create_line(last_x, last_y, event.x, event.y, fill="black", width=2, capstyle=tk.ROUND, smooth=tk.TRUE)
        # Update last_x and last_y to the current mouse position for the next segment.
        last_x, last_y = event.x, event.y

def start_drawing(event):
    """
    Callback function for left mouse button press (Button-1).
    Enables drawing and records the initial click coordinates.
    """
    global drawing_enabled, last_x, last_y
    if event.num == 1:
        drawing_enabled = True
        last_x, last_y = event.x, event.y

def stop_drawing(event):
    """
    Callback function for left mouse button release (Button-1).
    Disables drawing and resets last_x, last_y to prevent unintended lines
    when the mouse is moved without the button pressed.
    """
    global drawing_enabled, last_x, last_y
    if event.num == 1:
        drawing_enabled = False
        last_x, last_y = None, None

def export_canvas_as_image():
    """
    Exports the current content of the Tkinter canvas to an image file.
    It first saves the canvas as a PostScript (.ps) file, then converts this PostScript file into a grayscale PNG image using the Pillow library.
    The PNG image is saved in OUTPUT_FOLDER.
    """
    ps_path = os.path.join(OUTPUT_FOLDER, "canvas_output.ps")
    png_path = os.path.join(OUTPUT_FOLDER, "exported_canvas_fixed_size.png")

    # Tkinter's postscript method captures the canvas content
    canvas.postscript(file=ps_path, colormode='color')
    print(f"Canvas exported to PostScript at {ps_path}")

    # Convert the PostScript file to PNG using Pillow (PIL).
    # 'L' mode ensures it's converted to grayscale, compatible with the MNIST-based model.
    try:
        image = Image.open(ps_path)
        image = image.convert('L') # Convert to grayscale
        image.save(png_path)
        print(f"Converted to PNG and saved at {png_path}")
        return png_path # Return the path to the saved PNG image
    except Exception as e:
        print(f"Error converting PostScript to PNG: {e}")
        return None

def process_and_save_digits_from_image(image_path, output_dir):
    """
    Processes a given input image (expected to contain handwritten digits) to isolate and normalize individual digits. Each extracted digit is then saved as a 28x28 grayscale PNG image in the specified output directory.
    This prepares the images for input into the MNIST-trained model.
    """
    extracted_digit_paths = []
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"WARNING: Couldn't read {image_path}")
        return []

    # Invert colors, since MNIST digits are white on black but the canvas is black on white.
    if np.mean(image) > 127:
        image = cv2.bitwise_not(image)

    # Apply Otsu's thresholding to convert to a binary image, separating digits from background.
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours - distinct shapes (digits) in the binary image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found in {image_path}, skipping this image.")
        return []

    # Sort contours from left to right to maintain the correct order of digits in a number
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt) # Get bounding box coordinates
        area = cv2.contourArea(cnt) # Calculate contour area

        # Filter out very small or very thin/wide contours that are unlikely to be digits. These parameters should be tuned based on personal writing style
        if area < 50 or w < 10 or h < 10:
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            continue

        # Crop the digit using its bounding box
        cropped = image[y:y+h, x:x+w]
        if cropped.size == 0: # Skip empty crops
            continue

        # Resize the cropped digit to fit within a 20x20 pixel box while maintaining aspect ratio, then pad it to a 28x28 canvas. This mimics the preprocessing of MNIST dataset.
        max_dim = max(cropped.shape)
        scale = 20.0 / max_dim
        new_w, new_h = int(cropped.shape[1] * scale), int(cropped.shape[0] * scale)
        if new_w == 0 or new_h == 0:
            continue

        resized_digit = cv2.resize(cropped, (new_w, new_h), interpolation = cv2.INTER_AREA)
        canvas_28x28 = np.zeros((28, 28), dtype=np.uint8) # Create a black 28x28 canvas
        pad_x = (28 - new_w) // 2
        pad_y = (28 - new_h) // 2
        canvas_28x28[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_digit # Place resized digit on canvas
        normalized_digit = canvas_28x28 / 255.0 # Normalize pixel values to [0, 1]

        # Save the processed 28x28 digit image
        save_path = os.path.join(output_dir, f"digit_{i}.png")
        cv2.imwrite(save_path, (normalized_digit * 255).astype(np.uint8))
        extracted_digit_paths.append(save_path) # Convert back to 0-255 for saving

    return extracted_digit_paths

def predict_digits(digit_image_paths):
    """
    Takes a list of paths to individual digit images, loads them, and uses the loaded TensorFlow model to predict the numerical value of each digit.
    The predictions are concatenated into a single string.
    """
    predicted_number_string = ""
    for file_path in digit_image_paths:
        if os.path.isfile(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"WARNING: Could not read processed digit image {file_path}")
                continue
            # Reshape the image to (1, 28, 28, 1) for model input (batch_size, height, width, channels)
            img = img.reshape(1, 28, 28, 1)
            # Predict the digit. verbose=0 suppresses prediction output for each image
            prediction = model.predict(img, verbose=0)
            # Get the index of the highest probability, which corresponds to the predicted digit
            predicted_number_string += str(np.argmax(prediction))
    return predicted_number_string

def automate_web_with_number(number_to_enter, input_box_locator=None):
    """
    Uses Selenium to input the predicted number into the designated web element on the Zetamac website and then submits it.

    Arguments:
        number_to_enter (str): The numerical string to be entered.
        input_box_locator (tuple): A tuple (Element type, locator value) to find the input box.
    """
    print(f"Attempting to enter number: {number_to_enter}")
    try:
        if input_box_locator:
            # Wait for the input box to be visible and interactable
            answer_box = wait.until(EC.visibility_of_element_located(input_box_locator))
            answer_box.clear() # Clear any existing text in the input box
            answer_box.send_keys(number_to_enter) # Send the predicted number
            answer_box.send_keys(Keys.RETURN) # Submit the answer by pressing RETURN
            print(f"Entered '{number_to_enter}' and pressed RETURN.")
        else:
            print("No input box locator provided.")
    except Exception as e:
        print(f"Failed to input number into Zetamac: {e}")

def show_toast_notification(root, message, duration=1000):
    """
    Displays a small, temporary "toast" notification within the Tkinter window.
    This provides immediate feedback to the user without interrupting the workflow.

    Args:
        root (tk.Tk): The parent Tkinter window.
        message (str): The text message to display in the toast.
        duration (int): The duration (in milliseconds) before the toast disappears.
    """
    toast = tk.Toplevel(root)
    toast.overrideredirect(True)  # Remove window border
    toast.configure(bg = "#333333")  # Dark background
    toast.attributes("-topmost", True)  # Notification on top

    # Notification position: bottom-right corner of the root window
    x = root.winfo_rootx() + root.winfo_width() - 250
    y = root.winfo_rooty() + root.winfo_height() - 100
    toast.geometry(f"180x40+{x}+{y}") # Fixed size and position

    label = tk.Label(toast, text=message, bg="#333333", fg="white", font=("Segoe UI", 10))
    label.pack(fill="both", expand=True, padx=10, pady=5)

    # Notification disappears after the specified duration.
    toast.after(duration, toast.destroy)

def overall_control(event):
    """
    This is the core control function, triggered by a double-click on the canvas.
    This function controls the entire sequence: drawing cessation, canvas export, digit preprocessing, model prediction, web automation, and canvas clearing.
    """
    global drawing_enabled, last_x, last_y
    if event.num == 1: # Confirm left-button double click (end answer)
        drawing_enabled = False # Stop further drawing
        last_x, last_y = None, None # Reset drawing coordinates
        print("Drawing ended by double-click. Processing image...")

        # If empty canvas, return message
        if not canvas.find_all():
            print("Canvas is empty. No drawing to export.")
            canvas.delete("all")
            return

        # Step 1: Export the user's drawing from the canvas
        canvas_image_path = export_canvas_as_image()

        # Step 2: Clear previously processed digits to avoid mixing data from different drawings
        for f in os.listdir(PROCESSED_DIGITS_FOLDER):
            os.remove(os.path.join(PROCESSED_DIGITS_FOLDER, f))

        # Step 3: Process the exported canvas image to extract and normalize individual digits
        extracted_digit_paths = process_and_save_digits_from_image(canvas_image_path, PROCESSED_DIGITS_FOLDER)
        if not extracted_digit_paths:
            print("No valid digits found in the drawing after processing.")
            canvas.delete("all")
            return

        # Step 4: Use the pre-trained model to predict the numerical value from the extracted digits
        predicted_number = predict_digits(extracted_digit_paths)
        print(f"Predicted number: {predicted_number}")

        # Step 5: If a number was successfully predicted, automate its entry into the web game
        if predicted_number:
            input_locator = (By.CLASS_NAME, "answer") # Defines how to find the answer input box.
            automate_web_with_number(predicted_number, input_box_locator=input_locator)
        else:
            print("No number predicted, skipping web automation.")

        # Step 6: Clear the canvas for the next drawing input from the user
        canvas.delete("all")
        print("Canvas cleared for new input.")

        # Provide a visual confirmation to the user that the answer was submitted
        show_toast_notification(window, "Answer submitted")

# --- Main Tkinter Window ---
# Create the main Tkinter window
window = tk.Tk()
window.title("Answer Input")

# Configure the window to stay on top of other applications
window.lift()
window.attributes('-topmost', True)

# Create the canvas widget where users will draw
canvas = tk.Canvas(window, bg="white", width=CANVAS_DRAWING_WIDTH, height=CANVAS_DRAWING_HEIGHT, highlightthickness=0, bd=0)
canvas.pack()

# Bind mouse events to their respective callback functions
canvas.bind("<Motion>", while_drawing) # Mouse movement for drawing lines
canvas.bind("<ButtonPress-1>", start_drawing) # Left mouse button press to start drawing
canvas.bind("<ButtonRelease-1>", stop_drawing) # Left mouse button release to stop drawing
canvas.bind("<Double-Button-1>", overall_control) # Double-click to trigger processing and submission

# Start the Tkinter event loop. This keeps the GUI responsive
window.mainloop()
