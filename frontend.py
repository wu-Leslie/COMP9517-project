import tkinter as tk
from tkinter import filedialog
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import UseYolov5


# Define some global variables
WIDTH = 900
HEIGHT = 750
INPUT_IMAGE_PATH = None
CANVAS = None

def update_result_label_text(text):
    result_label.config(text=text, wraplength=WIDTH/2)

def update_display_image(image, processed_image):
    global CANVAS
    
    # Create a new canvas with the updated image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Display the original image on the left (ax1)
    ax1.imshow(image)
    ax1.set_title('Original')
    ax1.axis('off')  # Hide axis ticks and labels
    
    # Display the processed image on the right (ax2)
    ax2.imshow(processed_image)
    ax2.set_title('Processed')
    ax2.axis('off')  # Hide axis ticks and labels

    # Adjust the spacing between the images
    plt.tight_layout()
    
    new_canvas = FigureCanvasTkAgg(fig, master=root)
    
    # Replace the old canvas with the new one
    if CANVAS is not None:
        CANVAS.get_tk_widget().pack_forget()
    
    CANVAS = new_canvas
    CANVAS.draw()
    CANVAS.get_tk_widget().pack(side=tk.TOP, pady=5)
    
    plt.close()

    print("image display changed")

def check_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def process_image():
    global INPUT_IMAGE_PATH
    file_path = filedialog.askopenfilename()
    if show_text.get():
        update_result_label_text("Loading R-CNN...")
    else:
        update_result_label_text("Loading Yolo...")

    if file_path:
        INPUT_IMAGE_PATH = file_path
        image = cv2.imread(file_path)
        
        # Check the state of the switch (show_text) to determine the type of image processing
        if show_text.get():
            # R-CNN
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            def process_RCNN():
                try:
                    
                    update_display_image(image, processed_image)
                except Exception as e:
                    error_message = f"Error: Failed to process image in RCNN: {str(e)}"
                    error_label.config(text=error_message, wraplength=WIDTH/2)
                    return
                
                # Update the GUI when the function has finished
                update_result_label_text(f"Image Shown: {file_path}")
            
            # Schedule the process_yolo function to be called after a short delay
            root.after(100, process_RCNN)
        else:
            # Yolo-V5

            def process_yolo():
                process_image = image.copy()
                try:
                    processed_image = UseYolov5.detect_and_label_image_Yolov5(process_image)
                    update_display_image(image, processed_image)
                except Exception as e:
                    error_message = f"Error: Failed to process image in Yolo: {str(e)}"
                    error_label.config(text=error_message, wraplength=WIDTH/2)
                    return
                
                # Update the GUI when the function has finished
                update_result_label_text(f"Image Shown: {file_path}")
            
            # Schedule the process_yolo function to be called after a short delay
            root.after(100, process_yolo)
            
def update_text():
    if show_text.get():
        # result_label.config(text="Processing Complete - R-CNN")
        switch_label.config(text="Method of Image Processing: R-CNN")
    else:
        # result_label.config(text="Processing Complete - Yolo-V5")
        switch_label.config(text="Method of Image Processing: Yolo-V5")
        

def download_processed_image():
    global INPUT_IMAGE_PATH
    if INPUT_IMAGE_PATH is None:
        error_message = f"Error: No file has put in yet, please select an image first."
        error_label.config(text=error_message, wraplength=WIDTH/2)
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
    if file_path:
        processed_image = cv2.imread("processed_image.jpg")
        cv2.imwrite(file_path, processed_image)
    else:
        error_message = f"Error: Failed to save image, file path are incorrect, please try again."
        error_label.config(text=error_message, wraplength=WIDTH/2)
        
# Create the main GUI window
root = tk.Tk()
root.title("COMP9517 Group Project GUI")

root.tk_setPalette(background="#303030", foreground="white", activeBackground="#404040", activeForeground="white")

# Set the size of the main window
# root.geometry("800x500")
root.geometry(f"{WIDTH}x{HEIGHT}")

# Create a label for the text
text_label = tk.Label(root, text="COMP9517 Image Detection and Classification", font=("Arial", 20))
text_label.pack(pady=20)

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=process_image)
select_button.pack(pady=10)

# Create a frame to hold to result 
result_frame = tk.Frame(root, bd=2, relief="groove")
result_frame.pack(pady=10, padx=10, fill="x")
# Create a label for display result
result_label = tk.Label(result_frame, text="Result = ", fg="green")
result_label.pack(pady=5, padx=5, fill="x")

# Create a frame to hold the error label
error_frame = tk.Frame(root, bd=2, relief="groove")
error_frame.pack(pady=10, padx=10, fill="x")
# Create a label for displaying errors 
error_label = tk.Label(error_frame, text="Error = None", fg="red")
error_label.pack(pady=5, padx=5, fill="x")

# Create a switch (Checkbutton) to toggle text
show_text = tk.BooleanVar()
show_text.set(True)  # Default value (ON)
switch_label = tk.Label(root, text="Method of Image Processing: R-CNN")
switch_label.pack(side=tk.LEFT)
switch_button = tk.Checkbutton(root, variable=show_text, command=update_text)
switch_button.pack(side=tk.LEFT)

# Create a button to download the processed image
download_button = tk.Button(root, text="Download Processed Image", command=download_processed_image)
download_button.pack(pady=10)

# Create a footer
footer_frame = tk.Frame(root, bd=2, relief="groove")
footer_frame.pack(side=tk.BOTTOM, fill="x")
# Create a label for footer message
footer_label = tk.Label(footer_frame, text="COMP9517 Group Project by Raphael, Lesile, Toran, Xinran, Lili", fg="gray")
footer_label.pack(pady=5, fill="x")


# Start the GUI event loop
root.mainloop()

