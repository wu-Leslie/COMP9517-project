import tkinter as tk
from tkinter import filedialog
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

WIDTH = 800
HEIGHT = 500
def process_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        # Apply our model here
        
        # Check the state of the switch (show_text) to determine the type of image processing
        if show_text.get():
            # Apply one type of image processing algorithm (e.g., convert to grayscale)
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Apply a different type of image processing algorithm (e.g., apply edge detection)
            processed_image = cv2.Canny(image, 100, 200)

        try:
            # Display the processed image using matplotlib
            # Create separate figures for the original and processed images
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            
            # Display the original image on the left (ax1)
            ax1.imshow(image)
            ax1.axis('off')  # Hide axis ticks and labels
            
            # Display the processed image on the right (ax2)
            ax2.imshow(processed_image, cmap='gray')
            ax2.axis('off')  # Hide axis ticks and labels

            # Adjust the spacing between the images
            plt.tight_layout()
            
            # Convert the matplotlib figure to a Tkinter canvas
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, pady=10)
            plt.close()
            
            result_message = f"Image Shown: {file_path}"
            result_label.config(text=result_message, wraplength=WIDTH/2)
            
        except Exception as e:
            error_message = f"Error: Failed to display image: {str(e)}"
            error_label.config(text=error_message, wraplength=WIDTH/2)
            
def update_text():
    if show_text.get():
        # result_label.config(text="Processing Complete - R-CNN")
        switch_label.config(text="Method of Image Processing: R-CNN")
    else:
        # result_label.config(text="Processing Complete - Yolo-V5")
        switch_label.config(text="Method of Image Processing: Yolo-V5")
        
# Create the main GUI window
root = tk.Tk()
root.title("COMP9517 Group Project GUI")

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

# Start the GUI event loop
root.mainloop()

