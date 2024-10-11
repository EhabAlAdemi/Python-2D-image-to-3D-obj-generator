import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import pyvista as pv

# Setting up the depth estimation model
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# Function to perform depth estimation and 3D model generation
def generate_3d_model(image_path):
    image = Image.open(image_path)
    
    # Resizing the image
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)

    # Preprocessing the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Predicting depth
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Post-processing depth output
    pad = 16
    depth_map = predicted_depth.squeeze().cpu().numpy() * 1000.0
    depth_map = depth_map[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    # Visualizing depth
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax[1].imshow(depth_map, cmap='plasma')
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.show(block=False)

    # 3D mesh generation using Convex Hull
    depth_map_rescaled = depth_map / np.max(depth_map)  # Normalize
    points = np.column_stack(np.nonzero(depth_map_rescaled))
    points_3d = np.hstack([points, depth_map_rescaled[points[:, 0], points[:, 1]].reshape(-1, 1)])
    hull = ConvexHull(points_3d)

    # Create 3D mesh using pyvista
    mesh = pv.PolyData(points_3d[hull.vertices])
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue')
    plotter.show()

    # Option to save the OBJ file
    save_path = filedialog.asksaveasfilename(defaultextension=".obj", filetypes=[("OBJ files", "*.obj")])
    if save_path:
        mesh.save(save_path)

# Function to browse and select an image
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        generate_3d_model(file_path)

# Setting up the GUI
def create_gui():
    window = tk.Tk()
    window.title("3D Model Generator using Depth Estimation")

    # Description label
    description = """This project uses a pre-trained model to estimate depth from a 2D image 
    and generates a 3D model based on the depth map."""
    label = tk.Label(window, text=description, wraplength=400, justify="left")
    label.pack(pady=10)

    # Button to choose image
    select_image_btn = tk.Button(window, text="Choose Image", command=select_image, padx=10, pady=5)
    select_image_btn.pack(pady=20)

    # Start the GUI loop
    window.geometry("500x300")
    window.mainloop()

# Run the GUI
if __name__ == "__main__":
    create_gui()
