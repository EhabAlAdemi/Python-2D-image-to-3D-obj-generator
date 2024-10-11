# 3D Model Generator from Depth Estimation

## Overview

This project uses a pre-trained depth estimation model to convert 2D images into 3D models. It leverages a pre-trained model from the Hugging Face Transformers library, and generates a 3D mesh based on the predicted depth map. The project includes a user-friendly graphical interface (GUI) where users can upload an image, visualize the depth estimation, and save the generated 3D model as an `.obj` file.

## Features

- **Depth Estimation**: Utilizes the `GLPNForDepthEstimation` model from Hugging Face to predict depth from a 2D image.
- **GUI Interface**: A simple Tkinter-based interface allows users to select an image and save the 3D model with ease.
- **3D Model Generation**: Converts the depth map into a 3D mesh and visualizes it using PyVista.
- **OBJ Export**: Allows users to save the generated 3D model in `.obj` format for further use in 3D applications.

## How It Works

1. **Image Selection**: The user selects an image using a file dialog from the graphical interface.
2. **Depth Prediction**: The pre-trained `GLPNForDepthEstimation` model processes the image and generates a depth map.
3. **3D Mesh Creation**: A 3D point cloud is created based on the depth map, and a Convex Hull algorithm is applied to create a mesh from the 3D points.
4. **Visualization**: Both the original image and the generated depth map are displayed, and the 3D mesh is visualized in a 3D viewer.
5. **Saving the Model**: The user can save the generated 3D model as an `.obj` file using a file save dialog.

## Installation

### Prerequisites

To run the project, you need to install the following dependencies:

1. **Python 3.9 or higher**: Make sure Python is installed on your system. You can download it from [here](https://www.python.org/downloads/).

2. **Required Python Libraries**:
   - `matplotlib`
   - `Pillow`
   - `torch`
   - `transformers`
   - `scipy`
   - `pyvista`
   - `tkinter`

To install the required libraries, run:

```bash
pip install matplotlib Pillow torch transformers scipy pyvista tk
```

### Model Download
This project uses a pre-trained depth estimation model hosted on Hugging Face. The model will be automatically downloaded when you run the program for the first time. No additional steps are needed.

### GUI Instructions

- Upon running the script, a GUI window will appear.
- A description of the project will be displayed, along with a button labeled "Choose Image."
- Click the **Choose Image** button to browse and select a `.jpg` or `.png` image from your files.
- Once the image is selected, the depth estimation model will process the image, and a 3D model will be generated.
- After visualization, you will be prompted to **save the 3D model** in `.obj` format.

## Known Issues

- The quality of the generated 3D model may vary depending on the image used. For best results, try images with clear objects and backgrounds.
- Convex Hull may produce overly simplistic 3D models. Depending on your requirements, further post-processing might be needed.
