# Interactive-FT-Image-Mixer-Desktop-Application

This interactive image processing tool visually explores the roles of **magnitude** and **phase**, as well as **real** and **imaginary** components in the Fourier Transform of 2D signals (images). Users can load multiple images, extract and combine Fourier components, apply region-specific operations, and observe the reconstruction effects in real time. The tool is designed to support experimentation and enhance understanding of image analysis.

---

## Applications

* Visualize how each **frequency-domain component** contributes to the final image
* Visualize how **low vs. high frequency** content influences image features
* Understand the **importance of phase** in image structure
* Visualize the effect of **ROI-based fusion** on localized image structures

---

## Features

### 1. Image Viewers

* Load **up to 4 images** side-by-side
* Automatically converts colored images to **grayscale**
* Unifies image dimensions to the **smallest among the loaded**

### 2. Frequency Component Visualization

* For each image, using a combo box, switch between:
  * **FT Magnitude**
  * **FT Phase**
  * **FT Real**
  * **FT Imaginary**
* Displayed alongside the original image

### 3. Interactive Mixing

* Real-time mixing using the **inverse FFT** of selected components
* Choose to mix by:
  * **Magnitude + Phase**, or
  * **Real + Imaginary**
* Assign **custom weights** to each image and each component using sliders
* Two **output viewers** with selectable destination for the mixed result

### 4. Region-Based Control

* Select a **rectangular region** of the FT component to use in mixing
* Choose to mix with:
  * **Inner region** of the box, or
  * **Outer region** of the box
* Shared region across all images for fair comparison
* Visual overlay on selected region

### 5. Brightness / Contrast Adjustment

* Use **mouse drag** in any image viewer to interactively adjust:
  * Left/Right → Brightness
  * Up/Down → Contrast

---

## Logging for Debugging

* Using the ```logging``` package of Python, all major events and user actions are logged in `app.log`
* Helps trace bugs and understand unexpected behavior

---

## Demo Video

<div align="center">
  <a href="https://www.youtube.com/watch?v=gJun0gbB3IM&autoplay=1" target="_blank">
    <img src="https://img.youtube.com/vi/gJun0gbB3IM/0.jpg" alt="Demo Video Thumbnail" width="640">
  </a>
</div>

### Key Takeaways & Visual Observations

- Phase information carries most of the structural and positional cues in an image, whereas magnitude contributes less to shape perception.
- Low-frequency components are essential for reconstructing the overall shape and smooth gradients.
- High-frequency regions introduce fine details and texture but can make images appear noisy if used incorrectly.
- Real-time fusion and visualization help demonstrate how each frequency-domain component contributes to the final image.

---

## How to Run

### Requirements

- Python 3.7+
- PyQt5
- numpy
- scipy
- Pillow
- opencv-python
- pyqtgraph
- qimage2ndarray

### 1. Install dependencies:

   ```bash
   pip install pyqt5 numpy scipy pillow opencv-python pyqtgraph qimage2ndarray
   ```
   
### 2. Run the application:

   ```bash
   python main.py
   ```
