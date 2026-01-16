# Radar Signal Processing Portfolio

### 📡 Overview
This repository contains algorithmic implementations for **Synthetic Aperture Radar (SAR)** processing and **Interferometric SAR (InSAR)** analysis. 

The codebase demonstrates a complete signal processing pipeline. The focused SAR processor moves from raw range-compressed data to focused imagery. 
The interferogram project moves from raw dem and slc data to derived geophysical visualizations like Digital Elevation Models (DEMs) and surface deformation maps.

### ⚠️ Prerequisites: Data Download Required
Due to GitHub's file size limits, some of the raw radar data files (>100 MB) are hosted externally. To run the scripts, you must download the dataset.

1. **Clone this repository**:
   Run this command in the terminal window to download this repository
   
   ```bash
   git clone https://github.com/LBplano/Radar_Portfolio.git
   
2. **Download the Raw Data**:
   👉 NAVIGATE TO THE FOLLOWING URL TO DOWNLOAD THE REQUISITE RADAR FILES https://drive.google.com/drive/folders/1V8oYZuRLsGrPmgkwmyWR6mNQzcI2rK9I?usp=sharing
   
3. **Install**:
   Move slc.dem, slc1.dat, and slc2.dat into the Interferogram_Project folder (the same directory where Final.py is located). Move ersRC.hw4 into the Focused_SAR_Processor Folder.

### 📂 Project 1: Focused SAR Image Formation
**Goal:** Convert raw, range-compressed echo data into high-resolution imagery.

* **Input:** Raw range-compressed radar data.
* **Algorithm:** Implements azimuth compression and matched filtering for greater resolution.
* **Key Techniques:**
    * Fast Fourier Transforms (FFT/IFFT) for frequency domain analysis.
    * Doppler shift algorithms.
    * Image generation techniques.

### 🌍 Project 2: InSAR Pipeline (Interferogram Project)
**Goal:** Process Single Look Complex (SLC) image pairs to measure topography and ground movement.

* **Input:** Coregistered SLC image pairs (`slc1.dat`, `slc2.dat`).
* **Outputs:**
    * **Interferogram:** Phase difference map showing range differences between two satellites orbiting in tandem.
    * **DEM (Digital Elevation Model):** Topographic reconstruction from pure elevation data. 
    * **Deformation Map:** Sub-wavelength measurement of surface displacement.
* **Technical Highlights:**
    * **Resolution Enhancement:** Multi-looking final data based on viewing geometry to improve image resolution.
    * **Phase Processing:** Complex conjugate multiplication and phase wrapping visualizations.
    * **Geometric Reconstruction:** Derive satellite viewing parameters for an entire data set.  

---

### 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** `NumPy` (Vectorized matrix operations), `Matplotlib` (Signal visualization), `OS` (File I/O).

### 🚀 Usage
**To Run Projects 1 and 2 (SAR Processor and InSAR Pipeline):**
```bash
cd Focused_SAR_Project
python Focused_SAR_Processor.py
--------------------------------
cd Interferogram_Project
python Final.py