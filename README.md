# Integrating Multi-Source Satellite Time-Series and Continuous Wavelet Transform for Rice Cropping System Mapping using CNNs

This repository contains sample data and information related to the research paper titled **"Integrating Multi-Source Satellite Time-Series and Continuous Wavelet Transform for Rice Cropping System Mapping using Convolutional Neural Networks"**.

This study proposes an innovative approach to classify rice cropping systems at the pixel level in **Suphan Buri, Thailand**. By integrating **Continuous Wavelet Transform (CWT)** and **Convolutional Neural Networks (CNNs)**, we accurately identify four distinct rice cropping intensities:
* **SRC:** Single-crop system
* **DRC:** Double-crop system
* **HRC:** Two-and-a-half crop system
* **TRC:** Triple-crop system

## Dataset Description (Sample)
* **Satellite Imagery:** Sentinel-1 (SAR) and Sentinel-2 (MSI) data acquired via Google Earth Engine (GEE).
* **Vegetation Indices:** Enhanced Vegetation Index (**EVI**) and Normalized Difference Red Edge Index (**NDRE**).
* **SAR Features:** Vertical-Horizontal (**VH**) polarization from Sentinel-1.
* **Temporal Coverage:** Time-series data from April 2023 to March 2025.

## Methodology Summary
1.  **Preprocessing:** Smooth daily vegetation profiles were reconstructed using **Savitzky-Golay (SG) filtering** and linear interpolation to handle cloud contamination.
2.  **Feature Extraction:** The ID temporal signals were transformed into 2D **CWT scalograms** using the Morlet wavelet.
3.  **Classification:** A CNN model extracts robust patterns from 3-channel stacked scalograms (EVI, NDRE, and VH) to determine the cropping system.
