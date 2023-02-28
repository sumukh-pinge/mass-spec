''' import logging, time, os, glob
from typing import  Optional

import tqdm
import numpy as np
import numba as nb
import pandas as pd

import bottleneck as bn
from joblib import Parallel, delayed

from config import Config
from utils import load_mgf_file, export_mgf_file
 '''

import numpy as np
import pyteomics.mgf
import matplotlib.pyplot as plt

# Load the MGF file
filename = 'testfile.mgf'
spectra = pyteomics.mgf.read(filename)


''' 
# Get the first spectrum
spectrum = next(spectra)
print(spectrum.get('m/z array', None))

# Extract the m/z values and intensities from the spectrum
mz_values = []
intensities = []

mz_array = spectrum.get('m/z array', None)
intensity_array = spectrum.get('intensity array', None)

if mz_array is not None and intensity_array is not None:
    mz_values = mz_array
    intensities = intensity_array
else:
    print('Warning: Missing m/z array or intensity array')

# Plot the spectrum
plt.plot(mz_values, intensities)
plt.xlabel('m/z')
plt.ylabel('Intensity')
plt.title('First Spectrum')
plt.show()


 '''
 
''' with open('hypervector.txt', 'w') as outfile:
    # Loop over all spectra in the file
    for spectrum in spectra:
        # Extract the m/z values and intensities from the spectrum
        mz_array = spectrum.get('m/z array', None)
        intensity_array = spectrum.get('intensity array', None)

        if mz_array is not None and intensity_array is not None:
            # Create a hypervector for the spectrum
            hypervector = np.zeros(2048)
            for mz, intensity in zip(mz_array, intensity_array):
                hypervector[int(mz)] = intensity

            # Write the hypervector to the output file
            outfile.write(' '.join(str(x) for x in hypervector) + '\n')
        else:
            print('Warning: Missing m/z array or intensity array') '''
            
            
 # Define the preprocessing parameters
base_peak_percentage = 1.0
min_valid_peaks = 5
mass_range = 250
max_peaks = 50

# Open the output file for writing
with open('processed_spectra.mgf', 'w') as outfile:
    # Loop over all spectra in the file
    for spectrum in spectra:
        # Extract the m/z values and intensities from the spectrum
        mz_array = spectrum.get('m/z array', None)
        intensity_array = spectrum.get('intensity array', None)

        if mz_array is not None and intensity_array is not None:
            # Remove peaks below base peak percentage
            base_peak_intensity = np.max(intensity_array)
            threshold = base_peak_intensity * base_peak_percentage / 100.0
            valid_indices = np.where(intensity_array >= threshold)[0]
            mz_array = mz_array[valid_indices]
            intensity_array = intensity_array[valid_indices]

            # Remove spectra with too few valid peaks or mass range too small
            mass_range_check = np.max(mz_array) - np.min(mz_array)
            if len(mz_array) < min_valid_peaks or mass_range_check < mass_range:
                continue

            # Retain only the top max_peaks peaks and normalize intensities
            peak_indices = np.argsort(intensity_array)[-max_peaks:]
            mz_array = mz_array[peak_indices]
            intensity_array = intensity_array[peak_indices]
            norm = np.linalg.norm(intensity_array)
            intensity_array = intensity_array / norm

            # Write the processed spectrum to the output file
            spectrum['m/z array'] = mz_array
            spectrum['intensity array'] = intensity_array
            pyteomics.mgf.write([spectrum], outfile)
        else:
            print('Error')
            
            