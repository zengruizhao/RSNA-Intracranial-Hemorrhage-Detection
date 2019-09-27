import os
import pydicom


def get_dicom_data(filepath):
    # Return the dicom raw data of a given file
    return pydicom.dcmread(filepath)