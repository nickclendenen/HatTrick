"""Run a scan of optical spectrum vs driving current for a 2-element array."""

# Package for time/date info
import datetime
# Package for creating directory if it doesn't exist already
import os
from random import sample
# Package for progress bar
import tqdm
# Package to save run info
import h5py
# Instrument control packages
from instrument_scripts import yokogawa
from instrument_scripts import keysightB2912a
from instrument_scripts import pm100d
from instrument_scripts import thorcam_v2
#from instrument_scripts import thorlabs_MCM3000
from instrument_scripts.MCM301_COMMAND_LIB import *

# Current rastering functions
from math_scripts import raster2d
# Math
import numpy as np
from scipy import ndimage, signal
# Save Function
from Utilities import file_save_routine
# Plotting utilities
import matplotlib.pyplot as plt
from plotting import spectral
from plotting import liv3d_plot
from plotting import ff_plot

def check_input(istart, istop, istep, current_limit):
    """
    istart (numbers): Starting current value (in Amps)
    istop (numbers): Ending current value (in Amps)
    istep (numbers): Current sampling step (in Amps)
    current_limit (number): Maximal permitted current (in Amps) (default=20e-3)
    """
    assert istart < istop, "Start current is more than stop current."
    assert istep < istop - istart, "Step current is larger than stop-start current."
    assert istop <= current_limit, "Stop current is greater than than the current limit (20mA)"
    assert istep <= 0.1e-3, "istep cannot be larger than 0.1 mA"

def initialize_equipment(istart, istep, keysight23, keysight24, sense, center, span, scan_width, osa, pwr_wavelength, power_meter, camera, x_left_roi, y_top_roi, x_right_roi, y_bottom_roi, display_state, controller):
    keysight23.setup()
    keysight24.setup()
    keysight23.set_data_format("REAL,64")
    keysight24.set_data_format("REAL,64")
    keysight23.set_as_current_source(1)
    keysight24.set_as_current_source(1)
    keysight23.set_current(0.1e-3) if istart > 0.1e-3 else keysight23.set_current(istart)
    keysight24.set_current(0.1e-3) if istart > 0.1e-3 else keysight24.set_current(istart)
    keysight23.output_on()
    keysight24.output_on()
    if not display_state: 
        keysight23.display_state(0)
        keysight24.display_state(0)
    # Ensure that the Keithley is at an appropriate current
    if istart > 0.1:
        ramp = np.arange(0.1e-3, istart+istep/2, istep)
        for i in ramp:
            keysight23.set_current(i)
            keysight24.set_current(i)
        keysight23.set_current(istart)
        keysight24.set_current(istart)
    
    # Set the Camera 
    x_left_roi, y_top_roi, x_right_roi, y_bottom_roi = camera.set_roi(x_left_roi, y_top_roi, x_right_roi, y_bottom_roi)
    camera.prepare_for_taking_pictures()

    #Near-Field to Far-Field
    #OLD MCM3000: controller.move_um(2, -100)

    devs = MCM301.list_devices()
    if len(devs) <= 0:
        print('There is no devices connected')
        exit()
    device_info = devs[0]
    sn = device_info[0]

    # Set the slot number for the X-axis motor (replace with the correct slot, e.g., 4)
    x_axis_slot = 4

    # Open the connection to the device
    if controller.open(sn, 115200, x_axis_slot) >= 0:
        # Convert 593 microns to encoder counts
        FarFieldOffset = 100
        microns = 588 + FarFieldOffset
        
        encoder_count = [0]
        
        if controller.convert_nm_to_encoder(x_axis_slot, microns * 1000, encoder_count) == 0:
            # Move the X-axis motor to the target position (forward move)
            controller.move_absolute(x_axis_slot, encoder_count[0])


    # Set the Power Meter 
    power_meter.setup()
    power_meter.set_wavelength(pwr_wavelength)

    # Set the OSA scan settings
    osa.setup()
    osa.set_sense_mode(sense)
    osa.set_center_wavelength(center)
    osa.set_wavelength_span(span)
    osa.set_scan_spectral_width(scan_width)
    osa.auto_calibration(0)
    osa.set_data_format("REAL,64")
    if not display_state: osa.display_state("OFF")
    return x_left_roi, y_top_roi, x_right_roi, y_bottom_roi

def get_data(istart, istop, istep, username, rasterfunction, keysight23, keysight24, sense, center, span, scan_width, osa, power_meter, camera, x_left_roi, y_top_roi, x_right_roi, y_bottom_roi, file_path):
    """
    istart (numbers): Starting current value (in Amps)
    istop (numbers): Ending current value (in Amps)
    istep (numbers): Current sampling step (in Amps)
    username (string): Name of person running scan for metadata usage (default="NoNameUser")
    keysight: keysight object
    osa: osa object
    power_meter: power_meter object 
    camera: camera object
    file_path (string): file path for where to save the hdf5 file
    """
    # Create the range of currents (if statement is for the behavoir of arange not including endpoint like the SPA will)
    current_range = np.round(np.arange(istart, istop, istep), 5)
    if current_range[-1] + istep == istop:
        current_range = np.concatenate([current_range, [istop]])

    current_list = rasterfunction(current_range)
    pbar = tqdm.tqdm(total=len(current_list))

    # Unzip i1 and i2 lists
    i1_list, i2_list = np.array(current_list).T

    # Loop iterator for auto-calibration
    i = 0

    # Only get once
    freq_data = osa.get_scan_x()

    with h5py.File(file_path, "w") as outfile:
        outfile.attrs['Date Time'] = datetime.datetime.now().ctime()
        outfile.attrs['Sense'] = sense
        outfile.attrs['Spectral Scan Width'] = scan_width
        outfile.attrs['User'] = username
        outfile.attrs['Center Wavelength'] = center
        outfile.attrs['Span Wavelength'] = span
        outfile.create_dataset("I1", data=i1_list)
        outfile.create_dataset("I2", data=i2_list)
        # Run a scan to get the number of data points
        num_data_points = int(osa.get_num_data_points())
        spectral_data = outfile.create_dataset("Spectral Output", (len(current_list), num_data_points, 2))
        power_data = outfile.create_dataset("Light Output", len(current_list))
        voltage_1 = outfile.create_dataset("V1", len(current_list))
        voltage_2 = outfile.create_dataset("V2", len(current_list))
        pr = outfile.create_dataset("pr", (len(current_list),))
        freq = outfile.create_dataset("freq", (len(current_list), 2))
        phase = outfile.create_dataset("phase", (len(current_list),))
        frames = outfile.create_dataset("frame_data", (len(current_list), y_bottom_roi-y_top_roi+1,
                                                x_right_roi-x_left_roi+1,), dtype='u2')
        

        for i_vals in current_list:
            if i % 1000 == 0:
                osa.auto_calibration(2)
            i1, i2 = i_vals
            # Check to see if there is still a connection
            if keysight23.get_voltage() >= 9.98 or keysight24.get_voltage() >= 9.98:
                keysight23.output_off()
                keysight24.output_off()
                print("Current keysight237 current = " + str(i1))
                print("Current Keithley5 current = " + str(i2))
                input("Compliance voltage hit! Please check wiring and hit ENTER to continue.")
                input("Did you set the Keithleys back to current when stopped? Hit ENTER to continue.")
                keysight23.output_on()
                keysight24.output_on()
            # Set current on power source
            keysight23.set_current(i1)
            keysight24.set_current(i2)
            voltage_1[i] = keysight23.get_voltage(1)
            voltage_2[i] = keysight24.get_voltage(1)
            # Run a scan
            osa.run_single_scan()

            level_data = osa.get_scan_y()
            # Format data into spectral_data frame
            combined_data = np.stack((freq_data, level_data)).T
            spectral_data[i] = combined_data
            
            # Store power data
            power_data[i] = float(power_meter.get_power())

            # Get a frame from the camera
            img = camera.capture_image()
            # Check if any pixels are saturated or not enough
            while img.max() >= 2 ** camera.bit_depth() - 1:
                camera.change_exposure(-500)
                img = camera.capture_image()
                print("Max pixel value = " + str(img.max())
                      + " Exposure time: " + str(camera._exposure_time))
            while img.max() <= 250:
                camera.change_exposure(1000)
                img = camera.capture_image()
                print("Max pixel value = " + str(img.max())
                      + " Exposure time: " + str(camera._exposure_time))
            
            # Find center slice of beam
            centerX = np.ceil(ndimage.center_of_mass(img)[0]).astype('int')
            img_slice = img[centerX]

            # Calculate the Peak Ratio
            n = np.power(2, np.ceil(np.log2(len(img_slice)))).astype('int')
            ffFFT = np.fft.fftshift(np.fft.fft(img_slice, n))
            abs_fft = np.abs(ffFFT)
            pks, _ = signal.find_peaks(abs_fft)
            p = np.argsort(abs_fft[pks])[::-1]
            pr[i] = abs_fft[pks][p][1] / abs_fft[pks][p][0]
            phase[i] = np.angle(ffFFT[pks][p][1])
            
            frames[i] = img

            i += 1
            pbar.update(1)
    
    pbar.close()

def reset_equipment(keysight23, keysight24, osa, camera, power_meter, controller):
    osa.auto_calibration(1)
    osa.display_state("ON")
    keysight23.display_state(1)
    keysight24.display_state(1)
    
    # Ramp down and power off
    highest_curr1 = keysight23.get_current()
    for i in np.round(np.arange(highest_curr1, 0.1e-3, -0.1e-3), 5):
        keysight23.set_current(i)
    keysight23.set_current(0.1e-3)
    keysight23.output_off()
    highest_curr2 = keysight24.get_current()
    for i in np.round(np.arange(highest_curr2, 0.1e-3, -0.1e-3), 5):
        keysight24.set_current(i)
    keysight24.set_current(0.1e-3)
    keysight24.output_off()

    #Reset Camera to Near Field
    #controller.move_um(2, 100)
    x_axis_slot = 4
    controller.move_absolute(x_axis_slot, 0)

    #Back to local control
    osa.give_back_local_control()
    keysight23.give_back_local_control()
    keysight24.give_back_local_control()
    power_meter.give_back_local_control()
    camera.__del__()
    controller.close()

def plot(data_file_path, liv_3d_fig_save_path, spectral_scan_fig_save_path, peak_ratio_fig_save_path):
    # Plot and Save?
    plt.figure(figsize=(4, 4), dpi=300)
    spectral.get_num_peaks(data_file_path, 4, -70, spectral_scan_fig_save_path)
    plt.show()
    liv3d_plot.make_3dliv_plot(data_file_path, liv_3d_fig_save_path)
    plt.show()
    ff_plot.make_ff_pr_plot(data_file_path, peak_ratio_fig_save_path)
    plt.show()

def scan_hattrick(istart,
                  istop,
                  istep,
                  center=855e-9,
                  span=10e-9,
                  samplename="NoNameSample",
                  username="NoNameUser",
                  sense="HIGH2",
                  scan_width=0.02e-9,
                  rasterfunction=raster2d.rasteredge,
                  current_limit=20e-3,
                  pwr_wavelength = 940, #In nanometers!
                  x_left_roi = 0,
                  y_top_roi = 0, 
                  x_right_roi = 1920, 
                  y_bottom_roi = 1080,
                  display_state= True):
    """
    Scan a 2D space given a range of currents, saving spectra, power output, and farfield for each current set-point.

    Parameters:
    istart (numbers): Starting current value (in Amps)
    istop (numbers): Ending current value (in Amps)
    istep (numbers): Current sampling step (in Amps)
    center (number): Initial central wavelength in meters (default=855e-9)
    span (number): Initial wavelength span in meters (default=10e-9)
    samplename (string): Name of sample for metadata usage (default="NoNameSample")
    username (string): Name of person running scan for metadata usage (default="NoNameUser")
    sense (string): #Scan precision to use (either "NORM", "MID", "HIGH1", "HIGH2", "HIGH3") (default="HIGH2")
    scan_width(numbers): #Scan width to use (see OSA for options)
    rasterfunction (function): Function to generate the 2D current rastering order (default=raster2d.rasteredge)
    current_limit (number): Maximal permitted current (in Amps) (default=10e-3)
    pwr_wavelength (number): wavelength of measured power (in nanometers)
    x_left_roi (number): upper left x value of box to define the region of interest
    y_top_roi (number): upper left y value of box to define the region of interest
    x_right_roi (number): bottom right x value of box to define the region of interest
    y_bottom_roi (number): bottom right y value of box to define the region of interest
    display_state (boolean): True - display updates; False - display doesn't update (faster)
    """
    check_input(istart, istop, istep, current_limit)

    keysight23 = keysightB2912a.KeysightB2912a(Number=23)     # i1 is nominally the left element
    keysight24 = keysightB2912a.KeysightB2912a(Number=24)    # i2 is nominally the right element
    osa = yokogawa.Yokogawa()
    power_meter = pm100d.PM100D()
    #controller = thorlabs_MCM3000.Controller('COM3','MCM3000', ('ZFM2030','ZFM2030','ZFM2030'), (False,False,False),False,False)
    controller = MCM301()
    with thorcam_v2.ThorCam('CS2100M-USB') as camera: 

        x_left_roi, y_top_roi, x_right_roi, y_bottom_roi = initialize_equipment(istart, istep, keysight23, keysight24, sense, center, span, scan_width, osa, pwr_wavelength, power_meter, camera, x_left_roi, y_top_roi, x_right_roi, y_bottom_roi, display_state, controller)

        data_file_path, liv_3d_fig_save_path, spectral_scan_fig_save_path, peak_ratio_fig_save_path = file_save_routine.set_up_save_location(samplename, meas_name="hat_trick_", data_file_type=".hdf5", 
                                                                               fig_file_type=".png")

        get_data(istart, istop, istep, username, rasterfunction, keysight23, keysight24, sense, center, span, scan_width, osa, power_meter, camera, x_left_roi, y_top_roi, x_right_roi, y_bottom_roi, data_file_path)

        reset_equipment(keysight23, keysight24, osa, camera, power_meter, controller)

        plot(data_file_path, liv_3d_fig_save_path, spectral_scan_fig_save_path, peak_ratio_fig_save_path)