#!/usr/bin/env python
# coding: utf-8

##### Study of the dipoles directions #####


import healpy as hp
from astrotools import healpytools as hpt
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
import pymaster as nmt
from scipy.stats import binned_statistic
from numpy import loadtxt
import csv
from multiprocessing import Pool, cpu_count
#import def_functions_mp

import warnings
warnings.filterwarnings("ignore")


#########################

start_total_time = time.time()


### Retreive the masks with C1 apodization and sharp masks ###

masks_C1 = []
masks_sharp = []
for i in range(12):
    masks_C1.append(hp.read_map('./Masks/Mask%s_apoC1'%i))
    masks_sharp.append(hp.read_map('./Masks_sharp/Mask%s_sharp'%i))

#seed = 583
#np.random.seed(seed)


##### Some useful functions #####

def retrieve_angles(dip_vector):
    """ From a vector of 3 elements, this code can retrieve:
    - the coordinates of the dipole direction theta (latitude) and phi (longitude) in degrees
      with the convention 0° at the center of the map
    - the amplitude of the dipole in K
    """
    
    rlat = np.arctan(dip_vector[2]/np.sqrt(dip_vector[0]**2+dip_vector[1]**2))
    lat = np.degrees(rlat)
    #print('\nlat:',lat,'degrees')

    rlon = np.arctan(dip_vector[1]/dip_vector[0])
    lon = np.degrees(rlon)
    if dip_vector[0]<0:
        lon = lon + 180 # to recover the longitude with the convention where 0° is at the center of the map
    if lon < 0:
        lon = lon+360
    #print('lon:',lon,'degrees')

    amplitude = np.linalg.norm([dip_vector[0],dip_vector[1],dip_vector[2]])
    #print('amplitude:', amplitude*1e3, 'mK')
    
    return lat,lon,amplitude


def ring2nest_map(map):
    """ This code transforms a map in ring format into a map in nest format. """
    nested_map = np.zeros(len(map))    # Creation of a new map of zeros
    Nside = hp.npix2nside(len(map))
    new_place = hp.pixelfunc.ring2nest(Nside, np.arange(len(map))) # we calculate the new position of the pixels for the nested format
    for j in range(len(new_place)):
        nested_map[j] = map[new_place[j]] # and add the values of the mask for the new format.
    #hp.mollview(nested_map, title='map with dipole')
    return nested_map


def nest2ring_map(map):
    """ This code transforms a map in nest format into a map in ring format. """
    ring_map = np.zeros(len(map))    # Creation of a new map of zeros
    Nside = hp.npix2nside(len(map))
    new_place = hp.pixelfunc.nest2ring(Nside, np.arange(len(map))) # we calculate the new position of the pixels for the nested format
    for j in range(len(new_place)):
        ring_map[j] = map[new_place[j]] # and add the values of the mask for the new format.
    #hp.mollview(nested_map, title='map with dipole')
    return ring_map


def get_index(List,value):
    index = []
    for i in range(len(List)):
        if List[i] == value:
            index.append(i)
    return index


#########################

fwhm = 20 * np.pi / (60*180) # 20 arcmin in rad
sigma = fwhm/2.35

# binning
bin_size = 16
Nside = 512
lmax = 2*Nside
number_of_bins = int(lmax/bin_size)  # each bin contains bin_size values of l
ell = np.arange(lmax+1)
ell_edge = [2.+i*bin_size for i in range(number_of_bins+1)]

Cl_beam = np.exp(- ell*(ell+1) * sigma**2)
Bl_pixwin = hp.sphtfunc.pixwin(Nside, pol=True, lmax=lmax)
Cl_pixwin = [Bl_pixwin[i]**2 for i in range(len(Bl_pixwin))] # Cl = Bl^2

Cl_beam_bin = binned_statistic(ell, Cl_beam, statistic='mean', bins=ell_edge)
Cl_pixwin_bin = binned_statistic(ell, Cl_pixwin, statistic='mean', bins=ell_edge)

  
# Cls of Planck 2018
PLA_best_PS = pd.read_csv("COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01_copie.txt", delim_whitespace=True, index_col=0)
cl = PLA_best_PS.divide(PLA_best_PS.index * (PLA_best_PS.index+1) / (2*np.pi), axis="index")
cl /= 1e12 # in muK^2
cl = cl.reindex(np.arange(0, lmax+1)) # re index because it begins at 2
cl = cl.fillna(0) # replace the 1st values by 0 instead of NaN
Cl_TT_Planck = np.array(cl.TT)
Cl_TE_Planck = np.array(cl.TE)
Cl_EE_Planck = np.array(cl.EE)


#########################

start_time = time.time()

number_of_maps = 4 # number of simulations

CMB_maps = [] # list of the maps
Cls_C1 = [] # np.shape() = (number_of_maps, 12(number of patches), 3(TT,EE,TE), number_of_bins)
Dls_C1 = [] # np.shape() = (number_of_maps, 12(number of patches), 3(TT,EE,TE), number_of_bins)

delta_Cl_list = []
delta_average_list = []

y_average_PS = [] # list of the averaged power spectra for 1,2,... maps (np.shape=(number_of_maps,6,lmax))
y_subtraction = [] # list of the difference between the averaged PS and the Planck PS (np.shape=(number_of_maps,3,lmax))

nb_cores = cpu_count() # All the available cpus
pool = Pool(nb_cores) # Use available cpus



unconstrained_maps = []

for i in range(number_of_maps):
    #1st, let's simulate a random map from Planck 2018, with fwhm and pixwin
    alms = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True)
    map_random_Planck = hp.alm2map(alms, nside=Nside, lmax=lmax, pixwin=True, fwhm=fwhm) # T,Q,U (random)

    unconstrained_maps.append(map_random_Planck)

print('maps are simulated in', (time.time()-start_time)/60, 'min')


# if number_of_maps > len(masks_C1), then parallelisation of the maps
def get_Dls_map(unconstrained_map):
    for k in range(len(masks_C1)):
        Cls_C1_patches = []
        Dls_C1_patches = []

        f_0 = nmt.NmtField(masks_C1[k], [unconstrained_map[0]])
        f_2 = nmt.NmtField(masks_C1[k], [unconstrained_map[1],unconstrained_map[2]])

        b = nmt.NmtBin.from_nside_linear(Nside, bin_size)

        Cl_TT = nmt.compute_full_master(f_0, f_0, b)
        Cl_TE = nmt.compute_full_master(f_0, f_2, b)
        Cl_EE = nmt.compute_full_master(f_2, f_2, b)

        Cl_TT = Cl_TT[0][:number_of_bins]
        Cl_TE = Cl_TE[0][:number_of_bins]
        Cl_EE = Cl_EE[0][:number_of_bins]

        ell_arr = b.get_effective_ells()
        ell_arr = ell_arr[:number_of_bins]

        # correction for beam and pixel window function
        cl_C1 = []
        Cl_TT /= (Cl_beam_bin[0] * Cl_pixwin_bin[0][0]) # no need to divide by f_sky because the map and mask weren't multiplied
        cl_C1.append(Cl_TT)
        Cl_EE /= (Cl_beam_bin[0] * Cl_pixwin_bin[0][0])
        cl_C1.append(Cl_EE)
        Cl_TE /= (Cl_beam_bin[0] * Cl_pixwin_bin[0][0])
        cl_C1.append(Cl_TE)
        Cls_C1_patches.append(cl_C1)

        # Compute the Dls
        dl_C1 = []
        Dl_TT = Cl_TT * ell_arr * (ell_arr+1) * 1e12 / (2*np.pi)
        dl_C1.append(Dl_TT)
        Dl_EE = Cl_EE * ell_arr * (ell_arr+1) * 1e12 / (2*np.pi)
        dl_C1.append(Dl_EE)
        Dl_TE = Cl_TE * ell_arr * (ell_arr+1) * 1e12 / (2*np.pi)
        dl_C1.append(Dl_TE)
        Dls_C1_patches.append(dl_C1)
    
    return Dls_C1_patches


# if len(masks_C1) > number_of_maps, then parallelisation of the masks
def get_Dls_mask(unconstrained_map, mask):
    Cls_C1_patches = []
    Dls_C1_patches = []

    f_0 = nmt.NmtField(mask, [unconstrained_map[0]])
    f_2 = nmt.NmtField(mask, [unconstrained_map[1],unconstrained_map[2]])

    b = nmt.NmtBin.from_nside_linear(Nside, bin_size)

    Cl_TT = nmt.compute_full_master(f_0, f_0, b)
    Cl_TE = nmt.compute_full_master(f_0, f_2, b)
    Cl_EE = nmt.compute_full_master(f_2, f_2, b)

    Cl_TT = Cl_TT[0][:number_of_bins]
    Cl_TE = Cl_TE[0][:number_of_bins]
    Cl_EE = Cl_EE[0][:number_of_bins]

    ell_arr = b.get_effective_ells()
    ell_arr = ell_arr[:number_of_bins]

    # correction for beam and pixel window function
    cl_C1 = []
    Cl_TT /= (Cl_beam_bin[0] * Cl_pixwin_bin[0][0]) # no need to divide by f_sky because the map and mask weren't multiplied
    cl_C1.append(Cl_TT)
    Cl_EE /= (Cl_beam_bin[0] * Cl_pixwin_bin[0][0])
    cl_C1.append(Cl_EE)
    Cl_TE /= (Cl_beam_bin[0] * Cl_pixwin_bin[0][0])
    cl_C1.append(Cl_TE)
    Cls_C1_patches.append(cl_C1)

    # Compute the Dls
    dl_C1 = []
    Dl_TT = Cl_TT * ell_arr * (ell_arr+1) * 1e12 / (2*np.pi)
    dl_C1.append(Dl_TT)
    Dl_EE = Cl_EE * ell_arr * (ell_arr+1) * 1e12 / (2*np.pi)
    dl_C1.append(Dl_EE)
    Dl_TE = Cl_TE * ell_arr * (ell_arr+1) * 1e12 / (2*np.pi)
    dl_C1.append(Dl_TE)
    Dls_C1_patches.append(dl_C1)

    return dl_C1  #Dls_C1_patches



nb_cores = cpu_count() # All the available cpus
pool = Pool(nb_cores) # Use available cpus


if number_of_maps>len(masks_C1):
    Dls_C1 = pool.map(get_Dls_map, unconstrained_maps)

if len(masks_C1)>number_of_maps:
    Dls_C1 = []
    for i in range(number_of_maps):
        unconstrained_maps_masks = []
        for j in range(len(masks_C1)):
            unconstrained_maps_masks.append([unconstrained_maps[i],masks_C1[j]])
        Dls_C1.append(pool.starmap(get_Dls_mask, unconstrained_maps_masks))


print('Dls_C1 shape:', np.shape(Dls_C1))
print('\nSimulation of the maps and obtention of the Dls: Done in', (time.time()-start_time)/60, 'min')


#########################

List = [] # np.shape = (number_of_maps, number_of_bins, 3(TT,EE,TE), 12(patches))
# List contains all the Cls for the different maps, bins, temperature and polarisation, for the 12 patches
for maps in range(number_of_maps):
    bin_i = []
    for i in range(number_of_bins):
        TT_EE_TE = []
        for temp_pol in range(3):
            values = []
            for pixel in range(12):  
                value = Dls_C1[maps][pixel][temp_pol][i]
                values.append(value)
            TT_EE_TE.append(values)
        bin_i.append(TT_EE_TE)
    List.append(bin_i)


file = open('./results/SMICA_Dls_TT_EE_TE_mp_unconstrained.txt', 'a')
for map_ in range(len(List)):                       # number_of_maps values
    for bin_ in range(len(List[0])):                # number_of_bins values
        for TT_EE_TE in range(len(List[0][0])):     # 3 values (TT, EE, TE)
            for value in range(len(List[0][0][0])): # 12 values (12 patches)
                file.write(" ")
                file.write(str(List[map_][bin_][TT_EE_TE][value]))
file.close()

print('Saving the Dls into a file: Done')


#########################
##### For each bin, create a new map from the 12 patches: #####


start_time = time.time()

nside = Nside
nb_pix = hp.nside2npix(nside)

# Creation of new maps from the patches, for the different bins
bin_maps = [] # np.shape = (number_of_maps, 63 (number_of_bins,) 3 (TT,EE,TE), number_of_pixels)


def reconstruction_map(nb_bin, nb_maps):
    bin_map_TT_pix = []
    for pix in range(12):
        bin_map_TT_pix.append(masks_sharp[pix]*List[nb_maps][nb_bin][0][pix])
    bin_map_TT = sum(bin_map_TT_pix)
        
    bin_map_EE_pix = []
    for pix in range(12):
        bin_map_EE_pix.append(masks_sharp[pix]*List[nb_maps][nb_bin][1][pix])
    bin_map_EE = sum(bin_map_EE_pix)
        
    bin_map_TE_pix = []
    for pix in range(12):
        bin_map_TE_pix.append(masks_sharp[pix]*List[nb_maps][nb_bin][2][pix])
    bin_map_TE = sum(bin_map_TE_pix)
      
    bin_TT_EE_TE = [bin_map_TT, bin_map_EE, bin_map_TE]
    #bin_map_1.append(bin_TT_EE_TE)

    #return bin_map_1
    return bin_TT_EE_TE

nb_cores = cpu_count() # All the available cpus
pool = Pool(nb_cores) # Use available cpus

for nb_maps in range(number_of_maps):
    nb_bin_maps = []
    for nb_bin in range(number_of_bins):
        nb_bin_maps.append([nb_bin, nb_maps])
    bin_map_1 = pool.starmap(reconstruction_map, nb_bin_maps)
    bin_maps.append(bin_map_1)

print('Reconstruction of the maps: Done')


#########################

# Remove the dipoles of these new maps 
rm_dipoles = [] # np.shape = (number_of_maps, 63(number_of_bins), 3(TT,EE,TE), 3(map,monopole,dipole_vec))

for nb_maps in range(number_of_maps):
    rm_dipoles_1 = [] # np.shape = (63(number_of_bins), 3(TT,EE,TE), 3(map,monopole,dipole_vec))
    for k in range(len(bin_maps[0])): # for each bin
        TT_EE_TE = [] # np.shape = (3(TT,EE,TE), 3(map,monopole,dipole_vec))
        for j in range(3):
            rm_dipole = hp.remove_dipole(bin_maps[nb_maps][k][j], nest=False, fitval=True, copy=True) 
            TT_EE_TE.append(rm_dipole)
        rm_dipoles_1.append(TT_EE_TE)
    rm_dipoles.append(rm_dipoles_1)

# Save the dipole vectors of these new maps
rm_dipole_list = rm_dipoles # np.shape = (number_of_maps, 63 (number_of_bins), 3 (TT,EE,TE), 3 (map, monopole, dipole_vector))

print('Fitting dipoles: Done')

dipole_vector_list = [] # np.shape = (number_of_maps, 63 (number_of_dipoles), 3 (TT,EE,TE), 3 (dipole_vector))
for nb_maps in range(number_of_maps):
    dipole_vector_list_1 = [] # np.shape = (63 (number_of_dipoles), 3 (TT,EE,TE), 3 (dipole_vector))
    for i in range(len(rm_dipole_list[0])): # for each dipole
        dipole_vector = []
        dipole_vector.append(rm_dipole_list[nb_maps][i][0][2]) # add the dipole vector for TT
        dipole_vector.append(rm_dipole_list[nb_maps][i][1][2]) # add the dipole vector for EE
        dipole_vector.append(rm_dipole_list[nb_maps][i][2][2]) # add the dipole vector for TE
        dipole_vector_list_1.append(dipole_vector)
    dipole_vector_list.append(dipole_vector_list_1)

file = open('./results/time_unconstrained.txt', 'a')
file.write('\nTime to obtain the dipole directions of the simulated maps: ')
file.write(str((time.time() - start_time)/60))
file.write('min')
file.close()


##### Save data into a file: #####


file = open('./results/SMICA_dipole_vector_mp_unconstrained.txt', 'a')
for map_ in range(len(dipole_vector_list)):
    for bin_ in range(len(dipole_vector_list[0])):
        for TT_EE_TE in range(len(dipole_vector_list[0][0])):
            for value in range(len(dipole_vector_list[0][0][0])):
                file.write(" ")
                file.write(str(dipole_vector_list[map_][bin_][TT_EE_TE][value]))
file.close()

print("Saving the dipoles' vectors into a file: Done")


#########################

file = open('./results/time_unconstrained.txt', 'a')
file.write('\nTotal time: ')
file.write(str((time.time() - start_total_time)/60))
file.write('min \n')
file.close()
