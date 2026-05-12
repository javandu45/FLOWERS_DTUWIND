# FLOWERS AEP - Fast AEP

This repository contains the classes for the FLOWERS AEP model, with a demonstration file showing how AEP and gradients are computed, with a comparison to their respective conventional models results using PyWake.

Moreover, the posibility of plotting the AEP coming from each turbine is added at the end.

To see the demonstration, run `demo.py`. 

## FLOWERS

FLOWERS stands for FLOW Estimation and Rose Superposition.

FLOWERS is an AEP model which efficiently computes a wind farm's AEP. It uses numerous simplifications
and assumptions, but most importantly a Fourier transform turning discrete components, functions of wind
direction, into a continuous function. From this, it  solves the integral analytically, rather than 
numerically, also enabling the computation of analytical gradients.

## Flow models

FLOWERS is currently available for NO Jensen and Gaussian Bastankhah wake models.

Both cases use a Linear wake superposition model, and a single point rotor average model.

## Sources

**NO Jensen**: Based on "FLOWERS AEP: An Analytical Model for Wind Farm Layout Optimization" https://doi.org/10.1002/we.2954

**Gaussian Bastankhah**: Based on "Gaussian FLOWERS: Wind-rose-based analytical integration of Gaussian wake model for extremely fast AEP estimation" https://doi.org/10.1063/5.0245886