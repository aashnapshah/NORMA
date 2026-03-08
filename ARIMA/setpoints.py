# Code to calculate setpoints for EHRShot Data with set min. # of measurements and year cutoff

import pandas as pd
import numpy as np
import pickle as pkl
import sklearn as skl
from tqdm import tqdm
import sys
sys.path.append('../')
from process.config import *

# helper for calculate_setpoint
def calculate_gaussian_aic(measurements):
    num_points = len(measurements)
    mean = np.mean(measurements) 
    variance = np.var(measurements)  # MLE: ddof=0
    log_likelihood = -0.5 * num_points * (np.log(2 * np.pi * variance) + 1)
    num_parameters = 2
    aic_score = 2 * num_parameters - 2 * log_likelihood
    return aic_score

# basic fxn to calculate setpoints
def calculate_guassians(seq, max_gaussian_components=3, mixture_thresholds={2:0.7, 3:0.45}):
    # for 1 component model it is faster to use analytic expr for aic
    measurements = seq['x'] #.values
    best_aic_score = calculate_gaussian_aic(measurements)
    best_mixture_model = None

    # fit 2/3 component models
    for num_components in range(2, max_gaussian_components + 1):
        if len(measurements) < num_components:
            return measurements.mean(), measurements.var()
        gaussian_mixture = skl.mixture.GaussianMixture(
            n_components=num_components,
            covariance_type='full',
            max_iter=300,
            reg_covar=0.001,
            random_state=0
        ).fit(measurements.reshape(-1, 1))

        aic_score = gaussian_mixture.aic(measurements.reshape(-1, 1))
        if aic_score < best_aic_score and gaussian_mixture.converged_:
            best_aic_score = aic_score
            best_mixture_model = gaussian_mixture

    # Check if 2/3 component model beat 1 component aic
    if best_mixture_model is not None:
        component_weights = best_mixture_model.weights_
        dominant_component_idx = np.argmax(component_weights)
        dominant_weight = component_weights[dominant_component_idx]
        dominance_threshold = mixture_thresholds[best_mixture_model.n_components]

        # check dominant component has sufficient size
        if dominant_weight > dominance_threshold:
            mean = best_mixture_model.means_[dominant_component_idx, 0]
            covariance = best_mixture_model.covariances_[dominant_component_idx, 0, 0]
            std_dev = np.sqrt(covariance)
            coeff_variation = std_dev / mean
            variance = std_dev**2
            return mean, variance #std_dev, coeff_variation, variance

    # fallback to 1-component model statistics
    mean = np.mean(measurements)
    variance = np.var(measurements)
    return mean, variance #std_dev, coeff_variation, variance

def run_gmm_model(measurements_df):
    setpoint_records = []
    patient_test_groups = measurements_df.groupby(['subject_id', 'test_name'])

    for (patient_id, test_name), measurement_group in tqdm(patient_test_groups, desc="Calculating Setpoints"):
        setpoint_mean, setpoint_var = calculate_guassians(
            measurement_group['numeric_value'].values
        )
       # latest_measurement_date = measurement_group['time'].max()

        setpoint_record = {
            'subject_id': patient_id,
            'test_name': test_name,
            'setpoint_mean': setpoint_mean,
            'setpoint_std': setpoint_std,
            'setpoint_var': setpoint_var,
            'setpoint_cv': setpoint_cv,
            'setpoint_date': latest_measurement_date
        }
        setpoint_records.append(setpoint_record)

    setpoints_df = pd.DataFrame(setpoint_records)
    
    return setpoints_df
    
if __name__ == "__main__":
    measurements_df = pd.read_pickle("data/df_filtered_for_setpoints_2018.pkl")
    setpoints_df = run_gmm_model(measurements_df)
    setpoints_df.to_pickle("data/setpoints_df_2018.pkl")