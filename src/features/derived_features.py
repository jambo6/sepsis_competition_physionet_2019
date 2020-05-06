"""
derived_features.py
=================================
'Expert knowledge' type features found from the literature that can be computed from the pre-existing sepsis dataset
features.
"""
import torch


def shock_index(data):
    """ HR/SBP ratio. """
    return data['HR'] / data['SBP']


def age_normalised_shock_index(data):
    """ HR/(SBP * Age). """
    return data['HR'] / (data['SBP'] * data['Age'])


def modfied_shock_index(data):
    """ HR/(SBP * Age). """
    return data['HR'] / (data['MAP'] * data['Age'])


def bun_cr(data):
    """ BUN/Creatinine. """
    return data['BUN'] / data['Creatinine']


def sao2_fio2(data):
    """ SaO2/FiO2. """
    return data['SaO2'] / data['FiO2']


def urea_creatinine(data):
    """ BUN/Creatinine. """
    return data['BUN'] / data['Creatinine']


def pulse_pressure(data):
    """ SBP - DBP. """
    return data['SBP'] - data['DBP']


def cardiac_output(data):
    """ Pulse pressure * heart_rate. """
    pp = data['SBP'] - data['DBP']
    return pp * data['HR']


def partial_sofa(data):
    """ Partial reconstruction of the SOFA score from features available in the sepsis dataset. """
    # Init the tensor
    N, L, C = data.data.size()
    sofa = torch.zeros(N, L, 1)

    # Coagulation
    platelets = data['Platelets']
    sofa[platelets >= 150] += 0
    sofa[(100 <= platelets) & (platelets < 150)] += 1
    sofa[(50 <= platelets) & (platelets < 100)] += 2
    sofa[(20 <= platelets) & (platelets < 50)] += 3
    sofa[platelets < 20] += 4

    # Liver
    bilirubin = data['Bilirubin_total']
    sofa[bilirubin < 1.2] += 0
    sofa[(1.2 <= bilirubin) & (bilirubin <= 1.9)] += 1
    sofa[(1.9 < bilirubin) & (bilirubin <= 5.9)] += 2
    sofa[(5.9 < bilirubin) & (bilirubin <= 11.9)] += 3
    sofa[bilirubin > 11.9] += 4

    # Cardiovascular
    map = data['MAP']
    sofa[map >= 70] += 0
    sofa[map < 70] += 1

    # Creatinine
    creatinine = data['Creatinine']
    sofa[creatinine < 1.2] += 0
    sofa[(1.2 <= creatinine) & (creatinine <= 1.9)] += 1
    sofa[(1.9 < creatinine) & (creatinine <= 3.4)] += 2
    sofa[(3.4 < creatinine) & (creatinine <= 4.9)] += 3
    sofa[creatinine > 4.9] += 4

    return sofa







