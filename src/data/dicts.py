"""
dicts.py
==============================
Some dictionaries containing useful bits of information.
"""

feature_types = {
    'vitals': ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp'],
    'laboratory': [
        'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium',
        'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
        'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets'
    ],
    'demographics': ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'hospital'],
}

feature_types['all_changing'] = feature_types['vitals'] + feature_types['laboratory']