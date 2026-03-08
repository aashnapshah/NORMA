METADATAPATH = "../data/raw/meds_omop_ehrshot/metadata/codes.parquet"
DATAPATH = "../data/raw/meds_omop_ehrshot/data"

# Source: pg 63 of https://lhncbc.nlm.nih.gov/assets/legacy/files/LOINC_1.6_Top2000CommonLabResultsUS.pdf
CBC_LOINC_CODES = {
    'HGB': ["718-7"], 
    'HCT': ["4544-3", "20570-8", "4545-0"], 
    'RBC': ["789-8"], 
    'PLT': ["777-3", "26515-7"], 
    'MCH': ["785-6"], 
    'MCHC': ["786-4"], 
    'MCV': ["787-2", "30428-7"], 
    'MPV': ["776-5"], # Few measurements, LOINC code issues?
    'RDW': ["788-0"], 
    'WBC': ["6690-2", "20584-9"], #20584-9 is tech. for specimens not blood but #'s work out
}

BMP_LOINC_CODES = {
    'NA': ["2951-2", "2951-2"],  # Sodium
    'K': ["2823-3", "6298-4"],  # Potassium
    'CL': ["2075-0", "2069-3"],  # Chloride
    'CO2': ["2028-9"],  # Bicarbonate / Carbon Dioxide
    'BUN': ["3094-0"],  # Blood Urea Nitrogen
    'CRE': ["2160-0"],  # Creatinine
    'GLU': ["2345-7"],  # Glucose
    'A1C': ["4548-4"],  # Hemoglobin A1c
    'CA': ["17861-6"]   # Calcium, total
}

HFP_LOINC_CODES = {
    'ALT': ["1742-6"],  # Alanine Aminotransferase
    'GGT': ["2324-2"],  # Gamma Glutamyl Transferase
    'AST': ["1920-8"],  # Aspartate Aminotransferase
    'LDH': ["2532-0"],  # Lactate Dehydrogenase
    'PT': ["5902-2"],  # Prothrombin Time
    'ALP': ["6768-6"],  # Alkaline Phosphatase
    'TBIL': ["1975-2"],  # Total Bilirubin
    'DBIL': ["1968-7"],  # Direct Bilirubin
    'ALB': ["1751-7"],  # Albumin
    'TP': ["2885-2"],  # Total Protein
    'CRP': ["1988-5"],  # C-Reactive Protein
}

LIPID_LOINC_CODES = {
    'TC': ["2093-3"],  # Total Cholesterol
    'HDL': ["2085-9"],  # HDL Cholesterol
    'LDL': ["2089-1", "18262-6"],  # LDL Cholesterol (Calculated & Direct)
    'TGL': ["2571-8"]  # Triglycerides
}

LOINC_CODES = {**CBC_LOINC_CODES, **BMP_LOINC_CODES, **HFP_LOINC_CODES, **LIPID_LOINC_CODES}
REVERSE_LOINC_CODES = {item: key for key, values in LOINC_CODES.items() for item in values}

CBC_UNITS = {
    'HGB': ['g/dL', 'g/dl', None, 'G/DL'], # Can assume None is g/dl if val
    'HCT': ['%', 'g/dL', None], # Assume all in %
    'RBC': ['MIL/uL', 'x10E6/uL', 'Million/uL', '10x9/uL', 'M/uL', '10*6/uL',
       'MUL', None], # All same, 10^9 prolly mistake
    'PLT': ['K/uL', 'x10E3/uL', 'Thousand/uL', '10x3/uL', 'varies', None,
       'KUL'], #All same
    'MCH': ['pg', 'PG', None, 'g/dL'], #All same
    'MCHC': ['g/dL', 'G/DL', None], #ibid
    'MCV': ['fL', 'fl', 'FL', None], #ibid
    'MPV': ['fL', 'fl', 'FL'], #ibid
    'RDW': ['%', 'g/dL', None], #ibid (%) but mb sus a little, but only a small subset w bad units
    'WBC': ['/uL', 'K/uL', 'Thousand/uL', 'x10E3/uL', '10^3/mL', 'K/uL', 'x10E3/uL', 'Thousand/uL', '10x3/uL', '10*3/uL', None,
       'KUL', None], #None is K/uL but /uL must be converted
}

BMP_UNITS = {
   'NA': ['mg/dL', 'mg/dl', 'mg/dL', 'mg/dl', None],
   'K': ['mEq/L', 'mEq/L', None],
   'CL': ['mEq/L', 'mEq/L', None],
   'CO2': ['mEq/L', 'mEq/L', None],
   'BUN': ['mg/dL', 'mg/dl', None],
   'CRE': ['mg/dL', 'mg/dl', None],
   'GLU': ['mg/dL', 'mg/dl', None],
   'A1C': ['%', None],
   'CA': ['mg/dL', 'mg/dl', None],
}

HFP_UNITS = {
   'ALT': ['U/L', None],
   'GGT': ['U/L', None],
   'AST': ['U/L', None],
   'LDH': ['U/L', None],
   'PT': ['sec', None],
   'ALP': ['U/L', None],
   'TBIL': ['mg/dL', 'mg/dl', None],
   'DBIL': ['mg/dL', 'mg/dl', None],
   'ALB': ['g/dL', 'g/dl', None],
   'TP': ['g/dL', 'g/dl', None],
   'CRP': ['mg/dL', 'mg/dl', None],
}

LIPID_UNITS = {
   'TC': ['mg/dL', 'mg/dl', None],
   'HDL': ['mg/dL', 'mg/dl', None],
   'LDL': ['mg/dL', 'mg/dl', None],
   'TGL': ['mg/dL', 'mg/dl', None],
}

UNITS = {**CBC_UNITS, **BMP_UNITS, **HFP_UNITS, **LIPID_UNITS}

sex_codes = {
    'M': 8507,
    'F': 8532
}

sex_cats_regression = {
    'M': 0,
    'F': 1
}

# just for reference, see regex below for actual patterns
ICD_CODES = {
   't2d': ['ICD10CM/E11', 'ICD10CM/E11.xxx', 'ICD9CM/250.x0', 'ICD9CM/250.x2'],
   'ckd': ['ICD9CM/585', 'ICD10CM/N18'],
   'osteo': ['ICD9CM/733.0', 'ICD9CM/733.1', 'ICD10CM/M80', 'ICD10CM/M81'],
   'mds': ['ICD9CM/238.72-238.75', "ICD10CM/D46"],
   'mace':['ICD9CM/410, 428, 431, 432, 434', "ICD10CM/I21, I50, I61, I62, I63"],
   'af': ["ICD10CM/I48", "ICD9CM/427.3"], # all AF and AFlutter
}

ICD_CODES_REGEX = {
   't2d': r'^(ICD9CM/250\..[02].*|ICD10CM/E11.*)$',
   'ckd': r'^(ICD9CM/585.*|ICD10CM/N18.*)$',
   'osteo': r'^(ICD9CM/733\.0.*|ICD9CM/733\.1.*|ICD10CM/M80.*|ICD10CM/M81.*)$',
   'mds': r'^(ICD9CM/238\.7[2-5].*|ICD10CM/D46.*)$',
   'mace': r'^(ICD9CM/(410|428|431|432|434).*?|ICD10CM/I(21|50|61|62|63).*)$',
   'af': r'^(ICD9CM/427\.3.*|ICD10CM/I48.*)$',
   'hypertension': r'^(ICD9CM/401.*|ICD10CM/I10.*)$', 
   'hyperlipidemia': r'^(ICD9CM/272.*|ICD10CM/E78.*)$',
   'nafld|nash': r'^(ICD9CM/571.*|ICD10CM/K73.*|ICD10CM/K76.*|ICD10CM/K75.81)$'
}

# [W, A, BAA, NHPI, AIAN, O, U/ND]
# I count any tag that includes the race (ie biracials counted twice) - except for unknown/didnt disclose
race_dict = {
    "White": ['White |',
 'Other | White',
 'White | Other',
 'White | White',
 'White | Unknown',
 'White | American Indian or Alaska Native',
 'Declines to State | White',
 'Asian | White',
 'Unknown | White',
 'White | Black or African American',
 'Black or African American | White',
 'White | Declines to State',
 'White | Asian'],

    "Asian": ['Asian |',
 'Asian | Asian',
 'Declines to State | Asian',
 'Native Hawaiian or Other Pacific Islander | Asian',
 'Asian | Other',
 'Asian | Unknown',
 'Other | Asian',
 'Asian | White',
 'Asian | Declines to State',
 'White | Asian',
 'Asian | Native Hawaiian or Other Pacific Islander'],

    "Black or African American": ['Black or African American | Black or African American',
 'Black or African American | Other',
 'Black or African American |',
 'White | Black or African American',
 'Black or African American | White',
 'Black or African American | Unknown'],

    "Native Hawaiian or Other Pacific Islander": ['Native Hawaiian or Other Pacific Islander |',
 'Native Hawaiian or Other Pacific Islander | Asian',
 'Native Hawaiian or Other Pacific Islander | Native Hawaiian or Other Pacific Islander',
 'Native Hawaiian or Other Pacific Islander | Unknown',
 'Asian | Native Hawaiian or Other Pacific Islander'],

    'American Indian or Alaska Native': ['American Indian or Alaska Native |',
 'Unknown | American Indian or Alaska Native',
 'White | American Indian or Alaska Native',
 'American Indian or Alaska Native | American Indian or Alaska Native',
 'Other | American Indian or Alaska Native'],

    'Other': ['Other |',
 'Native Hawaiian or Other Pacific Islander |',
 'Other | White',
 'White | Other',
 'Other | Unknown',
 'Black or African American | Other',
 'Other | Other',
 'Native Hawaiian or Other Pacific Islander | Asian',
 'Asian | Other',
 'Other | Asian',
 'Native Hawaiian or Other Pacific Islander | Native Hawaiian or Other Pacific Islander',
 'Declines to State | Other',
 'Other | Declines to State',
 'Other | American Indian or Alaska Native',
 '| Other',
 'Native Hawaiian or Other Pacific Islander | Unknown',
 'Asian | Native Hawaiian or Other Pacific Islander',
 'Unknown | Other'],

    "Decline/Unknown": ['Unknown |',
 'Unknown | Unknown',
 'Declines to State |',
 'Declines to State | Declines to State']
}

### CODES ###


CBC_REFERENCE_INTERVALS = {
    'HCT': {'F': (37, 47, '%'), 'M': (42, 50, '%')},  
    'HGB': {'F': (12.0, 16.0, 'g/dL'), 'M': (14, 18, 'g/dL')},  
    'MCH': {'F': (28, 32, 'pg'), 'M': (28, 32, 'pg')},  
    'MCHC': {'F': (33, 36, 'g/dL'), 'M': (33, 36, 'g/dL')},  
    'MPV': {'F': (7.5, 12.5, 'fL'), 'M': (7.5, 12.5, 'fL')},  # Using Mass General Reference Range
    'PLT': {'F': (150, 450, '10³/µL'), 'M': (150, 450, '10³/µL')},  
    'RBC': {'F': (4.0, 5.2, '10⁶/µL'), 'M': (4.5, 5.9, '10⁶/µL')},  # double check
    'RDW': {'F': (9, 14.5, '%'), 'M': (9, 14.5, '%')},  
    'WBC': {'F': (4.5, 11.0, '10³/µL'), 'M': (4.5, 11.0, '10³/µL')},
    'MCV': {'F': (80, 98, 'fL'), 'M': (80, 98, 'fL')}
}

BMP_REFERENCE_INTERVALS = {
    'NA': {'F': (136, 145, 'mEq/L'), 'M': (136, 145, 'mEq/L')},
    'K': {'F': (3.5, 5.0, 'mEq/L'), 'M': (3.5, 5.0, 'mEq/L')},
    'CL': {'F': (98, 106, 'mEq/L'), 'M': (98, 106, 'mEq/L')},
    'CO2': {'F': (23, 28, 'mEq/L'), 'M': (23, 28, 'mEq/L')},
    'BUN': {'F': (8, 20, 'mg/dL'), 'M': (8, 20, 'mg/dL')},
    'CRE': {'F': (0.5, 1.1, 'mg/dL'), 'M': (0.7, 1.3, 'mg/dL')},
    'GLU': {'F': (70, 99, 'mg/dL'), 'M': (70, 99, 'mg/dL')},
    'A1C': {'F': (4.0, 5.6, '%'), 'M': (4.0, 5.6, '%')},
    'CA': {'F': (8.6, 10.2, 'mg/dL'), 'M': (8.6, 10.2, 'mg/dL')}
}

HFP_REFERENCE_INTERVALS = {
    'ALT': {'F': (10.0, 40.0, 'U/L'), 'M': (10.0, 40.0, 'U/L')},
    'GGT': {'F': (8.0, 40.0, 'U/L'), 'M': (9.0, 50.0, 'U/L')},
    'AST': {'F': (10.0, 40.0, 'U/L'), 'M': (10.0, 40.0, 'U/L')},
    'LDH': {'F': (105.0, 225.0, 'U/L'), 'M': (105.0, 225.0, 'U/L')},
    'PT': {'F': (10.0, 13.0, 'sec'), 'M': (10.0, 13.0, 'sec')},
    'ALP': {'F': (30, 110, 'U/L'), 'M': (30, 110, 'U/L')},
    'TBIL': {'F': (0.3, 1.1, 'mg/dL'), 'M': (0.3, 1.1, 'mg/dL')},
    'DBIL': {'F': (0.1, 0.3, 'mg/dL'), 'M': (0.1, 0.3, 'mg/dL')},
    'ALB': {'F': (3.5, 5.5, 'g/dL'), 'M': (3.5, 5.5, 'g/dL')},
    'TP': {'F': (6.0, 8.0, 'g/dL'), 'M': (6.0, 8.0, 'g/dL')},
    'CRP': {'F': (0.0, 3.0, 'mg/dL'), 'M': (0.0, 3.0, 'mg/dL')},
}

LIPID_REFERENCE_INTERVALS = {
    'TC': {'F': (100, 200, 'mg/dL'), 'M': (100, 200, 'mg/dL')},
    'HDL': {'F': (40, 100, 'mg/dL'), 'M': (40, 100, 'mg/dL')},
    'LDL': {'F': (0, 130, 'mg/dL'), 'M': (0, 130, 'mg/dL')},
    'TGL': {'F': (40, 150, 'mg/dL'), 'M': (40, 150, 'mg/dL')}
}

REFERENCE_INTERVALS = {**CBC_REFERENCE_INTERVALS, **BMP_REFERENCE_INTERVALS, **HFP_REFERENCE_INTERVALS, **LIPID_REFERENCE_INTERVALS}

CBC_MIN = [
    'HCT',
    'HGB',
    'MCH',
    'MCV',
    'MCHC',
    'PLT',
    'RBC',
]

CBC_MAX = [
    'RDW',
    'WBC',
]

BMP_MAX = [
    'NA',
    'K',
    'CL',
    'BUN',
    'CRE',
    'GLU',
    'A1C',
]

BMP_MIN = [
   'C02',
   'CA'
]

HFP_MIN = [
   'ALT',
   'GGT',
   'AST',
   'LDH',
   'ALP',
   'TBIL',
   'DBIL',
   'CRP',
   'PT',
   'ALB',
   'TP',
]

HFP_MAX = [
   'ALB',
   'TP']

LIPID_MAX = [
   'TC',
   'LDL',
   'TGL',
]

LIPID_MIN = [
   'HDL'
]

MIN = CBC_MIN + BMP_MIN + HFP_MIN + LIPID_MIN
MAX = CBC_MAX + BMP_MAX + HFP_MAX + LIPID_MAX

TEST_VOCAB = {test_name: i for i, test_name in enumerate(REFERENCE_INTERVALS.keys())}
INVERSE_TEST_VOCAB = {v: k for k, v in TEST_VOCAB.items()}