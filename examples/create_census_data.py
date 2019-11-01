"""
This script downloads 2017 ACS PUMS data for the state of Illinois from the
Census Bureau, extracts and recodes four variables
(age, schooling, marriage, and means of transportation to work),
and stores them in a CSV file.

Note that this script requires the requests library (~=2.7.0), which is
not installed by default with the dpwgan package.
"""

import io
import os
import zipfile

import pandas as pd
import requests

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
CENSUS_FILE = os.path.join(THIS_DIR, 'pums_il.csv')

schl = {
    -1: 'N/A',
    1: 'No schooling',
    2: 'Preschool',
    3: 'Kindergarten',
    4: 'Grade 1',
    5: 'Grade 2',
    6: 'Grade 3',
    7: 'Grade 4',
    8: 'Grade 5',
    9: 'Grade 6',
    10: 'Grade 7',
    11: 'Grade 8',
    12: 'Grade 9',
    13: 'Grade 10',
    14: 'Grade 11',
    15: 'Grade 12',
    16: 'High school diploma',
    17: 'GED',
    18: '< 1 year of college',
    19: '>= 1 year of college',
    20: "Associate's degree",
    21: "Bachelor's degree",
    22: "Master's degree",
    23: 'Professional degree',
    24: 'Doctorate degree'
}

mar = {
    -1: 'N/A',
    1: 'Married',
    2: 'Widowed',
    3: 'Divorced',
    4: 'Separated',
    5: 'Never married or <15'
}

jwtr = {
    -1: 'N/A',
    1: 'Car, truck, van',
    2: 'Bus',
    3: 'Streetcar',
    4: 'Subway',
    5: 'Railroad',
    6: 'Ferryboat',
    7: 'Taxicab',
    8: 'Motorcycle',
    9: 'Bicycle',
    10: 'Walked',
    11: 'Worked at home',
    12: 'Other method'
}


def bin_age(age):
    if age < 18:
        return '0-17'
    elif age < 26:
        return '18-25'
    elif age < 36:
        return '26-35'
    elif age < 46:
        return '36-45'
    elif age < 56:
        return '46-55'
    elif age < 66:
        return '56-65'
    elif age < 76:
        return '66-75'
    elif age < 86:
        return '76-85'
    else:
        return '85+'


def main():
    if os.path.isfile(CENSUS_FILE):
        print('Census data already exists')
        return

    print('Downloading Census data...')
    response = requests.get(
        'https://www2.census.gov/programs-surveys/acs/data/pums/2017/5-Year/'
        'csv_pil.zip'
    )

    print('Extracting csv...')
    zipfile.ZipFile(io.BytesIO(response.content)).extract('psam_p17.csv')
    pums_il = pd.read_csv('psam_p17.csv').fillna(-1)
    pums_il_small = pd.DataFrame()
    pums_il_small['AGE'] = [bin_age(age) for age in pums_il['AGEP']]
    pums_il_small['SCHL'] = [schl[val] for val in pums_il['SCHL']]
    pums_il_small['MAR'] = [mar[val] for val in pums_il['MAR']]
    pums_il_small['JWTR'] = [jwtr[val] for val in pums_il['JWTR']]

    print('Writing Census data...')
    pums_il_small.to_csv(CENSUS_FILE, index=False)

    print('Done')


if __name__ == '__main__':
    main()
