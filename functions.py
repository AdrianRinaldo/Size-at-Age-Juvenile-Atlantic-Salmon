def read_multiple_csv(path_to_folder, col1, col2, transform_func):
    '''
    Import multiple CSVs from the ISIMIP atmospheric variables, and allow for transformation of scale
    '''
    import glob
    import pandas as pd

    csv_files = sorted(glob.glob(path_to_folder + '/*.csv'))
    combined = pd.concat([pd.read_csv(file, header = None, names = [col1, col2]) for file in csv_files])
    if transform_func is not None:
        combined[col2] = transform_func(combined[col2])
    return combined

def read_multiple_excel(path_to_folder, sheet_name):
    '''
    Import multiple excel files and merge
    '''
    import glob
    import pandas as pd

    xlsx_files = sorted(glob.glob(path_to_folder+ '/*.xlsx'))
    combined = pd.concat([pd.read_excel(file, sheet_name = sheet_name) for file in xlsx_files])
    return combined


def kelvinToCelsius(kelvin):
    '''
    Transform kelvin to celsius
    '''
    return kelvin - 273.15

def persecondtoperday(persecond):
    '''
    Transform from per second to per day
    '''
    return persecond*(60*60*24)

def daterange(start_date, end_date):
    '''
    This function will generate a dataframe with all dates from start date to end date
    '''

    from datetime import date, timedelta
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def encode(data, col, max_val):
    '''
    Returns sine and cosine transformed time 
    '''
    import numpy as np
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def create_dataset(X, y, time_steps):
    import numpy as np
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)   

def Lagging(lag, cols, data):
    for i in range(1, lag+1, 1):
        for col in cols:
            data[f'{col}_lag_{i}'] = data[f'{col}'].shift(i)

def Periodicity(period, cols, data):
    for col in cols:
        data[f'{col}_lag_{period}'] = data[f'{col}'].shift(period)

def daylength(dayOfYear, lat):
    import numpy as np
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example,
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995.
    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    Returns
    -------
    d : float
        Daylength in hours.
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    if -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
        return 2.0*hourAngle/15.0
    
def get_distribution(X_new, beta_samp_mean, beta_samp_cov,  beta_samp_scale, num_samps):
    import numpy as np
    from scipy.stats.distributions import gamma
    import statsmodels.api as sm
    X_new = X_new
    X_new = sm.add_constant(X_new, has_constant = 'add')
    dispersion = beta_samp_scale
    possible_coefficients = np.random.multivariate_normal(mean = beta_samp_mean, cov = beta_samp_cov, size = num_samps)
    linear_predictions = [X_new@b for b in possible_coefficients] # Make the predictions
    y_hyp = gamma(a=1/dispersion, scale = np.exp(linear_predictions, order = 'K')*dispersion).rvs()
    return y_hyp
