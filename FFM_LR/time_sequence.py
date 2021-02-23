import pandas as pd
from astropy.time import Time


def generate(start_in, end_in):
    time_range = pd.date_range(start=start_in, end=end_in, freq='MS').to_frame()
    time_list = [str(time_range.iloc[i, 0])[0:10] for i in range(len(time_range))]
    time_jd = Time(time_list, format='iso').jd1
    jd_range = pd.DataFrame(time_jd, index=time_list)
    jd_range.columns = ['jd']
    return jd_range+0.5
