import pandas as pd
import datetime as dt
from tqdm import tqdm


def __date_string_to_date(date: str):
    date = str(date)
    if date == '':
        return None

    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])

    return dt.date(year, month, day)


# RCV_DATE 말고 DELIV_DATE로 바꾸기
def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[['DELIV_DATE', 'DELIV_CTRL_ZONE_NM', 'DELIV_EMD_NM', 'DLNM_DOLI_NM', 'EUPD_IVNM']]


# 이전 4주가 아니라 이후 4주로 변경
def filter_date(df: pd.DataFrame) -> pd.DataFrame:
    one_month = dt.timedelta(days=28)

    ranges = '20041022 20040928 20050209 20050918 20060129 20061006 20070218 20070925 ' \
             '20080207 20080914 20090126 20091003 20100214 20100922'
    ranges = [__date_string_to_date(date) for date in ranges.split()]
    ranges = [(date - one_month, date) for date in ranges]

    result = pd.DataFrame(columns=['DELIV_DATE', 'DELIV_CTRL_ZONE_NM', 'DELIV_EMD_NM', 'DLNM_DOLI_NM', 'EUPD_IVNM'])
    for row in tqdm(df.iterrows()):
        date = __date_string_to_date(list(row[1])[0])
        if date is None:
            continue

        for r in ranges:
            if r[0] < date < r[1]:
                result = result.append(row[1], ignore_index=True)
                break

    return result[['DELIV_CTRL_ZONE_NM', 'DELIV_EMD_NM', 'DLNM_DOLI_NM', 'EUPD_IVNM']]

