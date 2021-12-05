import os
from typing import Union, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import pandas as pd
from tqdm import tqdm

__coder = Nominatim(user_agent='South Korea')

__save_path = 'addr_coord_map.tar'
if os.path.exists(__save_path):
    with open(__save_path, 'r+') as file:
        __addr_coord_map = eval('\n'.join(file.readlines()))
else:
    __addr_coord_map = {}


def __merge_address(deliv_ctrl_zone_nm: str, deliv_emd_nm: str, deliv_dl_nm: str) -> str:
    """
    관할 구역을 주소로 변환한다.

    :param deliv_ctrl_zone_nm: 관할 구역 명
    :param deliv_emd_nm: 행정구역 읍면동 명
    :param deliv_dl_nm: 행정구역 동리 명
    :return: 관할 구역의 주소
    """

    deliv_ctrl_zone_nm = str(deliv_ctrl_zone_nm).strip()
    deliv_emd_nm = str(deliv_emd_nm).strip()
    deliv_dl_nm = str(deliv_dl_nm).strip()

    ctrl_zone, emd, dl = '', '', ''
    if deliv_emd_nm != '':
        emd = deliv_emd_nm
    if deliv_dl_nm != '':
        dl = deliv_dl_nm

    if emd != '' or dl != '':
        return f'{emd} {dl}'
    elif deliv_ctrl_zone_nm != '':
        return deliv_ctrl_zone_nm
    else:
        return ''


def __search_coordinates(address: str) -> Union[None, Tuple[float, float]]:
    """
    주소를 좌표로 변환한다. 주소를 찾지 못하면 None 을 반환한다.

    :param address: 대한민국 주소
    :return: 위도, 경도 순서쌍의 부동소수점 실수 표현
    """

    if address in __addr_coord_map:
        return __addr_coord_map[address]
    else:
        try:
            data = __coder.geocode(address)
            if data is None:
                coord = None
            else:
                coord = (data.latitude, data.longitude)
        except GeocoderTimedOut:
            coord = None
        except GeocoderServiceError:
            coord = None

        __addr_coord_map[address] = coord
        return coord


def __sending_address_to_coordinates(deliv_ctrl_zone_nm: str, deliv_emd_nm: str, deliv_dl_nm: str) -> Union[None, Tuple[float, float]]:
    """
    수신지의 대한민국 주소를 위도, 경도 좌표로 변환한다. 주소를 찾지 못하면 None 을 반환한다.

    :param deliv_ctrl_zone_nm: 관할 구역 명
    :param deliv_emd_nm: 행정구역 읍면동 명
    :param deliv_dl_nm: 행정구역 동리 명
    :return: 위도, 경도 순서쌍의 부동소수점 실수 표현
    """
    address = __merge_address(deliv_ctrl_zone_nm, deliv_emd_nm, deliv_dl_nm)
    if address == '':
        return None
    else:
        return __search_coordinates(address)


def __receiving_address_to_coordinates(eupd_ivnm: str) -> Union[None, Tuple[float, float]]:
    """
    발신지의 대한민국 주소를 위도, 경도 좌표로 변환한다. 주소를 찾지 못하면 None 을 반환한다.

    :param eupd_ivnm: 접수 관할 구멱 명
    :return: 위도, 경도 순서쌍의 부동소수점 실수 표현
    """
    if eupd_ivnm == '':
        return None
    else:
        return __search_coordinates(eupd_ivnm)


def __save_request_history():
    with open(__save_path, 'w+') as file:
        file.write(str(__addr_coord_map))


def address_to_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    for row in tqdm(df.iterrows()):
        row = list(row[1])
        receiver = row[-1]
        sender = row[:-1]

        rec_coord = __receiving_address_to_coordinates(receiver)
        snd_coord = __sending_address_to_coordinates(sender[0], sender[1], sender[2])

        if rec_coord is not None and snd_coord is not None:
            row = {
                'sender_latitude': snd_coord[0],
                'sender_longitude': snd_coord[1],
                'receiver_latitude': rec_coord[0],
                'receiver_longitude': rec_coord[1],
            }

            result = result.append(row, ignore_index=True)

    __save_request_history()
    return result
