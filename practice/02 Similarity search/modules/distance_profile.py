import numpy as np

from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance


def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: normalize or not time series and query

    Returns
    -------
    dist_profile: distance profile between query and time series
    """

    n = len(ts)
    m = len(query)
    N = n-m+1

    dist_profile = np.zeros(shape=(N,))

    # 1. Если требуется нормализация, нормализуем запрос один раз
    if is_normalize:
        query = z_normalize(query)

    # 2. Проходим скользящим окном по временному ряду
    for i in range(N):
        # Выделяем подпоследовательность
        subsequence = ts[i : i + m]
        
        # Если требуется, нормализуем подпоследовательность
        if is_normalize:
            subsequence = z_normalize(subsequence)
            
        # Считаем Евклидово расстояние
        dist = ED_distance(query, subsequence)
        
        # Сохраняем в профиль расстояний
        dist_profile[i] = dist

    return dist_profile
