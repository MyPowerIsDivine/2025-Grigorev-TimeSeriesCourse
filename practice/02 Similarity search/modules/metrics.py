import numpy as np

def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance
    """
    return np.linalg.norm(ts1 - ts2)


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance
    """
    # В контексте многих курсов это либо ED над нормализованными данными,
    # либо расстояние деленное на корень из длины.
    # Если данные уже нормализуются снаружи (как в задаче 1), здесь достаточно обычного ED.
    return np.linalg.norm(ts1 - ts2)


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n = len(ts1)
    m = len(ts2)
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Вычисляем ширину окна
    # Используем max(n, m), так как обычно r задается как доля от длины
    w = int(np.floor(r * max(n, m)))
    
    for i in range(1, n + 1):
        # Ограничение полосы Сако-Чиба
        start_j = max(1, i - w)
        end_j = min(m, i + w)
        
        for j in range(start_j, end_j + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            
            # DTW рекурсия
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # insertion
                                          dtw_matrix[i, j - 1],    # deletion
                                          dtw_matrix[i - 1, j - 1]) # match
            
    # ВОЗВРАЩАЕМ БЕЗ КОРНЯ (согласно формуле в задании)
    return dtw_matrix[n, m]