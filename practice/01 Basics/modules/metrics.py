import numpy as np

def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Вычисляет евклидово расстояние.
    Параметры
    ----------
    ts1: первый временной ряд
    ts2: второй временной ряд
    Возвращает
    -------
    ed_dist: евклидово расстояние между ts1 и ts2
    """
    
    # Вычисляем сумму квадратов разностей и извлекаем корень
    ed_dist = np.sqrt(np.sum((ts1 - ts2)**2))
    
    return ed_dist

def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Вычисляет нормализованное евклидово расстояние по формуле:
    sqrt( | 2n * (1 - (dot_prod - n*mu1*mu2) / (n*sigma1*sigma2)) | )
    """
    n = len(ts1)
    
    # 1. Вычисляем статистики
    mu1, sigma1 = np.mean(ts1), np.std(ts1)
    mu2, sigma2 = np.mean(ts2), np.std(ts2)
    
    if sigma1 == 0 or sigma2 == 0:
        return 0.0
        
    # 2. Скалярное произведение
    dot_product = np.dot(ts1, ts2)
    
    # 3. Вычисляем дробь из формулы
    correlation_term = (dot_product - n * mu1 * mu2) / (n * sigma1 * sigma2)
    
    # 4. Итоговая формула
    norm_ed_dist = np.sqrt(np.abs(2 * n * (1 - correlation_term)))
    
    return norm_ed_dist

def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Вычисляет DTW расстояние.
    Параметры
    ----------
    ts1: первый временной ряд
    ts2: второй временной ряд
    r: размер окна искривления (в данной реализации не используется)
    
    Возвращает
    -------
    dtw_dist: DTW расстояние между ts1 и ts2
    """
    
    n, m = len(ts1), len(ts2)
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            min_prev_cost = min(dtw_matrix[i - 1, j],
                                dtw_matrix[i, j - 1],
                                dtw_matrix[i - 1, j - 1])
            dtw_matrix[i, j] = cost + min_prev_cost
            
    dtw_dist = dtw_matrix[n, m]
    
    return dtw_dist