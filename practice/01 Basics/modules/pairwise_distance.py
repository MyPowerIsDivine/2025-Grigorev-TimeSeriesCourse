import numpy as np
from modules.metrics import ED_distance, DTW_distance, norm_ED_distance
from modules.utils import z_normalize

class PairwiseDistance:
    """
    Класс для вычисления матрицы расстояний между временными рядами.
    
    Параметры
    ----------
    metric: str, default='euclidean'
        Название метрики расстояния. Доступные опции: {'euclidean', 'dtw'}.
    is_normalize: bool, default=False
        Выполнять ли Z-нормализацию временных рядов перед вычислением расстояний.
    """
    def __init__(self, metric: str = 'euclidean', is_normalize: bool = False) -> None:
        self.metric: str = metric
        self.is_normalize: bool = is_normalize
    
    @property
    def distance_metric(self) -> str:
        """Возвращает описание используемой метрики расстояния."""
        norm_str = "normalized " if self.is_normalize else "non-normalized "
        return norm_str + self.metric + " distance"

    def _choose_distance(self):
        """
        Выбирает функцию для вычисления расстояния на основе self.metric и self.is_normalize.
        """
        if self.metric == 'euclidean':
            if self.is_normalize:
                return norm_ED_distance
            return ED_distance
            
        elif self.metric == 'dtw':
            return DTW_distance
        else:
            raise ValueError(f"Unknown metric: '{self.metric}'. Available options: 'euclidean', 'dtw'.")

    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """
        Вычисляет матрицу расстояний для набора временных рядов.
        """
        
        n = input_data.shape[0]
        matrix_values = np.zeros((n, n))
        
        data_to_process = input_data.copy()
        
        # ЛОГИКА НОРМАЛИЗАЦИИ:
        # 1. Если это евклидова метрика с нормализацией -> мы НЕ нормализуем данные здесь,
        #    так как функция norm_ED_distance работает с сырыми данными (считает статистики внутри).
        # 2. Если это любая другая метрика (DTW) с нормализацией -> нормализуем данные явно.
        if self.is_normalize and self.metric != 'euclidean':
            for i in range(n):
                data_to_process[i] = z_normalize(data_to_process[i])
        
        # Выбираем функцию расчета
        dist_func = self._choose_distance()
        
        # Вычисление расстояний
        for i in range(n):
            for j in range(i + 1, n):
                dist = dist_func(data_to_process[i], data_to_process[j])
                matrix_values[i, j] = dist
                matrix_values[j, i] = dist 
                
        return matrix_values