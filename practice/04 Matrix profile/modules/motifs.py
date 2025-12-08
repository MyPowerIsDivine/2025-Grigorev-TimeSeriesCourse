import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    mp = matrix_profile['mp'].copy()  # Копируем, чтобы не портить оригинальный профиль
    mpi = matrix_profile['mpi']
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        # 1. Находим индекс глобального минимума в текущем матричном профиле
        idx_1 = np.argmin(mp)
        min_dist = mp[idx_1]

        # Если все оставшиеся значения - бесконечность, выходим (мотивов меньше, чем top_k)
        if np.isinf(min_dist):
            break

        # 2. Находим индекс ближайшего соседа (парный мотив)
        idx_2 = int(mpi[idx_1])

        # 3. Сохраняем найденную пару индексов и расстояние
        # Порядок индексов не важен, но для красоты можно отсортировать пару
        motifs_idx.append(sorted([idx_1, idx_2]))
        motifs_dist.append(min_dist)

        # 4. Применяем зону исключения (Exclusion Zone)
        # Исключаем области вокруг обоих найденных индексов, заполняя их np.inf
        apply_exclusion_zone(mp, idx_1, excl_zone, np.inf)
        apply_exclusion_zone(mp, idx_2, excl_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
