# -*- coding: utf-8 -*-
"""Спільний модуль: контентні вектори вакансій (TF-IDF опису + one-hot атрибутів), використовується в ЛР2 та ЛР3."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

ATTR = ["field", "experience_level", "work_format", "work_language", "employment_type"]


def content_vectors(vacancies, X_text, attr_weight=0.5):
    A = pd.get_dummies(vacancies[ATTR])
    Xc = normalize(sp.hstack([X_text, sp.csr_matrix(normalize(A.values.astype(float)) * attr_weight)]).tocsr())
    return Xc, A.columns


def content_similarity(Xc):
    S = (Xc @ Xc.T).toarray().astype(np.float32)
    np.fill_diagonal(S, 0)
    return S
