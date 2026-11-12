# -*- coding: utf-8 -*-
"""
ЛР3. Колаборативна фільтрація (класичні підходи): user-based (кореляція Пірсона) – рекомендації 1–4,
item-based (скоригована косинусна схожість) – рекомендація 5. Аналіз для активних і пасивних користувачів.
"""
import os, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from content import content_vectors, content_similarity

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CLEAN = os.path.join(BASE, "data", "clean")
RES, FIG = os.path.join(BASE, "results", "lab3"), os.path.join(BASE, "figures", "lab3")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
REPORT = {}
users = pd.read_csv(os.path.join(CLEAN, "users_clean.csv"))
vacancies = pd.read_csv(os.path.join(CLEAN, "vacancies_clean.csv"))
R_full = np.load(os.path.join(CLEAN, "R_scores.npy"))          # матриця взаємодій U x C (оцінка 1..5, 0 – немає)
U, C = R_full.shape
clusters = vacancies["cluster"].values
K = clusters.max() + 1
uidx = {u: i for i, u in enumerate(users["user_id"])}
GROUPS = {"активні": "активний", "пасивні": "пасивний"}
DELTA = 0.5        # поріг: прогноз має перевищувати середню оцінку користувача щонайменше на DELTA
TOPN = 10
Xc, _ = content_vectors(vacancies, sp.load_npz(os.path.join(CLEAN, "X_text_tfidf.npz")))
S_content = content_similarity(Xc)

REPORT["matrix"] = dict(users=U, vacancies=C, nnz=int((R_full > 0).sum()), density=round(float((R_full > 0).mean() * 100), 3),
                        mean=round(float(R_full[R_full > 0].mean()), 3),
                        hist={str(k): int(v) for k, v in zip(*np.unique(R_full[R_full > 0], return_counts=True))})


# =============================================================== Подібності
def pearson_corated(R, w=None):
    """Кореляція Пірсона між користувачами, обчислена лише на спільно оцінених вакансіях (точна формула).
    w – ваги вакансій (напр. IUF); n – кількість спільно оцінених вакансій."""
    B = (R > 0).astype(np.float64)
    W = np.ones(R.shape[1]) if w is None else w
    BW, RW = B * W, R * W
    n_w = BW @ B.T                    # Σ w (спільні)
    Sx = RW @ B.T                     # Σ w r_u  на спільних
    Sxx = (R * RW) @ B.T              # Σ w r_u² на спільних
    Sxy = RW @ R.T                    # Σ w r_u r_v
    num = n_w * Sxy - Sx * Sx.T
    den = np.sqrt(np.clip(n_w * Sxx - Sx ** 2, 0, None) * np.clip(n_w * Sxx - Sx ** 2, 0, None).T)
    n = B @ B.T
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where((den > 1e-9) & (n >= 2), num / den, np.nan)
    np.fill_diagonal(P, np.nan)
    return P.astype(np.float32), n.astype(np.int32)


def adjusted_cosine_items(R):
    B = R > 0
    mu = np.where(B.sum(1) > 0, R.sum(1) / np.maximum(B.sum(1), 1), 0)
    Z = np.where(B, R - mu[:, None], 0)
    num = Z.T @ Z
    nrm = np.sqrt((Z ** 2).sum(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = num / np.outer(nrm, nrm)
    S = np.nan_to_num(S).astype(np.float32)
    np.fill_diagonal(S, 0)
    return S


class CF:
    def __init__(self, R):
        self.R = R
        self.B = R > 0
        self.mu = np.where(self.B.sum(1) > 0, R.sum(1) / np.maximum(self.B.sum(1), 1), 0)
        n_i = self.B.sum(0)
        self.iuf = np.log(max(1, (self.B.sum(1) > 0).sum()) / np.maximum(n_i, 1))      # зворотна частота користувача
        self.P, self.N = pearson_corated(R)
        self.P_iuf, _ = pearson_corated(R, self.iuf)
        self.S_item = adjusted_cosine_items(R)

    def predict_user_based(self, u, items, sim, k=None, min_sim=0.0, filt=None):
        s = np.nan_to_num(sim[u], nan=0.0)
        f = s if filt is None else np.nan_to_num(filt[u], nan=0.0)   # за чим відбираються сусіди
        out = np.full(len(items), np.nan)
        nb = np.zeros(len(items), int)
        for j, i in enumerate(items):
            raters = np.where(self.B[:, i] & (f > min_sim) & (s > 0))[0]
            if k is not None and len(raters) > k:
                raters = raters[np.argsort(-s[raters])[:k]]
            if len(raters) == 0:
                continue
            ww = s[raters]
            out[j] = self.mu[u] + (ww * (self.R[raters, i] - self.mu[raters])).sum() / np.abs(ww).sum()
            nb[j] = len(raters)
        return np.clip(out, 1, 5), nb

    def predict_item_based(self, u, items, k=20):
        rated = np.where(self.B[u])[0]
        out = np.full(len(items), np.nan)
        nb = np.zeros(len(items), int)
        for j, i in enumerate(items):
            s = self.S_item[i, rated]
            pos = s > 0
            if not pos.any():
                continue
            idx = np.where(pos)[0]
            idx = idx[np.argsort(-s[idx])[:k]]
            out[j] = (s[idx] * self.R[u, rated[idx]]).sum() / s[idx].sum()
            nb[j] = len(idx)
        return np.clip(out, 1, 5), nb

    def all_methods(self, u, items, beta, k_best):
        sig = np.minimum(self.N[u], beta) / beta                     # значущість ко-рейтингу
        res = {}
        res["R1"] = self.predict_user_based(u, items, self.P)
        res["R2"] = self.predict_user_based(u, items, self.P_iuf)
        res["R3"] = self.predict_user_based(u, items, _row_scaled(self.P, u, sig))
        res["R4"] = self.predict_user_based(u, items, _row_scaled(self.P_iuf, u, sig), k=k_best, min_sim=0.1, filt=self.P_iuf)
        res["R5"] = self.predict_item_based(u, items)
        return res


def _row_scaled(M, u, factor):
    """Повертає «матрицю» з одним зміненим рядком u (для передачі в predict_user_based без копіювання всієї M)."""
    class Row:
        def __getitem__(self, i):
            return M[u] * factor
    return Row()


# =============================================================== 1. Емпіричний підбір параметрів (hold-out 20 %)
rng = np.random.default_rng(42)
R_train = R_full.copy()
test = []
seg = users["segment"].values
for gname, s in GROUPS.items():
    for u in np.where((seg == s) & (users["n_vacancies"].values >= 2))[0]:
        items = np.where(R_full[u] > 0)[0]
        nt = max(1, int(round(0.2 * len(items))))
        for i in rng.choice(items, nt, replace=False):
            test.append((gname, u, i, R_full[u, i]))
            R_train[u, i] = 0
cf_tr = CF(R_train)
test = pd.DataFrame(test, columns=["group", "u", "i", "r"])

# розподіл кількості ко-рейтингових позицій
REPORT["corated"] = {}
for g, s in GROUPS.items():
    Nm = cf_tr.N[seg == s]
    REPORT["corated"][g] = {q: float(np.percentile(Nm[Nm > 0], q)) for q in (25, 50, 75, 90)}


def eval_variant(fn):
    rows = []
    for (g, u), d in test.groupby(["group", "u"]):
        p, _ = fn(u, d["i"].values)
        rows += [(g, pp, rr) for pp, rr in zip(p, d["r"].values)]
    df = pd.DataFrame(rows, columns=["group", "pred", "r"])
    out = {}
    for g, d in df.groupby("group"):
        ok = d["pred"].notna()
        e = d.loc[ok, "pred"] - d.loc[ok, "r"]
        out[g] = dict(MAE=round(float(e.abs().mean()), 4) if ok.any() else None,
                      RMSE=round(float(np.sqrt((e ** 2).mean())), 4) if ok.any() else None,
                      coverage=round(float(ok.mean()), 4), n=int(len(d)))
    return out


# baseline: середня оцінка користувача
REPORT["baseline_user_mean"] = eval_variant(lambda u, it: (np.full(len(it), cf_tr.mu[u]) if cf_tr.mu[u] > 0 else np.full(len(it), np.nan), None))
beta_grid = [2, 5, 10, 20, 30, 50, 75]
REPORT["beta_search"] = {}
for b in beta_grid:
    REPORT["beta_search"][b] = eval_variant(lambda u, it, b=b: cf_tr.predict_user_based(u, it, _row_scaled(cf_tr.P, u, np.minimum(cf_tr.N[u], b) / b)))
    print("beta", b, REPORT["beta_search"][b])
def wmae(res):  # MAE по всіх відкладених оцінках (зважена кількістю прогнозів у групах)
    return sum(res[g]["MAE"] * res[g]["n"] * res[g]["coverage"] for g in res) / sum(res[g]["n"] * res[g]["coverage"] for g in res)
BETA = min(beta_grid, key=lambda b: wmae(REPORT["beta_search"][b]))
k_grid = [5, 10, 20, 30, 50, 100]
REPORT["k_search"] = {}
sig_tr = lambda u: np.minimum(cf_tr.N[u], BETA) / BETA
for k in k_grid:
    REPORT["k_search"][k] = eval_variant(lambda u, it, k=k: cf_tr.predict_user_based(u, it, _row_scaled(cf_tr.P_iuf, u, sig_tr(u)), k=k, min_sim=0.1, filt=cf_tr.P_iuf))
    print("k", k, REPORT["k_search"][k])
K_BEST = min(k_grid, key=lambda k: wmae(REPORT["k_search"][k]))
REPORT["BETA"], REPORT["K_BEST"] = BETA, K_BEST
_cache = {}
def _all(u, it):
    if u not in _cache:
        _cache[u] = cf_tr.all_methods(u, it, BETA, K_BEST)
    return _cache[u]
REPORT["holdout"] = {}
for name in ["R1", "R2", "R3", "R4", "R5"]:
    REPORT["holdout"][name] = eval_variant(lambda u, it, name=name: _all(u, it)[name])
    print(name, REPORT["holdout"][name])

# =============================================================== 2. Рекомендації на повних даних
cf = CF(R_full)
REPORT["pearson_stats"] = {}
for g, s in GROUPS.items():
    m = np.where(seg == s)[0]
    P = cf.P[m]
    REPORT["pearson_stats"][g] = dict(valid_pairs_per_user=round(float(np.isfinite(P).sum(1).mean()), 1),
                                      positive_per_user=round(float((P > 0).sum(1).mean()), 1),
                                      mean_corr=round(float(np.nanmean(P)), 3),
                                      users_without_neighbors=int((np.nan_to_num(P) > 0).sum(1).__eq__(0).sum()))
# стратифікована вибірка 100 неоцінених вакансій пропорційно розмірам кластерів (ЛР1, I1);
# нові вакансії без жодної взаємодії не включаються – для них CF не може дати прогноз
has_inter = (R_full > 0).sum(0) > 0
cl_size = np.bincount(clusters[has_inter], minlength=K)
quota = np.floor(cl_size / cl_size.sum() * 100).astype(int)
quota[np.argsort(-(cl_size / cl_size.sum() * 100 - quota))[:100 - quota.sum()]] += 1
REPORT["quota"] = quota.tolist()


def sample_candidates(u, seed):
    r = np.random.default_rng(seed)
    out = []
    for c in range(K):
        pool = np.where((clusters == c) & has_inter & (R_full[u] == 0))[0]
        out += list(r.choice(pool, min(quota[c], len(pool)), replace=False))
    return np.array(out)


rec_rows, pred_rows = [], []
for g, s in GROUPS.items():
    for u in np.where(seg == s)[0]:
        cand = sample_candidates(u, 1000 + u)
        res = cf.all_methods(u, cand, BETA, K_BEST)
        for name, (p, nb) in res.items():
            for i, pp, n in zip(cand, p, nb):
                pred_rows.append((users.at[u, "user_id"], g, name, vacancies.at[i, "vacancy_id"], pp, n))
            ok = np.where(np.nan_to_num(p) >= cf.mu[u] + DELTA)[0]
            ok = ok[np.argsort(-p[ok], kind="stable")][:TOPN]
            for rank, j in enumerate(ok, 1):
                i = cand[j]
                rec_rows.append((users.at[u, "user_id"], g, name, rank, vacancies.at[i, "vacancy_id"], vacancies.at[i, "title"],
                                 int(clusters[i]), round(float(p[j]), 3), int(nb[j])))
preds = pd.DataFrame(pred_rows, columns=["user_id", "group", "method", "vacancy_id", "pred", "n_neighbors"])
preds.to_csv(os.path.join(RES, "predictions_100_candidates.csv.gz"), index=False, compression="gzip")
recs = pd.DataFrame(rec_rows, columns=["user_id", "group", "method", "rank", "vacancy_id", "title", "cluster", "pred", "n_neighbors"])
recs.to_csv(os.path.join(RES, "recommendations_cf.csv"), index=False)
assert recs.merge(pd.read_csv(os.path.join(CLEAN, "user_vacancy_scores.csv"))[["user_id", "vacancy_id"]], on=["user_id", "vacancy_id"]).empty

# =============================================================== 3. Порівняння
lab2 = pd.read_csv(os.path.join(BASE, "results", "lab2", "recommendations_top10.csv"))
lab2_pools = pd.read_csv(os.path.join(BASE, "results", "lab2", "pools", "pools_all.csv.gz"))
lab2 = lab2[lab2.group.isin(GROUPS)]
METHODS = ["R1", "R2", "R3", "R4", "R5"]
lists = {}
for (g, m, uid), d in recs.groupby(["group", "method", "user_id"]):
    lists.setdefault((g, m), {})[uid] = list(d.sort_values("rank")["vacancy_id"])
for (g, p, uid), d in lab2.groupby(["group", "pool", "user_id"]):
    lists.setdefault((g, "ЛР2-" + p), {})[uid] = list(d.sort_values("rank")["vacancy_id"])
cpos = {c: i for i, c in enumerate(vacancies["vacancy_id"])}
cl_of = dict(zip(vacancies["vacancy_id"], clusters))
seg_of = dict(zip(vacancies["vacancy_id"], vacancies["segment"]))
comp = {}
for g in GROUPS:
    n_users_g = int((seg == GROUPS[g]).sum())
    for m in METHODS + ["ЛР2-P1", "ЛР2-P2", "ЛР2-P3", "ЛР2-P4"]:
        L = lists.get((g, m), {})
        allrec = [c for l in L.values() for c in l]
        share = np.bincount([cl_of[c] for c in allrec], minlength=K) / max(1, len(allrec))
        # середня кількість спільних рекомендацій для пари користувачів групи
        ids = list(L)
        sets = [set(L[x]) for x in ids]
        rr = np.random.default_rng(0)
        pairs = [(a, b) for a, b in rr.integers(0, len(ids), size=(4000, 2)) if a != b] if len(ids) > 1 else []
        overlap = float(np.mean([len(sets[a] & sets[b]) for a, b in pairs])) if pairs else 0.0
        cnt = pd.Series(allrec).value_counts() if allrec else pd.Series(dtype=int)
        # різноманітність списку: 1 – середня контентна схожість пар у списку (ILD)
        ild = []
        for l in L.values():
            if len(l) >= 2:
                ix = [cpos[c] for c in l]
                sub = S_content[np.ix_(ix, ix)]
                ild.append(1 - sub[np.triu_indices(len(ix), 1)].mean())
        row = dict(users_with_recs=len(L), users_total=n_users_g, avg_len=round(len(allrec) / max(1, n_users_g), 2),
                   cluster_share=[round(float(x), 3) for x in share], overlap=round(overlap, 3),
                   repeat_ratio=round(len(allrec) / max(1, len(cnt)), 2), share_repeated=round(float((cnt >= 2).mean()), 3) if len(cnt) else 0,
                   ild=round(float(np.mean(ild)), 4) if ild else None,
                   n_clusters=round(float(np.mean([len({cl_of[c] for c in l}) for l in L.values()])), 2) if L else 0,
                   popular_share=round(float(np.mean([seg_of[c] == "популярний" for c in allrec])), 3) if allrec else 0,
                   rare_share=round(float(np.mean([seg_of[c] == "рідкісний" for c in allrec])), 3) if allrec else 0)
        if not m.startswith("ЛР2"):
            # частка рекомендацій ЛР3, що збігаються з ТОП-10 ЛР2 (будь-який пул) та що входять до пулів ЛР2
            l2 = {}
            for p in ["P1", "P2", "P3", "P4"]:
                for uid, l in lists.get((g, "ЛР2-" + p), {}).items():
                    l2.setdefault(uid, set()).update(l)
            pool_sets = {}
            for _, r in lab2_pools[(lab2_pools.group == g)].iterrows():
                if r["pool"] in ("P1", "P2", "P3", "P4_N50"):
                    pool_sets.setdefault((r["user_id"], r["pool"]), set(str(r["vacancy_ids"]).split(";")))
            tot = len(allrec)
            row["match_lr2_top10"] = round(sum(c in l2.get(uid, set()) for uid, l in L.items() for c in l) / max(1, tot), 4)
            for p in ("P1", "P2", "P3", "P4_N50"):
                row[f"in_pool_{p}"] = round(sum(c in pool_sets.get((uid, p), set()) for uid, l in L.items() for c in l) / max(1, tot), 4)
        comp[f"{g}|{m}"] = row
REPORT["comparison"] = comp

# =============================================================== 4. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, g in zip(axes, GROUPS):
    ms = METHODS + ["ЛР2-P1", "ЛР2-P3", "ЛР2-P4"]
    bottom = np.zeros(len(ms))
    for c in range(K):
        vals = np.array([comp[f"{g}|{m}"]["cluster_share"][c] for m in ms])
        ax.bar(ms, vals, bottom=bottom, color=plt.cm.tab20(c), label=f"кл. {c}")
        bottom += vals
    ax.set_title(f"{g.capitalize()}: частка рекомендацій з кожного кластера"); ax.tick_params(axis="x", rotation=40)
axes[1].legend(ncol=1, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "cluster_share.png"), dpi=140); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
ms = METHODS + ["ЛР2-P1", "ЛР2-P2", "ЛР2-P3", "ЛР2-P4"]
for ax, key, title in zip(axes, ["overlap", "ild", "avg_len"], ["Сер. к-сть спільних рекомендацій пари", "Різноманітність списку (ILD)", "Сер. довжина списку"]):
    for j, g in enumerate(GROUPS):
        ax.bar(np.arange(len(ms)) + (j - 0.5) * 0.4, [comp[f"{g}|{m}"][key] or 0 for m in ms], width=0.4, label=g)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=45, fontsize=7); ax.set_title(title, fontsize=9); ax.grid(axis="y", alpha=0.3)
axes[0].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "comparison_metrics.png"), dpi=140); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for g in GROUPS:
    ax[0].plot(beta_grid, [REPORT["beta_search"][b][g]["MAE"] for b in beta_grid], marker="o", label=g)
    ax[1].plot(k_grid, [REPORT["k_search"][k][g]["MAE"] for k in k_grid], marker="o", label=g)
ax[0].set_xlabel("β (поріг ко-рейтингових позицій)"); ax[0].set_ylabel("MAE"); ax[0].set_title("Підбір β для рекомендації 3"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].set_xlabel("k (кількість найкращих сусідів)"); ax[1].set_title("Підбір k для рекомендації 4"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "param_search.png"), dpi=140); plt.close()

# =============================================================== 5. Приклади
ex = {}
rs = np.random.default_rng(11)
for g, s in GROUPS.items():
    ids = [x for x in recs[(recs.group == g)].groupby("user_id")["method"].nunique().sort_values(ascending=False).index[:40]]
    ex[g] = []
    for uid in rs.choice(ids, 2, replace=False):
        u = uidx[uid]
        d = recs[recs.user_id == uid]
        ex[g].append(dict(user_id=uid, n_vacancies=int(users.at[u, "n_vacancies"]), mean=round(float(cf.mu[u]), 2), faculty=users.at[u, "faculty"],
                          recs={m: d[d.method == m].sort_values("rank")[["vacancy_id", "title", "cluster", "pred", "n_neighbors"]].values.tolist() for m in METHODS},
                          top_neighbors=[(users.at[v, "user_id"], round(float(cf.P[u, v]), 3), int(cf.N[u, v]))
                                         for v in np.argsort(-np.nan_to_num(cf.P[u], nan=-9))[:5]]))
REPORT["examples"] = ex
REPORT["n_rec_users"] = recs.groupby(["group", "method"])["user_id"].nunique().reset_index().values.tolist()
pp = preds.groupby(["group", "method"])["pred"].agg(lambda x: round(float(x.notna().mean()), 3)).reset_index().values.tolist()
REPORT["pred_coverage"] = pp
REPORT["pred_mean"] = preds.groupby(["group", "method"])["pred"].mean().round(3).reset_index().values.tolist()
json.dump(REPORT, open(os.path.join(RES, "lab3_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
          default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o))
print(json.dumps({k: REPORT[k] for k in ["BETA", "K_BEST", "n_rec_users", "pred_coverage"]}, ensure_ascii=False))
