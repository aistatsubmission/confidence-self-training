import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.special import softmax

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax

# --- data + model helpers (same as before) ---

from sklearn.datasets import make_moons

# --- new data generator for Appendix experiment ---
def make_moon_data(n=500, angle=0.0, seed=0):
    """
    Generates rotated two-moon data to mimic gradual domain shift.
    """
    np.random.seed(seed)
    X, y = make_moons(n_samples=n*2, noise=0.1, random_state=seed)
    # rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    X = X @ R.T
    return X, y

def make_gaussian_data(n=500, angle=0.0, seed=0):
    np.random.seed(seed)
    mean0, mean1 = np.array([-2, 0]), np.array([+2, 0])
    cov = np.eye(2)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    X0 = np.random.multivariate_normal(mean0, cov, n)
    X1 = np.random.multivariate_normal(mean1, cov, n)
    X = np.vstack([X0, X1]) @ R.T
    y = np.array([0]*n + [1]*n)
    return X, y

#def train_classifier(X, y):
#    clf = LogisticRegression(max_iter=200)
#    clf.fit(X, y)
#    return clf

from sklearn.svm import SVC

def train_classifier(X, y):
    clf = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale")
    clf.fit(X, y)
    return clf

def scores_preds(clf, X):
    # logits, probs, confidence, margin, preds
    logits = clf.decision_function(X)
    if logits.ndim == 1:  # binary case
        logits = np.stack([-logits, logits], axis=1)
    probs = softmax(logits, axis=1)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    top2 = np.sort(logits, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return conf, margin, preds

# 0-1 loss
def zero_one(preds, labels):
    return (preds != labels).astype(float)
import numpy as np
import matplotlib.pyplot as plt

# --- helper to plot decision boundary ---
def plot_decision_boundary(clf, X, y_true, accepted_mask, title, fname=None):
    # grid for contour
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    # background decision regions
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")

    # plot accepted vs rejected
    plt.scatter(X[~accepted_mask,0], X[~accepted_mask,1],
                c=y_true[~accepted_mask], cmap="coolwarm", 
                marker="x", alpha=0.3, label="rejected")
    plt.scatter(X[accepted_mask,0], X[accepted_mask,1],
                c=y_true[accepted_mask], cmap="coolwarm", 
                edgecolor="k", s=40, label="accepted")

    plt.title(title,fontsize=16)
    plt.legend()
    if fname:
        plt.savefig(fname, bbox_inches="tight") #,pad_inches=0.1
    plt.show()

# ---------------------------------------------------------
# GDA loop with CORRECT epsilon per your Definition (masked risks)
# ---------------------------------------------------------
def gradual_self_training(diss_func=make_moon_data,
    K=20, n_per_domain=300, filter_type="confidence", c=0.5, seed=0 , plot=False
):
    # Source (labeled)
    X0, y0 = diss_func(n_per_domain, angle=0.0, seed=seed)
    scaler = StandardScaler().fit(X0)
    X0 = scaler.transform(X0)
    h_prev = train_classifier(X0, y0)

    phi_list, eps_list, rho_list = [], [], []

    for k in range(1, K+1):
        # Unlabeled batch from domain μ_k
        angle = np.pi/4 * (k / K)    # gradual angle shift up to 45°
        Xk, yk = diss_func(n_per_domain, angle=angle, seed=seed + k)
        Xk = scaler.transform(Xk)

        # Scores & acceptance from h_{k-1}
        conf, margin, preds_prev = scores_preds(h_prev, Xk)
        scores = conf if filter_type == "confidence" else margin

        # Percentile schedule: q_k = 1 - c/k  (coverage target)
        qk = 1- c/k
        theta_k = np.quantile(scores, 1 - qk)
        A = (scores >= theta_k)          # acceptance mask 𝔄_k(x)
        rho_k = A.mean()                  # coverage ρ_k

        # Pseudo-labels from h_{k-1}
        y_tilde = preds_prev

        # ---- Train h_k on accepted pseudo-labels (ERM on \tilde S_k) ----
        if A.sum() > 0:
            h_curr = train_classifier(Xk[A], y_tilde[A])
        else:
            # Fallback: train on entire batch with pseudo-labels (shouldn't happen often)
            h_curr = train_classifier(Xk, y_tilde)

        # Predictions of h_k on X_k (for risks)
        _, _, preds_curr = scores_preds(h_curr, Xk)

        # -------- Compute φ_k (conditional error on ACCEPTED set) --------
        if A.sum() > 0:
            phi_k = zero_one(preds_prev[A], yk[A]).mean()
        else:
            phi_k = 0.0

        # -------- Compute ε_k exactly from masked risks --------
        # Masked true risk: E[ A * 1(h(x) != y) ]
        # Masked pseudo-risk: E[ A * 1(h(x) != \tilde y) ]
        # We estimate expectations by averages over the current unlabeled batch.

        # h = h_{k-1}
        over_true_prev   = (A * zero_one(preds_prev, yk)).mean()
        over_pseudo_prev = (A * zero_one(preds_prev, y_tilde)).mean()   # == 0 by construction
        diff_prev = abs(over_true_prev - over_pseudo_prev)

        # h = h_k
        over_true_curr   = (A * zero_one(preds_curr, yk)).mean()
        over_pseudo_curr = (A * zero_one(preds_curr, y_tilde)).mean()
        diff_curr = abs(over_true_curr - over_pseudo_curr)

        eps_k = max(diff_prev, diff_curr)
        if (plot):
            if k in [1, 5, 10, 20, 45]:  # pick rounds you want to visualize
                plot_decision_boundary(h_prev, Xk, yk, A,
                    title=f"Round {k}: Decision boundary ({filter_type})",
                    fname=f"./new/decision_boundary_{filter_type}_round{k}.png")

        # Store round stats
        phi_list.append(phi_k)
        eps_list.append(eps_k)
        rho_list.append(rho_k)

        # Next round's previous model
        h_prev = h_curr

    return np.array(phi_list), np.array(eps_list), np.array(rho_list)
def per_class_accept(scores, preds, qk, min_frac=0.10):
    """
    Accepts at least min_frac per predicted class, 
    and otherwise keeps the top qk fraction within each predicted class.
    """
    N = len(scores)
    mask = np.zeros(N, dtype=bool)
    for cls in np.unique(preds):
        idx = np.where(preds == cls)[0]
        if idx.size == 0:
            continue
        frac = max(qk, min_frac)
        k_keep = int(np.ceil(frac * idx.size))
        order = idx[np.argsort(-scores[idx])]
        mask[order[:k_keep]] = True
    return mask

def gradual_self_training_class( diss_func=make_moon_data,
    K=20, n_per_domain=300, filter_type="confidence", c=0.5, seed=0, plot=False
):
    # Source (labeled)
    X0, y0 = diss_func(n_per_domain, angle=0.0, seed=seed)
    scaler = StandardScaler().fit(X0)
    X0 = scaler.transform(X0)
    h_prev = train_classifier(X0, y0)

    phi_list, eps_list, rho_list = [], [], []
    for k in range(1, K+1):
        # Unlabeled batch from domain μ_k
        angle = np.pi/2 * (k / K)    # gradual angle shift up to 45°
        Xk, yk = diss_func(n_per_domain, angle=angle, seed=seed + k)
        Xk = scaler.transform(Xk)

        # Scores & acceptance from h_{k-1}
        conf, margin, preds_prev = scores_preds(h_prev, Xk)
        scores = conf if filter_type == "confidence" else margin

        # Percentile schedule: q_k = 1 - c/k
        qk = 1 - c/k
        #print (1-qk)
        A = per_class_accept(scores, preds_prev, qk, min_frac=0.10)
        rho_k = A.mean()  # coverage ρ_k

        # Pseudo-labels from h_{k-1}
        y_tilde = preds_prev

        # ---- Train h_k on accepted pseudo-labels (ERM on \tilde S_k) ----
        if A.sum() > 0:
            h_curr = train_classifier(Xk[A], y_tilde[A])
        else:
            h_curr = train_classifier(Xk, y_tilde)

        # Predictions of h_k on X_k
        _, _, preds_curr = scores_preds(h_curr, Xk)

        # -------- φ_k: conditional error on accepted set --------
        if A.sum() > 0:
            phi_k = zero_one(preds_prev[A], yk[A]).mean()
        else:
            phi_k = 0.0

        # -------- ε_k: masked risks --------
        over_true_prev   = (A * zero_one(preds_prev, yk)).mean()
        over_pseudo_prev = (A * zero_one(preds_prev, y_tilde)).mean()
        diff_prev = abs(over_true_prev - over_pseudo_prev)

        over_true_curr   = (A * zero_one(preds_curr, yk)).mean()
        over_pseudo_curr = (A * zero_one(preds_curr, y_tilde)).mean()
        diff_curr = abs(over_true_curr - over_pseudo_curr)

        eps_k = max(diff_prev, diff_curr)
        if (plot):
            if k in [1, 3, 5, 10, 20]:  # pick rounds you want to visualize
                plot_decision_boundary(h_prev, Xk, yk, A, #  h_prev or  h_curr? or both ?
                    title = f"Round {k}: Decision boundary ($h_{{{k-1}}}$ on $\\mu_{{{k}}}$)",
                    fname = f"./new/decision_boundary_{filter_type}_round{k}.png")

        # Store round stats
        phi_list.append(phi_k)
        eps_list.append(eps_k)
        rho_list.append(rho_k)

        h_prev = h_curr

    return np.array(phi_list), np.array(eps_list), np.array(rho_list)


def run_multi_seed(diss_func, K=45, n_per_domain=100, filter_type="confidence", c=0.5, n_seeds=10):
    all_phi, all_eps, all_rho = [], [], []
    for seed in range(n_seeds):
        phi, eps, rho = gradual_self_training_class(diss_func=diss_func,
            K=K, n_per_domain=n_per_domain, filter_type=filter_type, c=c, seed=seed, plot=False
        )
        all_phi.append(phi)
        all_eps.append(eps)
        all_rho.append(rho)

    # convert to arrays (n_seeds × K)
    all_phi = np.vstack(all_phi)
    all_eps = np.vstack(all_eps)
    all_rho = np.vstack(all_rho)

    # compute mean and std across seeds
    phi_mean, phi_std = all_phi.mean(axis=0), all_phi.std(axis=0)
    eps_mean, eps_std = all_eps.mean(axis=0), all_eps.std(axis=0)
    rho_mean, rho_std = all_rho.mean(axis=0), all_rho.std(axis=0)

    return phi_mean, eps_mean, rho_mean, phi_std, eps_std, rho_std


# ----------------------------
# 4. Run experiment
# ----------------------------
K = 20

diss = make_moon_data  #diss_func can be make_gaussian_data or make make_moon_data

phi_conf, eps_conf, rho_conf = gradual_self_training_class(diss_func= diss, K=K,n_per_domain=1000, filter_type="confidence", c=0.5, plot=True)
phi_margin, eps_margin, rho_margin = gradual_self_training_class(diss_func=diss, K=K,n_per_domain=1000, filter_type="margin" , c=0.7, plot=True)
