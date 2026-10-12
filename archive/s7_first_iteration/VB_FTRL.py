import numpy as np

class VB_FTRL:
    def __init__(self, d, mu=6, lambda_val=7, S=10, max_steps=1000):
        self.d = d
        self.mu = mu
        self.lambda_val = lambda_val
        self.S = S
        # Préallocation des retours
        self.max_steps = max_steps
        self.R = np.zeros((max_steps, d))       # (max_steps, d)
        self.RA = np.zeros((max_steps, d-1))    # (max_steps, d-1)
        self.t = 0                              # nombre de retours ajoutés

        # Poids initiaux uniformes
        self.w = np.full(d, 1/d)

        # Matrices affines pré-calculées
        I = np.eye(d - 1)
        self.A = np.vstack([I, -np.ones((1, d - 1))])  # (d, d-1)
        self.A_T = self.A.T                            # (d-1, d)
        self.A_pinv = np.linalg.pinv(self.A)           # (d-1, d)
        self.e_d = np.zeros(d); self.e_d[-1] = 1

    def affine_mapping(self, v):
        return self.A.dot(v) + self.e_d

    def inverse_affine_mapping(self, w):
        return self.A_pinv.dot(w - self.e_d)

    def update(self, return_t):
        # Stockage sans realloc
        if self.t >= self.max_steps:
            raise IndexError("Nombre de retours > max_steps, ajustez max_steps")
        self.R[self.t] = return_t
        self.RA[self.t] = return_t.dot(self.A)
        self.t += 1

        # Initialisation de v
        v = self.inverse_affine_mapping(self.w)

        # Boucle FTRL
        for _ in range(self.S):
            aff_v = self.affine_mapping(v)                 # (d,)

            # Construction de x_aff
            x_aff = np.empty(self.d + self.t)
            x_aff[:self.d] = aff_v
            x_aff[self.d:] = self.R[:self.t].dot(aff_v)

            # Construction de lambda_tau
            n = self.d + self.t
            lambda_tau = np.empty(n)
            lambda_tau[:self.d] = self.lambda_val
            lambda_tau[self.d:] = 1

            # Gradients
            grad0 = -self.A / aff_v[:, None]               # (d, d-1)
            grad_rest = -self.RA[:self.t] / x_aff[self.d:, None]  # (t, d-1)
            gradients = np.vstack([grad0, grad_rest])       # (n, d-1)

            # Hessien vectorisé
            weighted = gradients * lambda_tau[:, None]
            H = gradients.T.dot(weighted)                   # (d-1, d-1)

            # Inversion via solve
            H_inv = np.linalg.solve(H, np.eye(H.shape[0]))

            # Calcul de pi_tau
            Gh = gradients.dot(H_inv)                       # (n, d-1)
            pi_tau = lambda_tau * np.sum(Gh * gradients, axis=1)

            # Calcul du gradient g:
            # - Le terme 2 * A_T @ ones(d) vient du gradient de la régularisation quadratique ||v||^2,
            #   car ∇(vᵀv) = 2v, et v est mappé via A vers l'espace w.
            # - Le terme gradients.T.dot(lambda_tau + mu * pi_tau) collecte les contributions
            #   du log-loss sur chaque dimension avec pondération mu * pi_tau.
            g = 2 * self.A_T.dot(np.ones(self.d)) + gradients.T.dot(lambda_tau + self.mu * pi_tau)

            # Construction de la matrice M pour la mise à jour:
            # M = H + 3 * mu * Σ_i π_i ∇_i ∇_iᵀ
            # Le facteur 3 apparaît car, au second ordre, l'entropie FTRL induit un
            # terme supplémentaire proportionnel à 3 * mu dans la combinaison du Hessien
            M = H + 3 * self.mu * gradients.T.dot(gradients * pi_tau[:, None])

            # Mise à jour de v via résolution du système linéaire
            v -= np.linalg.solve(M, g)

        # Mise à jour finale de w
        self.w = self.affine_mapping(v)

