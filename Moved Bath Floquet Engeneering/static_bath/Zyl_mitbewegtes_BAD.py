#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, schur
from scipy.special import iv 

J = 1.0
bruch = 0.5            
#0.15 < Chern < 0.3333
#0.3333 < anomal < 0.66666
# 0.6666 < Chern
TJ3 = np.pi * bruch
T = 3*(TJ3) / J
omega = 2*np.pi / T
A = 1.7         # Modulationsstärke: klein = weich, groß = fast wie die 3 Stufen
dphi = -2*np.pi/3  # Phasenversatz der Bindungen (Vorzeichen = Reihenfolge = Drehsinn)
Nt = 600      # Zeitschritte pro Periode (für großes A mehr nehmen)
# im 3-Stufen-Modell. Mittel von exp(A cos) ist I_0(A), also norm = 3·I_0(A).
norm = 3*np.i0(A)
phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3   # −π/3: Pulse liegen bei T/6, T/2, 5T/6 (Mitte der Stufen)
w_c = 10 * omega

M, N = 20, 20                      # Zellen entlang b1 und b2
K = 2*M*N
idx = lambda x, y, z: x + y*M + z*M*N
T = 3*(TJ3)/J
                   
a1 = np.array([0, 1.0])
a2 = np.array([-np.sqrt(3)/2, -0.5])
a3 = np.array([ np.sqrt(3)/2, -0.5])
b1 = a1 - a2
b2 = a1 - a3
G = 2*np.pi*np.linalg.inv(np.array([b1, b2])).T
G1, G2 = G
 
def k_kartesisch(kx, ky):
    return (kx*G1 + ky*G2)/(2*np.pi)

phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3 

def f(i, t):
    return np.exp(A*np.cos(omega*t + phis[i]))/norm


# Bindungsprofile J_i(t) über eine Periode 
t_werte = np.linspace(0, T, 500)

plt.figure(figsize=(7, 3.5))

for i in range(3):
    plt.plot(t_werte/T, J*f(i, t_werte), label=f"Bindung {i+1} (a{i+1})")
for x in [1/3, 2/3]:
    plt.axvline(x, color='gray', ls=':', lw=1)      # Drittel der Periode
plt.xlabel("t / T")
plt.ylabel(r"$J_i(t)$")
plt.title(f"Bindungsprofile, A = {A}, Δφ = {dphi/np.pi:.2f}π")
plt.legend()
plt.tight_layout()
plt.show()
#%%
     # Zeitschritte pro Periode
 
# echte Positionen aller Plätze (für den Plot)
pos = np.zeros((K, 2))
for i in range(M):
    for j in range(N):
        pos[idx(i, j, 0)] = i*b1 + j*b2
        pos[idx(i, j, 1)] = i*b1 + j*b2 + a1
 
# Partner jedes Platzes in jeder Bindungsrichtung (−1 = kein Partner, Rand)
bindungen = [(0, 0), (-1, 0), (0, -1)]          # a1, a2, a3: A in Zelle (i,j) -> B in Zelle (i+di, j+dj)
partner = -np.ones((3, K), dtype=int)
for s, (di, dj) in enumerate(bindungen):
    for i in range(M):
        for j in range(N):
            iz, jz = i + di, j + dj
            if 0 <= iz < M and 0 <= jz < N:
                a, b = idx(i, j, 0), idx(iz, jz, 1)
                partner[s, a] = b
                partner[s, b] = a
hat_partner = partner >= 0
 
def bindung_anwenden(s, c, psi):
    """exp(-i c H_s) auf psi (Vektor oder Matrix mit Plätzen als erster Achse), exakt.
       Auf jedem Paar: [[cos(cJ), -i sin(cJ)], [-i sin(cJ), cos(cJ)]], Plätze ohne Partner bleiben."""
    neu = psi.copy()
    hp = hat_partner[s]
    neu[hp] = np.cos(c*J)*psi[hp] - 1j*np.sin(c*J)*psi[partner[s, hp]]
    return neu
 
# Floquet-Operator U über eine Periode (auf die Einheitsmatrix angewendet)
dt = T/Nt
U = np.eye(K, dtype=complex)
for n in range(Nt):
    t = (n + 0.5)*dt
    c1, c2, c3 = f(0, t)*dt, f(1, t)*dt, f(2, t)*dt
    U = bindung_anwenden(0, c1/2, U)
    U = bindung_anwenden(1, c2/2, U)
    U = bindung_anwenden(2, c3, U)
    U = bindung_anwenden(1, c2/2, U)
    U = bindung_anwenden(0, c1/2, U)
 
#Startzustände: ein Platz im Bulk, ein Platz am Rand
def start(platz):
    psi = np.zeros(K, complex); psi[platz] = 1.0
    return psi
starts = {"Bulk": idx(M//2, N//2, 0), "Rand": idx(M//2, 0, 0)}
 
perioden = [0, 3, 8, 15]
fig, ax = plt.subplots(2, len(perioden), figsize=(3.2*len(perioden), 6.4))
for zeile, (name, platz) in enumerate(starts.items()):
    psi = start(platz)
    n = 0
    for spalte, n_ziel in enumerate(perioden):
        while n < n_ziel:                                   # Periode für Periode weiterentwickeln
            psi = U @ psi
            n += 1
        p = np.abs(psi)**2
        a = ax[zeile, spalte]
        a.scatter(pos[:, 0], pos[:, 1], s=3, c='lightgray')
        a.scatter(pos[:, 0], pos[:, 1], s=np.where(p > 2e-3, 200*p, 0), c='crimson', linewidths=0)   # winzige p ausgeblendet
        a.set_title(f"{name}, n = {n}"); a.set_aspect('equal'); a.axis('off')
fig.suptitle(f"glatt moduliert: A = {A}, J·T/3 = π*{bruch}")
plt.tight_layout()
plt.show()
#%%
# =====================================================================================================
# ---------- Zylinder: periodisch entlang b1 (kx), offen entlang b2 (Nz Zellen) ----------
# =====================================================================================================
# Eigene Namen (Nz, Kz), damit K, N, idx, partner der Flocke nicht überschrieben werden.
# Plätze:  A in Zelle j -> Index j,   B in Zelle j -> Index Nz + j
# Braucht: J, A, T, omega, norm, phis, Nt, bruch, a1, b1, b2, k_kartesisch, iv

Nz = 30                                            # Zellen entlang b2 (Breite des Zylinders)
Kz = 2*Nz
x_A = np.array([j*b2 for j in range(Nz)])          # Positionen der A-Plätze (Zellursprung)
x_B = x_A + a1                                     # B-Plätze
bindungen_z = [(0, 0), (-1, 0), (0, -1)]           # a1, a2, a3: A(i,j) -> B(i+di, j+dj)

def bindungs_H(kx):
    """Die drei Bindungsrichtungen als getrennte Matrizen (Positions-Konvention), je mit Stärke J."""
    k = k_kartesisch(kx, 0.0)
    H = [np.zeros((Kz, Kz), complex) for _ in range(3)]
    for s, (di, dj) in enumerate(bindungen_z):
        for j in range(Nz):
            jz = j + dj
            if not (0 <= jz < Nz):
                continue
            delta = x_B[jz] + di*b1 - x_A[j]       # Vektor A -> B
            H[s][j, Nz + jz] = J*np.exp(1j*k @ delta)
    return [h + h.conj().T for h in H]

def f_alle(t):
    """Alle drei Bindungsstärken J_s(t)/J zur Zeit t."""
    return np.exp(A*np.cos(omega*t + phis))/norm

def bindungs_exp(Hs):
    """exp(−i c H_s) exakt: H_s besteht aus getrennten 2x2-Blöcken, H_s² = J² Π_s."""
    n = Hs[0].shape[0]
    Pis = [np.real(np.diag(h @ h))/J**2 for h in Hs]
    def expo(s, c):
        return np.eye(n) - np.diag((1 - np.cos(c*J))*Pis[s]) - 1j*np.sin(c*J)/J*Hs[s]
    return expo

def floquet_U(kx):
    """U(kx, T) mit Strang-Zerlegung (schnell)."""
    Hs = bindungs_H(kx)
    expo = bindungs_exp(Hs)
    dt = T/Nt
    U = np.eye(Kz, dtype=complex)
    for n in range(Nt):
        c = f_alle((n + 0.5)*dt)*dt
        U = expo(0, c[0]/2) @ U
        U = expo(1, c[1]/2) @ U
        U = expo(2, c[2])   @ U
        U = expo(1, c[1]/2) @ U
        U = expo(0, c[0]/2) @ U
    return U


#Floquet-Spektrum über kx, eingefärbt nach Rand
kxs = np.linspace(-np.pi, np.pi, 121)
rand_breite = 2
y_zelle = np.concatenate([np.arange(Nz), np.arange(Nz)])
unten_rand = y_zelle < rand_breite
oben_rand = y_zelle >= Nz - rand_breite

E_alle, farbe = [], []
for kx in kxs:
    lam, V = np.linalg.eig(floquet_U(kx))
    w = np.abs(V)**2
    E_alle.append(-np.angle(lam)/T)
    farbe.append(w[oben_rand].sum(0) - w[unten_rand].sum(0))   # +1 oberer Rand, −1 unterer Rand, 0 Bulk
E_alle, farbe = np.array(E_alle), np.array(farbe)

plt.figure(figsize=(6, 4))
plt.scatter(np.repeat(kxs, Kz), (E_alle*T/np.pi).ravel(), c=farbe.ravel(), cmap='coolwarm',
            vmin=-1, vmax=1, s=2)
plt.colorbar(label="unterer Rand (−) / oberer Rand (+) ")
plt.xlabel("$k_x$"); plt.ylabel("εT/π")
plt.title(f"Floquet-Spektrum Zylinder, A = {A}, JT/3 = {bruch}π")
plt.tight_layout(); plt.show()

#%%
# # ---------- Sambe-Raum (Fourier) zum Vergleich ----------
# Nf = 4

# def H_fourier(Hb, n):
#     return sum(iv(n, A)*np.exp(-1j*n*phis[i])/norm*Hb[i] for i in range(3))

# def floquet_H(kx):
#     """Sambe-Matrix: Block (m, m') = H_{m−m'} − m ω δ_{mm'}."""
#     Hb = bindungs_H(kx)
#     ms = np.arange(-Nf, Nf + 1)
#     D = len(ms)
#     Hn = {n: H_fourier(Hb, n) for n in range(-2*Nf, 2*Nf + 1)}
#     HF = np.zeros((D*Kz, D*Kz), complex)
#     for a_, m in enumerate(ms):
#         for b_, mp in enumerate(ms):
#             HF[a_*Kz:(a_+1)*Kz, b_*Kz:(b_+1)*Kz] = Hn[m - mp]
#         HF[a_*Kz:(a_+1)*Kz, a_*Kz:(a_+1)*Kz] -= m*omega*np.eye(Kz)
#     return HF

# kx_s = np.linspace(-np.pi, np.pi, 81)
# eps_sambe = np.array([np.linalg.eigvalsh(floquet_H(k)) for k in kx_s])
# plt.figure()
# plt.plot(kx_s, eps_sambe*T/np.pi, 'k.', ms=1)
# plt.ylim(-3, 3)
# plt.xlabel(r"$k_x$"); plt.ylabel(r"$\varepsilon T/\pi$")
# plt.title(f"Sambe, A = {A}, JT/3 = {bruch}π, Nf = {Nf}")
# plt.show()
# %%
#BAD:
# ---------- Ruhendes Bad, das an die Dichte koppelt (Dichtewellen e^{iq·r}), Bulk im k-Raum ----------
#
# Bad:  H_SB = η Σ_q Σ_i e^{iq·r_i} n_i B_q + h.c.     (jede Mode q ist ein eigenes Ohm'sches Bad)
# Unterschied zu bad_mod (σ_z):
#   - e^{iq·r} gibt dem Teilchen den Impuls q:  (k, β) -> (k+q, α).  k bleibt NICHT erhalten.
#   - Deshalb gibt es kein 2-Niveau-Problem pro k mehr, sondern eine Pauli-Gleichung über alle 2·Nk² Zustände.
#   - Matrixelement (Positions-Konvention): Überlapp der periodischen Anteile bei k+q und k;
#     läuft k+q aus der Brillouin-Zone heraus, bekommen die B-Plätze (bei R + a1) den Faktor e^{iG·a1}.
# Braucht: J, a1, a2, a3, G1, G2, dphi, schritt2, g_bath

def g_bath(E, kappa, wc, beta): #=Ohmsches Bad
    E = np.asarray(E, dtype=float)
    g = np.full(E.shape, 2*np.pi*kappa/beta)                 # Grenzwert E -> 0
    nz = np.abs(E) > 1e-12
    with np.errstate(over='ignore'):                         # große negative E: g -> 0
        g[nz] = 2*np.pi*kappa*E[nz]*np.exp(-np.abs(E[nz])/wc) / (-np.expm1(-beta*E[nz]))
    return g

#Zylinder:
# =====================================================================================================
# ---------- Zylinder-Version von schritt2 / zustaende_zeit ----------
# =====================================================================================================
# Geometrie wie in der Zylinder-Zelle: periodisch entlang b1 (Impuls kx), offen entlang b2 (Nz Zellen).
# Plätze:  A in Zelle j -> Index j,   B in Zelle j -> Index Nz + j        (D = 2·Nz Plätze pro kx)
# Konvention: Positions-Konvention, Bindung A -> B in Richtung s hat die Phase e^{+i k·a_s}.
# Braucht: J, a1, a2, a3, G1

def bindungen_zylinder(Nz):
    """Partner jedes Platzes je Bindungsrichtung (für Plätze ohne Partner: sich selbst),
       Maske 'hat Partner' und Vorzeichen der Bloch-Phase (+1 auf A, −1 auf B)."""
    D = 2*Nz
    partner = np.tile(np.arange(D), (3, 1))
    hat = np.zeros((3, D), bool)
    vorz = np.zeros((3, D))
    for s, (di, dj) in enumerate([(0, 0), (-1, 0), (0, -1)]):   # a1, a2, a3: A(i,j) -> B(i+di, j+dj)
        for j in range(Nz):
            jz = j + dj
            if 0 <= jz < Nz:                                      # offen entlang b2: sonst keine Bindung
                a, b = j, Nz + jz
                partner[s, a], partner[s, b] = b, a
                hat[s, a] = hat[s, b] = True
                vorz[s, a], vorz[s, b] = 1, -1
    return partner, hat, vorz


def schritt_zyl(fs, dt, X, partner, hat, ph):
    """Ein Zeitschritt X -> exp(−i H(t) dt) X für alle kx gleichzeitig (Strang: 1/2, 2/2, 3, 2/2, 1/2).
       Jede Bindungsrichtung H_s besteht aus getrennten Paaren -> exp(−i c H_s) exakt:
         neu_a = cos(cJ) X_a − i sin(cJ) e^{iθ_a} X_partner(a)       (Plätze ohne Partner bleiben)
       X: Form (Nkx, D, ...), fs: die drei Stärken J_s/J in der Mitte des Schritts,
       ph[s]: Bloch-Phasen e^{iθ_a}, Form (Nkx, D)."""
    def bindung(s, c, X):
        m = hat[s]
        neu = X.copy()
        phase = ph[s][:, m].reshape(ph[s][:, m].shape + (1,)*(X.ndim - 2))
        neu[:, m] = np.cos(c*J)*X[:, m] - 1j*np.sin(c*J)*phase*X[:, partner[s, m]]
        return neu
    c = fs*dt
    X = bindung(0, c[0]/2, X)
    X = bindung(1, c[1]/2, X)
    X = bindung(2, c[2],   X)
    X = bindung(1, c[1]/2, X)
    X = bindung(0, c[0]/2, X)
    return X


def zustaende_zeit_zyl(T, A, dphi, Nkx, Nz, Nt=300, statisch=False, jede=1):
    """Floquet-Zustände auf dem Zylinder.
       Gibt zurück:
         ts      Zeitpunkte, an denen u gespeichert ist (jeder 'jede'-te Schritt)
         u       periodische Anteile u[n, ikx, Platz, Zustand], Form (len(ts), Nkx, 2Nz, 2Nz)
         eps     Quasienergien eps[ikx, Zustand] in (−π/T, π/T], aufsteigend sortiert
         kxs     die kx-Werte (reduzierte Koordinate entlang G1, 0 … 2π)
         y       Zellindex entlang b2 für jeden Platz (zum Klassifizieren: Rand / Bulk)"""
    omega = 2*np.pi/T
    norm = 3*np.i0(A)
    phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
    D = 2*Nz
    dt = T/Nt

    partner, hat, vorz = bindungen_zylinder(Nz)
    kxs = 2*np.pi*np.arange(Nkx)/Nkx
    kc = kxs[:, None]*G1[None, :]/(2*np.pi)                     # kartesisches k, Form (Nkx, 2)
    # Bloch-Phase e^{±i k·a_s} für jeden Platz (Vorzeichen 0 auf Plätzen ohne Partner -> Phase 1)
    ph = [np.exp(1j*vorz[s][None, :]*(kc @ d)[:, None]) for s, d in enumerate((a1, a2, a3))]

    def fs_von(n):
        return np.full(3, 1/3) if statisch else np.exp(A*np.cos(omega*(n + 0.5)*dt + phis))/norm

    # ---- Durchlauf 1: U(kx, T) ----
    U = np.broadcast_to(np.eye(D, dtype=complex), (Nkx, D, D)).copy()
    for n in range(Nt):
        U = schritt_zyl(fs_von(n), dt, U, partner, hat, ph)

    # ---- Floquet-Zustände: Schur statt eig -> auch bei Entartung orthonormale Eigenvektoren ----
    eps = np.zeros((Nkx, D))
    Z = np.zeros((Nkx, D, D), complex)
    for ik in range(Nkx):
        Dm, Zk = schur(U[ik], output='complex')
        e = -np.angle(np.diag(Dm))/T
        r = np.argsort(e)
        eps[ik], Z[ik] = e[r], Zk[:, r]

    # ---- Durchlauf 2: Z mitentwickeln, u(t) = U(t) Z e^{iεt} speichern (spart ein zweites großes Array) ----
    n_speicher = np.arange(0, Nt, jede)
    ts = dt*n_speicher
    u = np.zeros((len(ts), Nkx, D, D), complex)
    X = Z.copy()
    zaehler = 0
    for n in range(Nt):
        if n % jede == 0:
            u[zaehler] = X*np.exp(1j*eps*n*dt)[:, None, :]
            zaehler += 1
        X = schritt_zyl(fs_von(n), dt, X, partner, hat, ph)

    y = np.concatenate([np.arange(Nz), np.arange(Nz)])
    return ts, u, eps, kxs, y


# =====================================================================================================
# ---------- Ruhendes Dichtebad auf dem Zylinder ----------
# =====================================================================================================
# Bad:  H_SB = η Σ_{q_x} Σ_j  S_{q_x, j} ⊗ B_{q_x, j} + h.c.,    S_{q_x, j} = Σ_{Plätze in Zeile j} e^{i q·r} n_r
#   - entlang x (periodisch): Dichtewellen e^{i q_x x}  ->  (kx, β) -> (kx + q_x, α)
#   - entlang y (offen):      ein unabhängiges Ohm'sches Bad pro Zeile j (kein q_y, weil kein k_y)
#   - q_x = 0 gehört jetzt dazu: die Zeilendichte Σ_x n_{x,j} vertauscht nicht mit H
#     (im k-Raum-Code wurde q = 0 weggelassen, weil die Gesamtdichte erhalten ist)
# Matrixelement für Zeile j (Positions-Konvention, wie bad_dichte):
#   <u_α(kx+q_x,t)| S_{q_x,j} |u_β(kx,t)> = u*_{A_j,α}(kx+q_x) u_{A_j,β}(kx) + e^{iwG1·a1} u*_{B_j,α}(kx+q_x) u_{B_j,β}(kx)
#   w = Anzahl, wie oft kx + q_x aus der Zone heraus gefaltet wurde
# Rate:  R[(kx+q_x, α) <- (kx, β)] = η² Σ_m ( Σ_j |X_j^(m)|² ) g(ε_β(kx) − ε_α(kx+q_x) + mω)
# Braucht: zustaende_zeit_zyl, g_bath, G1, a1

def bad_dichte_zyl(T, A, dphi, Nkx=12, Nz=20, eta=0.1, kappa=1.0, wc=None, kT=0.1,
                   Nt=300, jede=5, Nm=20, statisch=False, zeilen=None, daten=None, mit_R=False):
    """Pauli-Ratengleichung auf dem Zylinder, Bad = Dichtewellen in x, unabhängige Bäder pro Zeile in y.
       zeilen: bool-Maske der Länge Nz, welche Zeilen ans Bad koppeln (Standard: alle).
       daten:  Rückgabe von zustaende_zeit_zyl zum Wiederverwenden (spart die Zeitentwicklung).
       Gibt zurück: Besetzung p[ikx, Zustand], eps[ikx, Zustand], daten."""
    beta = 1/kT
    omega = 2*np.pi/T
    if wc is None:
        wc = 10*omega
    # ---------- Schritt 1 + 2: Floquet-Zustände und periodische Anteile auf dem Zylinder ----------
    if daten is None:
        daten = zustaende_zeit_zyl(T, A, dphi, Nkx, Nz, Nt, statisch, jede)
    ts, u, eps, kxs, y = daten
    Nkx, D = eps.shape
    Nz = D//2
    Nts = len(ts)
    if Nts < 2*Nm + 1:
        raise ValueError(f"nur {Nts} Zeitpunkte gespeichert, für Nm = {Nm} braucht es mindestens {2*Nm + 1}")
    zeilen = np.ones(Nz, bool) if zeilen is None else np.asarray(zeilen, bool)

    # ---------- Badmoden: q_x = −h … h (inkl. 0), zu jedem q_x auch −q_x ----------
    h = Nkx//2
    ms = np.arange(-Nm, Nm + 1)
    ik = np.arange(Nkx)
    R = np.zeros((Nkx*D, Nkx*D))                    # R[ziel, start] = Rate start -> ziel
    for n in range(-h, h + 1):
        iz = (ik + n) % Nkx                         # Index von kx + q_x (zurückgefaltet)
        w = (ik + n)//Nkx
        phaseB = np.exp(1j*w*(G1 @ a1))             # Faltungsphase auf B
        uz = u[:, iz]                               # u(kx + q_x, t), Form (Nts, Nkx, Platz, Zustand)

        # ---------- Schritt 3 + 4: Matrixelement pro Zeile, Seitenbänder, Σ_j |X_j^(m)|² ----------
        X2 = np.zeros((len(ms), Nkx, D, D))
        for j in np.nonzero(zeilen)[0]:
            Mel = (np.conj(uz[:, :, j, :])[..., :, None]*u[:, :, j, :][..., None, :]
                   + phaseB[None, :, None, None]
                   * np.conj(uz[:, :, Nz + j, :])[..., :, None]*u[:, :, Nz + j, :][..., None, :])   # (Nts, Nkx, α, β)
            X2 += np.abs(np.fft.ifft(Mel, axis=0)[ms % Nts])**2

        E = eps[None, :, None, :] - eps[iz][None, :, :, None] + ms[:, None, None, None]*omega
        rate = eta**2*np.sum(X2*g_bath(E, kappa, wc, beta), axis=0)     # (Nkx, α, β): (kx,β) -> (kx+q_x,α)
        ziel = (iz*D)[:, None, None] + np.arange(D)[None, :, None]
        start = (ik*D)[:, None, None] + np.arange(D)[None, None, :]
        R[np.broadcast_to(ziel, rate.shape), np.broadcast_to(start, rate.shape)] += rate   # Paare je q_x eindeutig

    # ---------- Schritt 5: stationärer Zustand  L p = 0,  Σ p = 1 ----------
    np.fill_diagonal(R, 0)
    L = R - np.diag(R.sum(axis=0))
    Lm = L.copy()
    Lm[-1] = 1.0                                    # eine (redundante) Gleichung durch die Normierung ersetzen
    rechts = np.zeros(len(L)); rechts[-1] = 1.0
    p = np.linalg.solve(Lm, rechts)
    #return p.reshape(Nkx, D), eps, daten #1 Teilchen
    erg = (p.reshape(Nkx, D), eps, daten) #viele Teilchen
    return erg + (R,) if mit_R else erg #viele Teilchen


def auswertung_zyl(p, eps, daten, rand_breite=2, schwelle=0.5):
    """Teilt die Zustände in Bulk / Rand bei y = 0 / Rand bei y = Nz−1 ein (zeitgemitteltes Gewicht)
       und gibt die Besetzungen zurück. p_unten_bulk ist auf die Bulk-Besetzung normiert."""
    ts, u, _, kxs, y = daten
    Nz = len(y)//2
    w = np.mean(np.abs(u)**2, axis=0)                          # (Nkx, Platz, Zustand)
    w_y0 = w[:, y < rand_breite, :].sum(1)
    w_yN = w[:, y >= Nz - rand_breite, :].sum(1)
    rand_y0, rand_yN = w_y0 > schwelle, w_yN > schwelle
    bulk = ~(rand_y0 | rand_yN)
    p_bulk = p[bulk].sum()
    return dict(p_unten_bulk=p[bulk & (eps < 0)].sum()/p_bulk,
                p_bulk=p_bulk, p_rand_y0=p[rand_y0].sum(), p_rand_yN=p[rand_yN].sum(),
                n_bulk=int(bulk.sum()), n_rand=int((rand_y0 | rand_yN).sum()),
                maske=dict(bulk=bulk, rand_y0=rand_y0, rand_yN=rand_yN))


# %%
# =====================================================================================================
# ---------- Mitbewegtes Bad:  n_i  ->  O_i(t) = P(t) n_i P†(t) ----------
# =====================================================================================================
#
# Idee (Iadecola et al., PRB 91, 235133, Sec. II.2 und IV):
#   U(t) = P(t) e^{-i H_F t},  P(t+T) = P(t),  P(0) = 1   (P = Mikrobewegung, unitär)
#   Ist P†(t) H_SB(t) P(t) zeitunabhängig, gibt es nur den Kanal m = 0  ->  Boltzmann bezüglich H_F.
#   Lösung: H_SB(t) = P(t) V P†(t) mit festem V.  Wir nehmen V = n_i (wie im Dichtebad).
#
#   Altes Bad:   V_q    = Σ_i e^{iq·r_i} n_i
#   Neues Bad:   V_q(t) = Σ_i e^{iq·r_i} O_i(t),     O_i(t) = |ψ_i(t)><ψ_i(t)|,   ψ_i(t) = P(t)|i>
#
# Varianten:
#   "n_i"          : altes ruhendes Dichtebad (Kontrolle, muss bad_dichte reproduzieren)
#   "ideal_exakt"  : exakt mitbewegt, im k-Raum: u(k,t) -> u(k,0) für alle t (schnell, ~1 s)
#   "voll"         : O_i(t) im Ortsraum, alle Plätze bis Abstand r_max (~2 min bei Nk = 12)
#   "r<=2"         : nur Plätze im eigenen Sechseck (Abstand ≤ 2)
#   "Dichte"       : nur Diagonale von O_i(t)     (wo ist das Gewicht)
#   "Kohärenz"     : nur Nicht-Diagonale von O_i  (Hüpfterme a†_j a_l)
#
# Braucht aus dem Skript: J, a1, b1, b2, G1, G2, dphi, g_bath, zustaende_zeit


## ---------- Mikrobewegung P(t) und Kopplungsprofil auf dem Zylinder ----------
# =====================================================================================================
# Geometrie (wie in der Zylinder-Zelle und in zylinder_zeit.py):
#   periodisch entlang b1 (M Zellen, Index i),  offen entlang b2 (Nz Zellen, Index j = 0 … Nz−1)
#   Plätze: idx(i, j, z) = (i mod M) + j·M + z·M·Nz,   z = 0: A, z = 1: B,   K = 2·M·Nz
# Unterschied zum Torus:
#   - Bindungen in Richtung a3 über den offenen Rand fehlen  ->  manche Plätze haben keinen Partner
#   - keine Translationsinvarianz entlang b2  ->  das Profil P(t)|i⟩ hängt von der Zeile j0 des Startplatzes ab
# Braucht: np, schur, J, a1, b1, b2

def zylinder(M, Nz):
    """Zylinder aus M×Nz Zellen. Gibt Anzahl Plätze, Index-Funktion, Positionen, Partner je Bindungsrichtung
       und die Maske hat[s, p] (True, wenn Platz p in Richtung s einen Partner hat).
       Plätze ohne Partner zeigen in partner auf sich selbst."""
    K = 2*M*Nz
    idx = lambda i, j, z: (i % M) + j*M + z*M*Nz       # j wird NICHT gefaltet (offen)
    pos = np.zeros((K, 2))
    for i in range(M):
        for j in range(Nz):
            pos[idx(i, j, 0)] = i*b1 + j*b2
            pos[idx(i, j, 1)] = i*b1 + j*b2 + a1
    bindungen = [(0, 0), (-1, 0), (0, -1)]              # a1, a2, a3: A(i,j) -> B(i+di, j+dj)
    partner = np.tile(np.arange(K), (3, 1))
    hat = np.zeros((3, K), bool)
    for s, (di, dj) in enumerate(bindungen):
        for i in range(M):
            for j in range(Nz):
                jz = j + dj
                if 0 <= jz < Nz:                         # über den offenen Rand: keine Bindung
                    a, b = idx(i, j, 0), idx(i + di, jz, 1)
                    partner[s, a], partner[s, b] = b, a
                    hat[s, a] = hat[s, b] = True
    return K, idx, pos, partner, hat


def P_von_t_zyl(T, A, dphi, M=10, Nz=12, Nt=300):
    """Mikrobewegung P(t_n) = U(t_n) e^{+i H_F t_n} auf dem Zylinder, t_n = n·T/Nt.
       Zweig von H_F: Quasienergien in (−π/T, π/T].
       Gibt zurück: ts, P (Nt, K, K), pos, idx, M, Nz, fehler, eps (Quasienergien), y (Zeile jedes Platzes)."""
    omega = 2*np.pi/T
    norm = 3*np.i0(A)
    phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
    K, idx, pos, partner, hat = zylinder(M, Nz)

    def bindung(s, c, X):                            # exp(−i c J H_s), exakt; Plätze ohne Partner bleiben
        form = (-1,) + (1,)*(X.ndim - 1)
        co = np.where(hat[s], np.cos(c*J), 1.0).reshape(form)
        si = np.where(hat[s], np.sin(c*J), 0.0).reshape(form)
        return co*X - 1j*si*X[partner[s]]

    def schritt(n, X):
        c = np.exp(A*np.cos(omega*(n + 0.5)*dt + phis))/norm*dt
        X = bindung(0, c[0]/2, X); X = bindung(1, c[1]/2, X); X = bindung(2, c[2], X)
        X = bindung(1, c[1]/2, X); X = bindung(0, c[0]/2, X)
        return X

    dt = T/Nt
    # Durchlauf 1: U(T)
    U = np.eye(K, dtype=complex)
    for n in range(Nt):
        U = schritt(n, U)
    D, Z = schur(U, output='complex')                # U(T) = Z D Z†
    eps = -np.angle(np.diag(D))/T
    # Durchlauf 2: X(t) = U(t) Z mitführen, P(t) = X(t) e^{iεt} Z†   (spart das Array U(t) für alle t)
    ts = dt*np.arange(Nt)
    P = np.zeros((Nt, K, K), complex)
    X = Z.copy()
    for n in range(Nt):
        P[n] = (X*np.exp(1j*eps*ts[n])[None, :]) @ Z.conj().T
        X = schritt(n, X)
    # Kontrolle: P(T) = U(T) e^{iH_F T} muss die Eins sein
    fehler = np.abs((X*np.exp(1j*eps*T)[None, :]) @ Z.conj().T - np.eye(K)).max()
    y = (np.arange(K) % (M*Nz))//M
    return ts, P, pos, idx, M, Nz, fehler, eps, y


def P_spalten_zyl(T, A, dphi, M, Nz, Nt, spalten, jede=1):
    """Nur die Spalten P(t_n)|i⟩ für die Startplätze i in 'spalten', an jedem 'jede'-ten Zeitschritt.
       Spart gegenüber P_von_t_zyl den Speicher für die volle Matrix P (Nt, K, K).
       Gibt zurück: ts, Ps (len(ts), K, len(spalten)), pos, idx, fehler, y."""
    omega = 2*np.pi/T
    norm = 3*np.i0(A)
    phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
    K, idx, pos, partner, hat = zylinder(M, Nz)
    dt = T/Nt

    def bindung(s, c, X):
        form = (-1,) + (1,)*(X.ndim - 1)
        co = np.where(hat[s], np.cos(c*J), 1.0).reshape(form)
        si = np.where(hat[s], np.sin(c*J), 0.0).reshape(form)
        return co*X - 1j*si*X[partner[s]]

    def schritt(n, X):
        c = np.exp(A*np.cos(omega*(n + 0.5)*dt + phis))/norm*dt
        X = bindung(0, c[0]/2, X); X = bindung(1, c[1]/2, X); X = bindung(2, c[2], X)
        X = bindung(1, c[1]/2, X); X = bindung(0, c[0]/2, X)
        return X

    U = np.eye(K, dtype=complex)
    for n in range(Nt):
        U = schritt(n, U)
    D, Z = schur(U, output='complex')
    eps = -np.angle(np.diag(D))/T
    Zs = Z.conj().T[:, spalten]                                   # Z† |i⟩
    n_sp = np.arange(0, Nt, jede)
    Ps = np.zeros((len(n_sp), K, len(spalten)), complex)
    X = Z.copy()
    zaehler = 0
    for n in range(Nt):
        if n % jede == 0:
            Ps[zaehler] = (X*np.exp(1j*eps*n*dt)[None, :]) @ Zs   # P(t) |i⟩ = U(t) Z e^{iεt} Z† |i⟩
            zaehler += 1
        X = schritt(n, X)
    fehler = np.abs((X*np.exp(1j*eps*T)[None, :]) @ Z.conj().T - np.eye(K)).max()
    y = (np.arange(K) % (M*Nz))//M
    return dt*n_sp, Ps, pos, idx, fehler, y


def kopplungs_profil_zyl(T, A, dphi, M=10, Nz=12, Nt=300, r_max=4.6, zeilen=None, P_daten=None, jede=1,
                         ausgabe=True):
    """ψ_i(t) = P(t)|i⟩ für Startplätze in den Zeilen j0 (je ein A- und ein B-Platz), auf allen Plätzen bis r_max.
       Entlang b1 periodisch (Minimalbild), entlang b2 offen.
       Rückgabe: dict  profile[j0] = [Profil A-Platz, Profil B-Platz]   (wie die Liste beim Torus, aber pro Zeile).
       Jedes Profil enthält zusätzlich y (Zeile jedes Platzes) und j0.
       P_daten: Rückgabe von P_von_t_zyl zum Wiederverwenden; sonst werden nur die nötigen Spalten von P berechnet.
       jede: nur jeden 'jede'-ten Zeitschritt behalten (muss zu zustaende_zeit_zyl passen)."""
    if zeilen is None:
        zeilen = range(Nz)
    zeilen = list(zeilen)
    if P_daten is None:
        starts = [(M//2) + j0*M + z*M*Nz for j0 in zeilen for z in (0, 1)]   # = idx(M//2, j0, z)
        ts, Ps, pos, idx, fehler, y = P_spalten_zyl(T, A, dphi, M, Nz, Nt, starts, jede)
        spalte = {s: c for c, s in enumerate(starts)}
        psi_von = lambda start: Ps[:, :, spalte[start]]
    else:
        ts, P, pos, idx, M, Nz, fehler, eps, y = P_daten
        psi_von = lambda start: P[::jede, :, start]
    if ausgabe:
        print(f"  Kontrolle |P(T) − 1| = {fehler:.1e}")
    L1 = M*b1
    s_j = (np.arange(len(pos)) >= M*Nz).astype(int)              # Untergitter jedes Platzes
    profile = {}
    for j0 in zeilen:
        profile[j0] = []
        for z in (0, 1):                                          # z = 0: A-Platz, z = 1: B-Platz
            start = idx(M//2, j0, z)
            tau_z = pos[start] - pos[idx(M//2, j0, 0)]            # Lage in der Zelle: 0 (A) oder a1 (B)
            disp = pos - pos[start]
            kand = np.stack([disp + u*L1 for u in (-1, 0, 1)])    # Minimalbild nur entlang b1
            disp = kand[np.argmin(np.linalg.norm(kand, axis=2), axis=0), np.arange(len(pos))]
            abstand = np.linalg.norm(disp, axis=1)
            sel = abstand <= r_max
            profile[j0].append(dict(r=(tau_z + disp)[sel],        # Position relativ zum Zellursprung von i
                                    s=s_j[sel], abstand=abstand[sel], y=y[sel], j0=j0,
                                    psi=psi_von(start)[:, sel]))  # ψ_j(t), Form (Nt/jede, n_j)
    return profile


BAUSTEINE = {"Dichte": 0.0, "NN": 1.0, "NNN": np.sqrt(3), "quer": 2.0}   # Abstand |r_j − r_l| je Baustein


def o_matrix(prof, variante):
    """o_jl(t) = ψ_j(t) ψ*_l(t) bzw. der gewählte Teil davon. Form (Nt, n, n).
       Arbeitet auf einem einzelnen Profil, egal ob Torus oder Zylinder.
       variante:
         "voll"                        alles (innerhalb r_max)
         "r<=2"                        nur Plätze im eigenen Sechseck (Abstand ≤ 2 vom Startplatz)
         "Kohärenz"                    alles außer der Diagonale
         Kombination mit "+" aus den Bausteinen
             "Dichte" (j = l), "NN" (Abstand 1), "NNN" (√3), "quer" (2, quer durchs Sechseck),
           jeweils optional mit Endung "_reell" (nur Re o) oder "_strom" (nur i·Im o),
           z. B. "NN", "Dichte+NN+quer", "NN_strom+quer".
       Unbekannte Namen geben einen Fehler (früher wurden sie stillschweigend wie "voll" behandelt)."""
    psi, ab, r = prof["psi"], prof["abstand"], prof["r"]
    o = psi[:, :, None]*np.conj(psi[:, None, :])
    n = len(ab)
    if variante == "voll":
        return o
    if variante == "r<=2":
        m = ab <= 2 + 1e-6
        return o*(m[:, None] & m[None, :])[None]
    if variante == "Kohärenz":
        return o*(1 - np.eye(n))[None]
    dist = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=2)
    ergebnis = np.zeros_like(o)
    for teil in variante.split("+"):
        teil = teil.strip()
        name, _, art = teil.partition("_")
        if name not in BAUSTEINE or art not in ("", "reell", "strom"):
            raise ValueError(f"unbekannte Variante '{variante}' (Teil '{teil}'). Erlaubt: 'voll', 'r<=2', 'Kohärenz' "
                             f"oder Kombinationen aus {list(BAUSTEINE)} mit '+', optional mit '_reell'/'_strom'.")
        m = np.isclose(dist, BAUSTEINE[name])[None]
        teil_o = o if art == "" else (o.real if art == "reell" else 1j*o.imag)
        ergebnis = ergebnis + teil_o*m
    return ergebnis


# =====================================================================================================
# ---------- Mitbewegtes Bad auf dem Zylinder ----------
# =====================================================================================================
# Badwellen wie in bad_prof_zyl (Standard bad_modell="wellen": e^{iq·r} entlang x UND y; alternativ "lokal", "zeile"),
# aber das Bad koppelt an die MITBEWEGTE Dichte:
#     V_{q_x, j0}(t) = Σ_i Σ_z e^{iq·r_{i,j0,z}} O_{i,j0,z}(t),     O(t) = P(t) n P†(t)   (bzw. eine Variante)
# Matrixelement (Positions-Konvention, ψ_site = e^{ik·r_site} u_site):
#     ⟨kx+q_x, α| V_{q_x,j0}(t) |kx, β⟩ = Σ_z e^{iq·τ_z} Σ_{j,l} o^{(j0,z)}_{jl}(t)
#                                         · e^{−i(k+q)·r_j} u*_{site_j, α}(k+q, t) · e^{ik·r_l} u_{site_l, β}(k, t)
#     r_j relativ zum Zellursprung des Startplatzes, site_j = y_j + s_j·Nz  (Platzindex in u)
# Rate ("wellen"):  R[(kx+q_x, α) <- (kx, β)] = η² Σ_m Σ_{q_⊥} |Σ_{j0,z} e^{iq_⊥·(j0·b2 + τ_z)} X_{j0,z}^(m)|² g(E_m)
#      ("zeile"):   Σ_{j0} |Σ_z X_{j0,z}|²        ("lokal"):  Σ_{j0,z} |X_{j0,z}|²
#
# Varianten:
#   "n_i"          ruhendes Dichtebad (Kontrolle, = bad_dichte_zyl)
#   "ideal_exakt"  exakt mitbewegt: u(kx,t) -> u(kx,0), nur der Kanal m = 0 bleibt
#   sonst          Ortsraum-Profil pro Zeile mit o_matrix(prof, variante): "voll", "r<=2", "Dichte", "Kohärenz", …
# Braucht: zustaende_zeit_zyl, g_bath, kopplungs_profil_zyl, o_matrix, G1, a1

def bad_mitbewegt_zyl(T, A, dphi, variante="ideal_exakt", eta=0.1, kappa=1.0, wc=None, kT=0.02,
                      Nkx=12, Nz=20, Nt=300, jede=5, Nm=20, M_prof=10, r_max=4.6,
                      zeilen=None, profile=None, daten=None, bad_modell="wellen", Nqy=None):
    """Pauli-Ratengleichung auf dem Zylinder mit mitbewegter Badkopplung.
       bad_modell: wie in bad_prof_zyl
         "wellen" (Standard): Badwellen e^{iq·r} auch entlang y (q_⊥ = n·G2/Nqy, Standard Nqy = 2·Nz)
         "lokal":             jeder Platz (A bzw. B einer Zeile) hat ein eigenes Bad
         "zeile":             alle Plätze einer Zeile koppeln kohärent an ein Bad (frühere Version)
       zeilen:  welche Zeilen ans Bad koppeln (Liste/Maske, Standard: alle)
       profile: Rückgabe von kopplungs_profil_zyl zum Wiederverwenden (gleiches T, A, Nt, jede, Nz!)
       daten:   Rückgabe von zustaende_zeit_zyl zum Wiederverwenden
       M_prof:  Umfang (in Zellen) des Zylinders, auf dem P(t) für das Profil berechnet wird
       Gibt zurück: p[ikx, Zustand], eps[ikx, Zustand], daten, profile."""
    beta = 1/kT
    omega = 2*np.pi/T
    if wc is None:
        wc = 10*omega

    # ---------- Schritt 1 + 2: Zylinder-Zustände u(kx, t) = P(kx, t) φ(kx) ----------
    if daten is None:
        daten = zustaende_zeit_zyl(T, A, dphi, Nkx, Nz, Nt, False, jede)
    ts, u, eps, kxs, y = daten
    Nkx, D = eps.shape
    Nz = D//2
    Nts = len(ts)
    if Nts < 2*Nm + 1:
        raise ValueError(f"nur {Nts} Zeitpunkte gespeichert, für Nm = {Nm} braucht es mindestens {2*Nm + 1}")
    if bad_modell not in ("wellen", "lokal", "zeile"):
        raise ValueError(f"unbekanntes bad_modell '{bad_modell}' (erlaubt: 'wellen', 'lokal', 'zeile')")
    if Nqy is None:
        Nqy = 2*Nz

    if zeilen is None:
        bad_zeilen = list(range(Nz))
    else:
        zeilen = np.asarray(zeilen)
        bad_zeilen = list(np.nonzero(zeilen)[0]) if zeilen.dtype == bool else [int(j) for j in zeilen]

    ortsraum = variante not in ("n_i", "ideal_exakt")
    if variante == "ideal_exakt":
        # P†(t)(P n P†)P(t) = n:  statt u(kx,t) überall u(kx,0) = φ(kx)  ->  nur m = 0
        u = np.broadcast_to(u[:1], u.shape).copy()
    kc = kxs[:, None]*G1[None, :]/(2*np.pi)                       # kartesisches k, Form (Nkx, 2)
    tau = [np.zeros(2), a1]                                       # Lage von A bzw. B in der Zelle
    if ortsraum:
        if profile is None:
            profile = kopplungs_profil_zyl(T, A, dphi, M=M_prof, Nz=Nz, Nt=Nt, r_max=r_max,
                                           zeilen=bad_zeilen, jede=jede, ausgabe=False)
        os_ = {j0: [o_matrix(pr, variante) for pr in profile[j0]] for j0 in bad_zeilen}
        site = {j0: [pr["y"] + pr["s"]*Nz for pr in profile[j0]] for j0 in bad_zeilen}   # Platzindex in u
        if os_[bad_zeilen[0]][0].shape[0] != Nts:
            raise ValueError("Profil und Zustände haben verschieden viele Zeitpunkte (gleiches Nt und jede nehmen)")

    def terme_der_zeile(j0, sl, uz, uq, phaseB, qv, kq):
        """Beiträge der Zeile j0 (A- und B-Platz) für die kx-Auswahl sl, mit ihrer Lage r = j0·b2 + τ_z."""
        terme, orte = [], []
        for z in (0, 1):
            if not ortsraum:                                      # ruhende Dichte n (bzw. n mit u(t) -> u(0))
                st = j0 + z*Nz
                fak = 1.0 if z == 0 else phaseB[None, sl, None, None]
                Mel = fak*np.conj(uz[:, sl, st, :])[..., :, None]*u[:, sl, st, :][..., None, :]
            else:                                                 # mitbewegte Dichte mit Profil der Zeile j0
                pr, o, st = profile[j0][z], os_[j0][z], site[j0][z]
                Aj = np.exp(1j*kq[sl] @ pr["r"].T)[None, :, :, None]*uq[:, sl][:, :, st, :]   # (Nts, nk, n, D)
                Bl = np.exp(1j*kc[sl] @ pr["r"].T)[None, :, :, None]*u[:, sl][:, :, st, :]
                oB = o[:, None] @ Bl                                                   # Σ_l o_jl B_l
                Mel = np.exp(1j*qv @ tau[z])*(np.conj(Aj).swapaxes(-1, -2) @ oB)       # (Nts, nk, α, β)
            terme.append(Mel)
            orte.append(j0*b2 + tau[z])
        return terme, orte

    h = Nkx//2
    ms = np.arange(-Nm, Nm + 1)
    ik = np.arange(Nkx)
    q_perp = [n2*G2/Nqy for n2 in range(-(Nqy//2), Nqy//2 + 1)]   # symmetrisch: zu jedem q auch −q
    R = np.zeros((Nkx*D, Nkx*D))                                  # R[ziel, start] = Rate start -> ziel
    for n in range(-h, h + 1):                                    # Badmoden q_x (inkl. 0)
        iz = (ik + n) % Nkx
        w = (ik + n)//Nkx
        phaseB = np.exp(1j*w*(G1 @ a1))                           # Faltungsphase auf B, Form (Nkx,)
        uz = u[:, iz]                                             # u(kx + q_x) zurückgefaltet
        qv = n*G1/Nkx                                             # kartesisches q_x
        kq = kc + qv                                              # k + q_x, nicht zurückgefaltet
        uq = uz.copy()
        uq[:, :, Nz:, :] *= np.conj(phaseB)[None, :, None, None]  # u_B(k+q) = e^{−iwG1·a1} u_B(zurückgefaltet)

        # ---------- Schritt 3 + 4: Matrixelemente, Seitenbänder, Σ |X^(m)|² ----------
        X2 = np.zeros((len(ms), Nkx, D, D))
        if bad_modell == "wellen":
            for k0 in range(Nkx):                                 # pro kx, damit der Speicher klein bleibt
                sl = slice(k0, k0 + 1)
                Xt, r_t = [], []
                for j0 in bad_zeilen:
                    terme, orte = terme_der_zeile(j0, sl, uz, uq, phaseB, qv, kq)
                    Xt += [np.fft.ifft(Mel, axis=0)[ms % Nts] for Mel in terme]
                    r_t += orte
                Xt, r_t = np.stack(Xt), np.array(r_t)             # (Terme, m, 1, α, β)
                for qp in q_perp:                                 # q_⊥: kohärent mit e^{iq_⊥·r} summieren
                    X2[:, sl] += np.abs(np.tensordot(np.exp(1j*(r_t @ qp)), Xt, axes=(0, 0)))**2
        else:
            for j0 in bad_zeilen:
                terme, _ = terme_der_zeile(j0, slice(None), uz, uq, phaseB, qv, kq)
                if bad_modell == "zeile":                         # ein Bad pro Zeile: kohärent summieren
                    terme = [sum(terme)]
                for Mel in terme:                                 # "lokal": jeder Platz ein eigenes Bad
                    X2 += np.abs(np.fft.ifft(Mel, axis=0)[ms % Nts])**2

        E = eps[None, :, None, :] - eps[iz][None, :, :, None] + ms[:, None, None, None]*omega
        rate = eta**2*np.sum(X2*g_bath(E, kappa, wc, beta), axis=0)
        ziel = (iz*D)[:, None, None] + np.arange(D)[None, :, None]
        start = (ik*D)[:, None, None] + np.arange(D)[None, None, :]
        R[np.broadcast_to(ziel, rate.shape), np.broadcast_to(start, rate.shape)] += rate

    # ---------- Schritt 5: stationärer Zustand ----------
    np.fill_diagonal(R, 0)
    L = R - np.diag(R.sum(axis=0))
    Lm = L.copy(); Lm[-1] = 1.0
    rechts = np.zeros(len(L)); rechts[-1] = 1.0
    p = np.linalg.solve(Lm, rechts)
    return p.reshape(Nkx, D), eps, daten, profile
#
# ---------- Realistische Kopplungen (Vorschläge Prof) auf dem Zylinder ----------
# =====================================================================================================
# Bad wie in bad_dichte_zyl: Dichtewellen q_x entlang x, ein unabhängiges Ohm'sches Bad pro Zeile j0.
#
# Form 1 "dichte":   S_{q_x,j0}(t) = Σ_i e^{iq·r} ( f_A(t) n_{A,i,j0} + f_B(t) n_{B,i,j0} )
#     Matrixelement:  f_A u*_{A_j0}(k+q) u_{A_j0}(k) + f_B e^{iwG1·a1} u*_{B_j0}(k+q) u_{B_j0}(k)
#
# Form 2 "bindung":  S_{q_x,j0}(t) = Σ_s f_s(t) Σ_i e^{iq·(r_A + a_s/2)} K_s(i, j0)
#     Zeile j0 koppelt an alle Bindungen, deren A-Platz in Zeile j0 liegt (wie "pro Einheitszelle" im Bulk-Code).
#     Bindung s: A in Zeile j0 -> B in Zeile j0 + dj_s  (a1: dj = 0, a2: dj = 0, a3: dj = −1; fehlt am Rand)
#     art="real":   K = a†_B a_A + a†_A a_B
#     art="strom":  K = i (a†_B a_A − a†_A a_B)
#     art=Zahl α:   K = cos α · reell + sin α · Strom
#     Matrixelemente (Positions-Konvention, k = kx·G1/2π, k+q nicht zurückgefaltet):
#       a†_B a_A:  e^{−i(k+q)·a_s} e^{iq·a_s/2} u*_{B, j0+dj}(k+q,t) u_{A, j0}(k,t)
#       a†_A a_B:  e^{+ik·a_s}     e^{iq·a_s/2} u*_{A, j0}(k+q,t)    u_{B, j0+dj}(k,t)
#
# Standard für f: Form 1: f_A = f_B = 1 (ruhendes Dichtebad),  Form 2: f_s(t) = J_s(t)/J (Pulsprofil)
#
# Bad-Modell entlang y (bad_modell):
#   "wellen" (Standard): dieselben Badwellen e^{iq·r} wie im Bulk-Code, auch entlang y.  q = q_x + q_⊥,
#            q_⊥ = n·G2/Nqy verschiebt kx nicht (G2·b1 = 0), steht aber als Phase e^{iq_⊥·r} an jedem Term:
#              Platz:    e^{iq_⊥·(j0·b2 + τ)}            Bindung:  e^{iq_⊥·(j0·b2)} · e^{iq·a_s/2}  (q voll)
#            Rate ∝ Σ_{q_⊥} |Σ_{j0, Terme} e^{iq_⊥·r} X|².  Wichtig: Weil q nur über EINE Brillouin-Zone läuft,
#            ist dieses Bad für Abstände, die kein Gittervektor sind (A–B, Bindungsmitten), räumlich korreliert.
#            Genau dieses Bad steckt im Bulk-Code -> nur damit stimmt der Bulk des Zylinders mit dem k-Raum überein.
#   "lokal": jede Bindung / jeder Platz einer Zeile hat ein eigenes Bad, Rate ∝ Σ_{j0} Σ_{s} |X_{j0,s}|².
#            Das ist der Grenzfall, gegen den der Bulk-Code für große Nk konvergiert: dort mitteln die Phasen
#            e^{iq·d} für Bindungsmitten/Plätze mit nicht-ganzzahligem Abstand d über alle q heraus.
#   "zeile": eine Zeile = ein Bad, alle Terme der Zeile kohärent: Rate ∝ Σ_{j0} |Σ_s X_{j0,s}|².
#            Weil q nur entlang G1 läuft (q·b2 = 0), bleiben Terme, die sich nur entlang b2 unterscheiden
#            (z. B. a1- und a3-Bindung desselben A-Platzes), immer kohärent -> systematisch anders als der Bulk.
# Braucht: zustaende_zeit_zyl, g_bath, J, a1, a2, a3, G1

def bad_prof_zyl(T, A, dphi, form="bindung", art="real", fA=None, fB=None, f_bind=None,
                 eta=0.1, kappa=1.0, wc=None, kT=0.02, Nkx=12, Nz=20, Nt=300, jede=5, Nm=20,
                 statisch=False, zeilen=None, daten=None, bad_modell="wellen", Nqy=None, mit_R=False): #mit_R=False für viele Fermionen
    """Ratengleichung auf dem Zylinder mit Badkopplung S_q(t) (Formen wie bad_prof).
       fA, fB, f_bind: Funktionen der Zeit (Array ts -> Array), f_bind gibt Form (len(ts), 3) zurück.
       bad_modell="wellen" (Standard): Badwellen e^{iq·r} auch entlang y (q_⊥ = n·G2/Nqy, Standard Nqy = 2·Nz).
                           Gleiches Bad wie im Bulk-Code -> der Bulk des Zylinders muss den k-Raum reproduzieren.
       bad_modell="lokal": jede Bindung (Form 2) bzw. jeder Platz (Form 1) hat ein eigenes Bad.
       bad_modell="zeile": alle Bindungen/Plätze einer Zeile koppeln kohärent an EIN Bad (wie bad_dichte_zyl).
       zeilen: welche Zeilen ans Bad koppeln (Liste oder bool-Maske, Standard: alle).
       daten:  Rückgabe von zustaende_zeit_zyl zum Wiederverwenden.
       Gibt zurück: p[ikx, Zustand], eps[ikx, Zustand], daten."""
    beta = 1/kT
    omega = 2*np.pi/T
    if wc is None:
        wc = 10*omega

    # ---- Schritt 1: Zylinder-Zustände u(kx,t) und Quasienergien ε(kx) ----
    if daten is None:
        daten = zustaende_zeit_zyl(T, A, dphi, Nkx, Nz, Nt, statisch, jede)
    ts, u, eps, kxs, y = daten
    Nkx, D = eps.shape
    Nz = D//2
    Nts = len(ts)
    if Nts < 2*Nm + 1:
        raise ValueError(f"nur {Nts} Zeitpunkte gespeichert, für Nm = {Nm} braucht es mindestens {2*Nm + 1}")
    if zeilen is None:
        bad_zeilen = list(range(Nz))
    else:
        zeilen = np.asarray(zeilen)
        bad_zeilen = list(np.nonzero(zeilen)[0]) if zeilen.dtype == bool else [int(j) for j in zeilen]
    if Nqy is None:
        Nqy = 2*Nz

    # ---- Schritt 2: Zeitprofil der Kopplung (an den gespeicherten Zeitpunkten ts) ----
    if form == "dichte":
        gA = np.ones(Nts) if fA is None else np.asarray(fA(ts), float)*np.ones(Nts)
        gB = np.ones(Nts) if fB is None else np.asarray(fB(ts), float)*np.ones(Nts)
    else:
        if f_bind is None:                                   # Standard: Pulsprofil f_s(t) = J_s(t)/J
            phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
            fs = np.exp(A*np.cos(omega*ts[:, None] + phis[None]))/(3*np.i0(A))
        else:
            fs = np.asarray(f_bind(ts), float)*np.ones((Nts, 3))
    kc = kxs[:, None]*G1[None, :]/(2*np.pi)                  # kartesisches k, Form (Nkx, 2)
    avec = [a1, a2, a3]
    dj = [0, 0, -1]                                          # Zeilenversatz A -> B je Bindungsrichtung

    def terme_der_zeile(j0, sl, uz, uq, phaseB, qv, kq):
        """Einzelne Beiträge (Plätze bzw. Bindungen) der Zeile j0 für die kx-Auswahl sl, mit ihrer Lage r."""
        terme, orte = [], []
        if form == "dichte":
            terme.append(gA[:, None, None, None]*np.conj(uz[:, sl, j0, :])[..., :, None]*u[:, sl, j0, :][..., None, :])
            orte.append(j0*b2)
            terme.append(gB[:, None, None, None]*phaseB[None, sl, None, None]
                         * np.conj(uz[:, sl, Nz + j0, :])[..., :, None]*u[:, sl, Nz + j0, :][..., None, :])
            orte.append(j0*b2 + a1)
            return terme, orte
        for s in range(3):                                   # drei Bindungsrichtungen
            jB = j0 + dj[s]
            if not (0 <= jB < Nz):                           # Bindung über den offenen Rand: gibt es nicht
                continue
            a_, b_ = j0, Nz + jB                             # Platzindizes in u
            ph = np.exp(1j*qv @ avec[s]/2)                   # Badwelle (q_x-Anteil) in der Bindungsmitte
            BA = (np.exp(-1j*kq[sl] @ avec[s])*ph)[None, :, None, None] \
                 * np.conj(uq[:, sl, b_, :])[..., :, None]*u[:, sl, a_, :][..., None, :]   # a†_B a_A
            AB = (np.exp(1j*kc[sl] @ avec[s])*ph)[None, :, None, None] \
                 * np.conj(uq[:, sl, a_, :])[..., :, None]*u[:, sl, b_, :][..., None, :]   # a†_A a_B
            # ======== HIER WIRD α GEWÄHLT ========
            if art == "real":
                K = BA + AB                                            # α = 0
            elif art == "strom":
                K = 1j*(BA - AB)                                       # α = 90°  -> Stromkopplung
            else:
                K = np.cos(art)*(BA + AB) + np.sin(art)*1j*(BA - AB)   # beliebiges α
            # =====================================
            terme.append(fs[:, s][:, None, None, None]*K)              # mal Zeitprofil f_s(t)
            orte.append(j0*b2 + avec[s]/2)                             # Bindungsmitte
        #return terme, orte #wenn 1 Teilchen
        erg = (p.reshape(Nkx, D), eps, daten) #wenn viele Teilchen
        return erg + (R,) if mit_R else erg #wenn viele Teilchen, mit_R=True gibt auch die Rate zurück

    h = Nkx//2
    ms = np.arange(-Nm, Nm + 1)
    ik = np.arange(Nkx)
    q_perp = [n2*G2/Nqy for n2 in range(-(Nqy//2), Nqy//2 + 1)]   # symmetrisch: zu jedem q auch −q
    R = np.zeros((Nkx*D, Nkx*D))                             # R[ziel, start] = Rate start -> ziel

    for n in range(-h, h + 1):                               # Badmoden q_x (inkl. 0, siehe bad_dichte_zyl)
        iz = (ik + n) % Nkx
        w = (ik + n)//Nkx
        phaseB = np.exp(1j*w*(G1 @ a1))                      # Faltungsphase auf B, Form (Nkx,)
        uz = u[:, iz]                                        # u(kx + q_x), zurückgefaltet
        qv = n*G1/Nkx                                        # kartesisches q_x
        kq = kc + qv                                         # k + q_x, nicht zurückgefaltet
        uq = uz.copy()
        uq[:, :, Nz:, :] *= np.conj(phaseB)[None, :, None, None]   # u_B(k+q) = e^{−iwG1·a1} u_B(zurückgefaltet)

        # ---- Schritt 3 + 4: Matrixelemente, Seitenbänder, Σ |X^(m)|² ----
        X2 = np.zeros((len(ms), Nkx, D, D))
        if bad_modell == "wellen":
            for k0 in range(Nkx):                            # pro kx, damit der Speicher klein bleibt
                sl = slice(k0, k0 + 1)
                Xt, r_t = [], []
                for j0 in bad_zeilen:
                    terme, orte = terme_der_zeile(j0, sl, uz, uq, phaseB, qv, kq)
                    Xt += [np.fft.ifft(Mel, axis=0)[ms % Nts] for Mel in terme]
                    r_t += orte
                if not Xt:
                    continue
                Xt, r_t = np.stack(Xt), np.array(r_t)        # (Terme, m, 1, α, β)
                for qp in q_perp:                            # q_⊥: kohärent mit e^{iq_⊥·r} summieren
                    X2[:, sl] += np.abs(np.tensordot(np.exp(1j*(r_t @ qp)), Xt, axes=(0, 0)))**2
        else:
            for j0 in bad_zeilen:
                terme, _ = terme_der_zeile(j0, slice(None), uz, uq, phaseB, qv, kq)
                if not terme:
                    continue
                if bad_modell == "zeile":                    # ein Bad pro Zeile: kohärent summieren
                    terme = [sum(terme)]
                for Mel in terme:                            # "lokal": jedes Glied ein eigenes Bad
                    X2 += np.abs(np.fft.ifft(Mel, axis=0)[ms % Nts])**2

        E = eps[None, :, None, :] - eps[iz][None, :, :, None] + ms[:, None, None, None]*omega
        rate = eta**2*np.sum(X2*g_bath(E, kappa, wc, beta), axis=0)       # (Nkx, α, β): (kx,β) -> (kx+q_x,α)
        ziel = (iz*D)[:, None, None] + np.arange(D)[None, :, None]
        start = (ik*D)[:, None, None] + np.arange(D)[None, None, :]
        R[np.broadcast_to(ziel, rate.shape), np.broadcast_to(start, rate.shape)] += rate

    # ---- Schritt 5: stationäre Besetzung der Pauli-Gleichung ----
    np.fill_diagonal(R, 0)
    L = R - np.diag(R.sum(axis=0))
    Lm = L.copy(); Lm[-1] = 1.0
    rechts = np.zeros(len(L)); rechts[-1] = 1.0
    p = np.linalg.solve(Lm, rechts)
    return p.reshape(Nkx, D), eps, daten

#%%
# Kontrolle: ohne Antrieb muss Boltzmann herauskommen (gilt für jede Zylindergröße und jedes Bad-Modell)
for form, art in [("dichte", "real"), ("bindung", "real"), ("bindung", "strom")]:
    p, eps_k, _ = bad_prof_zyl(0.5, 1.0, dphi, form=form, art=art,
                               f_bind=lambda t: np.full((len(t), 3), 1/3),
                               kT=0.5, wc=50.0, statisch=True, Nkx=8, Nz=6)
    boltz = np.exp(-eps_k/0.5); boltz /= boltz.sum()
    print(f"Kontrolle {form}/{art}: max |p − Boltzmann| = {np.abs(p - boltz).max():.1e}")
#%%
## ---------- Zylinder: verschiedene Badkopplungen ----------
JT3, A_wert = 0.5*np.pi, 3.0
kT_bad = 0.02
T_ = 3*JT3/J
omega_ = 2*np.pi/T_
Nkx_, Nz_ = 12, 12
modell = "wellen"          # "wellen" / "zeile" / "lokal"
'''
"wellen": dasselbe Bad wie im Bulk-Code. Man nimmt es, wenn der Zylinder-Bulk mit dem Torus vergleichbar sein soll.
"zeile": Alle Plätze bzw. Bindungen einer Zeile hängen kohärent an einem Bad. Weil q nur entlang x läuft, bleiben zum Beispiel die a1- und die a3-Bindung desselben A-Platzes immer kohärent. Deshalb weicht dieses Modell systematisch vom Bulk ab.
"lokal": Jeder Platz bzw. jede Bindung hat ein eigenes Bad.
'''

daten = zustaende_zeit_zyl(T_, A_wert, dphi, Nkx_, Nz_, 300, False, 5)   # Zustände einmal rechnen
fA_ = lambda t: 1 + 0.8*np.cos(omega_*t)
fB_ = lambda t: 1 - 0.8*np.cos(omega_*t)

def zeile(name, p_z):
    a = auswertung_zyl(p_z, daten[2], daten)
    print(f"  {name:42s}  {a['p_unten_bulk']:14.3f}  {a['p_bulk']:11.3f}  {a['p_rand_y0']:9.3f}  {a['p_rand_yN']:11.3f}",
          flush=True)

# --- Formen des Profs ---
faelle = [
    ("Dichte, ruhend (f_A = f_B = 1)",
     lambda: bad_prof_zyl(T_, A_wert, dphi, form="dichte", kT=kT_bad, daten=daten, bad_modell=modell)[0]),
    ("Form 1: f_A, f_B = 1 ± 0.8 cos(ωt) -Dichte",
     lambda: bad_prof_zyl(T_, A_wert, dphi, form="dichte", kT=kT_bad, fA=fA_, fB=fB_, daten=daten, bad_modell=modell)[0]),
    ("Form 2: Bindung reell, f_s = J_s(t)/J -NN",
     lambda: bad_prof_zyl(T_, A_wert, dphi, form="bindung", art="real", kT=kT_bad, daten=daten, bad_modell=modell)[0]),
    ("Form 2: Bindung Strom, f_s = J_s(t)/J -NN",
     lambda: bad_prof_zyl(T_, A_wert, dphi, form="bindung", art="strom", kT=kT_bad, daten=daten, bad_modell=modell)[0]),
]

# --- mitbewegtes Bad: nur die Namen der Varianten ---
varianten = ["voll", "r<=2",
             "NN", "NNN", "quer",
             "NN+NNN", "NN+quer", "NNN+quer",
             "Dichte+NN", "Dichte+NNN", "Dichte+quer",
             "Dichte+NN+NNN", "Dichte+NN+quer", "Dichte+NNN+quer"]

print(f"\nJT/3 = {JT3/np.pi:.2f}π, A = {A_wert}, kT = {kT_bad},  Zylinder Nkx = {Nkx_}, Nz = {Nz_}, Bad-Modell = {modell}")
print(f"  {'':42s}  {'p_unten (Bulk)':>14s}  {'Bulk gesamt':>11s}  {'Rand y=0':>9s}  {'Rand y=Nz-1':>11s}")

for name, zyl_fn in faelle:
    zeile(name, zyl_fn())

prof = None                                   # Profil wird beim ersten Durchlauf berechnet, danach wiederverwendet
for var in varianten:
    p, eps, _, prof = bad_mitbewegt_zyl(T_, A_wert, dphi, variante=var, kT=kT_bad, wc=10*omega_, daten=daten, profile=prof, bad_modell=modell)
    zeile(f"mitbewegt {var}", p)


# daten = zustaende_zeit_zyl(T_, A_wert, dphi, Nkx_, Nz_, 300, False, 5)   # einmal rechnen, überall wiederverwenden
# fA_ = lambda t: 1 + 0.8*np.cos(omega_*t)
# fB_ = lambda t: 1 - 0.8*np.cos(omega_*t)

# faelle = [
#     ("Dichte, ruhend (f_A = f_B = 1)",
#      lambda: bad_prof_zyl(T_, A_wert, dphi, form="dichte", kT=kT_bad, daten=daten)[0]),
#     ("Form 1: f_A, f_B = 1 ± 0.8 cos(ωt) -Dichte",
#      lambda: bad_prof_zyl(T_, A_wert, dphi, form="dichte", kT=kT_bad, fA=fA_, fB=fB_, daten=daten)[0]),
#     ("Form 2: Bindung reell, f_s = J_s(t)/J -NN",
#      lambda: bad_prof_zyl(T_, A_wert, dphi, form="bindung", art="real", kT=kT_bad, daten=daten)[0]),
#     ("Form 2: Bindung Strom, f_s = J_s(t)/J -NN",
#      lambda: bad_prof_zyl(T_, A_wert, dphi, form="bindung", art="strom", kT=kT_bad, daten=daten)[0]),
#     ("mitbewegt ideal (P n P†)",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="ideal_exakt", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt voll (Profil, r_max = 4.6)",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="voll", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt NN",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="NN", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt NNN",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="NNN", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt quer",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="quer", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt NN+NNN",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="NN+NNN", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt NN+quer",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="NN+quer", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt NNN+quer",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="NNN+quer", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt Dichte+NN",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="Dichte+NN", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt Dichte+NNN",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="Dichte+NNN", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt Dichte+quer",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="Dichte+quer", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt Dichte+NN+NNN",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="Dichte+NN+NNN", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt Dichte+NN+quer",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="Dichte+NN+quer", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
#     ("mitbewegt Dichte+NNN+quer",
#      lambda: bad_mitbewegt_zyl(T_, A_wert, dphi, variante="Dichte+NNN+quer", kT=kT_bad, wc=10*omega_, daten=daten)[0]),
# ]

# print(f"\nJT/3 = {JT3/np.pi:.2f}π, A = {A_wert}, kT = {kT_bad},  Zylinder Nkx = {Nkx_}, Nz = {Nz_}")
# print(f"  {'':42s}  {'p_unten (Bulk)':>14s}  {'Bulk gesamt':>11s}  {'Rand y=0':>9s}  {'Rand y=Nz-1':>11s}")
# for name, zyl_fn in faelle:
#     p_z = zyl_fn()
#     a = auswertung_zyl(p_z, daten[2], daten)
#     print(f"  {name:42s}  {a['p_unten_bulk']:14.3f}  {a['p_bulk']:11.3f}  {a['p_rand_y0']:9.3f}  {a['p_rand_yN']:11.3f}",
#           flush=True)

#%%











#%%
#%%
# ---------- P(t)|i> auf dem Zylinder, mit Gitter darunter ----------
# Eine Bildzeile pro Startzeile j0: z. B. Mitte (Nz//2) und Rand (0). Braucht: zylinder, P_spalten_zyl, J, dphi, b1
JT3, A_wert = 0.5*np.pi, 3.0
T_ = 3*JT3/J
Mt, Nz, Nt = 10, 12, 300
zeilen_start = [Nz//2, 0]                          # Startzeilen: Mitte und unterer Rand (A-Platz ohne a3-Partner)
radius = 3.2

K, idx, pos, partner, hat = zylinder(Mt, Nz)
starts = [idx(Mt//2, j0, 0) for j0 in zeilen_start]   # je ein A-Platz in der Mitte des Umfangs
ts, Ps, _, _, fehler, y = P_spalten_zyl(T_, A_wert, dphi, Mt, Nz, Nt, starts)   # nur die nötigen Spalten von P
print("Kontrolle |P(T) - 1| =", fehler)

L1 = Mt*b1                                          # Umfang des Zylinders (periodisch entlang b1)
def min_bild(v):
    """kürzester Vektor entlang des Umfangs (nur b1 ist periodisch)"""
    kand = np.stack([v + u*L1 for u in (-1, 0, 1)])
    return kand[np.argmin(np.linalg.norm(kand, axis=-1), axis=0), np.arange(len(v))]

zeiten = [0, 1/6, 1/3, 1/2, 2/3, 5/6]
farben_bindung = ['tab:blue', 'tab:orange', 'tab:green']   # a1, a2, a3
phis_ = np.array([0, dphi, 2*dphi]) - np.pi/3
fig, ax = plt.subplots(len(zeilen_start), len(zeiten), figsize=(2.8*len(zeiten), 3.0*len(zeilen_start)),
                       squeeze=False)
for zeile_bild, (j0, start) in enumerate(zip(zeilen_start, starts)):
    d = min_bild(pos - pos[start])                  # Lage relativ zum Startplatz (über den Saum gefaltet)
    nah = np.linalg.norm(d, axis=1) < radius
    # Bindungen: nur vorhandene (hat), beide Enden in der Nähe
    linien = []
    for s in range(3):
        for a_ in range(Mt*Nz):                     # A-Plätze sind 0 … Mt·Nz−1
            if not hat[s, a_]:
                continue                            # Bindung über den offenen Rand gibt es nicht
            b_ = partner[s, a_]
            if nah[a_] or nah[b_]:
                db = d[a_] + min_bild((pos[b_] - pos[a_])[None])[0]   # B-Ende passend zum A-Ende
                linien.append((d[a_], db, s))
    for spalte, tau in enumerate(zeiten):
        a = ax[zeile_bild, spalte]
        n = int(round(tau*len(ts))) % len(ts)
        fs = np.exp(A_wert*np.cos(2*np.pi/T_*ts[n] + phis_))/(3*np.i0(A_wert))   # momentane Bindungsstärken
        for p1, p2, s in linien:
            a.plot([p1[0], p2[0]], [p1[1], p2[1]], color=farben_bindung[s], lw=0.3 + 4*fs[s], alpha=0.6, zorder=1)
        a.scatter(d[nah, 0], d[nah, 1], s=12, facecolors='white', edgecolors='gray', lw=0.6, zorder=2)
        a.scatter(0, 0, s=60, facecolors='none', edgecolors='k', lw=1, zorder=3)   # Startplatz
        psi = Ps[n, :, zeile_bild]                  # P(t)|start>
        w = np.abs(psi)**2
        ph = np.angle(psi*np.conj(psi[np.argmax(w)]))
        sel = nah & (w > 2e-3)
        sc = a.scatter(d[sel, 0], d[sel, 1], s=500*w[sel], c=ph[sel], cmap='twilight',
                       vmin=-np.pi, vmax=np.pi, linewidths=0, zorder=4)
        rest = 1 - w[nah].sum()                     # Gewicht außerhalb des gezeigten Ausschnitts
        a.set_title(f"t = {tau:.2f} T" + (f"\n(außerhalb: {rest:.2f})" if rest > 0.01 else ""), fontsize=9)
        a.set_aspect('equal'); a.axis('off')
        a.set_xlim(-radius, radius); a.set_ylim(-radius, radius)
    ax[zeile_bild, 0].text(-radius, 0, f"Start in Zeile {j0}" + (" (Rand)" if j0 in (0, Nz-1) else " (Mitte)"),
                           rotation=90, va='center', ha='right', fontsize=10)
cb = fig.colorbar(sc, ax=ax, shrink=0.7, label="relative Phase")
cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cb.set_ticklabels(["−π", "−π/2", "0", "π/2", "π"])
fig.suptitle(f"P(t)|i⟩ auf dem Zylinder (A = {A_wert:g}, JT/3 = {JT3/np.pi:.2f}π): Größe = |ψ_j|², Farbe = Phase, Kreis = Startplatz")
plt.show()
#%%
# ---------- Lokalität der Mikrobewegung: zeitgemitteltes Gewicht gegen den Abstand ----------
# Braucht die Zelle davor (plot_P_zyl.py): Ps, ts, starts, zeilen_start, pos, min_bild, Nz, T_
fig, ax_ = plt.subplots(figsize=(6, 4))
for k, (j0, start) in enumerate(zip(zeilen_start, starts)):
    d_k = min_bild(pos - pos[start])                         # Abstand über den Saum gefaltet
    ab = np.linalg.norm(d_k, axis=1)
    w_mittel = np.mean(np.abs(Ps[:, :, k])**2, axis=0)       # ⟨|P_ji(t)|²⟩_t
    name = f"Start Zeile {j0}" + (" (Rand)" if j0 in (0, Nz - 1) else " (Mitte)")
    ax_.semilogy(ab, w_mittel, 'o', ms=4, alpha=0.7, label=name)
ax_.set_ylim(1e-8, 1.5)
ax_.grid(alpha=0.3); ax_.legend()
ax_.set_xlabel("Abstand vom Startplatz"); ax_.set_ylabel("⟨|P_ji(t)|²⟩_t")
ax_.set_title("Wie weit reicht die Mikrobewegung? (bestimmt, welches r_max nötig ist)", fontsize=9)
plt.show()

#%%
# ---------- Gewicht auf dem A-Untergitter während einer Periode ----------
# Bulk-Analogon zu ⟨|P_AA(k,t)|²⟩_k: Startplatz ist ein A-Platz, gemessen wird das Gewicht auf ALLEN A-Plätzen.
w_ = 2*np.pi/T_
A_maske = np.arange(Ps.shape[1]) < Ps.shape[1]//2            # A-Plätze sind die erste Hälfte der Indizes
fig, ax_ = plt.subplots(figsize=(7, 4))
for k, j0 in enumerate(zeilen_start):
    gew_A = np.sum(np.abs(Ps[:, A_maske, k])**2, axis=1)
    name = f"Start Zeile {j0}" + (" (Rand)" if j0 in (0, Nz - 1) else " (Mitte)")
    ax_.plot(ts/T_, gew_A, label=f"Gewicht auf A, {name}")
ax_.plot(ts/T_, np.cos(w_*ts/4)**2, 'k--', lw=0.8, label="cos²(ωt/4)")
ax_.plot(ts/T_, np.sin(w_*ts/4)**2, 'k:', lw=0.8, label="sin²(ωt/4)")
for t_puls, name in [(1/6, "a1"), (1/2, "a2"), (5/6, "a3")]:     # Pulsmaxima
    ax_.axvline(t_puls, color='gray', ls='--', lw=0.6)
    ax_.text(t_puls, 1.03, name, ha='center', fontsize=8, color='gray')   # jetzt in der Schleife: alle drei beschriftet
ax_.set_ylim(-0.02, 1.1)
ax_.set_xlabel("t / T"); ax_.set_ylabel("Gewicht"); ax_.grid(alpha=0.3); ax_.legend(fontsize=8)
plt.show()
#%%
# Gewicht auf dem Startplatz und mittlerer Abstand vom Startplatz, gegen t  (Zylinder)
# Braucht die Zelle plot_P_zyl.py: Ps, ts, starts, zeilen_start, pos, min_bild, Nz, T_
fig, ax2 = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
for k, (j0, start) in enumerate(zip(zeilen_start, starts)):
    ab = np.linalg.norm(min_bild(pos - pos[start]), axis=1)  # Abstand jedes Platzes vom Startplatz (über den Saum gefaltet)
    w_t = np.abs(Ps[:, :, k])**2                             # |P_ji(t)|² für alle j
    w_start = w_t[:, start]                                  # |P_ii(t)|²
    r_mittel = w_t @ ab                                      # ⟨Abstand⟩(t) = Σ_j |P_ji(t)|² · Abstand_j
    name = f"Start Zeile {j0}" + (" (Rand)" if j0 in (0, Nz - 1) else " (Mitte)")
    ax2[0].plot(ts/T_, w_start, label=name)
    ax2[1].plot(ts/T_, r_mittel, label=name)
for a in ax2:
    for t_puls in (1/6, 1/2, 5/6):                           # Pulsmaxima
        a.axvline(t_puls, color='gray', ls='--', lw=0.6)
    a.grid(alpha=0.3)
ax2[0].set_ylabel("Gewicht auf Startplatz"); ax2[0].legend(fontsize=8)
ax2[1].set_ylabel("mittlerer Abstand")
ax2[1].set_xlabel("t / T")
plt.tight_layout(); plt.show()
#%%



















#%%
# =====================================================================================================
# ---------- Viele (nicht wechselwirkende) Fermionen: Ratengleichung mit Pauli-Blockade ----------
# =====================================================================================================
# Ein Teilchen:   dp_a/dt = Σ_b [ R_ab p_b − R_ba p_a ]                      (linear)
# N Fermionen:    dn_a/dt = Σ_b [ R_ab n_b (1 − n_a) − R_ba n_a (1 − n_b) ]  (Pauli-Blockade, nichtlinear)
#   n_a ∈ [0, 1]  Besetzung des Einteilchen-Floquet-Zustands a = (kx, Zustand),  Σ_a n_a = N bleibt erhalten.
#   R ist DIESELBE Ratenmatrix wie für ein Teilchen (bad_dichte_zyl / bad_prof_zyl / bad_mitbewegt_zyl mit mit_R=True).
#   Näherung: Besetzungen faktorisieren (keine Korrelationen) – Standard für ein nicht wechselwirkendes System
#   mit schwacher Badkopplung.
# Kontrolle: ohne Antrieb muss Fermi-Dirac herauskommen, mit μ passend zu N.
# Braucht: zustaende_zeit_zyl, bad_prof_zyl, bad_mitbewegt_zyl (mit mit_R), auswertung_zyl
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def fermi_stationaer(R, N, n0=None, tol=1e-12, max_iter=200):
    """Stationäre Besetzung n (Länge D) der Ratengleichung mit Pauli-Blockade, Σ n = N.
       R: Ratenmatrix R[ziel, start] (Form (D, D)) aus einer Bad-Funktion mit mit_R=True.
       Vorgehen: erst steife Zeitentwicklung in die Nähe des stationären Zustands, dann Newton zum Polieren."""
    Rt = np.array(R, float)
    np.fill_diagonal(Rt, 0)
    D = Rt.shape[0]
    if not 0 < N < D:
        raise ValueError(f"Teilchenzahl N = {N} muss zwischen 0 und D = {D} liegen")

    def F(n):                                       # dn/dt
        return (1 - n)*(Rt @ n) - n*(Rt.T @ (1 - n))

    def Jac(n):                                     # ∂F_a/∂n_c
        Jm = (1 - n)[:, None]*Rt + n[:, None]*Rt.T
        Jm[np.diag_indices(D)] -= Rt @ n + Rt.T @ (1 - n)
        return Jm

    n = np.full(D, N/D) if n0 is None else np.array(n0, float)
    # 1) Zeitentwicklung (die Teilchenzahl bleibt dabei automatisch erhalten)
    t_end = 200.0/max(Rt.sum(axis=0).mean(), 1e-300)
    sol = solve_ivp(lambda t, x: F(x), (0, t_end), n, method='BDF', jac=lambda t, x: Jac(x),
                    rtol=1e-8, atol=1e-12)
    n = np.clip(sol.y[:, -1], 0, 1)
    # 2) Newton mit der Nebenbedingung Σ n = N (eine Gleichung wird durch sie ersetzt)
    for _ in range(max_iter):
        f = F(n); Jm = Jac(n)
        f[-1] = n.sum() - N; Jm[-1] = 1.0
        dn = np.linalg.solve(Jm, -f)
        schritt = 1.0
        while np.any(n + schritt*dn < 0) or np.any(n + schritt*dn > 1):
            schritt /= 2                            # in [0, 1] bleiben
        n = n + schritt*dn
        if np.abs(dn).max() < tol:
            break
    rest = np.abs(F(n)).max()/max(np.abs(Rt).max(), 1e-300)
    if rest > 1e-8:
        print(f"  Warnung: fermi_stationaer nicht ganz konvergiert (|dn/dt|/R_max = {rest:.1e})")
    return n


def fermi_dirac(eps, kT, N):
    """Fermi-Dirac-Besetzung zu den Energien eps (beliebige Form) mit μ so, dass Σ n = N."""
    e = np.ravel(eps)
    anzahl = lambda mu: np.sum(0.5*(1 - np.tanh((e - mu)/(2*kT)))) - N      # = Σ 1/(e^{(ε−μ)/kT} + 1) − N
    mu = brentq(anzahl, e.min() - 50*kT, e.max() + 50*kT)
    return (0.5*(1 - np.tanh((e - mu)/(2*kT)))).reshape(np.shape(eps)), mu


def auswertung_fermi(n, eps, daten, rand_breite=2, schwelle=0.5):
    """Füllung der Bänder: Anteil der besetzten Zustände im unteren / oberen Bulk-Band, Teilchen im Rand."""
    m = auswertung_zyl(n, eps, daten, rand_breite, schwelle)["maske"]
    unten, oben = m["bulk"] & (eps < 0), m["bulk"] & (eps >= 0)
    rand = m["rand_y0"] | m["rand_yN"]
    return dict(fuellung_unten=n[unten].mean(),            # 1 = unteres Bulk-Band voll
                fuellung_oben=n[oben].mean(),              # 0 = oberes Bulk-Band leer
                N_unten=n[unten].sum(), N_oben=n[oben].sum(), N_rand=n[rand].sum(),
                n_zust_unten=int(unten.sum()), n_zust_oben=int(oben.sum()), n_zust_rand=int(rand.sum()))


#%%
# ---------- Kontrollen ----------
# (1) ohne Antrieb: Fermi-Dirac;  (2) sehr wenige Teilchen: n/N = Ergebnis für ein Teilchen
for form, art in [("dichte", "real"), ("bindung", "strom")]:
    p1, eps_k, d_k, R = bad_prof_zyl(0.5, 1.0, dphi, form=form, art=art,
                                     f_bind=lambda t: np.full((len(t), 3), 1/3),
                                     kT=0.5, wc=50.0, statisch=True, Nkx=8, Nz=6, mit_R=True)
    N_test = eps_k.size//2
    n = fermi_stationaer(R, N_test).reshape(eps_k.shape)
    fd, mu = fermi_dirac(eps_k, 0.5, N_test)
    n_klein = fermi_stationaer(R, 1e-6).reshape(eps_k.shape)
    print(f"Kontrolle {form}/{art}: max |n − Fermi-Dirac| = {np.abs(n - fd).max():.1e},   "
          f"wenige Teilchen: max |n/N − p| = {np.abs(n_klein/1e-6 - p1).max():.1e}")


#%%
# ---------- Zylinder mit vielen Fermionen: verschiedene Badkopplungen ----------
JT3, A_wert = 0.5*np.pi, 3.0
kT_bad = 0.02
T_ = 3*JT3/J
omega_ = 2*np.pi/T_
Nkx_, Nz_ = 12, 12
modell = "wellen"                                   # oder "zeile" / "lokal"
fuellung = 0.5                                      # Teilchen pro Zustand: 0.5 = halbe Füllung (ein Band voll)

daten = zustaende_zeit_zyl(T_, A_wert, dphi, Nkx_, Nz_, 300, False, 5)
N_teilchen = fuellung*daten[2].size
fA_ = lambda t: 1 + 0.8*np.cos(omega_*t)
fB_ = lambda t: 1 - 0.8*np.cos(omega_*t)

def zeile_fermi(name, R):
    n = fermi_stationaer(R, N_teilchen).reshape(daten[2].shape)
    a = auswertung_fermi(n, daten[2], daten)
    print(f"  {name:42s}  {a['fuellung_unten']:12.3f}  {a['fuellung_oben']:11.3f}  {a['N_rand']:7.1f}", flush=True)

faelle = [
    ("Dichte, ruhend (f_A = f_B = 1)",
     lambda: bad_prof_zyl(T_, A_wert, dphi, form="dichte", kT=kT_bad, daten=daten, bad_modell=modell, mit_R=True)[-1]),
    ("Form 1: f_A, f_B = 1 ± 0.8 cos(ωt) -Dichte",
     lambda: bad_prof_zyl(T_, A_wert, dphi, form="dichte", kT=kT_bad, fA=fA_, fB=fB_, daten=daten,
                          bad_modell=modell, mit_R=True)[-1]),
    ("Form 2: Bindung reell, f_s = J_s(t)/J -NN",
     lambda: bad_prof_zyl(T_, A_wert, dphi, form="bindung", art="real", kT=kT_bad, daten=daten,
                          bad_modell=modell, mit_R=True)[-1]),
    ("Form 2: Bindung Strom, f_s = J_s(t)/J -NN",
     lambda: bad_prof_zyl(T_, A_wert, dphi, form="bindung", art="strom", kT=kT_bad, daten=daten,
                          bad_modell=modell, mit_R=True)[-1]),
]
varianten = ["ideal_exakt", "voll", "r<=2", "Dichte", "NN", "quer", "NN+quer", "Dichte+NN+quer"]

a0 = auswertung_fermi(np.ones_like(daten[2]), daten[2], daten)       # nur für die Anzahl der Zustände
print(f"\nJT/3 = {JT3/np.pi:.2f}π, A = {A_wert}, kT = {kT_bad}, Zylinder Nkx = {Nkx_}, Nz = {Nz_}, Bad: {modell}")
print(f"N = {N_teilchen:.0f} Fermionen auf {daten[2].size} Zuständen "
      f"(unteres Bulk-Band {a0['n_zust_unten']}, oberes {a0['n_zust_oben']}, Rand {a0['n_zust_rand']})")
print(f"  {'':42s}  {'Füllung unten':>12s}  {'Füllung oben':>11s}  {'N Rand':>7s}")
for name, R_fn in faelle:
    zeile_fermi(name, R_fn())
prof = None
for var in varianten:
    *_, prof_neu, R = bad_mitbewegt_zyl(T_, A_wert, dphi, variante=var, kT=kT_bad, daten=daten, profile=prof,
                                        bad_modell=modell, mit_R=True)
    if prof_neu is not None:
        prof = prof_neu
    zeile_fermi(f"mitbewegt {var}", R)
# %%
