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
wc_cutoff = 0.2

def g_bath(E, kappa, wc_cutoff, beta): #=Ohmsches Bad
    E = np.asarray(E, dtype=float)
    g = np.full(E.shape, 2*np.pi*kappa/beta)                 # Grenzwert E -> 0
    nz = np.abs(E) > 1e-12
    with np.errstate(over='ignore'):                         # große negative E: g -> 0
        g[nz] = 2*np.pi*kappa*E[nz]*np.exp(-np.abs(E[nz])/wc_cutoff) / (-np.expm1(-beta*E[nz]))
    return g

#Periodisch:
def schritt2(h, dt):
    """exp(-i H dt) für H = [[0, h], [h*, 0]], für alle k gleichzeitig."""
    r = np.abs(h)
    s = np.sin(r*dt)/np.where(r > 1e-14, r, 1)
    S = np.zeros(h.shape + (2, 2), complex)
    S[..., 0, 0] = S[..., 1, 1] = np.cos(r*dt)
    S[..., 0, 1] = -1j*s*h
    S[..., 1, 0] = -1j*s*np.conj(h)
    return S

def zustaende_zeit(T, A, dphi, Nk, Nt=300, statisch=False):
    """Periodische Anteile u[n, i, j, Platz, Band] zur Zeit t_n und Quasienergien eps[i, j, Band]."""
    omega = 2*np.pi/T
    norm = 3*np.i0(A)
    phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
    ks = 2*np.pi*np.arange(Nk)/Nk
    KX, KY = np.meshgrid(ks, ks, indexing='ij')
    kc = (KX[..., None]*G1 + KY[..., None]*G2)/(2*np.pi)
    p_bind = [J*np.exp(1j*kc @ d) for d in (a1, a2, a3)]
    dt = T/Nt
    U_t = np.zeros((Nt,) + KX.shape + (2, 2), complex)
    Ut = np.broadcast_to(np.eye(2, dtype=complex), KX.shape + (2, 2)).copy()
    for n in range(Nt):
        U_t[n] = Ut
        fs = np.full(3, 1/3) if statisch else np.exp(A*np.cos(omega*(n + 0.5)*dt + phis))/norm
        Ut = schritt2(sum(fs[b]*p_bind[b] for b in range(3)), dt) @ Ut
    lam, phi0 = np.linalg.eig(Ut)
    eps = -np.angle(lam)/T
    r = np.argsort(eps, axis=-1)
    eps = np.take_along_axis(eps, r, axis=-1)
    phi0 = np.take_along_axis(phi0, r[..., None, :], axis=-1)
    phi0 = phi0/np.linalg.norm(phi0, axis=-2, keepdims=True)
    ts = dt*np.arange(Nt)
    u = (U_t @ phi0[None]) * np.exp(1j*eps[None]*ts[:, None, None, None])[..., None, :]
    return ts, u, eps

def bad_dichte(T, A, dphi, eta=0.1, kappa=1.0, wc=wc_cutoff, kT=0.1, Nk=12, Nt=300, Nm=20, statisch=False):
    """Pauli-Ratengleichung über alle Zustände (k, Band), Bad = Dichtewellen e^{iq·r} mit allen q ≠ 0.
       Gibt zurück: Besetzung p (Nk, Nk, 2) [Spalte 0 unten, 1 oben], eps (Nk, Nk, 2)."""
    beta = 1/kT
    omega = 2*np.pi/T
    # ---------- Schritt 1 + 2 (wie bad_mod): U(k,t), Floquet-Zustände, periodische Anteile u(k,t) ----------
    ts, u, eps = zustaende_zeit(T, A, dphi, Nk, Nt, statisch)
    # ---------- Badmoden: alle q ≠ 0, zu jedem q auch −q (detaillierte Balance) ----------
    h = Nk//2
    q_liste = [(n1, n2) for n1 in range(-h, h+1) for n2 in range(-h, h+1) if (n1, n2) != (0, 0)]    
    D = 2*Nk*Nk
    R = np.zeros((D, D))   # R[ziel, start] = Rate start -> ziel
    ms = np.arange(-Nm, Nm+1)
    i_idx, j_idx = np.meshgrid(np.arange(Nk), np.arange(Nk), indexing='ij')
    for (n1, n2) in q_liste:
        # ---------- Schritt 3: Matrixelement <u_α(k+q,t)| e^{iq·r} |u_β(k,t)> ----------
        iz, jz = (i_idx + n1) % Nk, (j_idx + n2) % Nk
        w1, w2 = (i_idx + n1) // Nk, (j_idx + n2) // Nk
        phaseB = np.exp(1j*(w1[..., None]*G1 + w2[..., None]*G2) @ a1)
        uz = u[:, iz, jz]
        f = (np.conj(uz[..., 0, :])[..., :, None]*u[..., 0, :][..., None, :]
             + phaseB[None, ..., None, None]*np.conj(uz[..., 1, :])[..., :, None]*u[..., 1, :][..., None, :])
        # ---------- Schritt 4: Fourier-Kanäle und Raten ----------
        X2 = np.abs(np.fft.ifft(f, axis=0)[ms % Nt])**2                          # |X^(m)|²
        E = eps[None, :, :, None, :] - eps[iz, jz][None, :, :, :, None] + ms[:, None, None, None, None]*omega
        rate = eta**2*np.sum(X2*g_bath(E, kappa, wc_cutoff, beta), axis=0)              # Rate (k,β) -> (k+q,α)
        start = ((i_idx*Nk + j_idx)*2)[..., None, None] + np.arange(2)[None, None, None, :]
        ziel = ((iz*Nk + jz)*2)[..., None, None] + np.arange(2)[None, None, :, None]
        np.add.at(R, (np.broadcast_to(ziel, rate.shape).ravel(), np.broadcast_to(start, rate.shape).ravel()),
                  rate.ravel())
    np.fill_diagonal(R, 0)
    # ---------- Schritt 5: stationärer Zustand der Pauli-Gleichung dp/dt = L p ----------
    L = R - np.diag(R.sum(axis=0))                # dp/dt = L p
    w, v = np.linalg.eig(L)
    p = np.real(v[:, np.argmin(np.abs(w))])       # stationär: Eigenvektor zum Eigenwert 0
    return (p/p.sum()).reshape(Nk, Nk, 2), eps
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


# ---------- Teil A: Mikrobewegung P(t) auf einem Torus (kein Rand -> reiner Bulk) ----------

def torus(M):
    """Torus aus M×M Zellen. Gibt Anzahl Plätze, Index-Funktion, Positionen, Partner je Bindungsrichtung."""
    K = 2*M*M
    idx = lambda i, j, z: (i % M) + (j % M)*M + z*M*M
    pos = np.zeros((K, 2))
    for i in range(M):
        for j in range(M):
            pos[idx(i, j, 0)] = i*b1 + j*b2
            pos[idx(i, j, 1)] = i*b1 + j*b2 + a1
    bindungen = [(0, 0), (-1, 0), (0, -1)]          # a1, a2, a3: A(i,j) -> B(i+di, j+dj), periodisch
    partner = np.zeros((3, K), dtype=int)
    for s, (di, dj) in enumerate(bindungen):
        for i in range(M):
            for j in range(M):
                a, b = idx(i, j, 0), idx(i + di, j + dj, 1)
                partner[s, a] = b
                partner[s, b] = a
    return K, idx, pos, partner


def P_von_t(T, A, dphi, M=10, Nt=300):
    """Mikrobewegung P(t_n) = U(t_n) e^{+i H_F t_n} für t_n = n·T/Nt, n = 0 … Nt−1.
       Zweig von H_F: Quasienergien in (−π/T, π/T]  (wie überall im Code: 'unten' = ε < 0)."""
    omega = 2*np.pi/T
    norm = 3*np.i0(A)
    phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
    K, idx, pos, partner = torus(M)

    def bindung(s, c, psi):                          # exp(−i c J H_s), exakt auf jedem Paar
        return np.cos(c*J)*psi - 1j*np.sin(c*J)*psi[partner[s]]

    dt = T/Nt
    Us = np.zeros((Nt, K, K), complex)
    U = np.eye(K, dtype=complex)
    for n in range(Nt):
        Us[n] = U                                    # U(t_n)
        c = np.exp(A*np.cos(omega*(n + 0.5)*dt + phis))/norm*dt
        U = bindung(0, c[0]/2, U); U = bindung(1, c[1]/2, U); U = bindung(2, c[2], U)
        U = bindung(1, c[1]/2, U); U = bindung(0, c[0]/2, U)
    D, Z = schur(U, output='complex')                # U(T) = Z D Z†
    eps = -np.angle(np.diag(D))/T                    # Eigenwerte von H_F in (−π/T, π/T]
    ts = dt*np.arange(Nt)
    P = (Us @ Z[None]*np.exp(1j*np.outer(ts, eps))[:, None, :]) @ Z.conj().T[None]   # U(t) Z e^{iεt} Z†
    # Kontrolle: P(T) = U(T) e^{iH_F T} muss die Eins sein
    fehler = np.abs(U @ Z @ np.diag(np.exp(1j*eps*T)) @ Z.conj().T - np.eye(K)).max()
    return ts, P, pos, idx, M, fehler


def kopplungs_profil(T, A, dphi, M=10, Nt=300, r_max=4.6):
    """ψ_i(t) = P(t)|i> für je einen A- und einen B-Startplatz, auf allen Plätzen bis Abstand r_max.
       Wegen Translationsinvarianz reicht das: jeder andere Platz hat dasselbe Profil, nur verschoben."""
    ts, P, pos, idx, M, fehler = P_von_t(T, A, dphi, M, Nt)
    print(f"  Kontrolle |P(T) − 1| = {fehler:.1e}")
    L1, L2 = M*b1, M*b2
    profile = []
    for z in (0, 1):                                 # z = 0: A-Platz, z = 1: B-Platz
        start = idx(M//2, M//2, z)
        tau_z = pos[start] - pos[idx(M//2, M//2, 0)] # Lage in der Zelle: 0 (A) oder a1 (B)
        disp = pos - pos[start]
        kand = np.stack([disp + u*L1 + v*L2 for u in (-1, 0, 1) for v in (-1, 0, 1)])
        disp = kand[np.argmin(np.linalg.norm(kand, axis=2), axis=0), np.arange(len(pos))]  # Minimalbild
        abstand = np.linalg.norm(disp, axis=1)
        sel = abstand <= r_max
        s_j = (np.arange(len(pos)) >= M*M).astype(int)            # Untergitter jedes Platzes
        profile.append(dict(r=(tau_z + disp)[sel],                # Position relativ zum Zellursprung von i
                            s=s_j[sel], abstand=abstand[sel],
                            psi=P[:, sel, start]))                # ψ_j(t), Form (Nt, n_j)
    return profile


def o_matrix(prof, variante):
    """o_jl(t) = ψ_j(t) ψ_l(t)*  bzw. der gewählte Teil davon. Form (Nt, n, n)."""
    psi, ab = prof["psi"], prof["abstand"]
    o = psi[:, :, None]*np.conj(psi[:, None, :])
    if variante == "r<=2":
        m = ab <= 2 + 1e-6
        o = o*(m[:, None] & m[None, :])[None]
    elif variante == "Dichte":
        o = o*np.eye(len(ab))[None]
    elif variante == "Kohärenz":
        o = o*(1 - np.eye(len(ab)))[None]
    return o


# ---------- Teil B: Ratengleichung mit mitbewegter Kopplung ----------

def bad_mitbewegt(T, A, dphi, variante="ideal_exakt", eta=0.1, kappa=1.0, wc=None, kT=0.02, Nk=12, Nt=300,
                  Nm=20, M_torus=10, r_max=4.6, profile=None):
    """Wie bad_dichte, aber das Bad koppelt an O_i(t) = P(t) n_i P†(t) (bzw. eine Variante davon).
       Gibt zurück: Besetzung p (Nk, Nk, 2) [unten, oben], eps (Nk, Nk, 2), profile (zum Wiederverwenden)."""
    beta = 1/kT
    omega = 2*np.pi/T
    if wc is None:
        wc = 10*omega
    ts, u, eps = zustaende_zeit(T, A, dphi, Nk, Nt)          # u(k,t) = P(t) φ(k)

    #ortsraum = variante in ("voll", "r<=2", "Dichte", "Kohärenz")
    ortsraum = variante not in ("n_i", "ideal_exakt")                 # neu
    if variante == "ideal_exakt":
        # Matrixelement mit P†(t)(P n P†)P(t) = n:  statt u(k,t) überall u(k,0) = φ(k)  ->  nur m = 0
        u = np.broadcast_to(u[:1], u.shape).copy()
    if ortsraum:
        if profile is None:
            profile = kopplungs_profil(T, A, dphi, M=M_torus, Nt=Nt, r_max=r_max)
        os_ = [o_matrix(p, variante) for p in profile]
        ks = 2*np.pi*np.arange(Nk)/Nk
        KX, KY = np.meshgrid(ks, ks, indexing='ij')
        kc = (KX[..., None]*G1 + KY[..., None]*G2)/(2*np.pi)
        tau = [np.zeros(2), a1]

    h = Nk//2
    q_liste = [(n1, n2) for n1 in range(-h, h+1) for n2 in range(-h, h+1) if (n1, n2) != (0, 0)]
    D = 2*Nk*Nk
    R = np.zeros((D, D))                                      # R[ziel, start] = Rate start -> ziel
    ms = np.arange(-Nm, Nm+1)
    i_idx, j_idx = np.meshgrid(np.arange(Nk), np.arange(Nk), indexing='ij')
    for (n1, n2) in q_liste:
        iz, jz = (i_idx + n1) % Nk, (j_idx + n2) % Nk
        w1, w2 = (i_idx + n1) // Nk, (j_idx + n2) // Nk
        if not ortsraum:
            # ---------- Matrixelement wie in bad_dichte: Σ_s u*_s(k+q) u_s(k) ----------
            phaseB = np.exp(1j*(w1[..., None]*G1 + w2[..., None]*G2) @ a1)
            uz = u[:, iz, jz]
            Mel = (np.conj(uz[..., 0, :])[..., :, None]*u[..., 0, :][..., None, :]
                   + phaseB[None, ..., None, None]*np.conj(uz[..., 1, :])[..., :, None]*u[..., 1, :][..., None, :])
        else:
            # ---------- Matrixelement mit Profil ----------
            # <k+q,α|V_q(t)|k,β> = Σ_{s_i} e^{iq·τ_{s_i}} Σ_{j,l} o_jl(t)
            #                      · e^{-i(k+q)·r_j} u*_{s_j α}(k+q,t) · e^{ik·r_l} u_{s_l β}(k,t)
            qv = (n1*G1 + n2*G2)/Nk
            Gw = w1[..., None]*G1 + w2[..., None]*G2          # k+q = (k+q zurückgefaltet) + Gw
            kq = kc + qv                                      # k+q, nicht zurückgefaltet
            uq = u[:, iz, jz].copy()                          # u(k+q) = e^{-iGw·τ_s} u(zurückgefaltet)
            uq[..., 1, :] *= np.exp(-1j*(Gw @ a1))[None, ..., None]
            Mel = 0
            for s_i in (0, 1):
                pr, o = profile[s_i], os_[s_i]
                Aj = np.exp(1j*kq @ pr["r"].T)[None, ..., None]*uq[:, :, :, pr["s"], :]   # (Nt,Nk,Nk,n,2)
                Bl = np.exp(1j*kc @ pr["r"].T)[None, ..., None]*u[:, :, :, pr["s"], :]
                oB = np.einsum('tjl,tably->tabjy', o, Bl, optimize=True)                  # Σ_l o_jl B_l
                Mel = Mel + np.exp(1j*qv @ tau[s_i])*np.einsum('tabjx,tabjy->tabxy', np.conj(Aj), oB,
                                                                optimize=True)
        # ---------- Fourier-Kanäle, Raten (wie bad_dichte) ----------
        X2 = np.abs(np.fft.ifft(Mel, axis=0)[ms % Nt])**2
        E = eps[None, :, :, None, :] - eps[iz, jz][None, :, :, :, None] + ms[:, None, None, None, None]*omega
        rate = eta**2*np.sum(X2*g_bath(E, kappa, wc, beta), axis=0)        # hier wirklich wc, nicht global
        start = ((i_idx*Nk + j_idx)*2)[..., None, None] + np.arange(2)[None, None, None, :]
        ziel = ((iz*Nk + jz)*2)[..., None, None] + np.arange(2)[None, None, :, None]
        np.add.at(R, (np.broadcast_to(ziel, rate.shape).ravel(), np.broadcast_to(start, rate.shape).ravel()),
                  rate.ravel())
    np.fill_diagonal(R, 0)
    L = R - np.diag(R.sum(axis=0))
    w, v = np.linalg.eig(L)
    p = np.real(v[:, np.argmin(np.abs(w))])
    return (p/p.sum()).reshape(Nk, Nk, 2), eps, profile

#%%
# =====================================================================================================
# ---------- Realistische Kopplungen (Vorschläge Prof): Bad koppelt an S(t) mit einfachen Funktionen f(t) ----------
# =====================================================================================================
#
# Allgemein:  H_SB = η Σ_q S_q(t) ⊗ B_q + h.c.,   Bad wie bisher (ohmsch, g_bath), alle q mit +q und −q
#
# Form 1 "dichte":   S_q(t) = Σ_i e^{iq·r_i} f_{s_i}(t) n_i            f_A(t) auf A-Plätzen, f_B(t) auf B-Plätzen
#     Matrixelement:  <k+q,α|S_q|k,β> = f_A(t) u*_A(k+q,t) u_A(k,t) + f_B(t) e^{iG·a1} u*_B(k+q,t) u_B(k,t)
#
# Form 2 "bindung":  S_q(t) = Σ_s f_s(t) Σ_(Bindungen R in Richtung a_s) e^{iq·(R + a_s/2)} K_s(R)
#     art="real":   K = a†_B a_A + a†_A a_B          (b†_l' b_l + h.c., Vorschlag Prof)
#     art="strom":  K = i (a†_B a_A − a†_A a_B)      (gleiche Bindung, um 90° phasenverschoben)
#     Matrixelement (A bei R, B bei R + a_s):
#       a†_B a_A:  e^{−i(k+q)·a_s} e^{iq·a_s/2} u*_B(k+q,t) u_A(k,t)
#       a†_A a_B:  e^{+ik·a_s}     e^{iq·a_s/2} u*_A(k+q,t) u_B(k,t)
#
# Standard für f: Form 1: f_A = f_B = 1 (ruhendes Dichtebad),  Form 2: f_s(t) = J_s(t)/J (Pulsprofil)
# Braucht: zustaende_zeit, g_bath, J, a1, a2, a3, G1, G2

def bad_prof(T, A, dphi, form="bindung", art="real", fA=None, fB=None, f_bind=None,
             eta=0.1, kappa=1.0, wc=None, kT=0.02, Nk=12, Nt=300, Nm=20, statisch=False):
    """Ratengleichung mit Badkopplung S_q(t).
       form="dichte":  S = Σ_i e^{iq·r_i} f_{A/B}(t) n_i
       form="bindung": S = Σ_s f_s(t) Σ_Bindungen e^{iq·r_Mitte} K_s
           art="real":  K = a†_B a_A + a†_A a_B          (α = 0)
           art="strom": K = i(a†_B a_A − a†_A a_B)       (α = 90°)
           art=Zahl α:  K = cos α (reell) + sin α (Strom)"""
    beta = 1/kT
    omega = 2*np.pi/T
    if wc is None:
        wc = 10*omega

    # ---- Schritt 1: Floquet-Zustände u(k,t) und Quasienergien ε(k) ----
    #      Hier entstehen die 90° vom Tunneln (in u), automatisch aus der Zeitentwicklung
    ts, u, eps = zustaende_zeit(T, A, dphi, Nk, Nt, statisch)

    # ---- Schritt 2: Zeitprofil der Kopplung ----
    if form == "dichte":
        gA = np.ones(Nt) if fA is None else np.asarray(fA(ts), float)*np.ones(Nt)
        gB = np.ones(Nt) if fB is None else np.asarray(fB(ts), float)*np.ones(Nt)
    else:
        if f_bind is None:                                   # Standard: Pulsprofil f_s(t) = J_s(t)/J
            phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
            fs = np.exp(A*np.cos(omega*ts[:, None] + phis[None]))/(3*np.i0(A))
        else:
            fs = np.asarray(f_bind(ts), float)
        ks = 2*np.pi*np.arange(Nk)/Nk
        KX, KY = np.meshgrid(ks, ks, indexing='ij')
        kc = (KX[..., None]*G1 + KY[..., None]*G2)/(2*np.pi)
        avec = [a1, a2, a3]

    h = Nk//2
    q_liste = [(n1, n2) for n1 in range(-h, h+1) for n2 in range(-h, h+1) if (n1, n2) != (0, 0)]
    D = 2*Nk*Nk
    R = np.zeros((D, D))                                     # R[ziel, start] = Rate
    ms = np.arange(-Nm, Nm+1)
    i_idx, j_idx = np.meshgrid(np.arange(Nk), np.arange(Nk), indexing='ij')

    for (n1, n2) in q_liste:                                 # jede Badmode q
        iz, jz = (i_idx + n1) % Nk, (j_idx + n2) % Nk
        w1, w2 = (i_idx + n1) // Nk, (j_idx + n2) // Nk

        # ---- Schritt 3: Matrixelement Mel(t) = ⟨u_f(k+q,t)| S_q(t) |u_i(k,t)⟩ ----
        if form == "dichte":
            phaseB = np.exp(1j*(w1[..., None]*G1 + w2[..., None]*G2) @ a1)
            uz = u[:, iz, jz]
            Mel = (gA[:, None, None, None, None]*np.conj(uz[..., 0, :])[..., :, None]*u[..., 0, :][..., None, :]
                   + gB[:, None, None, None, None]*phaseB[None, ..., None, None]
                   * np.conj(uz[..., 1, :])[..., :, None]*u[..., 1, :][..., None, :])
        else:
            qv = (n1*G1 + n2*G2)/Nk
            Gw = w1[..., None]*G1 + w2[..., None]*G2
            kq = kc + qv
            uq = u[:, iz, jz].copy()
            uq[..., 1, :] *= np.exp(-1j*(Gw @ a1))[None, ..., None]
            Mel = 0
            for s in range(3):                               # drei Bindungsrichtungen
                ph = np.exp(1j*qv @ avec[s]/2)               # Badwelle in der Bindungsmitte
                BA = (np.exp(-1j*kq @ avec[s])*ph)[None, ..., None, None] \
                     * np.conj(uq[..., 1, :])[..., :, None]*u[..., 0, :][..., None, :]   # a†_B a_A
                AB = (np.exp(1j*kc @ avec[s])*ph)[None, ..., None, None] \
                     * np.conj(uq[..., 0, :])[..., :, None]*u[..., 1, :][..., None, :]   # a†_A a_B
                # ======== HIER WIRD α GEWÄHLT ========
                if art == "real":
                    K = BA + AB                                        # α = 0
                elif art == "strom":
                    K = 1j*(BA - AB)                                   # α = 90°  -> Stromkopplung
                else:
                    K = np.cos(art)*(BA + AB) + np.sin(art)*1j*(BA - AB)   # beliebiges α
                # =====================================
                Mel = Mel + fs[:, s][:, None, None, None, None]*K   # mal Zeitprofil f_s(t)

        # ---- Schritt 4: Seitenbänder X^(m) und Raten  R = Σ_m |X^(m)|² g(E) ----
        X2 = np.abs(np.fft.ifft(Mel, axis=0)[ms % Nt])**2                            # |X^(m)|²  (System, enthält α)
        E = eps[None, :, :, None, :] - eps[iz, jz][None, :, :, :, None] + ms[:, None, None, None, None]*omega
        rate = eta**2*np.sum(X2*g_bath(E, kappa, wc, beta), axis=0)                  # mal g(E)  (nur Bad)
        start = ((i_idx*Nk + j_idx)*2)[..., None, None] + np.arange(2)[None, None, None, :]
        ziel = ((iz*Nk + jz)*2)[..., None, None] + np.arange(2)[None, None, :, None]
        np.add.at(R, (np.broadcast_to(ziel, rate.shape).ravel(), np.broadcast_to(start, rate.shape).ravel()),
                  rate.ravel())

    # ---- Schritt 5: stationäre Besetzung der Pauli-Gleichung ----
    np.fill_diagonal(R, 0)
    L = R - np.diag(R.sum(axis=0))
    w, v = np.linalg.eig(L)
    p = np.real(v[:, np.argmin(np.abs(w))])
    return (p/p.sum()).reshape(Nk, Nk, 2), eps          # p[..., 0] = unten, p[..., 1] = oben


#%%
# ---------- Kontrolle: ohne Antrieb muss Boltzmann herauskommen ----------
for form, art in [("dichte", "real"), ("bindung", "real"), ("bindung", "strom")]:
    p, eps_k = bad_prof(0.5, 1.0, dphi, form=form, art=art, f_bind=lambda t: np.full((len(t), 3), 1/3),
                        kT=0.5, wc=50.0, statisch=True)
    boltz = np.exp(-eps_k/0.5); boltz /= boltz.sum()
    print(f"Kontrolle {form}/{art}: max |p − Boltzmann| = {np.abs(p - boltz).max():.1e}")

#%%
# ---------- Vergleich: ruhendes Dichtebad, Form 1, Form 2, mitbewegtes Bad ----------
JT3, A_wert = 0.5*np.pi, 3.0
kT_bad = 0.02
T_ = 3*JT3/J
omega_ = 2*np.pi/T_

ergebnisse = {}
p, _ = bad_prof(T_, A_wert, dphi, form="dichte", kT=kT_bad)
ergebnisse["Dichte, ruhend (f_A = f_B = 1)"] = p[..., 0].sum()
p, _ = bad_prof(T_, A_wert, dphi, form="dichte", kT=kT_bad,
                fA=lambda t: 1 + 0.8*np.cos(omega_*t), fB=lambda t: 1 - 0.8*np.cos(omega_*t))
ergebnisse["Form 1: f_A, f_B = 1 ± 0.8 cos(ωt)"] = p[..., 0].sum()
p, _ = bad_prof(T_, A_wert, dphi, form="bindung", art="real", kT=kT_bad)
ergebnisse["Form 2: Bindung reell, f_s = J_s(t)/J"] = p[..., 0].sum()
p, _ = bad_prof(T_, A_wert, dphi, form="bindung", art="strom", kT=kT_bad)
ergebnisse["Form 2: Bindung Strom, f_s = J_s(t)/J"] = p[..., 0].sum()
p, _, _ = bad_mitbewegt(T_, A_wert, dphi, variante="ideal_exakt", kT=kT_bad, wc=10*omega_)
ergebnisse["mitbewegt ideal (P n P†)"] = p[..., 0].sum()

print(f"\nJT/3 = {JT3/np.pi:.2f}π, A = {A_wert}, kT = {kT_bad}")
for name, wert in ergebnisse.items():
    print(f"  {name:42s}  p_unten = {wert:.3f}")
#%%
JT3, A_wert = 0.5*np.pi, 3.0
kT_bad = 0.02
T_ = 3*JT3/J
omega_ = 2*np.pi/T_

p, eps_b, _ = bad_mitbewegt(T_, A_wert, dphi, variante="voll", kT=kT_bad, wc=10*omega_, Nk=12)
print(f"JT/3 = {JT3/np.pi:.2f}π, A = {A_wert}, mitbewegtes Bad:  p_unten = {p[..., 0].sum():.6f}"
      f"   (p_oben = {p[..., 1].sum():.2e})")
#%%
# # ---------- Anwendung: alle Varianten für einen Parametersatz ----------
# JT3, A_wert = 0.5*np.pi, 3.0
# kT_bad = 0.02
# T_ = 3*JT3/J
# omega_ = 2*np.pi/T_

# profil = None
# for var in ["n_i", "ideal_exakt", "voll", "r<=2", "Dichte", "Kohärenz"]:
#     p, eps_b, prof_neu = bad_mitbewegt(T_, A_wert, dphi, variante=var, kT=kT_bad, wc=10*omega_, Nk=12,
#                                       profile=profil)
#     if prof_neu is not None:
#         profil = prof_neu                        # Profil nur einmal ausrechnen
#     print(f"JT/3 = {JT3/np.pi:.2f}π, A = {A_wert}, {var:12s}: p_oben = {p[..., 0].sum():.3e}", flush=True)

# boltz = np.exp(-eps_b/kT_bad); boltz /= boltz.sum()
# print(f"Boltzmann bezüglich H_F zum Vergleich: p_oben = {boltz[..., 0].sum():.3e}")

#%%
# ---------- P(t)|i> im Ortsraum, mit Gitter darunter ----------
JT3, A_wert = 0.5*np.pi, 3.0
T_ = 3*JT3/J
ts, P, pos, idx, Mt, fehler = P_von_t(T_, A_wert, dphi, M=10, Nt=300)
print("Kontrolle |P(T) - 1| =", fehler)

_, _, _, partner = torus(Mt)                       # Bindungspartner für die Gitterlinien
start = idx(Mt//2, Mt//2, 0)                       # ein A-Platz in der Mitte
d = pos - pos[start]
radius = 3.2
nah = np.linalg.norm(d, axis=1) < radius           # nur die Umgebung zeigen

# Bindungen sammeln: nur echte Nachbarn (Abstand 1), beide Enden in der Nähe
linien = []
farben_bindung = ['tab:blue', 'tab:orange', 'tab:green']   # a1, a2, a3
for s in range(3):
    for a_ in range(Mt*Mt):                        # A-Plätze
        b_ = partner[s, a_]
        if np.linalg.norm(pos[a_] - pos[b_]) < 1.1 and (nah[a_] or nah[b_]):
            linien.append((d[a_], d[b_], s))

zeiten = [0, 1/6, 1/3, 1/2, 2/3, 5/6]
fig, ax = plt.subplots(1, len(zeiten), figsize=(2.8*len(zeiten), 3.0))
for a, tau in zip(ax, zeiten):
    n = int(round(tau*len(ts))) % len(ts)
    # Gitter: Bindungen und alle Plätze
    for p1, p2, s in linien:
        fs = np.exp(A_wert*np.cos(2*np.pi/T_*ts[n] + np.array([0, dphi, 2*dphi]) - np.pi/3))/(3*np.i0(A_wert))
        a.plot([p1[0], p2[0]], [p1[1], p2[1]], color=farben_bindung[s], lw=0.3 + 4*fs[s], alpha=0.6, zorder=1)
    a.scatter(d[nah, 0], d[nah, 1], s=12, facecolors='white', edgecolors='gray', lw=0.6, zorder=2)
    a.scatter(0, 0, s=60, facecolors='none', edgecolors='k', lw=1, zorder=3)   # Startplatz markieren
    # P(t)|start>
    psi = P[n, :, start]
    w = np.abs(psi)**2
    ph = np.angle(psi*np.conj(psi[np.argmax(w)]))
    sel = nah & (w > 2e-3)
    sc = a.scatter(d[sel, 0], d[sel, 1], s=500*w[sel], c=ph[sel], cmap='twilight',
                   vmin=-np.pi, vmax=np.pi, linewidths=0, zorder=4)
    a.set_title(f"t = {tau:.2f} T"); a.set_aspect('equal'); a.axis('off')
    a.set_xlim(-radius, radius); a.set_ylim(-radius, radius)
cb = fig.colorbar(sc, ax=ax, shrink=0.7, label="relative Phase")
cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cb.set_ticklabels(["−π", "−π/2", "0", "π/2", "π"])
fig.suptitle("P(t)|i⟩ auf dem Gitter: Größe = Gewicht |ψ_j|², Farbe = Phase φ_j, Kreis = Startplatz")
plt.show()
#%%
ab = np.linalg.norm(d, axis=1)
w_mittel = np.mean(np.abs(P[:, :, start])**2, axis=0)     # zeitgemittelt
plt.semilogy(ab, w_mittel, 'o')
plt.grid()
plt.xlabel("Abstand vom Startplatz"); plt.ylabel("⟨|P_ji(t)|²⟩_t"); plt.show()
#%%
ts, u, eps = zustaende_zeit(T_, A_wert, dphi, Nk=12)
u0 = u[0]                                                  # u(k, 0) = φ(k)
Pk = u @ np.linalg.inv(u0)[None]                           # P(k,t) = U_t · u0^{-1}, Form (Nt, Nk, Nk, 2, 2)
# z. B. Betrag der Einträge bei einem k gegen die Zeit:
ik, jk = 2, 5
for a_, b_ in [(0, 0), (0, 1)]:
    plt.plot(ts/T_, np.abs(Pk[:, ik, jk, a_, b_]), label=f"|P_{a_}{b_}(k,t)|")
for t_puls, name in [(1/6, "a1"), (1/2, "a2"), (5/6, "a3")]:          # Maxima der Pulse
    plt.axvline(t_puls, color='gray', ls='--', lw=0.8)

plt.text(t_puls, 1.02, name, ha='center', fontsize=8, color='gray')

gew_A = np.mean(np.abs(Pk[..., 0, 0])**2, axis=(1, 2))     # Mittel über alle k
w = 2*np.pi/T_
plt.plot(ts/T_, gew_A, label="⟨|P_AA(k,t)|²⟩_k  (Gewicht auf A)")
plt.plot(ts/T_, np.cos(w*ts/4)**2, 'k--', lw=0.8, label="cos²(ωt/4)")
plt.plot(ts/T_, np.sin(w*ts/4)**2, 'k:',  lw=0.8, label="sin²(ωt/4)")
for x in [1/6, 1/2, 5/6]:
    plt.axvline(x, color='gray', ls='--', lw=0.6)
plt.xlabel("t / T"); plt.legend(); plt.grid(alpha=0.3)

plt.grid(); plt.xlabel("t / T"); plt.legend(); plt.show()
# %%
# (a) Gewicht auf dem Startplatz und mittlerer Abstand vom Startplatz, gegen t
w_start = np.abs(P[:, start, start])**2                      # |P_ii(t)|²
w_t = np.abs(P[:, :, start])**2                               # |P_ji(t)|² für alle j
r_mittel = w_t @ ab                                           # ⟨Abstand⟩(t) = Σ_j |P_ji(t)|² · Abstand_j

fig, ax2 = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
ax2[0].plot(ts/T_, w_start)
ax2[0].set_ylabel("Gewicht auf Startplatz")
ax2[1].plot(ts/T_, r_mittel)
ax2[1].set_ylabel("mittlerer Abstand")
ax2[1].set_xlabel("t / T")
plt.tight_layout(); plt.show()

#%%
# =====================================================================================================
# ---------- Zerlegung der idealen Kopplung P n_i P† in Bausteine (nach Abstand, reell / Strom) ----------
# =====================================================================================================
# Einfach unter die Zelle mit bad_mitbewegt hängen. Ersetzt o_matrix durch eine erweiterte Version.
# WICHTIG: in bad_mitbewegt die Zeile
#     ortsraum = variante in ("voll", "r<=2", "Dichte", "Kohärenz")
# ersetzen durch
#     ortsraum = variante not in ("n_i", "ideal_exakt")
# (sonst werden die neuen Varianten nicht im Ortsraum gerechnet) und die Zelle mit bad_mitbewegt neu ausführen.

avec = [a1, a2, a3]

def o_matrix(prof, variante):
    """o_jl(t) = ψ_j(t) ψ*_l(t) bzw. der gewählte Teil davon. Form (Nt, n, n)."""
    psi, ab, r = prof["psi"], prof["abstand"], prof["r"]
    o = psi[:, :, None]*np.conj(psi[:, None, :])
    dist = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=2)     # Abstand zwischen Platz j und l
    nn   = np.isclose(dist, 1.0)                                       # Nachbarbindungen
    nnn  = np.isclose(dist, np.sqrt(3))                                # übernächste Nachbarn
    quer = np.isclose(dist, 2.0)                                       # quer durchs Sechseck
    diag = np.eye(len(ab), dtype=bool)                                 # Dichte (j = l)
    oR, oI = o.real, 1j*o.imag                                         # reelle Bindung / Strom
    wahl = {"voll": o, "Dichte": o*diag, "Kohärenz": o*(~diag),
            "r<=2": o*((ab <= 2 + 1e-6)[:, None] & (ab <= 2 + 1e-6)[None, :]),
            "NN": o*nn, "NN_reell": oR*nn, "NN_strom": oI*nn, "quer": o*quer, "NNN": o*nnn,
            "NN+quer": o*(nn | quer), "NN+NNN+quer": o*(nn | nnn | quer),
            "NN+quer+Dichte": o*(nn | quer | diag)}
    return wahl[variante]

#%%
# ---------- Gewichtsanteile und Tabelle ----------
JT3, A_wert = 0.5*np.pi, 3.0
T_ = 3*JT3/J
prof = kopplungs_profil(T_, A_wert, dphi)

# Anteil jedes Bausteins an Σ_jl |o_jl|² (zeitgemittelt; für einen Projektor ist die Summe genau 1)
pr = prof[0]
for teil in ["Dichte", "NN", "NNN", "quer", "voll"]:
    anteil = np.mean(np.sum(np.abs(o_matrix(pr, teil))**2, axis=(1, 2)))
    print(f"  Anteil {teil:6s}: {anteil:.3f}")

# Bad mit den einzelnen Bausteinen (je etwa 1–2 min)
for var in ["NN", "quer", "NN+quer", "NN+NNN+quer", "NN+quer+Dichte", "voll"]:
    p, _, _ = bad_mitbewegt(T_, A_wert, dphi, variante=var, profile=prof, Nk=12)
    print(f"JT/3 = {JT3/np.pi:.2f}π, A = {A_wert}, {var:16s}: p_unten = {p[..., 0].sum():.4f}", flush=True)
# %%
#%%
# =====================================================================================================
# ---------- Kühlen in den Dunkelzustand: Lindblad-Gleichung mit einem Sprungoperator pro Sechseck ----------
# =====================================================================================================
#
#   dρ/dt = −i[H_S(t), ρ] + γ Σ_h ( L_h ρ L_h† − ½{L_h† L_h, ρ} ),      L_h = |D_h⟩⟨B_h|
#
#   |D_h⟩ = Wannier-Zustand des unteren Bandes auf dem Dimer h (Dunkelzustand, Ziel)
#   |B_h⟩ = Wannier-Zustand des oberen Bandes auf demselben Dimer ("hell", wird gepumpt)
#   Dimer: A-Platz a und der Platz b, auf den U(T) ihn schickt (gegenüber im Sechseck)
#
#   Versionen, wie L_h(t) der Mikrobewegung folgt:
#     "ideal":    L_h(t) = U(t) L_h(0) U†(t)        (exakt mitgeführt, bis auf eine Phase = P L P†)
#     "stufen":   3 feste Muster pro Periode, je auf die 2 Plätze mit dem größten Gewicht beschränkt
#                 (wie Laser, die bei jedem Puls auf das nächste Plätzepaar umschalten)
#     "ruhend":   L_h(t) = L_h(0) für alle t
#
#   Gemessen: Besetzung des unteren Floquet-Bandes  p_unten = Tr(ρ Π_unten)  stroboskopisch bei t = nT
# Braucht: torus, J, dphi

def dimere(UT, M):
    """Paare (a, b): A-Platz a und Platz b mit dem größten |U(T)_ba|. Dazu D (unteres) und B (oberes) Band."""
    K = UT.shape[0]
    D, B = np.zeros((K, M*M), complex), np.zeros((K, M*M), complex)
    T_b = []
    for h, a in enumerate(range(M*M)):                         # A-Plätze 0 … M²−1
        b = np.argmax(np.abs(UT[:, a]))
        blk = UT[np.ix_([a, b], [a, b])]                       # U(T) auf dem Dimer
        lam, vec = np.linalg.eig(blk)
        eps_d = -np.angle(lam)                                 # ε·T
        iu, io = np.argmin(eps_d), np.argmax(eps_d)            # unten / oben
        D[[a, b], h] = vec[:, iu]/np.linalg.norm(vec[:, iu])
        B[[a, b], h] = vec[:, io]/np.linalg.norm(vec[:, io])
        T_b.append(b)
    if len(set(T_b)) != M*M:
        print("  Warnung: Dimere nicht eindeutig")
    return D, B

def auf_zwei_plaetze(V):
    """Jede Spalte auf die zwei Plätze mit dem größten Gewicht beschränken und normieren."""
    W = np.zeros_like(V)
    for h in range(V.shape[1]):
        top = np.argsort(np.abs(V[:, h]))[-2:]
        W[top, h] = V[top, h]
    return W/np.linalg.norm(W, axis=0, keepdims=True)

def lindblad_dunkel(T, A, dphi, version="ideal", gamma=0.2, M=6, Nt=300, perioden=40, start=None,
                    breit=False, n_plaetze=2):
    """breit=True: Dunkel-/Hellzustände auf die echten Floquet-Bänder projiziert (breitere Wannier-Zustände).
       n_plaetze: auf wie viele Plätze die Muster bei "stufen" beschränkt werden."""
    omega = 2*np.pi/T
    norm = 3*np.i0(A)
    phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
    K, idx, pos, partner = torus(M)
    dt = T/Nt

    def bindung(s, c, X):                                      # exp(−i c J H_s) auf X (Plätze = erste Achse)
        return np.cos(c*J)*X - 1j*np.sin(c*J)*X[partner[s]]

    def schritt(n, X):                                         # X(t_n) -> X(t_n + dt)
        c = np.exp(A*np.cos(omega*(n + 0.5)*dt + phis))/norm*dt
        X = bindung(0, c[0]/2, X); X = bindung(1, c[1]/2, X); X = bindung(2, c[2], X)
        X = bindung(1, c[1]/2, X); X = bindung(0, c[0]/2, X)
        return X

    # --- U(T), Floquet-Projektor auf das untere Band, Dimere ---
    U = np.eye(K, dtype=complex)
    Us = []
    for n in range(Nt):
        Us.append(U)
        U = schritt(n, U)
    UT = U
    lam, Z = np.linalg.eig(UT)
    unten = -np.angle(lam) < 0
    Zu = np.linalg.qr(Z[:, unten])[0]
    Pi_unten = Zu @ Zu.conj().T
    D0, B0 = dimere(UT, M)
    if breit:                                                  # auf die Bänder projizieren und orthonormieren
        Zo = np.linalg.qr(Z[:, ~unten])[0]
        def loewdin(V):
            S = V.conj().T @ V
            w_, Q = np.linalg.eigh(S)
            return V @ (Q @ np.diag(w_**-0.5) @ Q.conj().T)
        D0 = loewdin(Pi_unten @ D0)
        B0 = loewdin((Zo @ Zo.conj().T) @ B0)

    # --- Muster für die Version "stufen": Zeitpunkte 0, T/3, 2T/3 (zwischen den Pulsen) ---
    if version == "stufen":
        muster = []
        for n_ref in (0, Nt//3, 2*Nt//3):
            muster.append((auf_zwei_plaetze(Us[n_ref] @ D0, n_plaetze), auf_zwei_plaetze(Us[n_ref] @ B0, n_plaetze)))
        def muster_von(n):                                     # Segmentgrenzen bei den Pulsen T/6, T/2, 5T/6
            t = (n % Nt)/Nt
            return muster[0] if (t < 1/6 or t >= 5/6) else (muster[1] if t < 1/2 else muster[2])

    # --- Startzustand: ein Teilchen auf einem A-Platz in der Mitte ---
    if start is None:
        start = idx(M//2, M//2, 0)
    rho = np.zeros((K, K), complex); rho[start, start] = 1.0
    Dt, Bt = D0.copy(), B0.copy()                              # für "ideal": mit der Dynamik mitgeführt

    verlauf = [np.real(np.trace(Pi_unten @ rho))]
    for p in range(perioden):
        for n in range(Nt):
            # Sprungoperatoren zu dieser Zeit
            if version == "ideal":
                Dn, Bn = Dt, Bt
            elif version == "stufen":
                Dn, Bn = muster_von(n)
            else:
                Dn, Bn = D0, B0
            # Dissipator: Σ_h ( |D⟩⟨B|ρ|B⟩⟨D| − ½{|B⟩⟨B|, ρ} )
            BrB = np.real(np.einsum('kh,kl,lh->h', Bn.conj(), rho, Bn))
            PB = Bn @ Bn.conj().T
            diss = (Dn*BrB) @ Dn.conj().T - 0.5*(PB @ rho + rho @ PB)
            rho = rho + gamma*dt*diss
            # unitärer Schritt: ρ -> V ρ V†
            rho = schritt(n, rho)
            rho = schritt(n, rho.conj().T).conj().T
            if version == "ideal":
                Dt, Bt = schritt(n, Dt), schritt(n, Bt)
        verlauf.append(np.real(np.trace(Pi_unten @ rho)))
    return np.array(verlauf), np.real(np.trace(rho))


#%%
# ---------- Vergleich der drei Versionen für mehrere A ----------
import time
JT3 = 0.5*np.pi
T_ = 3*JT3/J
ergebnis = {}
for A_wert in [6.0, 3.0, 1.0]:
    for version in ["ideal", "stufen", "ruhend"]:
        t0 = time.time()
        v, spur = lindblad_dunkel(T_, A_wert, dphi, version=version, gamma=0.2, M=6, perioden=40)
        ergebnis[(A_wert, version)] = v
        print(f"A = {A_wert:g}, {version:7s}: p_unten nach 0/5/10/20/40 Perioden = "
              f"{v[0]:.3f} {v[5]:.3f} {v[10]:.3f} {v[20]:.3f} {v[40]:.3f}   (Spur {spur:.4f}, {time.time()-t0:.0f}s)",
              flush=True)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for a, A_wert in zip(ax, [6.0, 3.0, 1.0]):
    for version, stil in [("ideal", '-'), ("stufen", '--'), ("ruhend", ':')]:
        a.plot(ergebnis[(A_wert, version)], stil, label=version)
    a.set_title(f"A = {A_wert:g}, JT/3 = 0.5π, γ = 0.2J"); a.set_xlabel("Perioden"); a.grid(alpha=0.3)
ax[0].set_ylabel("Besetzung unteres Band"); ax[0].legend()
plt.tight_layout(); plt.savefig("dunkelzustand.png", dpi=110); plt.show()
# %%
#%%
# =====================================================================================================
# ---------- (a) Warum kühlt die ruhende Version? Dunkelzustände pro k ----------
# =====================================================================================================
# Braucht: dunkelzustand_zelle.py (dimere, lindblad_dunkel), torus, zustaende_zeit, J, dphi, a1, b1, b2, G1, G2
#
# Idee: Der Sprungoperator L_h = |D_h⟩⟨B_h| regt einen Zustand φ an, wenn ⟨B_h|φ⟩ ≠ 0.
#   ideal (mitlaufend):  φ wird zu jeder Zeit mit dem Muster mitgeführt  -> |⟨B|φ_unten⟩|² = 0 für ALLE k
#   ruhend:              Muster fest, φ_unten(k,t) bewegt sich           -> angeregt, außer bei wenigen k
#   Maß pro k:  Anregung(k) = ⟨ |⟨B_0|φ_unten(k,t)⟩|² ⟩_t     (0 = Dunkelzustand)

def dimer_vektoren(T, A, dphi, M=6, Nt=300):
    """D und B eines Dimers (2 Plätze) in Positions-Konvention: Amplituden, Lagen r_j, Untergitter s_j."""
    omega = 2*np.pi/T
    norm = 3*np.i0(A)
    phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
    K, idx, pos, partner = torus(M)
    dt = T/Nt
    bind = lambda s, c, X: np.cos(c*J)*X - 1j*np.sin(c*J)*X[partner[s]]
    U = np.eye(K, dtype=complex)
    for n in range(Nt):
        c = np.exp(A*np.cos(omega*(n + 0.5)*dt + phis))/norm*dt
        U = bind(0, c[0]/2, U); U = bind(1, c[1]/2, U); U = bind(2, c[2], U)
        U = bind(1, c[1]/2, U); U = bind(0, c[0]/2, U)
    D0, B0 = dimere(U, M)
    a = 0                                                        # A-Platz in Zelle (0,0)
    b = [j for j in np.nonzero(np.abs(D0[:, 0]) > 1e-9)[0] if j != a][0]
    d = pos[b] - pos[a]
    kand = [d + u*M*b1 + v*M*b2 for u in (-1, 0, 1) for v in (-1, 0, 1)]
    d = min(kand, key=np.linalg.norm)                            # kürzester Abstand auf dem Torus
    return D0[[a, b], 0], B0[[a, b], 0], np.array([np.zeros(2), d]), [0, int(b >= M*M)]

JT3 = 0.5*np.pi
T_ = 3*JT3/J
Nk = 12
for A_wert in [6.0, 3.0]:
    Dv, Bv, rr, ss = dimer_vektoren(T_, A_wert, dphi)
    ts, u, eps = zustaende_zeit(T_, A_wert, dphi, Nk, 300)        # u[t, i, j, Platz, Band]
    ks = 2*np.pi*np.arange(Nk)/Nk
    KX, KY = np.meshgrid(ks, ks, indexing='ij')
    kc = (KX[..., None]*G1 + KY[..., None]*G2)/(2*np.pi)
    psi = np.exp(1j*kc @ rr.T)[None, :, :, :, None]*u[:, :, :, ss, :]   # Bloch-Amplituden auf den Dimer-Plätzen
    for version in ["ideal", "ruhend"]:
        p_use = np.broadcast_to(psi[:1], psi.shape) if version == "ideal" else psi
        anr_u = np.mean(np.abs(np.einsum('j,tabj->tab', Bv.conj(), p_use[..., 0]))**2, axis=0)   # unteres Band
        anr_o = np.mean(np.abs(np.einsum('j,tabj->tab', Bv.conj(), p_use[..., 1]))**2, axis=0)   # oberes Band
        dunkel = int((anr_u < 0.01*max(anr_u.mean(), 1e-12)).sum()) if version == "ruhend" else int((anr_u < 1e-3).sum())
        print(f"A = {A_wert:g}, {version:6s}: Anregung unteres Band im Mittel {anr_u.mean():.3f}, "
              f"oberes Band {anr_o.mean():.3f},  dunkle k im unteren Band: {dunkel} von {Nk*Nk}")
    if A_wert == 3.0:
        plt.figure(figsize=(4.2, 3.6))
        plt.imshow(anr_u.T, origin='lower', cmap='viridis')
        plt.colorbar(label="⟨|⟨B|φ_unten(k,t)⟩|²⟩_t")
        plt.xlabel("k₁-Index"); plt.ylabel("k₂-Index")
        plt.title(f"ruhend, A = {A_wert:g}: Anregung pro k\n(dunkel = wird nicht angeregt)", fontsize=9)
        plt.tight_layout(); plt.show()

# Langzeitlauf der ruhenden Version: kühlt trotzdem, aber langsam (Teilchen muss einen der wenigen dunklen k finden)
v, _ = lindblad_dunkel(T_, 3.0, dphi, version="ruhend", gamma=0.2, perioden=200)
print("ruhend, A = 3: p_unten nach 40/80/120/200 Perioden =", np.round(v[[40, 80, 120, 200]], 4))


#%%
# =====================================================================================================
# ---------- (b) Breitere Sprungoperatoren: Dunkelzustände auf die echten Floquet-Bänder projiziert ----------
# =====================================================================================================
#   breit=False: Zwei-Platz-Dimere (exakt nur für A -> ∞)
#   breit=True:  Π_unten·D und Π_oben·B, orthonormiert  -> echte Wannier-Zustände (auch für kleines A)
#   n_plaetze:   bei "stufen" werden die Muster auf so viele Plätze beschränkt (realistische Laser)

for A_wert in [1.0, 3.0]:
    print(f"\nA = {A_wert:g}, JT/3 = 0.5π, γ = 0.2")
    for version, breit, n in [("ideal", False, 2), ("ideal", True, 2), ("stufen", True, 2),
                              ("stufen", True, 6), ("stufen", True, 12), ("ruhend", True, 2)]:
        v, _ = lindblad_dunkel(T_, A_wert, dphi, version=version, breit=breit, n_plaetze=n, perioden=40)
        print(f"  {version:6s} breit={breit!s:5s} n={n:2d}:  p_unten nach 5/10/20/40 Perioden = "
              f"{v[5]:.3f} {v[10]:.3f} {v[20]:.3f} {v[40]:.3f}", flush=True)


#%%
# =====================================================================================================
# ---------- (c) Abhängigkeit von der Rate γ ----------
# =====================================================================================================
A_wert = 3.0
gammas = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for version, breit, stil in [("stufen", False, 'o-'), ("ideal", True, 's--')]:
    end, n95 = [], []
    for g in gammas:
        v, _ = lindblad_dunkel(T_, A_wert, dphi, version=version, breit=breit, gamma=g, perioden=40)
        end.append(v[40])
        n95.append(next((i for i, x in enumerate(v) if x >= 0.95), np.nan))
        print(f"{version:6s} γ = {g:4.2f}: p_unten nach 40 Perioden = {v[40]:.3f},  ≥ 0.95 nach {n95[-1]} Perioden",
              flush=True)
    ax[0].semilogx(gammas, n95, stil, label=version)
    ax[1].semilogx(gammas, end, stil, label=version)
ax[0].set_xlabel("γ (in J)"); ax[0].set_ylabel("Perioden bis p_unten ≥ 0.95"); ax[0].grid(alpha=0.3); ax[0].legend()
ax[1].set_xlabel("γ (in J)"); ax[1].set_ylabel("p_unten nach 40 Perioden"); ax[1].grid(alpha=0.3)
fig.suptitle(f"Sprungoperator-Kühlung, A = {A_wert:g}, JT/3 = 0.5π")
plt.tight_layout(); plt.show()
# %%
