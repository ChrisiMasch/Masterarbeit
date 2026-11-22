#%%
"""
=====================================================================================================
  Sprungoperatoren für den anomalen Floquet-Isolator auf dem ZYLINDER
=====================================================================================================

Wie sprungoperatoren_torus.py (Methoden A, B, C), aber auf einem Zylinder:
    periodisch entlang b1 (M Zellen), offen entlang b2 (Nz Zellen)  ->  zwei Ränder bei j = 0 und j = Nz−1.

Was sich gegenüber dem Torus ändert:
  * Gitter:     partner[s, p] = −1, wenn der Partner hinter dem Rand läge. Diese Plätze bleiben im
                Bindungsschritt unverändert (wie bei der Flocke).
  * Dimere:     nur im Inneren. Ein Paar (a, b) ist ein Dimer, wenn U(T) a fast vollständig nach b bringt
                UND b nach a (gegenseitig). Die A-Plätze der Randzeile j = 0 haben keinen Partner, sie
                gehören zum Randkanal. Anzahl Dimere: nd = M·(Nz−1).
  * Bänder:     Die Eigenzustände von U(T) zerfallen in
                    Bulk unten (ε < 0), Bulk oben (ε ≥ 0), Randzustände (chiral, laufen durch alle ε).
                Getrennt über das Gewicht auf den beiden äußersten Zeilen (Randzustände: > 0.6).
                Eigenvektoren aus der Schur-Zerlegung -> exakt orthonormal, die drei Projektoren
                summieren sich zur Eins.
  * Messung:    stroboskopisch p_unten, p_oben, p_rand = Tr(Π ρ).  verlauf hat die Form (perioden+1, 3).
  * Sprungoperatoren gibt es nur auf den Bulk-Dimeren. Auf den Randzuständen wirkt nichts -> was dort
    ist, bleibt dort (bzw. wird nur über die Kopplung an Nachbar-Dimere angeregt).

Analyse (a) aus der Torus-Datei (Dunkelzustände pro k) ist eine reine Bulk-Rechnung im k-Raum und hängt
nicht von den Randbedingungen ab; sie bleibt in der Torus-Datei.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import schur


#%%
# =====================================================================================================
# 1. Parameter und Gitter
# =====================================================================================================
J = 1.0                      # Hüpfstärke (Energieeinheit), ħ = 1
dphi = -2*np.pi/3            # Phasenversatz der drei Bindungen (Vorzeichen = Drehsinn)

a1 = np.array([0, 1.0])                       # die drei Bindungsvektoren A -> B
a2 = np.array([-np.sqrt(3)/2, -0.5])
a3 = np.array([ np.sqrt(3)/2, -0.5])
b1, b2 = a1 - a2, a1 - a3                     # Gittervektoren
G1, G2 = 2*np.pi*np.linalg.inv(np.array([b1, b2])).T   # reziproke Gittervektoren (b_i·G_j = 2π δ_ij)


def zylinder(M, Nz):
    """Zylinder aus M×Nz Zellen: periodisch entlang b1 (i mod M), offen entlang b2 (j = 0 … Nz−1).
       Plätze: A in Zelle (i,j) -> idx(i,j,0), B -> idx(i,j,1).  K = 2·M·Nz.
       partner[s, p] = Partner von Platz p in Richtung s, −1 = kein Partner (Rand)."""
    K = 2*M*Nz
    idx = lambda i, j, z: (i % M) + j*M + z*M*Nz
    pos = np.zeros((K, 2))
    zeile = np.zeros(K, dtype=int)                       # j jedes Platzes
    for i in range(M):
        for j in range(Nz):
            pos[idx(i, j, 0)] = i*b1 + j*b2
            pos[idx(i, j, 1)] = i*b1 + j*b2 + a1
            zeile[idx(i, j, 0)] = zeile[idx(i, j, 1)] = j
    bindungen = [(0, 0), (-1, 0), (0, -1)]          # a1, a2, a3: A(i,j) -> B(i+di, j+dj)
    partner = -np.ones((3, K), dtype=int)
    for s, (di, dj) in enumerate(bindungen):
        for i in range(M):
            for j in range(Nz):
                jz = j + dj
                if 0 <= jz < Nz:                        # entlang b2 offen, entlang b1 periodisch (idx macht i mod M)
                    a, b = idx(i, j, 0), idx(i + di, jz, 1)
                    partner[s, a] = b
                    partner[s, b] = a
    return K, idx, pos, partner, zeile


#%%
# =====================================================================================================
# 2. Gemeinsame Bausteine
# =====================================================================================================

def mache_zeitschritt(T, A, M, Nz, Nt, K_gitter=None):
    """Gibt schritt(n, X) zurück: X(t_n) -> X(t_n + dt) unter H(t) = Σ_s J_s(t) H_s.
       Auf jedem Paar exakt: cos(cJ) ψ − i sin(cJ) ψ[partner]; Plätze ohne Partner bleiben.
       Strang-Zerlegung s1/2, s2/2, s3, s2/2, s1/2.
       K_gitter: nur auf die ersten K_gitter Zeilen wirken (Hilfsplätze in B/C bleiben unverändert)."""
    omega = 2*np.pi/T
    norm = 3*np.i0(A)
    phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
    K, idx, pos, partner, zeile = zylinder(M, Nz)
    Kg = K if K_gitter is None else K_gitter
    dt = T/Nt
    hp = [np.nonzero(partner[s] >= 0)[0] for s in range(3)]     # Plätze mit Partner
    pp = [partner[s, hp[s]] for s in range(3)]                  # deren Partner

    def bindung(s, c, X):
        Y = X.copy()
        Y[hp[s]] = np.cos(c*J)*X[hp[s]] - 1j*np.sin(c*J)*X[pp[s]]
        return Y

    def schritt(n, X):
        c = np.exp(A*np.cos(omega*(n + 0.5)*dt + phis))/norm*dt
        X = bindung(0, c[0]/2, X); X = bindung(1, c[1]/2, X); X = bindung(2, c[2], X)
        X = bindung(1, c[1]/2, X); X = bindung(0, c[0]/2, X)
        return X
    return schritt


def floquet(T, A, M, Nz, Nt, rand_schwelle=0.6):
    """U(t_n) für eine Periode, U(T) und die Projektoren Π_unten, Π_oben (Bulk) und Π_rand.
       Eigenvektoren von U(T) aus der komplexen Schur-Zerlegung (U normal -> Schur-Form diagonal,
       Spalten exakt orthonormal). Randzustand = Gewicht auf den Zeilen j = 0 und j = Nz−1 > rand_schwelle."""
    K, idx, pos, partner, zeile = zylinder(M, Nz)
    schritt = mache_zeitschritt(T, A, M, Nz, Nt)
    U = np.eye(K, dtype=complex); Us = []
    for n in range(Nt):
        Us.append(U)
        U = schritt(n, U)
    Ts, Z = schur(U, output='complex')
    eps = -np.angle(np.diag(Ts))/T
    randplatz = (zeile == 0) | (zeile == Nz - 1)
    w_rand = (np.abs(Z)**2)[randplatz].sum(axis=0)
    rand = w_rand > rand_schwelle
    unten = (~rand) & (eps < 0)
    oben = (~rand) & (eps >= 0)
    proj = lambda m: Z[:, m] @ Z[:, m].conj().T
    info = dict(eps=eps, w_rand=w_rand, n_rand=int(rand.sum()), n_unten=int(unten.sum()), n_oben=int(oben.sum()))
    return Us, U, proj(unten), proj(oben), proj(rand), info


def dimere(UT, M, Nz, schwelle=0.5):
    """Bulk-Dimere aus U(T): Paar (a, b) mit |U_ba|² > schwelle und |U_ab|² > schwelle, gegenseitig
       (b ist der stärkste Zielplatz von a und umgekehrt). Auf dem 2×2-Block diagonalisieren:
       D = kleineres ε (unten), B = größeres ε (oben). Randplätze ohne Dimer werden übersprungen."""
    K = UT.shape[0]
    W = np.abs(UT)**2
    paare = []
    for a in range(M*Nz):                              # A-Plätze sind 0 … M·Nz−1
        b = int(np.argmax(W[:, a]))
        if int(np.argmax(W[:, b])) == a and W[b, a] > schwelle and W[a, b] > schwelle:
            paare.append((a, b))
    nd = len(paare)
    D, B = np.zeros((K, nd), complex), np.zeros((K, nd), complex)
    for h, (a, b) in enumerate(paare):
        blk = UT[np.ix_([a, b], [a, b])]
        lam, vec = np.linalg.eig(blk)
        eps_d = -np.angle(lam)
        iu, io = np.argmin(eps_d), np.argmax(eps_d)
        D[[a, b], h] = vec[:, iu]/np.linalg.norm(vec[:, iu])
        B[[a, b], h] = vec[:, io]/np.linalg.norm(vec[:, io])
    if nd != M*(Nz - 1):
        print(f"  Hinweis: {nd} Dimere gefunden, erwartet M·(Nz−1) = {M*(Nz - 1)}")
    return D, B


def loewdin(V):
    """Spalten von V orthonormieren, so dass sie möglichst wenig verändert werden: V S^{-1/2}, S = V†V."""
    S = V.conj().T @ V
    w, Q = np.linalg.eigh(S)
    return V @ (Q @ np.diag(w**-0.5) @ Q.conj().T)


def breite_dimere(D0, B0, Pi_unten, Pi_oben):
    """Dimer-Zustände auf das echte untere/obere Bulk-Band projizieren und orthonormieren."""
    return loewdin(Pi_unten @ D0), loewdin(Pi_oben @ B0)


def auf_zwei_plaetze(V, n=2):
    """Jede Spalte auf die n Plätze mit dem größten Gewicht beschneiden und normieren."""
    W = np.zeros_like(V)
    for h in range(V.shape[1]):
        top = np.argsort(np.abs(V[:, h]))[-n:]
        W[top, h] = V[top, h]
    return W/np.linalg.norm(W, axis=0, keepdims=True)


def sechsecke(Us, D0, B0, Nt):
    """Die 6 Plätze, die jeder Dimer-Zustand während einer Periode besucht. Form (6, Anzahl Dimere)."""
    gew = sum(np.abs(Us[n] @ D0)**2 + np.abs(Us[n] @ B0)**2 for n in range(0, Nt, 5))
    return np.argsort(gew, axis=0)[-6:, :]


def stufen_muster(Us, D0, B0, Nt, n_muster=3, n_plaetze=2):
    """n_muster feste Muster pro Periode, aufgenommen bei t = k·T/n_muster, auf n_plaetze beschnitten."""
    refs = [int(round(k*Nt/n_muster)) % Nt for k in range(n_muster)]
    muster = [(auf_zwei_plaetze(Us[r] @ D0, n_plaetze), auf_zwei_plaetze(Us[r] @ B0, n_plaetze)) for r in refs]
    def muster_von(n):
        t = (n % Nt)/Nt
        return muster[int(np.floor(t*n_muster + 0.5)) % n_muster]
    return muster_von


def laser_pause(n, Nt, n_muster, aus_fenster):
    """1, wenn der Laser an ist, 0 in einem Fenster ±aus_fenster·T um jeden Umschaltzeitpunkt."""
    if aus_fenster <= 0:
        return 1.0
    t = ((n + 0.5) % Nt)/Nt*n_muster
    abstand = abs((t % 1.0) - 0.5)/n_muster
    return 0.0 if abstand < aus_fenster else 1.0


def messen(rho_g, Pis):
    """p_unten, p_oben, p_rand für die Gitter-Dichtematrix."""
    return [np.real(np.trace(P @ rho_g)) for P in Pis]


#%%
# =====================================================================================================
# 3. Methode A: effektiver Sprungoperator  L_h = |D_h⟩⟨B_h|   (nur Bulk-Dimere)
# =====================================================================================================

def lindblad_dunkel(T, A, version="ideal", gamma=0.2, M=6, Nz=8, Nt=300, perioden=40, start=None,
                    breit=False, n_plaetze=2):
    """Gibt zurück: verlauf (perioden+1, 3) mit Spalten p_unten, p_oben, p_rand; Spur von ρ."""
    K, idx, pos, partner, zeile = zylinder(M, Nz)
    dt = T/Nt
    schritt = mache_zeitschritt(T, A, M, Nz, Nt)
    Us, UT, Pi_unten, Pi_oben, Pi_rand, info = floquet(T, A, M, Nz, Nt)
    Pis = (Pi_unten, Pi_oben, Pi_rand)
    D0, B0 = dimere(UT, M, Nz)
    if breit:
        D0, B0 = breite_dimere(D0, B0, Pi_unten, Pi_oben)
    if version == "stufen":
        muster_von = stufen_muster(Us, D0, B0, Nt, 3, n_plaetze)

    if start is None:
        start = idx(M//2, Nz//2, 0)                    # A-Platz in der Mitte des Zylinders
    rho = np.zeros((K, K), complex); rho[start, start] = 1.0
    Dt, Bt = D0.copy(), B0.copy()

    verlauf = [messen(rho, Pis)]
    for p in range(perioden):
        for n in range(Nt):
            if version == "ideal":
                Dn, Bn = Dt, Bt
            elif version == "stufen":
                Dn, Bn = muster_von(n)
            else:
                Dn, Bn = D0, B0
            BrB = np.real(np.einsum('kh,kl,lh->h', Bn.conj(), rho, Bn))
            PB = Bn @ Bn.conj().T
            rho = rho + gamma*dt*((Dn*BrB) @ Dn.conj().T - 0.5*(PB @ rho + rho @ PB))
            rho = schritt(n, rho)
            rho = schritt(n, rho.conj().T).conj().T
            if version == "ideal":
                Dt, Bt = schritt(n, Dt), schritt(n, Bt)
        verlauf.append(messen(rho, Pis))
    return np.array(verlauf), np.real(np.trace(rho))


#%%
# =====================================================================================================
# 4. Methode B: expliziter Hilfsplatz + Laser Ω + spontaner Zerfall Γ   (ein Hilfsplatz pro Bulk-Dimer)
# =====================================================================================================

def lindblad_hilfsplatz(T, A, version="ideal", zerfall="selektiv", Omega=0.2, Gamma=2.0,
                        M=6, Nz=8, Nt=300, perioden=30, start=None, n_muster=3, aus_fenster=0.0, n_plaetze=2):
    """Gibt zurück: verlauf (perioden+1, 3), Besetzung der Hilfsplätze, Spur von ρ."""
    K, idx, pos, partner, zeile = zylinder(M, Nz)
    dt = T/Nt
    Us, UT, Pi_unten, Pi_oben, Pi_rand, info = floquet(T, A, M, Nz, Nt)
    Pis = (Pi_unten, Pi_oben, Pi_rand)
    D0, B0 = dimere(UT, M, Nz)
    nd = D0.shape[1]
    N = K + nd
    schritt = mache_zeitschritt(T, A, M, Nz, Nt, K_gitter=K)
    sechseck = sechsecke(Us, D0, B0, Nt)
    if version == "stufen":
        muster_von = stufen_muster(Us, D0, B0, Nt, n_muster, n_plaetze)

    def generator(rho, Dn, Bn, Om):
        HLr = np.zeros_like(rho)
        HLr[K:] = Om*(Bn.conj().T @ rho[:K])
        HLr[:K] = Om*(Bn @ rho[K:])
        out = -1j*(HLr - HLr.conj().T)
        pa = np.real(np.diag(rho)[K:])
        out[K:, :] -= 0.5*Gamma*rho[K:, :]
        out[:, K:] -= 0.5*Gamma*rho[:, K:]
        if zerfall == "selektiv":
            out[:K, :K] += Gamma*(Dn*pa) @ Dn.conj().T
        else:
            g = np.zeros(K)
            for h in range(nd):
                g[sechseck[:, h]] += Gamma*pa[h]/6
            out[np.arange(K), np.arange(K)] += g
        return out

    if start is None:
        start = idx(M//2, Nz//2, 0)
    rho = np.zeros((N, N), complex); rho[start, start] = 1.0
    Dt, Bt = D0.copy(), B0.copy()
    verlauf, hilf = [messen(rho[:K, :K], Pis)], [0.0]
    for p in range(perioden):
        for n in range(Nt):
            if version == "ideal":
                Dn, Bn, Om = Dt, Bt, Omega
            elif version == "stufen":
                (Dn, Bn), Om = muster_von(n), Omega*laser_pause(n, Nt, n_muster, aus_fenster)
            else:
                Dn, Bn, Om = D0, B0, Omega
            k1 = generator(rho, Dn, Bn, Om);            k2 = generator(rho + dt/2*k1, Dn, Bn, Om)
            k3 = generator(rho + dt/2*k2, Dn, Bn, Om);  k4 = generator(rho + dt*k3, Dn, Bn, Om)
            rho = rho + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            rho = schritt(n, rho)
            rho = schritt(n, rho.conj().T).conj().T
            if version == "ideal":
                Dt = schritt(n, np.vstack([Dt, np.zeros((nd, nd))]))[:K]
                Bt = schritt(n, np.vstack([Bt, np.zeros((nd, nd))]))[:K]
        verlauf.append(messen(rho[:K, :K], Pis))
        hilf.append(np.real(np.trace(rho[K:, K:])))
    return np.array(verlauf), np.array(hilf), np.real(np.trace(rho))


#%%
# =====================================================================================================
# 5. Methode C: Hilfsplatz mit realistischem Laser (ebene Welle über alle 6 Sechseckplätze)
# =====================================================================================================
#   Kürzeste Abstände nur noch entlang b1 periodisch (Umfang M·b1), entlang b2 gibt es kein Bild.

def lindblad_ebene_welle(T, A, laser="ebene_welle", zerfall="selektiv", Omega=0.5, Gamma=0.5,
                         M=6, Nz=8, Nt=300, perioden=30, aus_fenster=0.0, start=None, eps_rest=1.0):
    """Gibt zurück: verlauf (perioden+1, 3), Hilfsplatz-Besetzung, Spur, k pro Muster."""
    K, idx, pos, partner, zeile = zylinder(M, Nz)
    dt = T/Nt
    L1 = M*b1                                           # Umfang des Zylinders
    def min_bild(v):
        """kürzester Verbindungsvektor auf dem Zylinder (nur entlang b1 periodisch)"""
        return min((v + n1*L1 for n1 in (-1, 0, 1)), key=np.linalg.norm)

    Us, UT, Pi_unten, Pi_oben, Pi_rand, info = floquet(T, A, M, Nz, Nt)
    Pis = (Pi_unten, Pi_oben, Pi_rand)
    D0, B0 = dimere(UT, M, Nz)
    nd = D0.shape[1]
    N = K + nd
    schritt = mache_zeitschritt(T, A, M, Nz, Nt, K_gitter=K)
    sechseck = sechsecke(Us, D0, B0, Nt)

    muster, k_liste = [], []
    for r in (0, Nt//3, 2*Nt//3):
        Dr = auf_zwei_plaetze(Us[r] @ D0)
        h0 = nd//2                                      # Referenz-Dimer aus der Mitte (weit weg vom Rand)
        x0, y0 = np.argsort(np.abs(Dr[:, h0]))[-2:]
        d = min_bild(pos[y0] - pos[x0])
        k = d*np.angle(-Dr[x0, h0]/Dr[y0, h0])/np.dot(d, d)
        C = np.zeros((nd, K), complex)
        for h in range(nd):
            paar = np.argsort(np.abs(Dr[:, h]))[-2:]
            x = paar[-1]
            plaetze = sechseck[:, h] if laser == "ebene_welle" else paar
            for j in plaetze:
                fak = 1.0 if j in paar else eps_rest
                C[h, j] = fak*Omega/np.sqrt(2)*np.exp(1j*k @ min_bild(pos[j] - pos[x]))
        muster.append((Dr, C))
        k_liste.append(k)
    def muster_von(n):
        t = (n % Nt)/Nt
        return muster[int(np.floor(t*3 + 0.5)) % 3]

    def generator(rho, Dn, C):
        HLr = np.zeros_like(rho)
        HLr[K:] = C @ rho[:K]
        HLr[:K] = C.conj().T @ rho[K:]
        out = -1j*(HLr - HLr.conj().T)
        pa = np.real(np.diag(rho)[K:])
        out[K:, :] -= 0.5*Gamma*rho[K:, :]
        out[:, K:] -= 0.5*Gamma*rho[:, K:]
        if zerfall == "selektiv":
            out[:K, :K] += Gamma*(Dn*pa) @ Dn.conj().T
        else:
            g = np.zeros(K)
            for h in range(nd):
                g[sechseck[:, h]] += Gamma*pa[h]/6
            out[np.arange(K), np.arange(K)] += g
        return out

    if start is None:
        start = idx(M//2, Nz//2, 0)
    rho = np.zeros((N, N), complex); rho[start, start] = 1.0
    verlauf, hilf = [messen(rho[:K, :K], Pis)], [0.0]
    for p in range(perioden):
        for n in range(Nt):
            Dn, C = muster_von(n)
            C = C*laser_pause(n, Nt, 3, aus_fenster)
            k1 = generator(rho, Dn, C);            k2 = generator(rho + dt/2*k1, Dn, C)
            k3 = generator(rho + dt/2*k2, Dn, C);  k4 = generator(rho + dt*k3, Dn, C)
            rho = rho + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            rho = schritt(n, rho)
            rho = schritt(n, rho.conj().T).conj().T
        verlauf.append(messen(rho[:K, :K], Pis))
        hilf.append(np.real(np.trace(rho[K:, K:])))
    return np.array(verlauf), np.array(hilf), np.real(np.trace(rho)), k_liste


#%%
# =====================================================================================================
# 6. Beispielaufrufe
# =====================================================================================================
T_ = 3*(0.5*np.pi)/J          # θ = JT/3 = π/2
A_wert = 3.0
M_, Nz_ = 6, 8

# --- Kontrolle: Spektrum des Zylinders (Bulk unten / oben / Rand) ---
_, _, _, _, _, info = floquet(T_, A_wert, M_, Nz_, 300)
print(f"Zylinder M = {M_}, Nz = {Nz_}: {info['n_unten']} Bulk unten, {info['n_oben']} Bulk oben, "
      f"{info['n_rand']} Randzustände")
plt.figure(figsize=(5, 3.5))
plt.scatter(info['w_rand'], info['eps']*T_/np.pi, s=10)
plt.axvline(0.6, color='gray', ls=':')
plt.xlabel("Gewicht auf den Randzeilen"); plt.ylabel("ε T / π")
plt.title("Floquet-Zustände: Bulk (links) und Rand (rechts)", fontsize=9)
plt.tight_layout(); plt.show()

# --- Methode A: die drei Versionen, Start im Inneren ---
for version in ["ideal", "stufen", "ruhend"]:
    v, spur = lindblad_dunkel(T_, A_wert, version=version, gamma=0.2, M=M_, Nz=Nz_, perioden=40)
    print(f"A  {version:6s}: p_unten nach 10/20/40 T = {v[10,0]:.3f} {v[20,0]:.3f} {v[40,0]:.3f}"
          f"   p_rand(40 T) = {v[40,2]:.3f}   (Spur {spur:.4f})", flush=True)

# --- Methode A, Start am Rand: der Randkanal hat keinen Sprungoperator ---
K_, idx_, _, _, _ = zylinder(M_, Nz_)
v, _ = lindblad_dunkel(T_, A_wert, version="ideal", gamma=0.2, M=M_, Nz=Nz_, perioden=40, start=idx_(M_//2, 0, 0))
print(f"A  ideal, Start am Rand: p_unten / p_oben / p_rand nach 40 T = {v[40,0]:.3f} / {v[40,1]:.3f} / {v[40,2]:.3f}")

# --- Methode B ---
for Om, Ga, fenster in [(0.2, 2.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.5, 1/12)]:
    v, h, spur = lindblad_hilfsplatz(T_, A_wert, version="stufen", Omega=Om, Gamma=Ga, M=M_, Nz=Nz_,
                                     perioden=30, aus_fenster=fenster)
    v_eff, _ = lindblad_dunkel(T_, A_wert, version="stufen", gamma=4*Om**2/Ga, M=M_, Nz=Nz_, perioden=30)
    print(f"B  Ω={Om}, Γ={Ga}, Laserpause ±{fenster:.3f}T: p_unten(30 T) = {v[30,0]:.3f}  "
          f"(effektiv {v_eff[30,0]:.3f}), Hilfsplatz max {h.max():.3f}", flush=True)

# --- Methode C ---
for laser, eps in [("nur_paar", 0.0), ("ebene_welle", 1.0), ("ebene_welle", 0.25)]:
    v, h, spur, ks = lindblad_ebene_welle(T_, A_wert, laser=laser, Omega=0.2, Gamma=2.0, M=M_, Nz=Nz_,
                                          perioden=40, eps_rest=eps)
    print(f"C  {laser:11s}, Restamplitude {eps}: p_unten(40 T) = {v[40,0]:.3f}, p_rand = {v[40,2]:.3f}", flush=True)
print("   k pro Muster:", [np.round(k, 3) for k in ks])

# --- Plot: Verlauf für die drei Methoden ---
fig, ax = plt.subplots(figsize=(7, 4))
vA, _ = lindblad_dunkel(T_, A_wert, version="stufen", gamma=0.08, M=M_, Nz=Nz_, perioden=40)
vB, _, _ = lindblad_hilfsplatz(T_, A_wert, version="stufen", Omega=0.2, Gamma=2.0, M=M_, Nz=Nz_, perioden=40)
vC, _, _, _ = lindblad_ebene_welle(T_, A_wert, Omega=0.2, Gamma=2.0, M=M_, Nz=Nz_, perioden=40)
ax.plot(vA[:, 0], label="A: effektiver Sprungoperator (γ = 4Ω²/Γ = 0.08)")
ax.plot(vB[:, 0], '--', label="B: Hilfsplatz, Laser nur auf dem Paar")
ax.plot(vC[:, 0], ':', label="C: Hilfsplatz, Laser als ebene Welle")
ax.set_xlabel("Perioden"); ax.set_ylabel("Besetzung unteres Bulk-Band"); ax.grid(alpha=0.3)
ax.set_title(f"Zylinder {M_}×{Nz_}, Stufenversion, A = {A_wert}, θ = π/2, Ω = 0.2, Γ = 2"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()


#%%
# =====================================================================================================
# 7. Analysen zu Methode A
# =====================================================================================================
# (b) Breitere Sprungoperatoren
T_ = 3*(0.5*np.pi)/J
for A_wert in [1.0, 3.0]:
    print(f"\nA = {A_wert:g}, JT/3 = 0.5π, γ = 0.2")
    for version, breit, n in [("ideal", False, 2), ("ideal", True, 2), ("stufen", True, 2),
                              ("stufen", True, 6), ("stufen", True, 12), ("ruhend", True, 2)]:
        v, _ = lindblad_dunkel(T_, A_wert, version=version, breit=breit, n_plaetze=n, M=M_, Nz=Nz_, perioden=40)
        print(f"  {version:6s} breit={breit!s:5s} n={n:2d}:  p_unten nach 5/10/20/40 Perioden = "
              f"{v[5,0]:.3f} {v[10,0]:.3f} {v[20,0]:.3f} {v[40,0]:.3f}   p_rand(40) = {v[40,2]:.3f}", flush=True)

#%%
# (c) Abhängigkeit von der Rate γ
T_ = 3*(0.5*np.pi)/J
A_wert = 3.0
gammas = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for version, breit, stil in [("stufen", False, 'o-'), ("ideal", True, 's--')]:
    end, n95 = [], []
    for g in gammas:
        v, _ = lindblad_dunkel(T_, A_wert, version=version, breit=breit, gamma=g, M=M_, Nz=Nz_, perioden=40)
        end.append(v[40, 0])
        n95.append(next((i for i, x in enumerate(v[:, 0]) if x >= 0.95), np.nan))
        print(f"{version:6s} γ = {g:4.2f}: p_unten nach 40 Perioden = {v[40,0]:.3f},  ≥ 0.95 nach {n95[-1]} Perioden",
              flush=True)
    ax[0].semilogx(gammas, n95, stil, label=version)
    ax[1].semilogx(gammas, end, stil, label=version)
ax[0].set_xlabel("γ (in J)"); ax[0].set_ylabel("Perioden bis p_unten ≥ 0.95"); ax[0].grid(alpha=0.3); ax[0].legend()
ax[1].set_xlabel("γ (in J)"); ax[1].set_ylabel("p_unten nach 40 Perioden"); ax[1].grid(alpha=0.3)
fig.suptitle(f"Sprungoperator-Kühlung auf dem Zylinder {M_}×{Nz_}, A = {A_wert:g}, JT/3 = 0.5π")
plt.tight_layout(); plt.show()
# %%