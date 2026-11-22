#%%
"""
=====================================================================================================
  Sprungoperatoren für den anomalen Floquet-Isolator: drei Stufen der Realistik
=====================================================================================================

Ziel: Ein Teilchen im getriebenen Wabengitter soll ins UNTERE Floquet-Band gepumpt werden.
Idee (Diehl/Zoller): gerichtete Dissipation mit Dunkelzustand. Pro Dimer h (zwei gegenüberliegende
Plätze x, y eines Sechsecks) gibt es
    |D_h⟩ = Dimer-Zustand im unteren Band  (soll gefüllt werden, "dunkel")
    |B_h⟩ = Dimer-Zustand im oberen Band   (soll geleert werden, "hell")
und einen Prozess, der nur B -> D bringt.
 
Diese Datei enthält drei Methoden, von idealisiert bis realistisch:

  Methode A  lindblad_dunkel        effektiver Sprungoperator L_h = |D_h⟩⟨B_h| mit Rate γ
                                    (früher: dunkelzustand_zelle.py)
  Methode B  lindblad_hilfsplatz    explizit: Hilfsplatz |h⟩ pro Sechseck, Laser Ω koppelt nur B_h an |h⟩,
                                    spontaner Zerfall Γ von |h⟩ nach D_h (oder unselektiv aufs Sechseck)
                                    (früher: hilfsplatz_zelle.py)
  Methode C  lindblad_ebene_welle   wie B, aber der Laser ist eine ebene Welle Ω₀ e^{ik·r} und erreicht
                                    ALLE 6 Plätze des Sechsecks, nicht nur das Paar x, y
                             .       (früher: ebene_welle_zelle.py)

Zusammenhang: B geht für Γ ≫ Ω in A über, mit γ_eff = 4Ω²/Γ. C ist B mit realistischerem Laser.

Alle drei rechnen ein Teilchen auf einem Torus mit M×M Zellen (K = 2M² Gitterplätze, M² Sechsecke =
M² Dimere) und integrieren die volle Dichtematrix ρ. Gemessen wird stroboskopisch (t = nT)
    p_unten = Tr(Π_unten ρ),   Π_unten = Projektor auf alle Floquet-Zustände mit ε < 0.

Aufbau der Datei:
  1. Parameter und Gitter
  2. Gemeinsame Bausteine (Zeitschritt, Floquet-Operator, Dimere, Hilfsfunktionen)
  3. Methode A
  4. Methode B
  5. Methode C
  6. Beispielaufrufe (mit den Ergebnissen, die wir bekommen haben)
  7. Analysen zu Methode A: (a) Dunkelzustände pro k, (b) breitere Sprungoperatoren, (c) Abhängigkeit von γ
"""
import numpy as np
import matplotlib.pyplot as plt


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

def torus(M):
    """Torus aus M×M Zellen (periodisch, kein Rand).
       Plätze: A in Zelle (i,j) -> idx(i,j,0), B -> idx(i,j,1).  K = 2M².
       partner[s, p] = Partner von Platz p in Bindungsrichtung s (jeder Platz hat genau einen pro Richtung)."""
    K = 2*M*M
    idx = lambda i, j, z: (i % M) + (j % M)*M + z*M*M
    pos = np.zeros((K, 2))
    for i in range(M):
        for j in range(M):
            pos[idx(i, j, 0)] = i*b1 + j*b2
            pos[idx(i, j, 1)] = i*b1 + j*b2 + a1
    bindungen = [(0, 0), (-1, 0), (0, -1)]          # a1, a2, a3: A(i,j) -> B(i+di, j+dj)
    partner = np.zeros((3, K), dtype=int)
    for s, (di, dj) in enumerate(bindungen):
        for i in range(M):
            for j in range(M):
                a, b = idx(i, j, 0), idx(i + di, j + dj, 1)
                partner[s, a] = b
                partner[s, b] = a
    return K, idx, pos, partner


#%%
# =====================================================================================================
# 2. Gemeinsame Bausteine
# =====================================================================================================

def mache_zeitschritt(T, A, M, Nt, K_gitter=None):
    """Gibt die Funktion schritt(n, X) zurück: X(t_n) -> X(t_n + dt) unter H(t).

       H(t) = Σ_s J_s(t) H_s,   J_s(t) = J exp(A cos(ωt + φ_s)) / (3 I₀(A)),   φ_s = (0, Δφ, 2Δφ) − π/3
       Jede Bindungsrichtung s besteht aus getrennten 2×2-Blöcken -> exakt:
           exp(−i c J H_s) ψ = cos(cJ) ψ − i sin(cJ) ψ[partner]
       Strang-Zerlegung pro Zeitschritt: s1/2, s2/2, s3, s2/2, s1/2 (Fehler O(dt³)).

       K_gitter: falls X mehr Zeilen hat als Gitterplätze (z. B. Hilfsplätze in Methode B/C),
                 wird nur auf die ersten K_gitter Zeilen gewirkt, die Hilfsplätze bleiben unverändert."""
    omega = 2*np.pi/T
    norm = 3*np.i0(A)
    phis = np.array([0.0, dphi, 2*dphi]) - np.pi/3
    K, idx, pos, partner = torus(M)
    Kg = K if K_gitter is None else K_gitter
    dt = T/Nt

    def bindung(s, c, X):
        Y = X.copy()
        Y[:Kg] = np.cos(c*J)*X[:Kg] - 1j*np.sin(c*J)*X[:Kg][partner[s]]
        return Y

    def schritt(n, X):
        c = np.exp(A*np.cos(omega*(n + 0.5)*dt + phis))/norm*dt        # Pulsfläche jeder Bindung in diesem dt
        X = bindung(0, c[0]/2, X); X = bindung(1, c[1]/2, X); X = bindung(2, c[2], X)
        X = bindung(1, c[1]/2, X); X = bindung(0, c[0]/2, X)
        return X
    return schritt


def floquet(T, A, M, Nt):
    """U(t_n) für eine Periode, U(T), Projektoren auf unteres/oberes Floquet-Band.
       Quasienergien ε = −arg(λ)/T in (−π/T, π/T];  'unten' = ε < 0."""
    K, idx, pos, partner = torus(M)
    schritt = mache_zeitschritt(T, A, M, Nt)
    U = np.eye(K, dtype=complex); Us = []
    for n in range(Nt):
        Us.append(U)
        U = schritt(n, U)
    lam, Z = np.linalg.eig(U)
    unten = -np.angle(lam) < 0
    Zu = np.linalg.qr(Z[:, unten])[0]                  # orthonormieren (U(T) ist unitär, eig liefert nicht immer orthonormal)
    Zo = np.linalg.qr(Z[:, ~unten])[0]
    return Us, U, Zu @ Zu.conj().T, Zo @ Zo.conj().T


def dimere(UT, M):
    """Dimere aus U(T): für jeden A-Platz a ist b = argmax |U(T)_ba| sein Partner (gegenüber im Sechseck).
       Auf dem 2×2-Block von U(T) diagonalisieren: D = Eigenvektor mit kleinerem ε (unten), B mit größerem (oben).
       Bei großem A und θ = π/2 ist U(T) dort i·σ_x, also D, B = (|a⟩ ± |b⟩)/√2."""
    K = UT.shape[0]
    D, B = np.zeros((K, M*M), complex), np.zeros((K, M*M), complex)
    partnerliste = []
    for h, a in enumerate(range(M*M)):                 # A-Plätze sind 0 … M²−1
        b = np.argmax(np.abs(UT[:, a]))
        blk = UT[np.ix_([a, b], [a, b])]
        lam, vec = np.linalg.eig(blk)
        eps_d = -np.angle(lam)
        iu, io = np.argmin(eps_d), np.argmax(eps_d)
        D[[a, b], h] = vec[:, iu]/np.linalg.norm(vec[:, iu])
        B[[a, b], h] = vec[:, io]/np.linalg.norm(vec[:, io])
        partnerliste.append(b)
    if len(set(partnerliste)) != M*M:
        print("  Warnung: Dimere nicht eindeutig")
    return D, B


def loewdin(V):
    """Spalten von V orthonormieren, so dass sie möglichst wenig verändert werden: V S^{-1/2}, S = V†V."""
    S = V.conj().T @ V
    w, Q = np.linalg.eigh(S)
    return V @ (Q @ np.diag(w**-0.5) @ Q.conj().T)


def breite_dimere(D0, B0, Pi_unten, Pi_oben):
    """'breite' Version: Dimer-Zustände auf das echte untere/obere Band projizieren und orthonormieren.
       Nötig bei kleinem A, wo die Wannier-Zustände Ausläufer über die zwei Plätze hinaus haben."""
    return loewdin(Pi_unten @ D0), loewdin(Pi_oben @ B0)


def auf_zwei_plaetze(V, n=2):
    """Jede Spalte auf die n Plätze mit dem größten Gewicht beschneiden und normieren
       (realistisches Laser-Muster: nur wenige Plätze werden angesprochen)."""
    W = np.zeros_like(V)
    for h in range(V.shape[1]):
        top = np.argsort(np.abs(V[:, h]))[-n:]
        W[top, h] = V[top, h]
    return W/np.linalg.norm(W, axis=0, keepdims=True)


def sechsecke(Us, D0, B0, Nt):
    """Die 6 Plätze, die jeder Dimer-Zustand während einer Periode besucht = sein Sechseck. Form (6, Anzahl Dimere)."""
    gew = sum(np.abs(Us[n] @ D0)**2 + np.abs(Us[n] @ B0)**2 for n in range(0, Nt, 5))
    return np.argsort(gew, axis=0)[-6:, :]


def stufen_muster(Us, D0, B0, Nt, n_muster=3, n_plaetze=2):
    """Für die Version 'stufen': n_muster feste Muster pro Periode, aufgenommen bei t = k·T/n_muster
       (bei 3 Mustern: t = 0, T/3, 2T/3, also zwischen den Pulsen) und auf n_plaetze beschnitten.
       Muster k gilt im Intervall um seinen Aufnahmezeitpunkt; umgeschaltet wird dazwischen
       (bei 3 Mustern genau in den Pulsmitten T/6, T/2, 5T/6)."""
    refs = [int(round(k*Nt/n_muster)) % Nt for k in range(n_muster)]
    muster = [(auf_zwei_plaetze(Us[r] @ D0, n_plaetze), auf_zwei_plaetze(Us[r] @ B0, n_plaetze)) for r in refs]
    def muster_von(n):
        t = (n % Nt)/Nt
        return muster[int(np.floor(t*n_muster + 0.5)) % n_muster]
    return muster_von


def laser_pause(n, Nt, n_muster, aus_fenster):
    """1, wenn der Laser an ist, 0 in einem Fenster ±aus_fenster·T um jeden Umschaltzeitpunkt.
       (Bei 3 Mustern liegen die Umschaltzeitpunkte in den Pulsmitten: Laser nur zwischen den Pulsen an.)"""
    if aus_fenster <= 0:
        return 1.0
    t = ((n + 0.5) % Nt)/Nt*n_muster                   # in Einheiten des Musterabstands
    abstand = abs((t % 1.0) - 0.5)/n_muster            # Abstand (in T) zur nächsten Umschaltung
    return 0.0 if abstand < aus_fenster else 1.0


#%%
# =====================================================================================================
# 3. Methode A: effektiver Sprungoperator  L_h = |D_h⟩⟨B_h|
# =====================================================================================================
#   dρ/dt = −i[H(t), ρ] + γ Σ_h ( L_h ρ L_h† − ½{L_h† L_h, ρ} )
#   Für ein Teilchen:  L_h ρ L_h† = ⟨B_h|ρ|B_h⟩ |D_h⟩⟨D_h|,   L_h† L_h = |B_h⟩⟨B_h|
#   D ist dunkel: L_h|D_h⟩ = 0. Was in D ist, bleibt dort; alles andere wird gepumpt -> unteres Band voll.
#
#   Versionen, wie L_h der Mikrobewegung folgt:
#     "ideal":  L_h(t) = U(t)|D_h⟩⟨B_h|U†(t)   (exakt mitgeführt; = P L P† bis auf eine Phase)
#     "stufen": 3 feste Muster pro Periode, in den Pulsmitten umgeschaltet (realistischer)
#     "ruhend": L_h fest, folgt gar nicht (Kontrolle)
#   Integration pro Zeitschritt dt: erst Dissipator (Euler), dann unitärer Schritt ρ -> V ρ V†.

def lindblad_dunkel(T, A, version="ideal", gamma=0.2, M=6, Nt=300, perioden=40, start=None,
                    breit=False, n_plaetze=2):
    K, idx, pos, partner = torus(M)
    dt = T/Nt
    schritt = mache_zeitschritt(T, A, M, Nt)
    Us, UT, Pi_unten, Pi_oben = floquet(T, A, M, Nt)
    D0, B0 = dimere(UT, M)
    if breit:
        D0, B0 = breite_dimere(D0, B0, Pi_unten, Pi_oben)
    if version == "stufen":
        muster_von = stufen_muster(Us, D0, B0, Nt, 3, n_plaetze)

    if start is None:
        start = idx(M//2, M//2, 0)                     # ein Teilchen auf einem A-Platz in der Mitte
    rho = np.zeros((K, K), complex); rho[start, start] = 1.0
    Dt, Bt = D0.copy(), B0.copy()                      # für "ideal": werden mit U(t) mitgeführt

    verlauf = [np.real(np.trace(Pi_unten @ rho))]
    for p in range(perioden):
        for n in range(Nt):
            if version == "ideal":
                Dn, Bn = Dt, Bt
            elif version == "stufen":
                Dn, Bn = muster_von(n)
            else:
                Dn, Bn = D0, B0
            # Dissipator:  Σ_h ⟨B_h|ρ|B_h⟩ |D_h⟩⟨D_h|  −  ½{Σ_h |B_h⟩⟨B_h|, ρ}
            BrB = np.real(np.einsum('kh,kl,lh->h', Bn.conj(), rho, Bn))
            PB = Bn @ Bn.conj().T
            rho = rho + gamma*dt*((Dn*BrB) @ Dn.conj().T - 0.5*(PB @ rho + rho @ PB))
            # unitärer Schritt
            rho = schritt(n, rho)
            rho = schritt(n, rho.conj().T).conj().T
            if version == "ideal":
                Dt, Bt = schritt(n, Dt), schritt(n, Bt)
        verlauf.append(np.real(np.trace(Pi_unten @ rho)))
    return np.array(verlauf), np.real(np.trace(rho))


#%%
# =====================================================================================================
# 4. Methode B: expliziter Hilfsplatz + Laser Ω + spontaner Zerfall Γ
# =====================================================================================================
#   Zustandsraum: K Gitterplätze + nd Hilfsplätze (ein Hilfsplatz |h⟩ pro Sechseck = pro Dimer).
#   Die Hilfsplätze nehmen nicht am Hüpfen im Gitter teil.
#
#   Laser:   H_L(t) = Ω Σ_h ( |h⟩⟨B_h(t)| + |B_h(t)⟩⟨h| )
#            koppelt NUR den hellen Zustand an |h⟩. D_h ist dunkel, weil das Laser-Muster ∝ B_h*
#            die beiden Wege über x und y für D genau auslöscht (Interferenz).
#   Zerfall: "selektiv":     L_h = √Γ |D_h⟩⟨h|
#            "inkohaerent6": L_{h,j} = √(Γ/6) |j⟩⟨h| für die 6 Sechseckplätze j (unselektiv)
#
#   Grenzfall Γ ≫ Ω: |h⟩ kaum besetzt (~(Ω/Γ)²), adiabatisch eliminierbar -> Methode A mit γ_eff = 4Ω²/Γ.
#
#   Integration pro dt: Laser + Zerfall mit Runge-Kutta 4, dann unitärer Gitterschritt.
#   Optionen für "stufen": n_muster, n_plaetze, aus_fenster (Laser um die Umschaltung aus).

def lindblad_hilfsplatz(T, A, version="ideal", zerfall="selektiv", Omega=0.2, Gamma=2.0,
                        M=6, Nt=300, perioden=30, start=None, n_muster=3, aus_fenster=0.0, n_plaetze=2):
    K, idx, pos, partner = torus(M)
    dt = T/Nt
    Us, UT, Pi_unten, Pi_oben = floquet(T, A, M, Nt)
    D0, B0 = dimere(UT, M)
    nd = D0.shape[1]
    N = K + nd                                          # Gitter + Hilfsplätze
    schritt = mache_zeitschritt(T, A, M, Nt, K_gitter=K)   # wirkt nur auf die Gitterzeilen
    sechseck = sechsecke(Us, D0, B0, Nt)
    if version == "stufen":
        muster_von = stufen_muster(Us, D0, B0, Nt, n_muster, n_plaetze)

    def generator(rho, Dn, Bn, Om):
        """dρ/dt aus Laser und Zerfall:  −i[H_L, ρ] + Σ_c D[L_c]ρ."""
        HLr = np.zeros_like(rho)                       # H_L ρ, ohne die volle Matrix H_L aufzubauen:
        HLr[K:] = Om*(Bn.conj().T @ rho[:K])           #   Hilfszeilen:  Ω ⟨B_h| ρ
        HLr[:K] = Om*(Bn @ rho[K:])                    #   Gitterzeilen: Ω |B_h⟩ ⟨h|ρ
        out = -1j*(HLr - HLr.conj().T)                 # −i[H_L, ρ] = −i(H_L ρ − (H_L ρ)†)
        pa = np.real(np.diag(rho)[K:])                 # Besetzung der Hilfsplätze
        out[K:, :] -= 0.5*Gamma*rho[K:, :]             # −½{Γ P_Hilf, ρ}: Verlust aus den Hilfsplätzen
        out[:, K:] -= 0.5*Gamma*rho[:, K:]
        if zerfall == "selektiv":                      # Gewinn: Γ ρ_hh |D_h⟩⟨D_h|
            out[:K, :K] += Gamma*(Dn*pa) @ Dn.conj().T
        else:                                          # Gewinn: Γ ρ_hh/6 auf jeden Sechseckplatz
            g = np.zeros(K)
            for h in range(nd):
                g[sechseck[:, h]] += Gamma*pa[h]/6
            out[np.arange(K), np.arange(K)] += g
        return out

    if start is None:
        start = idx(M//2, M//2, 0)
    rho = np.zeros((N, N), complex); rho[start, start] = 1.0
    Dt, Bt = D0.copy(), B0.copy()
    verlauf, hilf = [np.real(np.trace(Pi_unten @ rho[:K, :K]))], [0.0]
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
            rho = schritt(n, rho)                       # unitärer Gitterschritt (Hilfsplätze unverändert)
            rho = schritt(n, rho.conj().T).conj().T
            if version == "ideal":
                Dt = schritt(n, np.vstack([Dt, np.zeros((nd, nd))]))[:K]
                Bt = schritt(n, np.vstack([Bt, np.zeros((nd, nd))]))[:K]
        verlauf.append(np.real(np.trace(Pi_unten @ rho[:K, :K])))
        hilf.append(np.real(np.trace(rho[K:, K:])))
    return np.array(verlauf), np.array(hilf), np.real(np.trace(rho))


#%%
# =====================================================================================================
# 5. Methode C: Hilfsplatz mit realistischem Laser (ebene Welle über alle 6 Sechseckplätze)
# =====================================================================================================
#   Laser = ebene (Raman-)Welle mit effektivem Wellenvektor k = k₁ − k₂:
#       H_L = Σ_h Σ_{j ∈ Sechseck h} Ω_j |h⟩⟨j| + h.c.,     Ω_j = Ω₀ e^{i k·r_j}
#   Wahl von k (pro Stufen-Muster): D_h soll dunkel sein  ->  Ω_x D_x + Ω_y D_y = 0
#       ->  e^{i k·(r_y − r_x)} = −D_x/D_y,  k entlang der Verbindung x–y, kleinstes |k|
#       (Ergebnis bei uns: |k| = π/2 pro Gitterabstand, Richtung dreht sich pro Muster um 60°;
#        ein einziges k passt für alle Dimere -> ein Strahlenpaar pro Muster genügt).
#   Die anderen 4 Plätze bekommen, was die Welle dort hat. Sie gehören zu Nachbar-Dimeren, von denen
#   jeweils nur EIN Platz im Sechseck liegt -> deren Dunkelzustände werden mit angeregt.
#   eps_rest: Amplitude auf diesen 4 Plätzen relativ zu x, y (1 = ebene Welle, 0 = nur das Paar).
#   Ω₀ = Ω/√2, damit die Kopplung an B gleich stark ist wie in Methode B.

def lindblad_ebene_welle(T, A, laser="ebene_welle", zerfall="selektiv", Omega=0.5, Gamma=0.5,
                         M=6, Nt=300, perioden=30, aus_fenster=0.0, start=None, eps_rest=1.0):
    K, idx, pos, partner = torus(M)
    dt = T/Nt
    L1, L2 = M*b1, M*b2                                 # Periodenvektoren des Torus
    def min_bild(v):
        """kürzester Verbindungsvektor auf dem Torus"""
        kand = [v + n1*L1 + n2*L2 for n1 in (-1, 0, 1) for n2 in (-1, 0, 1)]
        return min(kand, key=np.linalg.norm)

    Us, UT, Pi_unten, Pi_oben = floquet(T, A, M, Nt)
    D0, B0 = dimere(UT, M)
    nd = D0.shape[1]
    N = K + nd
    schritt = mache_zeitschritt(T, A, M, Nt, K_gitter=K)
    sechseck = sechsecke(Us, D0, B0, Nt)

    # ---- 3 Stufen-Muster: Dunkelzustand Dr und Laserkopplung C[h, j] = Ω_j ----
    muster, k_liste = [], []
    for r in (0, Nt//3, 2*Nt//3):
        Dr = auf_zwei_plaetze(Us[r] @ D0)
        # k aus Dimer 0 bestimmen (gilt für alle Dimere, siehe Kommentar oben)
        x0, y0 = np.argsort(np.abs(Dr[:, 0]))[-2:]
        d = min_bild(pos[y0] - pos[x0])
        k = d*np.angle(-Dr[x0, 0]/Dr[y0, 0])/np.dot(d, d)
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
        HLr[K:] = C @ rho[:K]                           # ⟨h|H_L = Σ_j Ω_j ⟨j|
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
        start = idx(M//2, M//2, 0)
    rho = np.zeros((N, N), complex); rho[start, start] = 1.0
    verlauf, hilf = [np.real(np.trace(Pi_unten @ rho[:K, :K]))], [0.0]
    for p in range(perioden):
        for n in range(Nt):
            Dn, C = muster_von(n)
            C = C*laser_pause(n, Nt, 3, aus_fenster)
            k1 = generator(rho, Dn, C);            k2 = generator(rho + dt/2*k1, Dn, C)
            k3 = generator(rho + dt/2*k2, Dn, C);  k4 = generator(rho + dt*k3, Dn, C)
            rho = rho + dt/6*(k1 + 2*k2 + 2*k3 + k4)
            rho = schritt(n, rho)
            rho = schritt(n, rho.conj().T).conj().T
        verlauf.append(np.real(np.trace(Pi_unten @ rho[:K, :K])))
        hilf.append(np.real(np.trace(rho[K:, K:])))
    return np.array(verlauf), np.array(hilf), np.real(np.trace(rho)), k_liste


#%%
# =====================================================================================================
# 6. Beispielaufrufe
# =====================================================================================================
#   Ergebnisse aus unseren Rechnungen (A = 3, θ = JT/3 = π/2, M = 6), p_unten am Ende:
#     Methode A, γ = 0.2, 40 T:      ideal 0.992, stufen 0.991, ruhend 0.95 (40 T) bzw. 0.9999 (200 T)
#                                    (A = 1 breit: 1.000; A = 6: ideal 1.000, stufen 0.994)
#     Methode B, 30 T:               Ω = 0.2, Γ = 2: ideal 0.992, stufen 0.997 (wie A mit γ_eff = 4Ω²/Γ)
#                                    Ω = 0.5–1, Γ = 0.5, stufen: 0.95–0.97; mit Laserpause ±T/12 … ±T/8: 0.997
#     Methode C, 40 T:               ebene Welle über alle 6 Plätze: 0.81–0.90
#                                    Restamplitude 0.5 / 0.25 / 0.1 / 0: 0.93 / 0.98 / 0.99 / 0.994

T_ = 3*(0.5*np.pi)/J          # θ = JT/3 = π/2
A_wert = 3.0

# --- Methode A: die drei Versionen ---
for version in ["ideal", "stufen", "ruhend"]:
    v, spur = lindblad_dunkel(T_, A_wert, version=version, gamma=0.2, perioden=40)
    print(f"A  {version:6s}: p_unten nach 10/20/40 T = {v[10]:.3f} {v[20]:.3f} {v[40]:.3f}   (Spur {spur:.4f})", flush=True)

# --- Methode B: Hilfsplatz, Vergleich mit A bei γ_eff = 4Ω²/Γ ---
for Om, Ga, fenster in [(0.2, 2.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.5, 1/12)]:
    v, h, spur = lindblad_hilfsplatz(T_, A_wert, version="stufen", Omega=Om, Gamma=Ga, perioden=30, aus_fenster=fenster)
    v_eff, _ = lindblad_dunkel(T_, A_wert, version="stufen", gamma=4*Om**2/Ga, perioden=30)
    print(f"B  Ω={Om}, Γ={Ga}, Laserpause ±{fenster:.3f}T: p_unten(30 T) = {v[30]:.3f}  "
          f"(effektiv {v_eff[30]:.3f}), Hilfsplatz max {h.max():.3f}", flush=True)

# --- Methode C: ebene Welle und Restamplitude ---
for laser, eps in [("nur_paar", 0.0), ("ebene_welle", 1.0), ("ebene_welle", 0.25)]:
    v, h, spur, ks = lindblad_ebene_welle(T_, A_wert, laser=laser, Omega=0.2, Gamma=2.0, perioden=40, eps_rest=eps)
    print(f"C  {laser:11s}, Restamplitude {eps}: p_unten(40 T) = {v[40]:.3f}", flush=True)
print("   k pro Muster:", [np.round(k, 3) for k in ks])

# --- Plot: Verlauf für die drei Methoden ---
fig, ax = plt.subplots(figsize=(7, 4))
vA, _ = lindblad_dunkel(T_, A_wert, version="stufen", gamma=0.08, perioden=40)
vB, _, _ = lindblad_hilfsplatz(T_, A_wert, version="stufen", Omega=0.2, Gamma=2.0, perioden=40)
vC, _, _, _ = lindblad_ebene_welle(T_, A_wert, Omega=0.2, Gamma=2.0, perioden=40)
ax.plot(vA, label="A: effektiver Sprungoperator (γ = 4Ω²/Γ = 0.08)")
ax.plot(vB, '--', label="B: Hilfsplatz, Laser nur auf dem Paar")
ax.plot(vC, ':', label="C: Hilfsplatz, Laser als ebene Welle")
ax.set_xlabel("Perioden"); ax.set_ylabel("Besetzung unteres Band"); ax.grid(alpha=0.3)
ax.set_title(f"Stufenversion, A = {A_wert}, θ = π/2, Ω = 0.2, Γ = 2"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()


#%%
# =====================================================================================================
# 7. Analysen zu Methode A
# =====================================================================================================
# Bausteine für (a): Floquet-Zustände im k-Raum (Bulk, entspricht dem Torus) und die Dimer-Vektoren eines Dimers.

def schritt2(h, dt):
    """exp(−i H dt) für H = [[0, h], [h*, 0]], für alle k gleichzeitig (H² = |h|²·1)."""
    r = np.abs(h)
    s = np.sin(r*dt)/np.where(r > 1e-14, r, 1)
    S = np.zeros(h.shape + (2, 2), complex)
    S[..., 0, 0] = S[..., 1, 1] = np.cos(r*dt)
    S[..., 0, 1] = -1j*s*h
    S[..., 1, 0] = -1j*s*np.conj(h)
    return S


def zustaende_zeit(T, A, Nk, Nt=300, statisch=False):
    """Periodische Anteile u[n, i, j, Platz, Band] zur Zeit t_n und Quasienergien eps[i, j, Band] (Band 0 = unten)."""
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


def dimer_vektoren(T, A, M=6, Nt=300):
    """D und B eines Dimers (2 Plätze) in Positions-Konvention: Amplituden, Lagen r_j, Untergitter s_j."""
    K, idx, pos, partner = torus(M)
    _, UT, _, _ = floquet(T, A, M, Nt)
    D0, B0 = dimere(UT, M)
    a = 0                                                        # A-Platz in Zelle (0,0)
    b = [j for j in np.nonzero(np.abs(D0[:, 0]) > 1e-9)[0] if j != a][0]
    d = pos[b] - pos[a]
    kand = [d + u*M*b1 + v*M*b2 for u in (-1, 0, 1) for v in (-1, 0, 1)]
    d = min(kand, key=np.linalg.norm)                            # kürzester Abstand auf dem Torus
    return D0[[a, b], 0], B0[[a, b], 0], np.array([np.zeros(2), d]), [0, int(b >= M*M)]


#%%
# =====================================================================================================
# (a) Warum kühlt die ruhende Version? Dunkelzustände pro k
# =====================================================================================================
# Idee: Der Sprungoperator L_h = |D_h⟩⟨B_h| regt einen Zustand φ an, wenn ⟨B_h|φ⟩ ≠ 0.
#   ideal (mitlaufend):  φ wird zu jeder Zeit mit dem Muster mitgeführt  -> |⟨B|φ_unten⟩|² = 0 für ALLE k
#   ruhend:              Muster fest, φ_unten(k,t) bewegt sich           -> angeregt, außer bei wenigen k
#   Maß pro k:  Anregung(k) = ⟨ |⟨B_0|φ_unten(k,t)⟩|² ⟩_t     (0 = Dunkelzustand)

T_ = 3*(0.5*np.pi)/J
Nk = 12
for A_wert in [6.0, 3.0]:
    Dv, Bv, rr, ss = dimer_vektoren(T_, A_wert)
    ts, u, eps = zustaende_zeit(T_, A_wert, Nk, 300)              # u[t, i, j, Platz, Band]
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
    if A_wert == 3.0:                                             # anr_u ist hier die ruhende Version
        plt.figure(figsize=(4.2, 3.6))
        plt.imshow(anr_u.T, origin='lower', cmap='viridis')
        plt.colorbar(label="⟨|⟨B|φ_unten(k,t)⟩|²⟩_t")
        plt.xlabel("k₁-Index"); plt.ylabel("k₂-Index")
        plt.title(f"ruhend, A = {A_wert:g}: Anregung pro k\n(dunkel = wird nicht angeregt)", fontsize=9)
        plt.tight_layout(); plt.show()

# Langzeitlauf der ruhenden Version: kühlt trotzdem, aber langsam (Teilchen muss einen der wenigen dunklen k finden)
v, _ = lindblad_dunkel(T_, 3.0, version="ruhend", gamma=0.2, perioden=200)
print("ruhend, A = 3: p_unten nach 40/80/120/200 Perioden =", np.round(v[[40, 80, 120, 200]], 4))

#%%
# =====================================================================================================
# (b) Breitere Sprungoperatoren: Dunkelzustände auf die echten Floquet-Bänder projiziert
# =====================================================================================================
#   breit=False: Zwei-Platz-Dimere (exakt nur für A -> ∞)
#   breit=True:  Π_unten·D und Π_oben·B, orthonormiert  -> echte Wannier-Zustände (auch für kleines A)
#   n_plaetze:   bei "stufen" werden die Muster auf so viele Plätze beschränkt (realistische Laser)

T_ = 3*(0.5*np.pi)/J
for A_wert in [1.0, 3.0]:
    print(f"\nA = {A_wert:g}, JT/3 = 0.5π, γ = 0.2")
    for version, breit, n in [("ideal", False, 2), ("ideal", True, 2), ("stufen", True, 2),
                              ("stufen", True, 6), ("stufen", True, 12), ("ruhend", True, 2)]:
        v, _ = lindblad_dunkel(T_, A_wert, version=version, breit=breit, n_plaetze=n, perioden=40)
        print(f"  {version:6s} breit={breit!s:5s} n={n:2d}:  p_unten nach 5/10/20/40 Perioden = "
              f"{v[5]:.3f} {v[10]:.3f} {v[20]:.3f} {v[40]:.3f}", flush=True)


#%%
# =====================================================================================================
# (c) Abhängigkeit von der Rate γ
# =====================================================================================================
T_ = 3*(0.5*np.pi)/J
A_wert = 3.0
gammas = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for version, breit, stil in [("stufen", False, 'o-'), ("ideal", True, 's--')]:
    end, n95 = [], []
    for g in gammas:
        v, _ = lindblad_dunkel(T_, A_wert, version=version, breit=breit, gamma=g, perioden=40)
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