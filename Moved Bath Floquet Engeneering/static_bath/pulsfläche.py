#%%
import numpy as np
import matplotlib.pyplot as plt

J = 1.0; dphi = -2*np.pi/3; br = 0.5; T = 3*br*np.pi/J; om = 2*np.pi/T
ph = np.array([0, dphi, 2*dphi]) - np.pi/3
t = np.linspace(0, T, 2000); farben = ['C0', 'C1', 'C2']
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)

for col, A in enumerate([6.0, 1.0]):
    Js = J*np.exp(A*np.cos(om*t[:, None] + ph[None]))/(3*np.i0(A))
    a = ax[0, col]
    for s in range(3):
        a.plot(t/T, Js[:, s], color=farben[s], label=f"J_{s+1}(t)  (Bindung a{s+1})")
        a.fill_between(t/T, 0, Js[:, s], color=farben[s], alpha=0.18)
    flaeche = np.trapezoid(Js[:, 0], t)
    a.set_title(f"A = {A:g}: Fläche unter jeder Kurve = θ = JT/3 = {flaeche/np.pi:.2f}π")
    a.set_ylabel("Hüpfstärke J_s(t)"); a.legend(fontsize=8, loc='upper right')
    for x in [1/6, 1/2, 5/6]:
        a.axvline(x, color='gray', ls=':', lw=0.8)
    # laufende Fläche von Bindung 1 und Besetzung (nur Bindung 1 an, Start auf Platz 1)
    b = ax[1, col]
    th = np.cumsum(Js[:, 0])*(t[1] - t[0])
    b.plot(t/T, th/np.pi, 'C0', lw=2, label="θ₁(t)/π = ∫ J₁ dt / π  (angesammelte Fläche)")
    b.plot(t/T, np.cos(th)**2, 'k--', label="|c₁|² = cos² θ₁  (noch auf Platz 1)")
    b.plot(t/T, np.sin(th)**2, 'r--', label="|c₂|² = sin² θ₁  (hinübergetunnelt)")
    b.axhline(0.5, color='gray', lw=0.6)
    b.set_xlabel("t / T"); b.set_ylim(-0.05, 1.05); b.legend(fontsize=8, loc='center right')
    if col == 0:
        b.set_ylabel("nur Bindung 1 betrachtet")

fig.suptitle("Pulsfläche θ: Fläche unter J_s(t). Bei θ = π/2 ist der Übertrag vollständig (cos² = 0, sin² = 1)")
fig.tight_layout()
plt.show()
# %%
