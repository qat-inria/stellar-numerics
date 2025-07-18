# stellar-numerics

## Objective

Solve for fidelity/stellar profiles

$$\sup_{\rho \in \mathcal S_k} F(\rho, \ket{\psi}) = \sup_{\Phi \in \mathcal S_k^{\text{pure}}} F(\Phi, \psi) =
\sup_{\hat{G}} {\rm Tr}[\underbrace{\hat{\Pi}_k}_{=\sum_{n=0}^k |n\times n|} \hat G^\dagger\ket\psi\bra\psi G]
$$

For other witnesses replace $\ket\psi \bra\psi \to {\hat W} := \sum w_{kl}\ket k \bra l$ with ${\hat W}$ Hermitian. An example: ${\hat W} = a \ket \psi \bra \psi + b \ket\varphi \bra \varphi$ with $\langle \varphi \vert \psi \rangle = 0$ and $a,b \in \mathbb{R}$ (not necessarily positive!!). Or truncated parity operator $(-1)^{\hat n} =  |0><0|-|1><1|+|2><2|-...$

### Conventions


- Gaussian operators (displacement, squeezing)
  - single mode convention [1, 4, draft] ${\hat G}(\gamma, \xi) \coloneqq {\hat D}(\gamma){\hat S}(\xi)$ (oppo to [CMG20, C+20])
  - ${\hat D}(\gamma) \coloneqq e^{\gamma {\hat a} - \gamma^* {\hat a}^\dagger}$
  - ${\hat S}(\xi) \coloneqq e^{\frac12 (\xi^*{\hat a}^2 - \xi ({\hat a}^\dagger)^2)}$ (related by conjugation to UCDraft and - sign to [MJD96])
- Gaussian states
  - $\vert \gamma , \xi \rangle = {\hat G}(\gamma, \xi) \, \vert 0 \rangle$
- multimode?
  - Fock basis
  - Beam-splitters  

## Code Structure

## How to/Tutorial

## To do

- [ ] auto doc generation
- [ ] add gaussian state computation
- [ ] generate (multi) cat profiles
- [ ] try other states
- [ ] Go to witnesses

## Contributions

- Project idea: Ulysse Chabaud
- Qode: Maxime Garnier & Thierry Martinez
- QATnowledgements: Zacharie van Herstraeten

## References

- [1, FQ21] [F. Miatto & N. Quesada, Fast Optimization of Parametrized Quantum Optical Circuits. *Quantum* **4**, 366 (2020)](https://doi.org/10.22331/q-2020-11-30-366). Or [preprint](https://arxiv.org/abs/2004.11002).
- [2, CMG20] [U. Chabaud *et al.*, Certification of Non-Gaussian States with Operational Measurements, *PRX Quantum* **2**, 020333 (2021)](https://doi.org/10.1103/PRXQuantum.2.020333). Or [preprint](https://arxiv.org/abs/2011.04320).
- [3, C+20] [U. Chabaud *et al.*, Stellar Representation of Non-Gaussian Quantum States, *Physical Review Letters* **124**, 063605 (2020)](https://doi.org/10.1103/PhysRevLett.124.063605).
‌or [preprint](https://arxiv.org/abs/1907.11009).
- [4, MJD96] [K. B. Møller *et al.*, Displaced Squeezed Number States: Position Space Representation, Inner Product, and Some Applications, *Physical Review A* **54**, 5378 (1996)](https://doi.org/10.1103/PhysRevA.54.5378).
- [5, TAFZ20] [D. Martínez-Tibaduiza *et al.*, New BCH-like relations of the su(1,1), su(2) and so(2,1) Lie algebras, *Physics Letters A* **384**, 36, 126937 (2020)](https://doi.org/10.1016/j.physleta.2020.126937)
‌
