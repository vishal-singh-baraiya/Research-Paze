# Paper 3 — Thermodynamics
## Unit 2: Complete Notes with Derivations

---

# 1. LAWS OF THERMODYNAMICS

---

## Zeroth Law of Thermodynamics

**Statement:** If two systems are each in thermal equilibrium with a third system, then they are in thermal equilibrium with each other.

**Significance:** This law defines **temperature** as a thermodynamic property. It is the basis of thermometry.

---

## First Law of Thermodynamics

**Statement:** Energy can neither be created nor destroyed, only converted from one form to another.

**Mathematical Form:**

If a system absorbs heat **dQ**, and does work **dW**, then the change in internal energy is:

> **dU = dQ − dW**

Or: **dQ = dU + dW**

Since work done by gas: dW = PdV

> **dQ = dU + PdV**

**Key Points:**
- dU is a **state function** (path independent)
- dQ and dW are **path functions** (path dependent)
- For a cyclic process: ΔU = 0, so dQ = dW

---

## Second Law of Thermodynamics

There are two equivalent statements:

**Kelvin-Planck Statement:**
It is impossible to construct a heat engine that operates in a cycle and converts all the heat absorbed from a reservoir into work without any other effect.

> No engine can have 100% efficiency.

**Clausius Statement:**
It is impossible for heat to flow spontaneously from a cold body to a hot body without external work being done.

> A refrigerator cannot work without external energy.

**Equivalence:** Both statements are equivalent — violation of one leads to violation of the other.

---

## Third Law of Thermodynamics

**Statement (Nernst Heat Theorem):**
The entropy of a perfect crystalline substance approaches **zero** as temperature approaches **absolute zero (0 K).**

> **S → 0 as T → 0 K**

**Significance:** It is impossible to reach absolute zero in a finite number of steps.

---
---

# 2. HEAT ENGINE

---

## What is a Heat Engine?

A heat engine is a device that **converts heat energy into mechanical work** by operating in a cyclic process between two temperature reservoirs.

**Components:**
- **Source** (Hot reservoir) at temperature T₁ — supplies heat Q₁
- **Working substance** — performs the cycle
- **Sink** (Cold reservoir) at temperature T₂ — absorbs heat Q₂
- **Work output** W = Q₁ − Q₂

---

## Efficiency of Heat Engine

Efficiency is defined as the ratio of work done to heat absorbed:

> **η = W / Q₁**

Since W = Q₁ − Q₂:

> **η = (Q₁ − Q₂) / Q₁**

> **η = 1 − (Q₂/Q₁)**

**For a perfect engine:** Q₂ = 0 → η = 1 (100%), which is **impossible** by Second Law.

So always: **η < 1**

---
---

# 3. CARNOT ENGINE AND ITS FOUR PROCESSES

---

## What is Carnot Engine?

Carnot engine is a **theoretical ideal heat engine** that operates on the **Carnot cycle** — the most efficient possible cycle between two temperature reservoirs T₁ (hot) and T₂ (cold).

It was proposed by **Sadi Carnot in 1824.**

**Working substance:** Ideal gas
**Condition:** All processes are quasi-static and reversible.

---

## The Four Processes of Carnot Cycle

---

### Process 1 → 2: Isothermal Expansion (at T₁)

- Gas expands at **constant temperature T₁** (hot source)
- Gas absorbs heat **Q₁** from source
- Internal energy unchanged (ΔU = 0 for isothermal, ideal gas)
- By First Law: **Q₁ = W₁₂**

**Work done:**

$$W_{12} = \int_{V_1}^{V_2} P\, dV$$

Using PV = nRT₁:

$$W_{12} = nRT_1 \ln\left(\frac{V_2}{V_1}\right) = Q_1$$

Since V₂ > V₁, W₁₂ > 0 ✓ (work done BY gas)

---

### Process 2 → 3: Adiabatic Expansion

- Gas expands **without exchanging heat** (dQ = 0)
- Temperature falls from **T₁ to T₂**
- By First Law: dU = −dW → internal energy decreases
- Work is done at the **expense of internal energy**

**Work done:**

$$W_{23} = nC_v(T_1 - T_2)$$

**Relation between V and T (Adiabatic condition):**

$$T_1 V_2^{\gamma-1} = T_2 V_3^{\gamma-1}$$

$$\Rightarrow \frac{T_1}{T_2} = \left(\frac{V_3}{V_2}\right)^{\gamma - 1} \quad \cdots (i)$$

---

### Process 3 → 4: Isothermal Compression (at T₂)

- Gas is compressed at **constant temperature T₂** (cold sink)
- Gas **releases heat Q₂** to sink
- ΔU = 0 (isothermal)
- Work is done ON the gas

**Work done (on gas, so negative work by gas):**

$$W_{34} = -nRT_2 \ln\left(\frac{V_3}{V_4}\right) = -Q_2$$

Or heat rejected:

$$Q_2 = nRT_2 \ln\left(\frac{V_3}{V_4}\right)$$

---

### Process 4 → 1: Adiabatic Compression

- Gas is compressed **without heat exchange** (dQ = 0)
- Temperature rises from **T₂ back to T₁**
- Work is done ON the gas

**Adiabatic condition:**

$$T_2 V_4^{\gamma-1} = T_1 V_1^{\gamma-1}$$

$$\Rightarrow \frac{T_1}{T_2} = \left(\frac{V_4}{V_1}\right)^{\gamma-1} \quad \cdots (ii)$$

---

## Important Relation Between Volumes

From equations (i) and (ii):

$$\left(\frac{V_3}{V_2}\right)^{\gamma-1} = \left(\frac{V_4}{V_1}\right)^{\gamma-1}$$

$$\Rightarrow \frac{V_3}{V_2} = \frac{V_4}{V_1}$$

$$\Rightarrow \boxed{\frac{V_2}{V_1} = \frac{V_3}{V_4}}$$

This is a **key result** used in deriving efficiency.

---

## Efficiency of Carnot Engine — Full Derivation

**Total work done in one cycle:**

$$W = W_{12} + W_{23} + W_{34} + W_{41}$$

Since W₂₃ and W₄₁ are adiabatic and cancel each other out in the efficiency calculation:

$$W = Q_1 - Q_2$$

Now:

$$\frac{Q_1}{Q_2} = \frac{nRT_1 \ln(V_2/V_1)}{nRT_2 \ln(V_3/V_4)}$$

Since V₂/V₁ = V₃/V₄ (proved above), the log terms cancel:

$$\frac{Q_1}{Q_2} = \frac{T_1}{T_2}$$

**Efficiency:**

$$\eta = 1 - \frac{Q_2}{Q_1} = 1 - \frac{T_2}{T_1}$$

$$\boxed{\eta_{Carnot} = 1 - \frac{T_2}{T_1}}$$

**Conclusions:**
- η = 1 only if T₂ = 0 K (absolute zero) — impossible
- η increases as T₁ increases or T₂ decreases
- Carnot engine gives **maximum possible efficiency** between two given temperatures
- All **reversible engines** between same temperatures have **same efficiency**
- All **irreversible engines** have efficiency **less than Carnot**

---

## Carnot's Theorem

**Statement:** No heat engine operating between two given temperatures can be more efficient than a reversible (Carnot) engine operating between the same temperatures.

> **η_irreversible < η_Carnot = η_reversible**

---

## Summary Table — Carnot Cycle

| Process | Type | Heat | Work | ΔT |
|---|---|---|---|---|
| 1→2 | Isothermal Expansion | +Q₁ absorbed | +W₁₂ | 0 |
| 2→3 | Adiabatic Expansion | 0 | +W₂₃ | T₁→T₂ |
| 3→4 | Isothermal Compression | −Q₂ released | −W₃₄ | 0 |
| 4→1 | Adiabatic Compression | 0 | −W₄₁ | T₂→T₁ |

---

# Quick Revision — Key Formulas

| Concept | Formula |
|---|---|
| First Law | dQ = dU + PdV |
| Engine Efficiency | η = 1 − Q₂/Q₁ |
| Carnot Efficiency | η = 1 − T₂/T₁ |
| Heat ratio | Q₁/Q₂ = T₁/T₂ |
| Isothermal Work | W = nRT ln(V₂/V₁) |
| Adiabatic relation | TV^(γ-1) = constant |

---

Unit 2 is now complete! Shall I proceed with **Unit 3 (Entropy, Maxwell's Equations, TdS)**?

# Paper 3 — Thermodynamics
## Unit 3: Complete Notes with Derivations

---

# 1. ENTROPY

---

## What is Entropy?

Entropy is a **thermodynamic state function** that measures the **degree of disorder or randomness** in a system.

- Denoted by **S**
- It is a **state function** (depends only on initial and final states)
- SI Unit: **J/K (Joules per Kelvin)**

---

## Mathematical Definition of Entropy

For a reversible process, the change in entropy is defined as:

$$dS = \frac{dQ_{rev}}{T}$$

For a finite process from state 1 to state 2:

$$\Delta S = S_2 - S_1 = \int_1^2 \frac{dQ_{rev}}{T}$$

**Key Points:**
- For **reversible process:** dS = dQ/T
- For **irreversible process:** dS > dQ/T
- Combined (Clausius Inequality): **dS ≥ dQ/T**

---

## Clausius Inequality

For any cyclic process:

$$\oint \frac{dQ}{T} \leq 0$$

- **= 0** for reversible (Carnot) cycle
- **< 0** for irreversible cycle
- **> 0** is impossible (violates Second Law)

---

## Entropy and Second Law

The Second Law can be restated in terms of entropy:

> **For an isolated system, entropy never decreases.**

$$\Delta S_{universe} \geq 0$$

- Reversible process: ΔS = 0
- Irreversible process: ΔS > 0
- Impossible process: ΔS < 0

---

## Entropy Change — Important Cases

---

### Case 1: Ideal Gas (General)

From First Law: dQ = dU + PdV

$$dS = \frac{dQ}{T} = \frac{dU}{T} + \frac{P\,dV}{T}$$

Using dU = nCᵥdT and P/T = nR/V:

$$dS = \frac{nC_v\,dT}{T} + \frac{nR\,dV}{V}$$

Integrating from state 1 to state 2:

$$\boxed{\Delta S = nC_v \ln\frac{T_2}{T_1} + nR\ln\frac{V_2}{V_1}}$$

---

### Case 2: Isothermal Process (T = constant)

dT = 0, so:

$$\Delta S = nR\ln\frac{V_2}{V_1}$$

---

### Case 3: Isochoric Process (V = constant)

dV = 0, so:

$$\Delta S = nC_v \ln\frac{T_2}{T_1}$$

---

### Case 4: Isobaric Process (P = constant)

Using V/T = nR/P = constant → V₂/V₁ = T₂/T₁:

$$\Delta S = nC_v \ln\frac{T_2}{T_1} + nR\ln\frac{T_2}{T_1}$$

$$\boxed{\Delta S = nC_p \ln\frac{T_2}{T_1}}$$

---

### Case 5: Adiabatic Reversible Process

dQ = 0, so:

$$\Delta S = 0$$

> Reversible adiabatic process is also called **Isentropic process** (constant entropy).

---
---

# 2. IRREVERSIBLE PROCESSES IN IDEAL GAS

---

## What is an Irreversible Process?

A process that **cannot be reversed** without leaving a permanent change in the system or surroundings.

**Examples:**
- Free expansion of gas
- Heat conduction across finite temperature difference
- Mixing of two gases
- Friction

---

## Key Property of Irreversible Processes

For irreversible processes:

$$dS > \frac{dQ}{T}$$

$$\Delta S_{universe} > 0$$

Entropy of the universe **always increases** in irreversible processes.

---

## Entropy Change in Free Expansion (Irreversible)

When an ideal gas expands freely into vacuum (Joule's expansion):

- No heat exchange: Q = 0
- No work done: W = 0
- Temperature unchanged: ΔT = 0 (for ideal gas)
- But **entropy increases!**

**Calculation:**

Since T is same and V increases from V₁ to V₂:

$$\Delta S = nR\ln\frac{V_2}{V_1}$$

Since V₂ > V₁:

$$\boxed{\Delta S > 0}$$

This confirms the process is **irreversible** — entropy increased even though Q = 0.

---

## Why Free Expansion is Irreversible

- Gas never spontaneously compresses back
- It represents maximum disorder
- ΔS_universe > 0 confirms irreversibility

---
---

# 3. ENTROPY OF MIXTURE

---

## Gibbs Paradox and Entropy of Mixing

When two **different ideal gases** are mixed at the same temperature T and pressure P:

**Setup:**
- Gas 1: n₁ moles, volume V₁
- Gas 2: n₂ moles, volume V₂
- After mixing: total volume V = V₁ + V₂

---

## Derivation of Entropy of Mixing

For each gas, during mixing it effectively undergoes isothermal expansion into the full volume V:

**For Gas 1:**

$$\Delta S_1 = n_1 R \ln\frac{V}{V_1}$$

**For Gas 2:**

$$\Delta S_2 = n_2 R \ln\frac{V}{V_2}$$

**Total entropy of mixing:**

$$\Delta S_{mix} = \Delta S_1 + \Delta S_2$$

$$\Delta S_{mix} = n_1 R \ln\frac{V}{V_1} + n_2 R \ln\frac{V}{V_2}$$

Since V/V₁ = (n₁+n₂)/n₁ = 1/x₁, where x₁ is mole fraction:

$$\boxed{\Delta S_{mix} = -R(n_1 \ln x_1 + n_2 \ln x_2)}$$

Since x₁ < 1 and x₂ < 1, ln terms are negative:

$$\Delta S_{mix} > 0 \quad \text{always}$$

**Conclusion:** Mixing of different gases always **increases entropy** — it is a naturally irreversible process.

---

## Gibbs Paradox

If two gases being mixed are **identical**, then ΔS_mix should be zero (no change in disorder). But the classical formula gives a non-zero result. This contradiction is the **Gibbs Paradox**, resolved only by quantum statistics (indistinguishability of identical particles).

---
---

# 4. THERMODYNAMIC WORK AND THERMODYNAMIC POTENTIALS (Maxwell's Equations) ⭐ V.IMP

---

## The Four Thermodynamic Potentials

These are energy functions that describe the state of a thermodynamic system:

| Potential | Symbol | Definition |
|---|---|---|
| Internal Energy | U | Fundamental |
| Enthalpy | H | H = U + PV |
| Helmholtz Free Energy | F (or A) | F = U − TS |
| Gibbs Free Energy | G | G = H − TS = U + PV − TS |

---

## Fundamental Relations (Starting Points)

From First and Second Law combined:

$$dU = TdS - PdV \quad \cdots (1)$$

This is the **master equation.**

---

### Enthalpy: H = U + PV

Differentiating:

$$dH = dU + PdV + VdP$$

Substituting dU = TdS − PdV:

$$dH = TdS - PdV + PdV + VdP$$

$$\boxed{dH = TdS + VdP} \quad \cdots (2)$$

---

### Helmholtz Free Energy: F = U − TS

Differentiating:

$$dF = dU - TdS - SdT$$

Substituting dU = TdS − PdV:

$$dF = TdS - PdV - TdS - SdT$$

$$\boxed{dF = -SdT - PdV} \quad \cdots (3)$$

---

### Gibbs Free Energy: G = H − TS

Differentiating:

$$dG = dH - TdS - SdT$$

Substituting dH = TdS + VdP:

$$dG = TdS + VdP - TdS - SdT$$

$$\boxed{dG = -SdT + VdP} \quad \cdots (4)$$

---

## Deriving Maxwell's Relations

Maxwell's relations come from the mathematical property that for any exact differential:

If **dZ = MdX + NdY**, then:

$$\left(\frac{\partial M}{\partial Y}\right)_X = \left(\frac{\partial N}{\partial X}\right)_Y$$

Applying this to each thermodynamic potential:

---

### Maxwell Relation 1 — From dU = TdS − PdV

Here: M = T, X = S, N = −P, Y = V

$$\boxed{\left(\frac{\partial T}{\partial V}\right)_S = -\left(\frac{\partial P}{\partial S}\right)_V}$$

---

### Maxwell Relation 2 — From dH = TdS + VdP

Here: M = T, X = S, N = V, Y = P

$$\boxed{\left(\frac{\partial T}{\partial P}\right)_S = \left(\frac{\partial V}{\partial S}\right)_P}$$

---

### Maxwell Relation 3 — From dF = −SdT − PdV

Here: M = −S, X = T, N = −P, Y = V

$$\boxed{\left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial P}{\partial T}\right)_V}$$

⭐ **This is the most commonly used Maxwell relation**

---

### Maxwell Relation 4 — From dG = −SdT + VdP

Here: M = −S, X = T, N = V, Y = P

$$\boxed{\left(\frac{\partial S}{\partial P}\right)_T = -\left(\frac{\partial V}{\partial T}\right)_P}$$

⭐ **This is also very frequently used**

---

## Summary of All Maxwell Relations

$$\left(\frac{\partial T}{\partial V}\right)_S = -\left(\frac{\partial P}{\partial S}\right)_V \quad \text{[from U]}$$

$$\left(\frac{\partial T}{\partial P}\right)_S = \left(\frac{\partial V}{\partial S}\right)_P \quad \text{[from H]}$$

$$\left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial P}{\partial T}\right)_V \quad \text{[from F]}$$

$$\left(\frac{\partial S}{\partial P}\right)_T = -\left(\frac{\partial V}{\partial T}\right)_P \quad \text{[from G]}$$

**Memory Trick — "USTH GVPF":**
U→S,V | H→S,P | F→T,V | G→T,P

---
---

# 5. TdS EQUATIONS

---

## What are TdS Equations?

TdS equations express **TdS (heat) in terms of measurable quantities** like T, V, P — because entropy S cannot be measured directly.

There are **two TdS equations.**

---

## TdS Equation 1 — In terms of T and V

Consider S as a function of T and V: S = S(T,V)

$$dS = \left(\frac{\partial S}{\partial T}\right)_V dT + \left(\frac{\partial S}{\partial V}\right)_T dV$$

Multiply both sides by T:

$$TdS = T\left(\frac{\partial S}{\partial T}\right)_V dT + T\left(\frac{\partial S}{\partial V}\right)_T dV$$

Now use:
- $T\left(\frac{\partial S}{\partial T}\right)_V = C_v$ (heat capacity at constant volume)
- $\left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial P}{\partial T}\right)_V$ (from Maxwell Relation 3)

$$\boxed{TdS = C_v\,dT + T\left(\frac{\partial P}{\partial T}\right)_V dV}$$

This is the **First TdS Equation.**

---

## TdS Equation 2 — In terms of T and P

Consider S as a function of T and P: S = S(T,P)

$$dS = \left(\frac{\partial S}{\partial T}\right)_P dT + \left(\frac{\partial S}{\partial P}\right)_T dP$$

Multiply both sides by T:

$$TdS = T\left(\frac{\partial S}{\partial T}\right)_P dT + T\left(\frac{\partial S}{\partial P}\right)_T dP$$

Now use:
- $T\left(\frac{\partial S}{\partial T}\right)_P = C_p$ (heat capacity at constant pressure)
- $\left(\frac{\partial S}{\partial P}\right)_T = -\left(\frac{\partial V}{\partial T}\right)_P$ (from Maxwell Relation 4)

$$\boxed{TdS = C_p\,dT - T\left(\frac{\partial V}{\partial T}\right)_P dP}$$

This is the **Second TdS Equation.**

---

## Applications of TdS Equations

### Application 1: Difference between Cₚ and Cᵥ

Equating both TdS equations (for same dS):

$$C_v\,dT + T\left(\frac{\partial P}{\partial T}\right)_V dV = C_p\,dT - T\left(\frac{\partial V}{\partial T}\right)_P dP$$

After mathematical manipulation using cyclic relations:

$$\boxed{C_p - C_v = -T\frac{\left(\frac{\partial P}{\partial V}\right)_T}{\left[\left(\frac{\partial P}{\partial T}\right)_V\right]^2} \cdot \left(\frac{\partial P}{\partial T}\right)_V^2}$$

Simplified using volume expansivity β and isothermal compressibility κ:

$$\boxed{C_p - C_v = \frac{TV\beta^2}{\kappa}}$$

Where:
- $\beta = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P$ → volume expansivity
- $\kappa = -\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_T$ → isothermal compressibility

**For ideal gas:** Cₚ − Cᵥ = nR ✓ (consistent result)

---

### Application 2: Entropy change using TdS Eq. 1

For an ideal gas, $\left(\frac{\partial P}{\partial T}\right)_V = \frac{nR}{V}$

From TdS Eq. 1:

$$TdS = C_v\,dT + \frac{nRT}{V}dV$$

$$dS = \frac{C_v\,dT}{T} + \frac{nR\,dV}{V}$$

Integrating:

$$\Delta S = nC_v\ln\frac{T_2}{T_1} + nR\ln\frac{V_2}{V_1}$$

This matches what we derived earlier ✓

---

## Summary of TdS Equations

$$\text{TdS Eq. 1: } TdS = C_v\,dT + T\left(\frac{\partial P}{\partial T}\right)_V dV$$

$$\text{TdS Eq. 2: } TdS = C_p\,dT - T\left(\frac{\partial V}{\partial T}\right)_P dP$$

---
---

# COMPLETE UNIT 3 — QUICK REVISION TABLE

| Topic | Key Formula |
|---|---|
| Entropy definition | dS = dQ_rev/T |
| Clausius Inequality | ∮dQ/T ≤ 0 |
| Entropy of ideal gas | ΔS = nCᵥln(T₂/T₁) + nRln(V₂/V₁) |
| Free Expansion | ΔS = nRln(V₂/V₁) > 0 |
| Entropy of Mixing | ΔS = −R(n₁lnx₁ + n₂lnx₂) |
| Master Equation | dU = TdS − PdV |
| Enthalpy | dH = TdS + VdP |
| Helmholtz | dF = −SdT − PdV |
| Gibbs | dG = −SdT + VdP |
| Maxwell 3 (most used) | (∂S/∂V)_T = (∂P/∂T)_V |
| Maxwell 4 (most used) | (∂S/∂P)_T = −(∂V/∂T)_P |
| TdS Eq. 1 | TdS = CᵥdT + T(∂P/∂T)_V dV |
| TdS Eq. 2 | TdS = CₚdT − T(∂V/∂T)_P dP |
| Cₚ − Cᵥ | = TVβ²/κ |

---

# 🎯 EXAM STRATEGY

**V. Important derivations to practice:**
1. Carnot efficiency derivation (Unit 2)
2. All 4 Maxwell relations derivation from potentials
3. Both TdS equations derivation
4. Cₚ − Cᵥ derivation
5. Entropy of mixing derivation
6. Entropy change in free expansion

**Both Units are now complete!** Ask me if you need any concept explained in more depth or any numerical problems solved!
