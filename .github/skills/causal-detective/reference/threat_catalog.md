# Threat Catalog

A comprehensive catalog of threats to causal claims, organized by type. For each threat: what it is, how to detect it, and what to do about it.

## Confounders (Omitted Variable Bias)

A third variable W that affects BOTH the cause (X) and the outcome (Y), creating a spurious association.

**Detection questions:**
- "Is there something else explaining or affecting this relationship?"
- Does W satisfy both conditions: (1) affects X, and (2) affects Y?
- Draw the causal graph — are there backdoor paths from X to Y?

**Examples:**
- Ice cream and shark attacks (weather is the confounder)
- Class size and academic performance (socio-economic factors)
- Cold showers and health (health-conscious lifestyle)

**Important distinctions:**
- **Mediators** (X -> M -> Y): Not confounders — they're part of the causal pathway. Don't control for them.
- **Colliders** (X -> C <- Y): Not confounders — controlling for them CREATES bias. Never include in the model.

**CausalPy response:** Include confounders in the formula. Use `OutcomeFalsification` to test if the effect disappears on unrelated outcomes.

## Reverse Causation

The arrow between X and Y points the wrong way. Y causes X, not X causes Y.

**Detection questions:**
- "Could it be the reverse?"
- Does the timeline make sense? (cause must precede effect)
- Are there feedback loops? (simultaneity: X -> Y -> X)

**Examples:**
- Police officers and crime (more crime -> more police, not the reverse)
- Swimming daily at 95 and longevity (good genes -> can swim, not swimming -> longevity)
- Weapons and wars (both directions simultaneously)

**CausalPy response:** Use methods with clear temporal separation (ITS, DiD). `PlaceboInTime` tests whether effects appear before treatment (which would suggest reverse causation or confounding).

## Measurement Error

What we measure doesn't perfectly capture the true variable.

**Classical measurement error:** Random noise — adds variance but doesn't bias results (in the outcome). In the cause of interest, creates attenuation bias (effect appears smaller than reality).

**Non-classical measurement error:** Systematic — the measurement error is correlated with the treatment or outcome. This biases results.

**Detection questions:**
- Could the measurement be systematically related to the treatment?
- Is the quality/frequency of measurement changing over time?
- Are we measuring what we think we're measuring?

**Examples:**
- COVID cases and vaccination rates (testing rates correlated with vaccination rates)
- Mental health reports increasing (stigma decreasing, not illness increasing)

**CausalPy response:** `OutcomeFalsification` with an alternative measurement. `PriorSensitivity` to check robustness.

## External Validity

The causal effect is real in this context, but doesn't generalize.

**Detection questions:**
- "Can we extrapolate?"
- Would the effect hold in a different population, time period, or scale?
- Was the study done in a very specific context?
- Is the relationship linear, or could there be diminishing returns?

**Dimensions of extrapolation risk:**
- **Population**: Results from men may not apply to women (drug side effects example)
- **Time**: A marketing campaign that worked in 2020 may not work in 2025
- **Geography/culture**: Results from one region may not transfer
- **Scale**: Effect of reducing class from 25 to 20 may not predict effect of reducing from 10 to 5
- **Treatment intensity**: Diminishing returns

**CausalPy response:** `PlaceboInSpace` across different units. Subgroup analysis. `PersistenceCheck` for temporal stability.

## Selection Bias

Systematic differences between treated and untreated groups that exist before treatment.

**Detection questions:**
- Are the groups comparable at baseline?
- Is treatment assignment related to the outcome?
- Could there be survivorship bias?

**CausalPy response:** `PreTreatmentPlaceboCheck` to verify no pre-existing differences. `ConvexHullCheck` for synthetic control.
