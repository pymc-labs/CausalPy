# PrePostNEGD vs DiD

Use this card when the user has baseline and post-treatment outcomes for treated and control or comparison groups.

## Deciding Question

Is the data a one-pretest/one-posttest snapshot, or a repeated time panel where pre-treatment trends can be assessed?

## Choose `PrePostNEGD`

- Each unit or group has a baseline outcome and a post outcome.
- There is a nonequivalent comparison group.
- The estimand is a baseline-adjusted post-treatment group contrast.
- The design story is that baseline adjustment captures pre-existing group differences.

## Choose `DifferenceInDifferences`

- The data has repeated observations over time before and after treatment.
- The estimand is a treated-group by post-period contrast.
- The design can assess or defend parallel pre-treatment trends.

## Choose Neither

- Use `StaggeredDifferenceInDifferences` if units adopt at different times.
- Return Not identifiable yet if there is neither a credible baseline adjustment story nor enough repeated pre-period structure for DiD.
