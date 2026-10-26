# How certified exclusion works

Exact search does not need every distance; it needs a proof, for every vector that it skips, that the vector cannot be
among the k nearest. certex obtains such proofs from a lower bound that is cheap enough to evaluate for every vector.

## A lower bound from a few directions

Each vector $x$ is lifted to
$\ell(x) = [V(x - c);\ \|(I - V^\top V)(x - c)\|]$, its coordinates along $m$ principal directions plus the length of
what remains. For any query $q$, $\|q - x\|^2 \ge \|\ell(q)\|^2 + \|\ell(x)\|^2 - 2\langle \ell(q), \ell(x) \rangle$,
because the two residuals can at best point the same way. For inner products, certex lifts around the origin, where the
same argument bounds $\|q\|^2 - 2\langle q, x \rangle$ from below.

## A threshold from a few exact distances

The exact distances from the query to a strided sample of the stored
vectors give $U$, the k-th smallest of them. Any k stored vectors are at least as far as the true k-th neighbor, so
that $U$ is an upper bound on the true k-th distance, and every vector whose lower bound exceeds $U$ is excluded.

## A certificate for rounding

The lifted rows are stored in half precision and compared with one tensor-core product
per pair. Four extra coordinates in every row carry bounds on the rounding errors of the stored values and of the
fp32 accumulation, so that the computed product can only overestimate what the exact bound would keep: rounding may
keep an extra vector but never excludes a neighbor.

## Exact verification of the survivors

The few vectors that survive are verified with exact double-precision
distances, in increasing order of their certified lower bounds, until no remaining survivor can enter the top k.

The proofs are given in the CertEx paper ([arXiv:2610.21325](https://arxiv.org/abs/2610.21325)) for certified screening and in the Zarc paper
([doi:10.5281/zenodo.23674583](https://doi.org/10.5281/zenodo.23674583)) for the rest of the pipeline.
