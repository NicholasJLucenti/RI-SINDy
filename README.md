# RI-SINDy

RI-SINDy (Regulation-Informed SINDy) is a sparse regression framework for identifying dynamical models of gene regulatory networks. It extends standard SINDy by treating known regulatory interactions (Hill-type candidate terms) as physically constrained rather than as ordinary regression candidates. Here, a regulatory term's coefficient is set by a drift-balance condition: the value that balances it against the already fitted drain/production terms in its equation, instead of being thresholded like every other library term.

This repository contains the RI-SINDy implementation and the three biological systems used to validate it: the Hes1 mRNA-protein oscillator (real experimental data), a synthetic Goodwin oscillator, and a synthetic NF-kB signaling network.

## Repository structure

```
src/                      Core algorithm -- system-agnostic
  risindy.m                 the solver
  drift_balance_generic.m   reusable drift-balance calculation
  drift_balance_template.m  documented template for a new system
  build_poly_library.m, poly_library_row.m, get_delayed.m, smooth_derivative.m

examples/                 self-contained teaching example (no paper data)

paper/
  hes1_app/, goodwin_app/, nfkb_app/   RI-SINDy fit for each system
  comparisons/               SR3, Nullcline-SINDy, and Traditional SINDy
                             baselines run against each system
  noise_sensitivity/         noise-robustness sweep (Traditional SINDy,
                             E-SINDy, RI-SINDy) on the Goodwin system
  supplementary_figures/     extra figures not in the paper (phase-space
                             field decomposition)

Hes1 Data/                Experimental Hes1 mRNA/protein data (Hirata et al.)
```

Every `drift_balance_{system}.m` file follows the same pattern: identify which of the model's other terms the regulatory term needs to stay balanced against, evaluate those terms on the data, and pass them to `drift_balance_generic.m`, which does the actual calculation. Start with `src/drift_balance_template.m` (a documented, fill-in-the-blanks version) or `examples/getting_started_toy_system.m` (a full example) if you're adapting RI-SINDy to a new system.

## Running it

Each script in `paper/` is self contained and can be run directly--it sets its own path, loads or generates its own data, and produces its own figures. 

## Requirements

MATLAB with the Optimization Toolbox (`lsqlin`) and Signal Processing Toolbox (`sgolayfilt`).

## Related methods

Baselines compared against in `paper/comparisons/`: Traditional SINDy (Brunton, Proctor & Kutz 2016), SR3 (Champion et al. 2020), Nullcline-Reconstruction SINDy (Prokop, Frolov & Gelens 2024), and Ensemble-SINDy (Fasel et al. 2022).

## License

MIT -- see `LICENSE`.