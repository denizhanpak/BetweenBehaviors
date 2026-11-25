# Between Behaviors: code for the figures

Three [Pluto](https://plutojl.org) notebooks that make the paper's figures. Each notebook
embeds its own package environment, so opening it in Pluto installs exactly the package
versions it was run with.

| Notebook | Figures (`figures/`) |
|---|---|
| `notebooks/FittedModel.jl` | `exp_data` (a worm's forward and reverse command-neuron activity and its velocity), `fitted_model` (time series of the fitted CTRNN and heteroclinic network), `fitted_dt` (both models at five Euler–Maruyama step sizes), `fitted_dwell_times` (dwell times of Roberts et al.'s worms and of the two models) |
| `notebooks/SimpleModelGLV.jl` | the 3-node GLV: noisy and deterministic time series of the excitable and heteroclinic settings, basins of the excitable setting, the heteroclinic phase portrait, the bifurcation chart, the metastability heatmap with the bifurcation curves, and the saddle-value figure |
| `notebooks/SimpleModelCTRNN.jl` | the 3-node CTRNN: noisy and deterministic time series of the excitable and limit-cycle settings, basins of the excitable setting, the bifurcation chart, the metastability heatmap with the bifurcation curves, the limit cycle with its ghosts, and its speed and distance from the ghosts |

## Running

With Julia 1.13 (the version used):

```julia
import Pkg; Pkg.add("Pluto")
import Pluto; Pluto.run(threads=8)
```

Then open a notebook from `notebooks/`. Each one runs top to bottom and its last cell writes
its figures to `figures/` as PNG and PDF. The metastability heatmaps (300 × 300 settings × 20
noisy runs) take a few minutes on 8 threads; everything else takes seconds to a minute.

## Code and data

- `src/`: the modules the notebooks `include`.
  - `Common.jl` (all three notebooks): the model-independent tools (equilibria by ChaosTools'
    interval root finding, manifolds, limit cycles, continuation with BifurcationKit, ghosts,
    and noisy runs with StochasticDiffEq's Euler–Maruyama), the colours and plotting helpers
    the figures share, plus the CTRNN field and Horchler et al.'s heteroclinic network.
  - `SimpleModels.jl` (the two simple-model notebooks): the 3-node rings and their figures. The
    basins are computed by Attractors.jl and their boundaries drawn with Meshing.jl.
  - `FittedModels.jl` (FittedModel): Roberts et al.'s dwell times, the two fits and their
    runs, and the WormWideWeb recording shown in `exp_data`.
- Both models are fitted to the dwell times of Roberts et al.'s (2016) wild-type worms in Forward,
  Pause 1, Reverse and Pause 2: the Table 1 mean of each state, with an SD equal to it, as the
  exponential dwell times of their Markov model imply (they report no SD of the bouts).
- `data/CTRNNRobertsFit.txt`: the 2-node CTRNN fitted (CMA-ES) to those means and SDs, its
  behaviour read from which node's output is above ½. The fit starts just past the saddle-node on
  invariant circle bifurcation of an excitable network, where it begins to oscillate, and adds a
  penalty to networks that do not oscillate without noise; its noise is fitted too. It was
  first fitted with SDs from Roberts et al.'s decoded bouts, then refitted from that fit to the
  targets above.
- `data/HorchlerRobertsFit.txt`: the heteroclinic network's rates and the noise on each node
  fitted (CMA-ES) to the same means and SDs. It started from Horchler et al.'s (2015) design rule
  (their SHCTools `shc_lv_params`) with one noise level for all nodes, fitted to the decoded-bout
  SDs, and was then refitted from that fit with a noise level per node (equivalent to fitting
  Horchler's node sizes βᵢ).
- The WormWideWeb recordings (Atanas, Kim et al. 2023; about 0.5 GB) are downloaded to
  `data/wormwideweb/` the first time `FittedModel.jl` runs; one of them is shown in `exp_data`.

Both fits are saved, so the figures do not rerun them. To refit, set `refit = true` in
`FittedModel.jl`: each model is fitted anew from its design rule, the CTRNN from just past the
SNIC (`snic_start`) and the heteroclinic network from Horchler et al.'s design at noise 0.3
(`horchler_design`). This takes an hour or more and overwrites the two fit files. The saved fits
were made in the two stages described above, so a new fit will not reproduce them exactly.

## Notes

- The metastability score is the one in the paper's `heatmapsV2.py`: a run is metastable if
  it has at least 5 dominant and 5 transition bouts, and its dominant bouts last on average at
  least twice as long as its transition bouts. Here runs are sampled every 0.1 time units
  instead of every 1, so a hand-over shorter than one time unit no longer merges two dominant
  bouts into one.
- In the GLV's noisy runs the noise is added to u = √x, the same flow in other variables, so
  the activities x = u² stay non-negative without a reflecting wall.

## References

The simple models build on Ashwin and Postlethwaite's work on heteroclinic and excitable
networks:

- Ashwin & Postlethwaite (2013), On designing heteroclinic networks from graphs, *Physica D* 265, 26–39, [doi:10.1016/j.physd.2013.09.006](https://doi.org/10.1016/j.physd.2013.09.006).
- Ashwin & Postlethwaite (2016), Designing heteroclinic and excitable networks in phase space using two populations of coupled cells, *J. Nonlinear Sci.* 26, 345–364, [doi:10.1007/s00332-015-9277-2](https://doi.org/10.1007/s00332-015-9277-2).
- Ashwin & Postlethwaite (2016), Quantifying noisy attractors: from heteroclinic to excitable networks, *SIAM J. Appl. Dyn. Syst.* 15, 1989–2016, [doi:10.1137/16M1061813](https://doi.org/10.1137/16M1061813).
- Ashwin & Postlethwaite (2021), Excitable networks for finite state computation with continuous time recurrent neural networks, *Biol. Cybern.* 115, 519–538, [doi:10.1007/s00422-021-00895-5](https://doi.org/10.1007/s00422-021-00895-5).

## License

The code is under the MIT license (`LICENSE`).
