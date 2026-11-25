"""
    FittedModels

The fitted models and the data of the FittedModel notebook:

- the dwell times of Roberts et al.'s (2016) wild-type worms in Forward → Pause 1 → Reverse →
  Pause 2, which both models are fitted to (`roberts_target`);
- the 2-node CTRNN, its behaviour read from which node's output is above ½ (Roberts et al.'s two
  command units), and its fit (`ctrnn_dwell_fit`);
- the 4-state heteroclinic network and its fit (`horchler_dwell_fit`);
- a WormWideWeb recording (Atanas, Kim et al. 2023) for the activity-above-velocity figure: its
  forward and reverse command neurons (`COMMAND_CLASSES`) and its velocity, as published on
  wormwideweb.org (Flavell lab). Each dataset JSON follows `dataset_json_readme.md` of
  github.com/flavell-lab/WormWideWeb: traces are neuron × frame, neuron rows and frame indices are
  1-based, and the NeuroPAL `label` table is keyed by neuron row.

Both fits are saved in `data/`; the README tells how the saved ones were made.
"""
module FittedModels

using Random
using Statistics
using StatsBase: rle
using Plots
using JSON
using Downloads
using Evolutionary: Evolutionary
using Dates: Dates
using ..Common: VectorField, sde_trajectory, find_equilibria, σ, ctrnn_system, squared_glv_rates, horchler_parameters,
    INK, PALETTE, shade_spans!

export COMMAND_CLASSES, used_pieces, cluster_rows, zscore, cluster_activity
export CYCLE_STATES, CYCLE_SHADES, PAUSE_SPEED, ROBERTS_RATE, TRANSIENT, ROBERTS_TABLE1_MEAN, roberts_target
export label_dwell, dwell_loss
export network, ctrnn, ctrnn_rates, noise_free_cycle, cycle_period, ctrnn_noisy_run, roberts_label, ctrnn_labels
export ctrnn_dwell, EXCITABLE_GENOME, snic_start, read_ctrnn_fit, ctrnn_dwell_fit
export stone_holmes, horchler_period, horchler_design, read_horchler_fit, horchler_noisy_run, horchler_labels
export horchler_velocity, horchler_dwell, horchler_dwell_fit
export NeuronLabel, WormDataset
export download_paper, dataset_files, load_dataset
export FORWARD, PAUSE, REVERSE
export behavior_panel, behavior_legend, behavior_figure

# ══ The command neurons ═════════════════════════════════════════════════════════════════

"""
The neuron classes averaged into each node: AVB drives forward locomotion; AVA and AVE drive
reversals (r ≈ 0.9 with each other in these recordings).
"""
const COMMAND_CLASSES = (forward=["AVB"], reverse=["AVA", "AVE"])

zscore(x) = (x .- mean(x)) ./ std(x)

"""
The frames of a recording that are used: those before the first heat pulse (heat recordings are
cut there), split at recording gaps, each piece longer than `min_frames`.
"""
function used_pieces(r; min_frames=20)
    last_frame = "heat" in r.types ? minimum(r.events["heat"]) - 1 : n_frames(r)
    cuts = [0; filter(<(last_frame), r.gaps); last_frame]
    return [a+1:b for (a, b) in zip(cuts[1:end-1], cuts[2:end]) if b - a > min_frames]
end

"""
The rows of a recording whose NeuroPAL class is one of `classes`: confident labels only (no
'?'), and the surer row of a label given twice.
"""
function cluster_rows(r, classes)
    best = Dict{String,Tuple{Float64,Int}}()
    for (row, l) in r.labels
        (occursin('?', l.label) || !(l.class in classes)) && continue
        (!haskey(best, l.label) || l.confidence > best[l.label][1]) && (best[l.label] = (l.confidence, row))
    end
    return [v[2] for v in values(best)]
end

"""
The average activity of the neurons of `classes` over `frames`: each row z-scored over those
frames, averaged, and the average z-scored.
"""
cluster_activity(r, classes, frames) =
    zscore(vec(mean(reduce(hcat, zscore(r.traces[i, frames]) for i in cluster_rows(r, classes)); dims=2)))


# ══ Roberts et al.'s (2016) dwell times ═════════════════════════════════════════════════

const CYCLE_STATES = ["Forward", "Pause 1", "Reverse", "Pause 2"]

"""The behaviour each state of `CYCLE_STATES` is shaded as (`BEHAVIOR_SHADES`): both pauses alike."""
const CYCLE_SHADES = [:forward, :pause, :reversal, :pause]

"""Speed (mm/s) below which a WormWideWeb worm is pausing (the example recording's shading)."""
const PAUSE_SPEED = 0.02

"""Roberts et al.'s video frame rate (Hz): both models are read every frame."""
const ROBERTS_RATE = 29.97

"""Time (s) each noisy run of either model is simulated and dropped before it is read, so it forgets its random initial state."""
const TRANSIENT = 50.0

# Mean velocity (mm/s) of Roberts et al.'s (2016) wild-type worms in their Forward and Reverse
# states, for reading a model's states as a velocity (v = forward·x_F + reverse·x_R): the mean over
# the five wild-type cohorts (C15, C21, C57, C162, C186; SEM over cohorts 0.009 and 0.010) of
# each cohort's mean over its Viterbi-decoded Forward or Reverse frames. Each frame's velocity is
# its tracking-spot displacement (mm) × 29.97 fps, projected onto the worm's heading over ±5
# frames, as in Roberts et al.'s wormtracks.ipf. Computed from their wormtracks archive
# (doi:10.5061/dryad.35qv6), which is not in this repo.
const ROBERTS_FORWARD_VELOCITY = 0.181
const ROBERTS_REVERSE_VELOCITY = -0.258

"""
Roberts et al.'s (2016) Table 1 mean dwell times (s) of wild-type worms, in the order of
`CYCLE_STATES`: Forward, Pause 1 (both command units off, their state X), Reverse, Pause 2 (both
on, their state Y). Each is the mean over the five cohorts' fits (their ± is the SEM over cohorts,
not a spread of the bouts).
"""
const ROBERTS_TABLE1_MEAN = [5.329, 0.441, 1.945, 0.208]

"""
What both models are fitted to: the Table 1 mean (`ROBERTS_TABLE1_MEAN`) of each state and an SD
equal to it. Roberts et al. report no SD of the bouts; their dwell times are those of a Markov
model, exponential, whose SD is its mean.
"""
roberts_target() = (; mean=ROBERTS_TABLE1_MEAN, sd=ROBERTS_TABLE1_MEAN)

"""
    label_dwell(series, dt)

The mean, SD and number of the bouts of each state 1:4 over label series sampled every `dt`
(each a state per sample), dropping each series' first and last bout, cut short by its ends;
NaN for a state seen too rarely.
"""
function label_dwell(series, dt)
    lengths = [Float64[] for _ in 1:4]
    for s in series
        v, l = rle(s)
        for k in 2:length(v)-1
            push!(lengths[v[k]], l[k] * dt)
        end
    end
    return (; mean=[isempty(x) ? NaN : mean(x) for x in lengths], sd=[length(x) < 2 ? NaN : std(x) for x in lengths],
        n=length.(lengths))
end

"""
    dwell_loss(model, target)

Σ log(mean / target mean)² + Σ log(SD / target SD)² over the four states, from `label_dwell`
and `roberts_target`; 10⁶ when a state is missing.
"""
function dwell_loss(d, target)
    e = sum(abs2, log.(d.mean ./ target.mean)) + sum(abs2, log.(d.sd ./ target.sd))
    return isfinite(e) ? e : 1e6
end

# ══ What the two models share: seeded runs, their dwell times, and the fit ═══════════════

"""
    seeded_run(rates, p, draw, seed; noise, sample_dt, dt, transient, duration)

A noisy run (`sde_trajectory`: Euler–Maruyama, step `dt`) from the state `draw(rng)` drawn with
`seed`: `transient` dropped, then `duration` sampled every `sample_dt`, as `(; t, Y, y0)` with
`y0` the state drawn.
"""
function seeded_run(rates, p, draw, seed; noise, sample_dt, dt, transient, duration)
    rng = Xoshiro(seed)
    y0 = draw(rng)
    return (; sde_trajectory(rates, p, y0, transient + duration; noise, dt, rng, saveat=sample_dt, transient)..., y0)
end

"""
    seeded_dwell(run, labels; seeds, sample_dt)

The dwell times (`label_dwell`) of the label series `labels(run(seed).Y)` over `seeds`, run in
parallel; NaN throughout when a run diverges.
"""
function seeded_dwell(run, labels; seeds, sample_dt)
    series = Vector{Vector{Int}}(undef, length(seeds))
    Threads.@threads for k in eachindex(seeds)
        Y = run(seeds[k]).Y
        series[k] = any(!isfinite, Y) ? Int[] : labels(Y)
    end
    any(isempty, series) && return (; mean=fill(NaN, 4), sd=fill(NaN, 4), n=zeros(Int, 4))
    return label_dwell(series, sample_dt)
end

"""The numbers of a fit file such as `data/CTRNNRobertsFit.txt`: its last line that is not a `#` comment."""
read_fit(path) = parse.(Float64, split(last(filter(line -> !startswith(line, "#"), readlines(path)))))

"""
    cmaes_fit(loss, box, x0, method; file, header, start, legend, runs, iterations, trace, saved=identity)

Minimise `loss` within `box` from `x0` by CMA-ES (Evolutionary's `method`), once per optimiser
seed in `runs` with at most `iterations` generations each, and write the best minimiser, as
`saved(x)`, to `file` under four comment lines: `header`, the `start`, each seed's loss with the
date, and the `legend` naming the numbers.
"""
function cmaes_fit(loss, box, x0, method; file, header, start, legend, runs, iterations, trace, saved=identity)
    fits = map(runs) do run
        res = Evolutionary.optimize(loss, box, x0, method,
            Evolutionary.Options(; iterations, rng=Xoshiro(run), show_trace=trace, callback=_ -> (flush(stdout); false)))
        (; run, loss=Evolutionary.minimum(res), x=Evolutionary.minimizer(res), generations=Evolutionary.iterations(res))
    end
    losses = join(["$(f.run): $(f.loss) ($(f.generations) generations)" for f in fits], ", ")
    open(file, "w") do io
        println(io, "# ", header)
        println(io, "# Started from ", start)
        println(io, "# Losses by optimiser seed: ", losses, "; fitted $(Dates.today()).")
        println(io, "# ", legend)
        println(io, join(saved(argmin(f -> f.loss, fits).x), " "))
    end
end

# ══ The 2-node CTRNN ════════════════════════════════════════════════════════════════════

"""
The network of a genome: weights W (W[i, j]: j → i; node 1 forward, node 2 reverse), biases θ,
log time constants (s) and log noise strengths, one per node.
"""
network(genome) = (W=reshape(genome[1:4], 2, 2), θ=genome[5:6], τ=exp.(genome[7:8]), noise=Tuple(exp.(genome[9:10])))

"""τ ⊙ ẏ = −y + W σ(y + θ), for a network from [`network`](@ref)."""
# |y*| ≤ Σⱼ |Wᵢⱼ| ≤ 2 × 20 within the fit's bounds.
const ctrnn = VectorField(ctrnn_system(net -> net.W; bias=net -> net.θ, tau=net -> net.τ)..., fill(-41.0, 2), fill(41.0, 2))

"""
    ctrnn_rates(y, net)

The field of [`ctrnn`](@ref), (−y + W σ(y + θ)) ./ τ, at a tuple `y` and as a tuple, for
`sde_trajectory`.
"""
ctrnn_rates(y, net) =
    ntuple(i -> (-y[i] + net.W[i, 1] * σ(y[1] + net.θ[1]) + net.W[i, 2] * σ(y[2] + net.θ[2])) / net.τ[i], Val(2))

"""
    noise_free_cycle(net; dt = 0.005)

The network's states along its noise-free oscillation, sampled every 0.05 s for 300 s + 20 τ
after 100 s + 40 τ (τ the larger time constant), or `nothing` when it does not oscillate with
both nodes switching: when it has a stable (or non-hyperbolic) equilibrium, or when either node's
input y + θ crosses 0 fewer than 4 times in that window. Without a stable equilibrium the bounded
2-D flow has a limit cycle (Poincaré–Bendixson).
"""
function noise_free_cycle(net; dt=0.005)
    any(e -> e.stability in (:stable, :nonhyperbolic), find_equilibria(ctrnn, net)) && return nothing
    T = maximum(net.τ)
    settle, window = 100.0 + 40T, 300.0 + 20T
    Y = sde_trajectory(ctrnn_rates, net, .-net.θ .+ [1.0, -1.0], settle + window; noise=0.0, dt, saveat=0.05,
        transient=settle).Y
    any(!isfinite, Y) && return nothing
    all(i -> count(diff(sign.(Y[i, :] .+ net.θ[i])) .!= 0) >= 4, 1:2) || return nothing
    return Y
end

"""The period (s) of a `noise_free_cycle` `Y`: the mean time between the forward node's upward threshold crossings."""
function cycle_period(Y, θ; saveat=0.05)
    up = findall(i -> Y[1, i-1] + θ[1] < 0 <= Y[1, i] + θ[1], 2:size(Y, 2)) .+ 1
    return length(up) < 2 ? NaN : mean(diff(up)) * saveat
end

"""
    ctrnn_noisy_run(net, seed; sample_dt, dt, transient, duration)

A [`seeded_run`](@ref) of the CTRNN with its own additive noise `net.noise` on the states, from
states drawn from a standard normal.
"""
ctrnn_noisy_run(net, seed; kwargs...) = seeded_run(ctrnn_rates, net, rng -> randn(rng, 2), seed; noise=net.noise, kwargs...)

"""
The behaviour of the CTRNN at a state `y`, from which node's output σ(y + θ) is above ½, as
Roberts et al.'s two command units: 1 Forward alone, 2 neither (Pause 1), 3 Reverse alone,
4 both (Pause 2).
"""
roberts_label(y, θ) = (f = y[1] + θ[1] > 0; r = y[2] + θ[2] > 0; f ? (r ? 4 : 1) : (r ? 3 : 2))

"""The CTRNN's behaviour at every sample of a run (`roberts_label`)."""
ctrnn_labels(Y, net) = [roberts_label(y, net.θ) for y in eachcol(Y)]

"""
    ctrnn_dwell(net; seeds = 1:4, sample_dt = 1 / ROBERTS_RATE, dt = 1 / (6 ROBERTS_RATE), transient = TRANSIENT, duration = 1000.0)

The CTRNN's dwell times over runs with the `seeds` given ([`seeded_dwell`](@ref) of `ctrnn_noisy_run` and `ctrnn_labels`).
"""
ctrnn_dwell(net; seeds=1:4, sample_dt=1 / ROBERTS_RATE, dt=1 / (6ROBERTS_RATE), transient=TRANSIENT, duration=1000.0) =
    seeded_dwell(seed -> ctrnn_noisy_run(net, seed; sample_dt, dt, transient, duration), Y -> ctrnn_labels(Y, net); seeds, sample_dt)

"""
The excitable 2-node network the fit starts from (an earlier fit to WormWideWeb activity,
rounded): a stable node next to a saddle, with an unstable focus inside. The genome is
W_FF W_RF W_FR W_RR (column-major), θ_F θ_R, log τ_F log τ_R (s), log noise_F log noise_R.
"""
const EXCITABLE_GENOME = [7.75, -4.87, 3.31, 6.1, -4.78, -2.05, log(2.901), log(2.918), log(2.473), log(0.689)]

"""
    snic_start(genome, cycle_length; past = 0.025)

Where the CTRNN fit starts. Lowering θ_F from the excitable `genome` merges its stable node with
the saddle beside it, a saddle-node on the invariant circle (SNIC): past it the network
oscillates, slowly near the vanished equilibria (the Forward state). The bifurcation is found by
bisection on θ_F, and the start is `past` beyond it. Its time constants are then divided by the
factor k that makes the noise-free period `cycle_length`, and its noise multiplied by √k, which
speeds the noisy network up by k without changing it otherwise. Returns the genome.
"""
function snic_start(genome, cycle_length; past=0.025)
    stable(θF) = any(e -> e.stability == :stable, find_equilibria(ctrnn, network([genome[1:4]; θF; genome[6:10]])))
    stable(genome[5]) || error("the network has no stable equilibrium to start from")
    # Step down in 0.01 to the first θ_F without a stable equilibrium (the oscillating band is narrow), then bisect.
    hi = genome[5]
    lo = hi - 0.01
    while stable(lo)
        hi, lo = lo, lo - 0.01
        lo < genome[5] - 2 && error("no SNIC within 2 below θ_F")
    end
    for _ in 1:50
        m = (lo + hi) / 2
        stable(m) ? (hi = m) : (lo = m)
    end
    g = [genome[1:4]; lo - past; genome[6:10]]
    net = network(g)
    k = cycle_period(noise_free_cycle(net), net.θ) / cycle_length
    return [g[1:6]; g[7:8] .- log(k); g[9:10] .+ log(k) / 2]
end

"""The network of a CTRNN fit file ([`read_fit`](@ref))."""
read_ctrnn_fit(path) = network(read_fit(path))

"""
    ctrnn_dwell_fit(; file, target, runs = 2:5, iterations = 400, penalty = 2.0, refit = false, trace = false)

The 2-node CTRNN (`read_ctrnn_fit`) whose dwell times (`ctrnn_dwell`) match `target`
(`roberts_target`). Read from `file` when it exists; otherwise (or with `refit = true`) fitted
and the best genome written there, which takes about 10–15 minutes per optimiser seed (about an
hour for the default four).

A fit starts from `snic_start(EXCITABLE_GENOME, sum(target.mean))`: just past the SNIC, with the
noise-free period equal to the cycle of the four target means. CMA-ES (`cmaes_fit`; λ = 40,
σ₀ = 0.3) then minimises `dwell_loss` of `ctrnn_dwell`, plus `penalty` for a network that does not
oscillate without noise (`noise_free_cycle`), with weights and biases within ±20, time constants
within [0.05, 20] s and noise within [0.01, 20].
"""
function ctrnn_dwell_fit(; file, target, runs=2:5, iterations=400, penalty=2.0, refit=false, trace=false)
    if refit || !isfile(file)
        start = snic_start(EXCITABLE_GENOME, sum(target.mean))
        loss(g) = (net = network(g); dwell_loss(ctrnn_dwell(net), target) + (isnothing(noise_free_cycle(net)) ? penalty : 0.0))
        box = Evolutionary.BoxConstraints([fill(-20.0, 6); fill(log(0.05), 2); fill(log(0.01), 2)],
            [fill(20.0, 6); fill(log(20.0), 2); fill(log(20.0), 2)])
        cmaes_fit(loss, box, start, Evolutionary.CMAES(λ=40, sigma0=0.3); file, runs, iterations, trace,
            header="2-node CTRNN fitted to Roberts et al.'s (2016) Table 1 mean dwell times, with SD = mean (FittedModels.ctrnn_dwell_fit).",
            start="the SNIC start, genome " * join(start, " "),
            legend="Genome: W_FF W_RF W_FR W_RR (column-major), θ_F θ_R, log τ_F log τ_R (s), log noise_F log noise_R")
    end
    return read_ctrnn_fit(file)
end

# ══ The 4-state heteroclinic network ════════════════════════════════════════════════════

"""
    stone_holmes(δ, ε, λu, λs)

The mean time to leave a δ-neighbourhood of a saddle with unstable and stable eigenvalues λu and
−λs under noise ε (Stone and Holmes 1990, as in Horchler's SHCTools `shc_passagetime`).
"""
stone_holmes(δ, ε, λu, λs) = (2 * (log(δ) - log(ε)) + log(4λu) - log1p(λu / λs) + Base.MathConstants.eulergamma) / (2λu)

"""
    horchler_period(α, ε; δ = 0.1)

Horchler et al.'s (2015) time spent at a node with rate α (β = ν = 1) under noise ε: the
transition to the next node, 2 log(1/δ − 1) / α, plus the passage time past the saddle
(`stone_holmes`, both eigenvalues α), with δ = 0.1 β the neighbourhood (SHCTools `shc_lv_params`).
"""
horchler_period(α, ε; δ=0.1) = 2 * log(1 / δ - 1) / α + stone_holmes(δ, ε, α, α)

"""
    horchler_design(dwell, ε)

Horchler et al.'s (2015) design rule: the rates α whose `horchler_period` at noise ε are the
dwell times `dwell`, by bisection on log α (the period falls as α grows).
"""
function horchler_design(dwell, ε)
    return map(dwell) do τ
        lo, hi = log(1e-4), log(1e4)
        for _ in 1:200
            m = (lo + hi) / 2
            horchler_period(exp(m), ε) > τ ? (lo = m) : (hi = m)
        end
        exp((lo + hi) / 2)
    end
end

"""The rates α and noise levels of a fit file such as `data/HorchlerRobertsFit.txt` ([`read_fit`](@ref)): `α₁ … α₄ ε₁ … ε₄`."""
read_horchler_fit(path) = (x = read_fit(path); (; α=x[1:4], noise=x[5:8]))

"""
    horchler_noisy_run(α, seed; noise, sample_dt, dt, transient, duration)

A [`seeded_run`](@ref) of the 4-state heteroclinic network with rates `α` (`squared_glv_rates`,
`horchler_parameters`) and additive noise on u, `noise[i]` on node i, from u drawn uniformly in
[0, 1]⁴.
"""
horchler_noisy_run(α, seed; kwargs...) = seeded_run(squared_glv_rates, horchler_parameters(α), rng -> rand(rng, 4), seed; kwargs...)

"""The heteroclinic network's state at every sample of a run: the node with the largest activity uᵢ²."""
horchler_labels(Y) = [argmax(abs.(u)) for u in eachcol(Y)]

"""
The heteroclinic network's states as a velocity (mm/s) at every sample of a run: Roberts et al.'s
mean forward and reverse velocities, weighted by the Forward and Reverse activities u².
"""
horchler_velocity(Y) = ROBERTS_FORWARD_VELOCITY .* Y[1, :] .^ 2 .+ ROBERTS_REVERSE_VELOCITY .* Y[3, :] .^ 2

"""
    horchler_dwell(α, noise; seeds = 1001:1064, sample_dt = 1 / ROBERTS_RATE, dt = 1 / (20 ROBERTS_RATE), transient = TRANSIENT, duration = 2000.0)

The heteroclinic network's dwell times over runs with the `seeds` given ([`seeded_dwell`](@ref) of `horchler_noisy_run` and `horchler_labels`).
"""
horchler_dwell(α, noise; seeds=1001:1064, sample_dt=1 / ROBERTS_RATE, dt=1 / (20ROBERTS_RATE), transient=TRANSIENT, duration=2000.0) =
    seeded_dwell(seed -> horchler_noisy_run(α, seed; noise, sample_dt, dt, transient, duration), horchler_labels; seeds, sample_dt)

"""
    horchler_dwell_fit(; file, target, runs = 1:1, iterations = 150, refit = false, trace = false)

The rates α and per-node noise levels ε (on u) of the 4-state heteroclinic network (Horchler
construction, β = ν = 1) whose dwell times (`horchler_dwell`) match `target` (`roberts_target`),
as `(; α, noise)`. Per-node noise is what Horchler's node sizes βᵢ would add: with uᵢ = √βᵢ vᵢ the
network with βᵢ and noise ε is this one with noise ε / √βᵢ on node i. Read from `file` when it
exists; otherwise (or with `refit = true`) fitted and written there, which takes about an hour.

A fit starts from Horchler et al.'s design (`horchler_design`) of the target means at noise 0.3
on every node. CMA-ES (`cmaes_fit`; 8 offspring, 4 parents, initial step 0.3) then minimises
`dwell_loss` of `horchler_dwell` (deterministic: fixed seeds) over log α within
[log 0.01, log 200] and log ε within [log 0.001, 0].
"""
function horchler_dwell_fit(; file, target, runs=1:1, iterations=150, refit=false, trace=false)
    if refit || !isfile(file)
        α, noise = horchler_design(target.mean, 0.3), fill(0.3, 4)
        loss(x) = dwell_loss(horchler_dwell(exp.(x[1:4]), exp.(x[5:8])), target)
        box = Evolutionary.BoxConstraints([fill(log(0.01), 4); fill(log(1e-3), 4)], [fill(log(200.0), 4); zeros(4)])
        cmaes_fit(loss, box, log.([α; noise]), Evolutionary.CMAES(μ=4, λ=8, sigma0=0.3); file, runs, iterations, trace,
            saved=x -> exp.(x),
            header="4-state heteroclinic GLV (Horchler construction, β = ν = 1) fitted to Roberts et al.'s (2016) Table 1 mean dwell times, with SD = mean (FittedModels.horchler_dwell_fit).",
            start="Horchler et al.'s design, α = $(join(α, " ")), ε = $(join(noise, " "))",
            legend="α: Forward, Pause 1, Reverse, Pause 2; then the noise ε on u of each node, same order")
    end
    return read_horchler_fit(file)
end

# ══ wormwideweb.org recordings ══════════════════════════════════════════════════════════

# Colours of the behaviours.
const FORWARD = PALETTE.blue
const PAUSE = PALETTE.orange
const REVERSE = PALETTE.green

# ── Data ────────────────────────────────────────────────────────────────────────────────

"""NeuroPAL identity of one recorded neuron (`confidence` runs 1–5, higher is surer)."""
struct NeuronLabel
    label::String
    class::String
    confidence::Float64
end

"""
One recorded animal. Frames are 1-based; `dt` is seconds per frame with recording gaps
excluded, and `gaps` holds the last frame before each gap (the time axis skips them).
"""
struct WormDataset
    uid::String
    types::Vector{String}
    dt::Float64
    traces::Matrix{Float64}
    velocity::Vector{Float64}
    events::Dict{String,Vector{Int}}
    gaps::Vector{Int}
    labels::Dict{Int,NeuronLabel}
end

n_frames(ds::WormDataset) = size(ds.traces, 2)

"""
    download_paper(paper_id, dir)

Download every dataset JSON of one paper from wormwideweb.org and unpack it into
`dir/paper_id`. Skips the download when that directory already holds JSON files.
Paper ids are listed in `activity/papers.json` of github.com/flavell-lab/WormWideWeb-data.
"""
function download_paper(paper_id::AbstractString, dir::AbstractString)
    target = joinpath(dir, paper_id)
    isdir(target) && !isempty(dataset_files(target)) && return target

    mkpath(target)
    archive = joinpath(dir, "$(paper_id).tar.bz2")
    url = "https://wormwideweb.org/activity/api/data/download/paper/$(paper_id)/"
    isfile(archive) || Downloads.download(url, archive)
    run(`tar -xjf $archive -C $target`)
    return target
end

"""Sorted paths of the dataset JSON files in `dir`."""
dataset_files(dir::AbstractString) =
    sort(filter(f -> endswith(f, ".json"), readdir(dir; join = true)))

_float(x) = x === nothing ? NaN : Float64(x)
_vector(xs) = xs === nothing ? Float64[] : _float.(xs)
_matrix(rows) = isempty(rows) ? zeros(0, 0) : permutedims(reduce(hcat, _vector.(rows)))

"""Last frame before each jump in `timestamps` longer than `factor` median frame steps."""
function _gaps(timestamps; factor = 5)
    steps = diff(timestamps)
    isempty(steps) && return Int[]
    findall(>(factor * median(steps)), steps)
end

"""Parse one dataset JSON into a `WormDataset`."""
function load_dataset(path::AbstractString)
    raw = JSON.parsefile(path)
    meta, timing = raw["metadata"], raw["timing"]
    events = Dict{String,Vector{Int}}(String(k) => Int.(v) for (k, v) in something(get(timing, "event", nothing), Dict()))
    labels = Dict{Int,NeuronLabel}()
    for (row, entry) in something(get(raw, "label", nothing), Dict())
        labels[parse(Int, row)] = NeuronLabel(String(entry["label"]), String(entry["neuron_class"]), _float(entry["confidence"]))
    end
    return WormDataset(String(meta["uid"]), String.(meta["dataset_type"]), Float64(timing["mean_timestep"]),
        _matrix(raw["gcamp"]["trace_array"]), _vector(get(raw["behavior"], "velocity", nothing)), events,
        _gaps(_vector(timing["timestamp_confocal"])), labels)
end

# ── Behaviour and velocity figures ──────────────────────────────────────────────────────

"""The shaded behaviours of `behavior_figure`, with their colours."""
const BEHAVIOR_SHADES = (forward=FORWARD, reversal=REVERSE, pause=PAUSE)

"""
A shaded span over each stretch of one behaviour in `states` (`:forward`, `:reversal` or `:pause`
at each time of `t`; anything else is left unshaded).
"""
shade_behaviors!(panel, t, states) = shade_spans!(panel, t, states, s -> get(BEHAVIOR_SHADES, s, nothing); alpha=0.25)

"""
    behavior_panel(t, traces, colors, states; linewidth = 2, kwargs...)

A panel of `traces` in `colors` against `t` over a zero line, shaded by the behaviour `states` at
every sample (`shade_behaviors!`); `kwargs` go to the plot.
"""
function behavior_panel(t, traces, colors, states; linewidth=2, kwargs...)
    panel = plot(; titlefontsize=10, legend=false, kwargs...)
    shade_behaviors!(panel, t, states)
    hline!(panel, [0]; color=:gray, alpha=0.5)
    for (trace, color) in zip(traces, colors)
        plot!(panel, t, trace; color, linewidth)
    end
    return panel
end

"""A legend in a row of its own: the shaded behaviours, then a line in each of `colors` named by `labels`."""
function behavior_legend(labels, colors)
    row = plot(; framestyle=:none, legend=:top, legend_columns=-1, xlims=(0, 1), ylims=(0, 1))
    for (name, color) in pairs(BEHAVIOR_SHADES)
        vspan!(row, [NaN, NaN]; color, alpha=0.25, linewidth=0, label=string(name))
    end
    for (label, color) in zip(labels, colors)
        plot!(row, [NaN]; color, linewidth=2, label)
    end
    return row
end

"""
    behavior_figure(t, traces, velocity; trace_labels, trace_colors, activity_states,
                    velocity_states = activity_states, velocity_t = t,
                    activity_ylabel = "GCaMP fluorescence (z-score)", activity_title = "Neuronal activity",
                    velocity_title = "Freely-moving velocity")

A worm's (or a model's) activity and velocity: the rows of `traces` (activity) above `velocity`,
sharing the time axis, each a [`behavior_panel`](@ref) shaded by its own behaviour, under a
[`behavior_legend`](@ref). `velocity_t` and `velocity_states` let the velocity come from a
separate run sampled at other times.
"""
function behavior_figure(t, traces, velocity; trace_labels, trace_colors, activity_states,
    velocity_states=activity_states, velocity_t=t, activity_ylabel="GCaMP fluorescence (z-score)",
    activity_title="Neuronal activity", velocity_title="Freely-moving velocity")
    activity_panel = behavior_panel(t, eachrow(traces), trace_colors, activity_states; ylabel=activity_ylabel,
        title=activity_title, xformatter=_ -> "")
    velocity_panel = behavior_panel(velocity_t, [velocity], [INK], velocity_states; xlabel="time (s)",
        ylabel="velocity (mm/s)", title=velocity_title, bottom_margin=4Plots.mm)
    t_lims = (min(first(t), first(velocity_t)), max(last(t), last(velocity_t)))
    # The legend sits in a row of its own above both panels, over the top panel's title.
    return plot(behavior_legend(trace_labels, trace_colors), activity_panel, velocity_panel;
        layout=grid(3, 1; heights=[0.04, 0.48, 0.48]), link=:x, size=(1100, 640), xlims=t_lims, left_margin=4Plots.mm)
end

end # module
