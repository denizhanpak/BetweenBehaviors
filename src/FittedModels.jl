"""
    FittedModels

The fitted models and the data of the FittedModel notebook, all on the WormWideWeb recordings
(Atanas, Kim et al. 2023):

- the forward and reverse command neurons of a recording (`COMMAND_CLASSES`: AVB; AVA and AVE),
  averaged into one z-scored trace each, and the statistics of such a pair of traces
  (`activity_stats`);
- the 2-node CTRNN fitted to those statistics, one node per cluster, restricted to networks that
  oscillate without noise (`ctrnn_activity_fit`, saved in `data/`);
- the locomotion cycle Forward → Pause 1 → Reverse → Pause 2 read from a sequence of 'F', 'P' and
  'R' (`cycle_bouts`): from the worms' velocity, and from the command neurons, the worms' or the
  CTRNN's (`neural_sequence`);
- the 4-state heteroclinic network's rates fitted to the worms' dwell times in that cycle
  (`horchler_dwell_fit`, saved in `data/`);
- the recordings themselves, as published on wormwideweb.org (Flavell lab), and the figure of
  activity above velocity. Each dataset JSON follows `dataset_json_readme.md` of
  github.com/flavell-lab/WormWideWeb: traces are neuron × frame, neuron rows and frame indices are
  1-based, the NeuroPAL `label` table is keyed by neuron row, and the head-angle sign is already
  corrected.
"""
module FittedModels

using Random
using Statistics
using StatsBase: rle, autocor
using Plots
using JSON
using Downloads
using Evolutionary: Evolutionary
using Dates: Dates
using ..Common: VectorField, sde_trajectory, find_equilibria, σ, ctrnn_system, squared_glv_rates, horchler_parameters

export COMMAND_CLASSES, used_pieces, cluster_rows, has_command_neurons, zscore, cluster_activity
export CYCLE_STATES, PAUSE_SPEED, velocity_sequence, neural_sequence, label_sequence, cycle_bouts, dwell_summary
export ACTIVITY_STAT_NAMES, activity_stats, activity_target, activity_loss
export network, ctrnn, ctrnn_rates, noise_free_cycle, cycle_period, ctrnn_noisy_run, ctrnn_activity_stats
export CTRNN_START, read_ctrnn_fit, ctrnn_activity_fit
export read_horchler_fit, horchler_noisy_run, horchler_labels, horchler_mean_dwell, horchler_dwell_fit
export NeuronLabel, WormDataset
export download_paper, dataset_files, load_dataset
export FORWARD, PAUSE, REVERSE
export mean_state_velocities, shade_behaviors!, behavior_figure

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

"""Whether a recording has both a forward and a reverse command neuron labelled."""
has_command_neurons(r) = all(!isempty(cluster_rows(r, c)) for c in COMMAND_CLASSES)

"""
The average activity of the neurons of `classes` over `frames`: each row z-scored over those
frames, averaged, and the average z-scored.
"""
cluster_activity(r, classes, frames) =
    zscore(vec(mean(reduce(hcat, zscore(r.traces[i, frames]) for i in cluster_rows(r, classes)); dims=2)))

# ══ The locomotion cycle ════════════════════════════════════════════════════════════════

const CYCLE_STATES = ["Forward", "Pause 1", "Reverse", "Pause 2"]

"""Speed (mm/s) below which a worm is pausing."""
const PAUSE_SPEED = 0.02

"""'F', 'R' or 'P' at every sample of a velocity: forward or backward faster than `pause_speed`, or pausing."""
velocity_sequence(v; pause_speed=PAUSE_SPEED) = [x > pause_speed ? 'F' : x < -pause_speed ? 'R' : 'P' for x in v]

"""
'F', 'R' or 'P' at every sample of a pair of z-scored traces, forward and reverse: 'F' when only
the forward trace is above 0, 'R' when only the reverse one is, 'P' otherwise. Used on the
worms' command neurons and on the CTRNN alike.
"""
neural_sequence(f, r) = [a && !b ? 'F' : b && !a ? 'R' : 'P' for (a, b) in zip(f .> 0, r .> 0)]

"""'F', 'R' or 'P' for the heteroclinic network's states (1 Forward, 3 Reverse, 2 and 4 pauses)."""
label_sequence(labels) = [l == 1 ? 'F' : l == 3 ? 'R' : 'P' for l in labels]

"""
    cycle_bouts(sequence, dt)

The bouts of the cycle in a sequence of 'F', 'P' and 'R' sampled every `dt`, as `(state, length)`
with state 1 Forward, 2 Pause 1, 3 Reverse, 4 Pause 2 and length in s. A pause between Forward
and Reverse is Pause 1 and one between Reverse and Forward Pause 2; a pause between two runs in
the same direction is merged into them (F P F is one Forward bout). The first and last bouts,
cut short by the ends of the sequence, are dropped.
"""
function cycle_bouts(sequence, dt)
    v, l = rle(sequence)
    v, l = collect(v), collect(l)
    i = 2
    while i < length(v)
        if v[i] == 'P' && v[i-1] == v[i+1]
            l[i-1] += l[i] + l[i+1]
            deleteat!(v, i:i+1)
            deleteat!(l, i:i+1)
        else
            i += 1
        end
    end
    out = Tuple{Int,Float64}[]
    for k in 2:length(v)-1
        state = v[k] == 'F' ? 1 : v[k] == 'R' ? 3 : (v[k-1], v[k+1]) == ('F', 'R') ? 2 : (v[k-1], v[k+1]) == ('R', 'F') ? 4 : 0
        state > 0 && push!(out, (state, l[k] * dt))
    end
    return out
end

"""The mean, SD and number of the bouts `(state, length)` of each state 1:4 (NaN for a state never seen)."""
function dwell_summary(bouts)
    lengths = [[d for (s, d) in bouts if s == i] for i in 1:4]
    return (; mean=[isempty(x) ? NaN : mean(x) for x in lengths], sd=[length(x) < 2 ? NaN : std(x) for x in lengths],
        n=length.(lengths))
end

# ══ Activity statistics ═════════════════════════════════════════════════════════════════

const QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
const ACF_LAGS = (1.2, 3.0, 6.0, 12.0, 24.0)                  # s
const CCF_LAGS = (-12.0, -6.0, -3.0, 0.0, 3.0, 6.0, 12.0)      # s; positive: reverse after forward

"""The names of the entries of `activity_stats`."""
const ACTIVITY_STAT_NAMES = [
    ["forward q$(q)" for q in QUANTILES]; ["reverse q$(q)" for q in QUANTILES];
    ["forward ACF $(l) s" for l in ACF_LAGS]; ["reverse ACF $(l) s" for l in ACF_LAGS];
    ["CCF $(l) s" for l in CCF_LAGS];
    ["log forward high (s)", "log forward low (s)", "log reverse high (s)", "log reverse low (s)"]]

"""The mean length (s) of the stretches above and below 0, all but the first and last."""
function high_low(x, dt)
    v, l = rle(x .> 0)
    v, l = v[2:end-1], l[2:end-1]
    return (mean(l[v]) * dt, mean(l[.!v]) * dt)
end

"""
    activity_stats(f, r, dt)

The statistics of a pair of z-scored traces, forward `f` and reverse `r`, sampled every `dt` s,
as one vector (names in `ACTIVITY_STAT_NAMES`): the quantiles of each trace, its autocorrelation
at 1.2–24 s, their cross-correlation at −12 to 12 s (positive lags: reverse after forward), and
the log mean time each spends above and below 0.
"""
function activity_stats(f, r, dt)
    lag(s) = round(Int, s / dt)
    cc(k) = k >= 0 ? cor(f[1:end-k], r[1+k:end]) : cor(f[1-k:end], r[1:end+k])
    return [[quantile(f, q) for q in QUANTILES]; [quantile(r, q) for q in QUANTILES];
        autocor(f, [lag(s) for s in ACF_LAGS]); autocor(r, [lag(s) for s in ACF_LAGS]);
        [cc(lag(s)) for s in CCF_LAGS];
        log.([high_low(f, dt)..., high_low(r, dt)...])]
end

"""
    activity_target(recordings)

What the CTRNN is fitted to, from the recordings with both command clusters labelled
(`has_command_neurons`): the `activity_stats` of every used piece (`used_pieces`), their mean
weighted by length (`target`) and their SD across pieces, at least 0.05 (`spread`), with the mean
frame step `dt` of those recordings and their number `n`.
"""
function activity_target(recordings)
    rs = filter(has_command_neurons, recordings)
    pieces = [(r, p) for r in rs for p in used_pieces(r)]
    S = reduce(hcat, [activity_stats(cluster_activity(r, COMMAND_CLASSES.forward, p),
        cluster_activity(r, COMMAND_CLASSES.reverse, p), r.dt) for (r, p) in pieces])
    w = [length(p) for (_, p) in pieces]
    return (; target=S * w / sum(w), spread=max.(vec(std(S; dims=2)), 0.05), dt=mean(r.dt for r in rs), n=length(rs))
end

"""Σ ((model − target) / spread)² over the statistics; 10⁶ for a model without them (`nothing`)."""
activity_loss(stats, t) = isnothing(stats) ? 1e6 : sum(abs2, (stats .- t.target) ./ t.spread)

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
    ctrnn_noisy_run(net, seed; sample_dt, dt = 0.005, transient = 50.0, duration = 1000.0)

A run of the CTRNN with its own additive noise `net.noise` on the states (Euler–Maruyama, step
`dt`), from states drawn from a standard normal with `seed`: `transient` dropped, then `duration`
sampled every `sample_dt`, as `(; t, Y)`.
"""
function ctrnn_noisy_run(net, seed; sample_dt, dt=0.005, transient=50.0, duration=1000.0)
    rng = Xoshiro(seed)
    y0 = randn(rng, 2)
    return sde_trajectory(ctrnn_rates, net, y0, transient + duration; noise=net.noise, dt, rng, saveat=sample_dt, transient)
end

"""
The CTRNN's `activity_stats` (states z-scored per run, sampled like the worms) averaged over runs
with `seeds`, or `nothing` for a run that diverges or stays flat.
"""
function ctrnn_activity_stats(net, sample_dt; seeds=1:4, run_settings...)
    S = map(seeds) do seed
        Y = ctrnn_noisy_run(net, seed; sample_dt, run_settings...).Y
        (any(!isfinite, Y) || any(std(y) < 1e-9 for y in eachrow(Y))) && return nothing
        activity_stats(zscore(Y[1, :]), zscore(Y[2, :]), sample_dt)
    end
    return any(isnothing, S) ? nothing : mean(S)
end

"""
Where the activity fit starts: the 2-node CTRNN once fitted to Roberts et al.'s (2016) dwell
times, an oscillator, with its time constants doubled and noise 0.15 of each state's range.
"""
const CTRNN_START = [7.445739172768187, -3.7203945373732137, 1.994685020079162, 5.329814365907088,
    -4.635460230491318, -1.330625360589267, -1.173409328204415, -1.5901249715194448, -0.26299070823917686,
    0.17133145472373434]

"""The network of a CTRNN fit file such as `data/CTRNNActivityFit.txt`: its last line that is not a `#` comment."""
read_ctrnn_fit(path) = network(parse.(Float64, split(last(filter(line -> !startswith(line, "#"), readlines(path))))))

"""
    ctrnn_activity_fit(path, target; runs = 2:5, iterations = 400)

Refits the 2-node CTRNN to the worms' activity statistics `target` (`activity_target`) and
writes the best genome to `path`, as in `data/CTRNNActivityFit.txt`. CMA-ES (Evolutionary: λ = 40,
σ₀ = 1, at most `iterations` generations, offspring evaluated on all threads) minimises
`activity_loss` of `ctrnn_activity_stats` over networks that oscillate without noise
(`noise_free_cycle`; any other scores 10⁶), once per optimiser seed in `runs`, each from
`CTRNN_START`, with weights and biases within ±20, time constants within [0.05, 20] s and noise
within [0.01, 20]. Takes about 20 minutes a run on 2 threads.
"""
function ctrnn_activity_fit(path, target; runs=2:5, iterations=400)
    loss(g) = (net = network(g); isnothing(noise_free_cycle(net)) ? 1e6 : activity_loss(ctrnn_activity_stats(net, target.dt), target))
    box = Evolutionary.BoxConstraints([fill(-20.0, 6); fill(log(0.05), 2); fill(log(0.01), 2)],
        [fill(20.0, 6); fill(log(20.0), 2); fill(log(20.0), 2)])
    fits = map(runs) do run
        res = Evolutionary.optimize(loss, box, CTRNN_START, Evolutionary.CMAES(λ=40, sigma0=1.0),
            Evolutionary.Options(iterations=iterations, rng=Xoshiro(run), parallelization=:thread))
        (; run, loss=Evolutionary.minimum(res), genome=Evolutionary.minimizer(res))
    end
    best = argmin(f -> f.loss, fits)
    open(path, "w") do io
        println(io, "# 2-node CTRNN fitted to the activity statistics of AVB (node 1) and AVA+AVE (node 2), $(target.n) WormWideWeb recordings (FittedModels.ctrnn_activity_fit).")
        println(io, "# Losses by optimiser seed: ", join(["$(f.run): $(f.loss)" for f in fits], ", "), "; fitted $(Dates.today()).")
        println(io, "# Genome: W_FF W_RF W_FR W_RR (column-major), θ_F θ_R, log τ_F log τ_R (s), log noise_F log noise_R")
        println(io, join(best.genome, " "))
    end
    return network(best.genome)
end

# ══ The 4-state heteroclinic network ════════════════════════════════════════════════════

"""The α of a fit file such as `data/HorchlerFitWWWNoise0.1.txt`: its last line that is not a `#` comment."""
read_horchler_fit(path) = parse.(Float64, split(last(filter(line -> !startswith(line, "#"), readlines(path)))))

"""
    horchler_noisy_run(α, seed; noise, sample_dt, dt, transient, duration)

A run of the 4-state heteroclinic network with rates `α` (`squared_glv_rates`,
`horchler_parameters`) and additive noise `noise` on u (Euler–Maruyama, step `dt`), from u drawn
uniformly in [0, 1]⁴ with `seed`: `transient` dropped, then `duration` sampled every `sample_dt`,
as `(; t, Y)`.
"""
function horchler_noisy_run(α, seed; noise, sample_dt, dt, transient, duration)
    rng = Xoshiro(seed)
    return sde_trajectory(squared_glv_rates, horchler_parameters(α), rand(rng, 4), transient + duration; noise, dt, rng,
        saveat=sample_dt, transient)
end

"""The heteroclinic network's state at every sample of a run: the node with the largest activity uᵢ²."""
horchler_labels(Y) = [argmax(abs.(u)) for u in eachcol(Y)]

"""
    horchler_mean_dwell(α, seeds; settings...)

The heteroclinic network's mean dwell time in each state of the cycle over runs with the `seeds`
given (`horchler_noisy_run`, in parallel), read like the worms' (`cycle_bouts`): NaN for a state
never visited.
"""
function horchler_mean_dwell(α, seeds; settings...)
    series = Vector{Vector{Tuple{Int,Float64}}}(undef, length(seeds))
    Threads.@threads for k in eachindex(seeds)
        labels = horchler_labels(horchler_noisy_run(α, seeds[k]; settings...).Y)
        series[k] = cycle_bouts(label_sequence(labels), settings[:sample_dt])
    end
    return dwell_summary(reduce(vcat, series)).mean
end

"""
    horchler_dwell_fit(; file, target, noise, sample_dt, dt, transient, duration, seeds = 1001:1064, refit = false, miss = 0.01)

The rates α of the 4-state heteroclinic network (Horchler construction, β = ν = 1) whose mean
dwell times with additive noise `noise`, sampled every `sample_dt` and read like the worms'
(`horchler_mean_dwell`), match `target`, the worms' mean dwell times. Read from `file` when it
exists; otherwise (or with `refit = true`) fitted and written there, which takes about an hour.

The fit minimises Σᵢ log(mᵢ / dᵢ)² (mᵢ on the fixed `seeds`, so the objective is deterministic;
a state never visited scores 10⁶) by CMA-ES (Evolutionary: 8 offspring, 4 parents, initial step
0.3, at most 150 generations, optimiser seed 1) on log α within [log 0.01, log 200]. It starts
from αᵢ = ln(1/noise) / dᵢ, the passage time past a saddle (Stone–Holmes). A fit whose objective
is above `miss` (0.01: about 5 % off per state) is returned with a warning.
"""
function horchler_dwell_fit(; file, target, noise, sample_dt, dt, transient, duration, seeds=1001:1064, refit=false,
    miss=0.01)
    if refit || !isfile(file)
        settings = (; noise, sample_dt, dt, transient, duration)
        function objective(logα)
            e = sum(abs2, log.(horchler_mean_dwell(exp.(logα), seeds; settings...) ./ target))
            return isfinite(e) ? e : 1e6
        end
        start = log.(log(1 / noise) ./ target)
        cma = Evolutionary.optimize(objective, Evolutionary.BoxConstraints(fill(log(0.01), 4), fill(log(200.0), 4)),
            start, Evolutionary.CMAES(μ=4, λ=8, sigma0=0.3), Evolutionary.Options(iterations=150, rng=Xoshiro(1)))
        open(file, "w") do io
            println(io, "# 4-state heteroclinic GLV (Horchler construction, β = ν = 1), α fitted for additive noise $noise on u (FittedModels.horchler_dwell_fit)")
            println(io, "# to the WormWideWeb dwell times $(round.(target; digits=3)) s. Euler–Maruyama dt = $dt s, sampled every $sample_dt s; seeds $seeds, each $duration s after $transient s.")
            println(io, "# Objective Σ log(mean dwell / target)² = $(Evolutionary.minimum(cma)), " *
                        "$(Evolutionary.iterations(cma)) generations of CMA-ES; fitted $(Dates.today()).")
            println(io, "# α: Forward, Pause 1, Reverse, Pause 2")
            println(io, join(exp.(Evolutionary.minimizer(cma)), " "))
        end
    end
    reached = match(r"Objective .* = ([0-9.eE+-]+),", read(file, String))
    if !isnothing(reached) && parse(Float64, reached[1]) > miss
        @warn "The heteroclinic network's fit misses the worms' dwell times: objective $(reached[1]) > $miss ($(basename(file)))."
    end
    return read_horchler_fit(file)
end

# ══ wormwideweb.org recordings ══════════════════════════════════════════════════════════

# Colours of the behaviours and of the velocity trace.
const FORWARD = "#2a78d6"
const PAUSE = "#eb6834"
const REVERSE = "#1baf7a"
const INK = "#0b0b0b"

# ── Data ────────────────────────────────────────────────────────────────────────────────

"""NeuroPAL identity of one recorded neuron (`confidence` runs 1–5, higher is surer)."""
struct NeuronLabel
    label::String
    class::String
    lr::String
    confidence::Float64
end

"""
One recorded animal. Frames are 1-based; `dt` is seconds per frame with recording gaps
excluded, and `gaps` holds the last frame before each gap (the time axis skips them).
"""
struct WormDataset
    paper_id::String
    uid::String
    types::Vector{String}
    dt::Float64
    traces::Matrix{Float64}
    traces_raw::Matrix{Float64}
    behavior::Dict{String,Vector{Float64}}
    reversals::Vector{UnitRange{Int}}
    events::Dict{String,Vector{Int}}
    gaps::Vector{Int}
    labels::Dict{Int,NeuronLabel}
end

n_frames(ds::WormDataset) = size(ds.traces, 2)

function Base.show(io::IO, ds::WormDataset)
    print(io, "WormDataset($(ds.paper_id) $(ds.uid) [$(join(ds.types, ","))]: ",
        "$(size(ds.traces, 1)) neurons ($(length(ds.labels)) labeled), ",
        "$(n_frames(ds)) frames × $(round(ds.dt; digits = 3)) s)")
end

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
    meta, timing, behavior, gcamp = raw["metadata"], raw["timing"], raw["behavior"], raw["gcamp"]

    traces = _matrix(gcamp["trace_array"])
    traces_raw = haskey(gcamp, "trace_array_original") ?
        _matrix(gcamp["trace_array_original"]) : fill(NaN, size(traces))

    beh = Dict{String,Vector{Float64}}()
    for key in ("velocity", "head_angle", "angular_velocity", "pumping")
        series = get(behavior, key, nothing)
        series === nothing || (beh[key] = _vector(series))
    end

    reversals = [Int(r[1]):Int(r[2]) for r in get(behavior, "reversal_events", [])]
    event_table = something(get(timing, "event", nothing), Dict())
    events = Dict{String,Vector{Int}}(String(k) => Int.(v) for (k, v) in event_table)

    labels = Dict{Int,NeuronLabel}()
    for (row, entry) in something(get(raw, "label", nothing), Dict())
        labels[parse(Int, row)] = NeuronLabel(
            String(entry["label"]),
            String(entry["neuron_class"]),
            String(something(entry["LR"], "undefined")),
            _float(entry["confidence"]),
        )
    end

    WormDataset(
        String(meta["paper_id"]),
        String(meta["uid"]),
        String.(meta["dataset_type"]),
        Float64(timing["mean_timestep"]),
        traces,
        traces_raw,
        beh,
        reversals,
        events,
        _gaps(_vector(timing["timestamp_confocal"])),
        labels,
    )
end

# ── Behaviour and velocity figures ──────────────────────────────────────────────────────

"""
    mean_state_velocities(recordings; pause_speed = 0.02)

Mean velocity (mm/s) over every frame of `recordings` faster than `pause_speed` forward, and
over every frame faster than `pause_speed` backward (negative): `(; forward, reverse)`. A
model's velocity readout is v = forward·x_F + reverse·x_R.
"""
function mean_state_velocities(recordings; pause_speed = 0.02)
    v = reduce(vcat, [r.behavior["velocity"] for r in recordings])
    (forward = mean(v[v .> pause_speed]), reverse = mean(v[v .< -pause_speed]))
end

"""The shaded behaviours of `behavior_figure`, with their colours."""
const BEHAVIOR_SHADES = [(:forward, "forward", FORWARD), (:reversal, "reversal", REVERSE), (:pause, "pause", PAUSE)]

"""
    shade_behaviors!(panel, t, states; alpha = 0.25)

A shaded span over each stretch of one behaviour in `states` (`:forward`, `:reversal` or
`:pause` at each time of `t`; anything else is left unshaded), half a sample wider each side.
"""
function shade_behaviors!(panel, t, states; alpha = 0.25)
    dt = t[2] - t[1]
    start = 1
    for k in eachindex(states)
        (k == length(states) || states[k+1] != states[k]) || continue
        shade = findfirst(s -> s[1] == states[k], BEHAVIOR_SHADES)
        isnothing(shade) || vspan!(panel, [t[start] - dt / 2, t[k] + dt / 2];
            color = BEHAVIOR_SHADES[shade][3], alpha, linewidth = 0, linealpha = 0, label = "")
        start = k + 1
    end
    panel
end

"""
    behavior_figure(t, traces, velocity; trace_labels, trace_colors, activity_states,
                    velocity_states = activity_states, velocity_t = t, zero_line = true,
                    activity_ylabel = "GCaMP fluorescence (z-score)", activity_title = "Neuronal activity",
                    velocity_ylabel = "velocity (mm/s)", velocity_title = "Freely-moving velocity",
                    xlabel = "time (s)", activity_ylims = :auto, size = (1100, 640))

A worm's (or a model's) activity and velocity: the rows of `traces` (activity) above
`velocity`, sharing the time axis, each panel shaded by its own behaviour at every sample
(see [`shade_behaviors!`](@ref)), with the legend in a row of its own above both. `velocity_t`
and `velocity_states` let the velocity come from a separate run sampled at other times.
"""
function behavior_figure(t, traces, velocity; trace_labels, trace_colors, activity_states,
    velocity_states = activity_states, velocity_t = t, zero_line = true,
    activity_ylabel = "GCaMP fluorescence (z-score)", activity_title = "Neuronal activity",
    velocity_ylabel = "velocity (mm/s)", velocity_title = "Freely-moving velocity",
    xlabel = "time (s)", activity_ylims = :auto, size = (1100, 640))
    activity_panel = plot(; ylabel = activity_ylabel, title = activity_title, titlefontsize = 10,
        legend = false, xformatter = _ -> "", ylims = activity_ylims)
    velocity_panel = plot(; xlabel, ylabel = velocity_ylabel, title = velocity_title,
        titlefontsize = 10, legend = false, bottom_margin = 4Plots.mm)
    shade_behaviors!(activity_panel, t, activity_states)
    shade_behaviors!(velocity_panel, velocity_t, velocity_states)

    zero_line && hline!(activity_panel, [0]; color = :gray, alpha = 0.5)
    for (k, trace) in enumerate(eachrow(traces))
        plot!(activity_panel, t, trace; color = trace_colors[k], linewidth = 2)
    end
    hline!(velocity_panel, [0]; color = :gray, alpha = 0.5)
    plot!(velocity_panel, velocity_t, velocity; color = INK, linewidth = 2)

    # The legend gets a row of its own above both panels, so it sits over the top panel's title.
    legend_row = plot(; framestyle = :none, legend = :top, legend_columns = -1, xlims = (0, 1), ylims = (0, 1))
    for (_, name, color) in BEHAVIOR_SHADES
        vspan!(legend_row, [NaN, NaN]; color, alpha = 0.25, linewidth = 0, label = name)
    end
    for (label, color) in zip(trace_labels, trace_colors)
        plot!(legend_row, [NaN]; color, linewidth = 2, label)
    end

    t_lims = (min(first(t), first(velocity_t)), max(last(t), last(velocity_t)))
    plot(legend_row, activity_panel, velocity_panel; layout = grid(3, 1; heights = [0.04, 0.48, 0.48]),
        link = :x, size, xlims = t_lims, left_margin = 4Plots.mm)
end

end # module
