"""
    FittedModels

The fitted models and the data of the FittedModel notebook:

- the dwell times of Roberts et al.'s (2016) wild-type worms in Forward → Pause 1 → Reverse →
  Pause 2: their Table 1 means, with SDs equal to those means as their Markov model's exponential
  dwell times imply (`roberts_target`);
- the 2-node CTRNN fitted to those means and SDs, with its behaviour read from which node's
  output is above ½ (Roberts et al.'s two command units). The fit starts just past the
  saddle-node on invariant circle (SNIC) bifurcation of an excitable network, on the side where
  the network oscillates, and prefers networks that oscillate without noise (`ctrnn_dwell_fit`,
  saved in `data/`);
- the 4-state heteroclinic network's rates and noise fitted to the same means and SDs, starting
  from Horchler et al.'s (2015) design rule (`horchler_design`, `horchler_dwell_fit`, saved in
  `data/`);
- a WormWideWeb recording (Atanas, Kim et al. 2023) for the activity-above-velocity figure: its
  forward and reverse command neurons (`COMMAND_CLASSES`) and its velocity, as published on
  wormwideweb.org (Flavell lab). Each dataset JSON follows `dataset_json_readme.md` of
  github.com/flavell-lab/WormWideWeb: traces are neuron × frame, neuron rows and frame indices are
  1-based, the NeuroPAL `label` table is keyed by neuron row, and the head-angle sign is already
  corrected.
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
using ..Common: VectorField, sde_trajectory, find_equilibria, σ, ctrnn_system, squared_glv_rates, horchler_parameters

export COMMAND_CLASSES, used_pieces, cluster_rows, has_command_neurons, zscore, cluster_activity
export CYCLE_STATES, PAUSE_SPEED, ROBERTS_RATE, ROBERTS_TABLE1_MEAN, roberts_target
export label_dwell, dwell_loss
export network, ctrnn, ctrnn_rates, noise_free_cycle, cycle_period, ctrnn_noisy_run, roberts_label, ctrnn_labels
export ctrnn_dwell, EXCITABLE_GENOME, snic_start, read_ctrnn_genome, read_ctrnn_fit, ctrnn_dwell_fit
export stone_holmes, horchler_period, horchler_design, read_horchler_fit, horchler_noisy_run, horchler_labels
export horchler_dwell, horchler_dwell_fit
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


# ══ Roberts et al.'s (2016) dwell times ═════════════════════════════════════════════════

const CYCLE_STATES = ["Forward", "Pause 1", "Reverse", "Pause 2"]

"""Speed (mm/s) below which a WormWideWeb worm is pausing (the example recording's shading)."""
const PAUSE_SPEED = 0.02

"""Roberts et al.'s video frame rate (Hz): both models are read every frame."""
const ROBERTS_RATE = 29.97

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

A run of the CTRNN with its own additive noise `net.noise` on the states (Euler–Maruyama, step
`dt`), from states drawn from a standard normal with `seed`: `transient` dropped, then `duration`
sampled every `sample_dt`, as `(; t, Y)`.
"""
function ctrnn_noisy_run(net, seed; sample_dt, dt, transient, duration)
    rng = Xoshiro(seed)
    y0 = randn(rng, 2)
    return sde_trajectory(ctrnn_rates, net, y0, transient + duration; noise=net.noise, dt, rng, saveat=sample_dt, transient)
end

"""
The behaviour of the CTRNN at a state `y`, from which node's output σ(y + θ) is above ½, as
Roberts et al.'s two command units: 1 Forward alone, 2 neither (Pause 1), 3 Reverse alone,
4 both (Pause 2).
"""
roberts_label(y, θ) = (f = y[1] + θ[1] > 0; r = y[2] + θ[2] > 0; f ? (r ? 4 : 1) : (r ? 3 : 2))

"""The CTRNN's behaviour at every sample of a run (`roberts_label`)."""
ctrnn_labels(Y, net) = [roberts_label(y, net.θ) for y in eachcol(Y)]

"""
    ctrnn_dwell(net; seeds = 1:4, sample_dt = 1 / ROBERTS_RATE, dt = 1 / (6 ROBERTS_RATE), transient = 50.0, duration = 1000.0)

The CTRNN's dwell times (`label_dwell` of `ctrnn_labels`) over runs with the `seeds` given
(`ctrnn_noisy_run`); NaN throughout when a run diverges.
"""
function ctrnn_dwell(net; seeds=1:4, sample_dt=1 / ROBERTS_RATE, dt=1 / (6ROBERTS_RATE), transient=50.0, duration=1000.0)
    series = map(seeds) do seed
        r = ctrnn_noisy_run(net, seed; sample_dt, dt, transient, duration)
        any(!isfinite, r.Y) ? Int[] : ctrnn_labels(r.Y, net)
    end
    any(isempty, series) && return (; mean=fill(NaN, 4), sd=fill(NaN, 4), n=zeros(Int, 4))
    return label_dwell(series, sample_dt)
end

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

"""The genome of a CTRNN fit file such as `data/CTRNNRobertsFit.txt`: its last line that is not a `#` comment."""
read_ctrnn_genome(path) = parse.(Float64, split(last(filter(line -> !startswith(line, "#"), readlines(path)))))

"""The network of a CTRNN fit file (`read_ctrnn_genome`)."""
read_ctrnn_fit(path) = network(read_ctrnn_genome(path))

"""
    ctrnn_dwell_fit(path, target; start = snic_start(EXCITABLE_GENOME, sum(target.mean)), runs = 2:5, iterations = 400,
                    penalty = 2.0, trace = false)

Refits the 2-node CTRNN to `target` (`roberts_target`) and writes the best genome to `path`, as
in `data/CTRNNRobertsFit.txt`. CMA-ES (Evolutionary: λ = 40, σ₀ = 0.3, at most `iterations`
generations) minimises `dwell_loss` of `ctrnn_dwell`, plus `penalty` for a network that does not
oscillate without noise (`noise_free_cycle`), once per optimiser seed in `runs`, each from the
genome `start`, with weights and biases within ±20, time constants within [0.05, 20] s and noise
within [0.01, 20]. Takes about an hour a run.

The saved fit was evolved in two stages: from the SNIC start, then from that fit
(`start = read_ctrnn_genome(path)`) once the target SDs became the Table 1 means.
"""
function ctrnn_dwell_fit(path, target; start=snic_start(EXCITABLE_GENOME, sum(target.mean)), runs=2:5, iterations=400,
    penalty=2.0, trace=false)
    loss(g) = (net = network(g); dwell_loss(ctrnn_dwell(net), target) + (isnothing(noise_free_cycle(net)) ? penalty : 0.0))
    box = Evolutionary.BoxConstraints([fill(-20.0, 6); fill(log(0.05), 2); fill(log(0.01), 2)],
        [fill(20.0, 6); fill(log(20.0), 2); fill(log(20.0), 2)])
    fits = map(runs) do run
        res = Evolutionary.optimize(loss, box, start, Evolutionary.CMAES(λ=40, sigma0=0.3),
            Evolutionary.Options(iterations=iterations, rng=Xoshiro(run), show_trace=trace, callback=_ -> (flush(stdout); false)))
        (; run, loss=Evolutionary.minimum(res), genome=Evolutionary.minimizer(res))
    end
    best = argmin(f -> f.loss, fits)
    open(path, "w") do io
        println(io, "# 2-node CTRNN fitted to Roberts et al.'s (2016) Table 1 mean dwell times, with SD = mean (FittedModels.ctrnn_dwell_fit).")
        println(io, "# Started from genome ", join(start, " "))
        println(io, "# Losses by optimiser seed: ", join(["$(f.run): $(f.loss)" for f in fits], ", "), "; fitted $(Dates.today()).")
        println(io, "# Genome: W_FF W_RF W_FR W_RR (column-major), θ_F θ_R, log τ_F log τ_R (s), log noise_F log noise_R")
        println(io, join(best.genome, " "))
    end
    return network(best.genome)
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

"""The rates α and noise levels of a fit file such as `data/HorchlerRobertsFit.txt`: its last line that is not a `#` comment, `α₁ … α₄ ε₁ … ε₄`."""
function read_horchler_fit(path)
    x = parse.(Float64, split(last(filter(line -> !startswith(line, "#"), readlines(path)))))
    return (; α=x[1:4], noise=x[5:8])
end

"""
    horchler_noisy_run(α, seed; noise, sample_dt, dt, transient, duration)

A run of the 4-state heteroclinic network with rates `α` (`squared_glv_rates`,
`horchler_parameters`) and additive noise on u, `noise[i]` on node i (Euler–Maruyama, step `dt`), from u drawn
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
    horchler_dwell(α, noise; seeds = 1001:1064, sample_dt = 1 / ROBERTS_RATE, dt = 1 / (20 ROBERTS_RATE), transient = 20.0, duration = 2000.0)

The heteroclinic network's dwell times (`label_dwell` of `horchler_labels`) over runs with the
`seeds` given (`horchler_noisy_run`, in parallel); NaN throughout when a run diverges.
"""
function horchler_dwell(α, noise; seeds=1001:1064, sample_dt=1 / ROBERTS_RATE, dt=1 / (20ROBERTS_RATE), transient=20.0,
    duration=2000.0)
    series = Vector{Vector{Int}}(undef, length(seeds))
    Threads.@threads for k in eachindex(seeds)
        r = horchler_noisy_run(α, seeds[k]; noise, sample_dt, dt, transient, duration)
        series[k] = any(!isfinite, r.Y) ? Int[] : horchler_labels(r.Y)
    end
    any(isempty, series) && return (; mean=fill(NaN, 4), sd=fill(NaN, 4), n=zeros(Int, 4))
    return label_dwell(series, sample_dt)
end

"""
    horchler_dwell_fit(; file, target, start = (; α = horchler_design(target.mean, 0.3), noise = fill(0.3, 4)),
                       refit = false, trace = false)

The rates α and per-node noise levels ε (on u) of the 4-state heteroclinic network (Horchler
construction, β = ν = 1) whose dwell times (`horchler_dwell`) match `target` (`roberts_target`),
as `(; α, noise)`. Per-node noise is what Horchler's node sizes βᵢ would add: with uᵢ = √βᵢ vᵢ the
network with βᵢ and noise ε is this one with noise ε / √βᵢ on node i. Read
from `file` when it exists; otherwise (or with `refit = true`) fitted and written there, which
takes about an hour.

The fit starts from `start`, by default Horchler et al.'s design (`horchler_design`) of the
target means at noise 0.3 on every node. The saved fit was evolved in two stages: from that
design with one noise level shared by the nodes, fitted to SDs from Roberts et al.'s decoded
bouts; then from that fit, with its noise on every node, with a noise level per node, once the
target SDs became the Table 1 means. CMA-ES
(Evolutionary: 8 offspring, 4 parents, initial step 0.3, at most 150 generations, optimiser seed
1) then minimises `dwell_loss` (deterministic: fixed seeds) over log α within [log 0.01, log 200]
and log ε within [log 0.001, 0].
"""
function horchler_dwell_fit(; file, target, start=(; α=horchler_design(target.mean, 0.3), noise=fill(0.3, 4)),
    refit=false, trace=false)
    if refit || !isfile(file)
        objective(x) = dwell_loss(horchler_dwell(exp.(x[1:4]), exp.(x[5:8])), target)
        x0 = [log.(start.α); log.(start.noise)]
        box = Evolutionary.BoxConstraints([fill(log(0.01), 4); fill(log(1e-3), 4)], [fill(log(200.0), 4); zeros(4)])
        cma = Evolutionary.optimize(objective, box, x0, Evolutionary.CMAES(μ=4, λ=8, sigma0=0.3),
            Evolutionary.Options(iterations=150, rng=Xoshiro(1), show_trace=trace, callback=_ -> (flush(stdout); false)))
        x = Evolutionary.minimizer(cma)
        open(file, "w") do io
            println(io, "# 4-state heteroclinic GLV (Horchler construction, β = ν = 1) fitted to Roberts et al.'s (2016) Table 1 mean dwell times, with SD = mean (FittedModels.horchler_dwell_fit),")
            println(io, "# from α = $(join(start.α, " ")), ε = $(join(start.noise, " ")). dwell_loss = $(Evolutionary.minimum(cma)), " *
                        "$(Evolutionary.iterations(cma)) generations of CMA-ES; fitted $(Dates.today()).")
            println(io, "# α: Forward, Pause 1, Reverse, Pause 2; then the noise ε on u of each node, same order")
            println(io, join(exp.(x), " "))
        end
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
