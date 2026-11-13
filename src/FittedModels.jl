"""
    FittedModels

The fitted models and the data of the FittedModel notebook:

- the 2-neuron Forward/Reverse CTRNN fitted to Roberts et al.'s dwell times: its genome, field
  and outputs, the four behaviours read from which neurons are on (Forward on: Forward; both
  off: Pause 1; Reverse on: Reverse; both on: Pause 2), and the dwell scores of a genome;
- the dwell times of the fits (the 2-node CTRNN, `ctrnn_dwell_fit`, and the 4-state
  heteroclinic network's rates for a given noise level, `horchler_dwell_fit`, saved in
  `data/`), noisy runs of both sampled at a fixed rate with the behaviour at every sample
  (1 Forward, 2 Pause 1, 3 Reverse, 4 Pause 2), and Roberts et al.'s stochastic neuronal model
  (a 4-state Markov chain), run exactly and sampled at the same rate;
- the whole-brain GCaMP recordings published on wormwideweb.org (Flavell lab) and the figure of
  a worm's activity above its velocity. Each dataset JSON follows `dataset_json_readme.md` of
  github.com/flavell-lab/WormWideWeb: traces are neuron × frame, neuron rows and frame indices
  are 1-based, the NeuroPAL `label` table is keyed by neuron row, and the head-angle sign is
  already corrected.
"""
module FittedModels

using Random
using Statistics
using Plots
using JSON
using Downloads
using Evolutionary: Evolutionary
using Dates: Dates
using ..Common: VectorField, crossings, sde_trajectory, bouts
using ..Common: σ, ctrnn_system, DATA_DWELL, dwell_scores, mean_bout_lengths, dwell_fitness,
    squared_glv_rates, horchler_parameters, HORCHLER_FIT

export network, ctrnn, output, logit, start, beer_oscillator
export behavior, behavior_at
export FIT_TOL, behavior_switches, behavior_bouts, log_timing, dwell_evaluation
export ctrnn_rates
export ctrnn_dwell_fit, dwell_fit_network, ctrnn_noisy_run, ctrnn_labels, ctrnn_state_range, ctrnn_matched_noise
export read_horchler_fit, horchler_noisy_run, horchler_labels, horchler_mean_dwell, horchler_fit_file, horchler_dwell_fit
export ROBERTS_MODEL, roberts_rates, roberts_model_run
export bout_lengths
export NeuronLabel, WormDataset
export download_paper, dataset_files, load_dataset
export FORWARD, PAUSE, REVERSE
export mean_state_velocities, shade_behaviors!, behavior_figure

# ══ The 2-node CTRNN ═══════════════════════════════════════════════════════════════════

# ── The network ─────────────────────────────────────────────────────────────────────────

network(genome) = (W=reshape(genome[1:4], 2, 2), θ=genome[5:6], τ=exp.(genome[7:8]))

"""τ ⊙ ẏ = -y + W σ(y + θ), for a network from [`network`](@ref)."""
# |y*| ≤ Σⱼ |Wᵢⱼ| ≤ 2 × 20 within the fits' bounds.
const ctrnn = VectorField(ctrnn_system(net -> net.W; bias=net -> net.θ, tau=net -> net.τ)..., fill(-41.0, 2), fill(41.0, 2))

"""
    ctrnn_rates(y, net)

The field of [`ctrnn`](@ref), (−y + W σ(y + θ)) ./ τ, at a tuple `y` and as a tuple, for
`sde_trajectory`.
"""
ctrnn_rates(y, net) =
    ntuple(i -> (-y[i] + net.W[i, 1] * σ(y[1] + net.θ[1]) + net.W[i, 2] * σ(y[2] + net.θ[2])) / net.τ[i], Val(2))

output(y, net) = σ.(y .+ net.θ)
logit(o) = log(o / (1 - o))

# The nullclines in output space, each solved for the other neuron's output: ẏ_F = 0 is
# logit(o_F) - θ_F = W_FF o_F + W_FR o_R, and likewise for Reverse.
"""Every run starts with Forward on and Reverse off: outputs (0.9, 0.1)."""
start(net) = logit.([0.9, 0.1]) .- net.θ

"""
Beer's two-neuron oscillator as a genome: W_FF = W_RR = 4.5, Forward inhibiting Reverse (-1),
Reverse exciting Forward (1), θ = (-2.75, -1.75). Forward is Beer's neuron 1 and Reverse his
neuron 2, which makes it cycle Forward → Pause 1 (both off) → Reverse → Pause 2 (both on), the
order of Roberts et al. 2016. It spends 7.25 τ in each behaviour, so both time constants are
7.92 / 29, which makes its cycle as long as the data's.
"""
const beer_oscillator = [4.5, -1.0, 1.0, 4.5, -2.75, -1.75, fill(log(sum(DATA_DWELL.mean) / 29), 2)...]

# ── Behaviours ──────────────────────────────────────────────────────────────────────────

"""
The behaviour, from which neurons are on (output above ½), in the order of `DATA_DWELL`: 1
Forward alone, 2 neither (Pause 1), 3 Reverse alone, 4 both (Pause 2). This is Roberts et al.
2016's model (eLife, PMC4798983), whose fitted dwell times `DATA_DWELL` holds: Forward is
followed by the longer pause, with both command units off (their state X, 0.44 s), and
Reverse by the shorter one, with both on (their state Y, 0.21 s).
"""
behavior(on) = on[1] ? (on[2] ? 4 : 1) : (on[2] ? 3 : 2)
behavior_at(y, net) = behavior(y .+ net.θ .> 0)

"""Looser tolerances than `ODE_TOL`, for the fits' many runs."""
const FIT_TOL = (abstol=1e-8, reltol=1e-6)

"""
When the behaviour changes on a run of `duration` from `start`, and what it changes to;
the first entry is the start. A neuron switches when its state crosses -θ.
"""
function behavior_switches(net, duration)
    y0 = start(net)
    on = collect(y0 .+ net.θ .> 0)
    t, label = [0.0], [behavior(on)]
    for c in crossings(ctrnn, net, y0, duration; levels=-net.θ, tol=FIT_TOL)
        on[c.i] = c.up
        push!(t, c.t)
        push!(label, behavior(on))
    end
    return (; t, label)
end

"""The bouts that start after `offset` and end before `duration`."""
function behavior_bouts(net; duration=80.0, offset=20.0)
    s = behavior_switches(net, duration)
    return [(label=s.label[k], length=s.t[k+1] - s.t[k]) for k in 1:length(s.t)-1 if s.t[k] >= offset]
end

"""
The dwell fits' timing score, exp(−mean of log²(m_b / d_b)) over the behaviours, with m_b the
mean bout length of behaviour b and d_b the target's: smooth where a dwell time matches.
"""
log_timing(mean_bout, target) = exp(-sum(log.(mean_bout ./ target) .^ 2) / length(target))

"""
A genome's dwell scores against `DATA_DWELL` (`dwell_scores` with `log_timing`, from
`behavior_bouts`) and its mean dwell time in each behaviour: what `ctrnn_dwell_fit`
fits and how the fits are compared.
"""
function dwell_evaluation(genome)
    runs = behavior_bouts(network(genome))
    return (; dwell_scores(runs, DATA_DWELL.mean; timing=log_timing)..., dwell=mean_bout_lengths(runs, 4))
end

# ══ Dwell times of the fits and of Roberts et al.'s model ══════════════════════════════

# ── Model runs ──────────────────────────────────────────────────────────────────────────

"""
    ctrnn_dwell_fit(path; runs = 1:10)

Refits the 2-node CTRNN to Roberts et al.'s mean dwell times (`DATA_DWELL`) and writes the
fitted genomes to `path`, one per line, as in `data/CTRNN2NeuronOpt.txt`. CMA-ES (Evolutionary:
λ = 100, σ₀ = 1, at most 2000 generations, offspring evaluated on all threads) maximises
`dwell_fitness ∘ dwell_evaluation`, one run per optimiser seed in `runs`, each from Beer's
two-neuron oscillator (`beer_oscillator`), with weights and biases within ±20 and time
constants within [0.01, 20]. Takes hours.
"""
function ctrnn_dwell_fit(path; runs=1:10)
    objective(genome) = -dwell_fitness(dwell_evaluation(genome))
    box = Evolutionary.BoxConstraints([fill(-20.0, 6); fill(log(0.01), 2)], [fill(20.0, 6); fill(log(20.0), 2)])
    fitted = [Evolutionary.minimizer(Evolutionary.optimize(objective, box, beer_oscillator,
        Evolutionary.CMAES(λ=100, sigma0=1.0), Evolutionary.Options(iterations=2000, rng=Xoshiro(run), parallelization=:thread)))
              for run in runs]
    open(io -> foreach(g -> println(io, join(g, " ")), fitted), path, "w")
end

"""
The network of the fittest genome (`dwell_fitness ∘ dwell_evaluation`) in a file of 2-node
dwell fits such as `data/CTRNN2NeuronOpt.txt`, one genome per line.
"""
function dwell_fit_network(path)
    genomes = [parse.(Float64, split(line)) for line in eachline(path)]
    return network(argmax(g -> dwell_fitness(dwell_evaluation(g)), genomes))
end

"""
    ctrnn_noisy_run(net, seed; noise, rate, dt, transient, duration)

A run of the 2-node CTRNN `net` with additive noise of strength `noise` on its states (one
number for both, or one per state: `ctrnn_matched_noise`) (Euler–Maruyama, step `dt`), from
outputs drawn uniformly in [0.05, 0.95]² with `seed`:
`transient` dropped, then `duration` sampled `rate` times per unit time, as `(; t, Y)`.
"""
function ctrnn_noisy_run(net, seed; noise, rate, dt, transient, duration)
    rng = Xoshiro(seed)
    y0 = logit.(0.05 .+ 0.9 .* rand(rng, 2)) .- net.θ
    return sde_trajectory(ctrnn_rates, net, y0, transient + duration; noise, dt, rng, saveat=1 / rate, transient)
end

"""The CTRNN's behaviour at every sample of a run (`behavior_at`)."""
ctrnn_labels(Y, net) = [behavior_at(y, net) for y in eachcol(Y)]

"""
    ctrnn_state_range(net; dt)

The range (maximum − minimum) of each state yᵢ of the 2-node CTRNN `net` over its deterministic
limit cycle, as a tuple: Euler steps of `dt` from `start(net)`, 40 s dropped, then 80 s (about
ten cycles of the dwell fits).
"""
function ctrnn_state_range(net; dt)
    Y = sde_trajectory(ctrnn_rates, net, start(net), 120.0; noise=0.0, dt, transient=40.0).Y
    return Tuple(vec(maximum(Y; dims=2) .- minimum(Y; dims=2)))
end

"""
    ctrnn_matched_noise(noise, net; dt)

The noise on the 2-node CTRNN's states equivalent to additive noise `noise` on the 4-state
heteroclinic network: the same fraction of each variable's range. The network's saddles are at
uᵢ = 1 (β = 1), so each uᵢ spans 0 to 1 and `noise` is that fraction; each CTRNN state gets
`noise` × its range over the limit cycle (`ctrnn_state_range`). A tuple, one per state, for
`ctrnn_noisy_run`.
"""
ctrnn_matched_noise(noise, net; dt) = noise .* ctrnn_state_range(net; dt)

"""The α of a fit file such as `data/HorchlerFitNoise0.1.txt`: its last line that is not a `#` comment."""
read_horchler_fit(path) = parse.(Float64, split(last(filter(line -> !startswith(line, "#"), readlines(path)))))

"""
    horchler_noisy_run(α, seed; noise, rate, dt, transient, duration)

As [`ctrnn_noisy_run`](@ref), for the 4-state heteroclinic network with rates `α`
(`squared_glv_rates`, `horchler_parameters`), from u drawn uniformly in [0, 1]⁴.
"""
function horchler_noisy_run(α, seed; noise, rate, dt, transient, duration)
    rng = Xoshiro(seed)
    return sde_trajectory(squared_glv_rates, horchler_parameters(α), rand(rng, 4), transient + duration; noise, dt, rng,
        saveat=1 / rate, transient)
end

"""The heteroclinic network's state at every sample of a run: the node with the largest activity uᵢ²."""
horchler_labels(Y) = [argmax(abs.(u)) for u in eachcol(Y)]

"""
    horchler_mean_dwell(α, seeds; noise, rate, dt, transient, duration)

The heteroclinic network's mean dwell time in each state over runs with the `seeds` given
(`horchler_noisy_run`, in parallel), every bout but each run's first and last: NaN for a
state never visited.
"""
function horchler_mean_dwell(α, seeds; noise, rate, dt, transient, duration)
    series = Vector{Vector{Int}}(undef, length(seeds))
    Threads.@threads for k in eachindex(seeds)
        series[k] = horchler_labels(horchler_noisy_run(α, seeds[k]; noise, rate, dt, transient, duration).Y)
    end
    return [isempty(l) ? NaN : mean(l) for l in bout_lengths(series, 4; rate)]
end

"""Where the heteroclinic network's fit for additive noise `noise` is kept: `dir/HorchlerFitNoise<noise>.txt`."""
horchler_fit_file(dir, noise) = joinpath(dir, "HorchlerFitNoise$(noise).txt")

"""
    horchler_dwell_fit(; dir, noise, rate, dt, transient, duration, seeds = 1001:1064, refit = false, miss = 0.01)

The rates α of the 4-state heteroclinic network (Horchler construction, β = ν = 1) whose mean
dwell times with additive noise `noise` match Roberts et al.'s (`DATA_DWELL`). Each noise level
has its own fit, kept in `horchler_fit_file(dir, noise)`: read from there when it exists, and
otherwise (or with `refit = true`) fitted and written there, which takes minutes.

The fit minimises Σᵢ log(mᵢ / dᵢ)², with mᵢ from `horchler_mean_dwell` on the fixed `seeds`
(so the objective is deterministic; a state never visited scores 10⁶) and dᵢ = `DATA_DWELL.mean`,
by CMA-ES (Evolutionary: 8 offspring, 4 parents, initial step 0.2, at most 150 generations,
optimiser seed 1) on log α within [log 0.05, log 200]. It starts from `HORCHLER_FIT.α`, fitted
for noise 0.01, × ln(1/0.01) / ln(1/noise): the passage time past a saddle grows as
ln(1/noise) / α (Stone–Holmes), so the start keeps the dwell times for small noise. At noise
0.1 the start is `HORCHLER_FIT.α` / 2. The file records the run settings, the objective reached
and the date; a fit is looked up by its noise level alone. A fit whose objective is above
`miss` (0.01: about 5 % off per state) is returned with a warning: the network could not be
fitted at that noise (at noise 0.4 the best found is 30–80 % short).
"""
function horchler_dwell_fit(; dir, noise, rate, dt, transient, duration, seeds=1001:1064, refit=false, miss=0.01)
    file = horchler_fit_file(dir, noise)
    if refit || !isfile(file)
        0 < noise < 1 || error("horchler_dwell_fit: the starting point needs 0 < noise < 1, not $noise")
        settings = (; noise, rate, dt, transient, duration)
        function objective(logα)
            e = sum(abs2, log.(horchler_mean_dwell(exp.(logα), seeds; settings...) ./ DATA_DWELL.mean))
            return isfinite(e) ? e : 1e6
        end
        start = log.(HORCHLER_FIT.α .* log(1 / HORCHLER_FIT.noise) ./ log(1 / noise))
        cma = Evolutionary.optimize(objective, Evolutionary.BoxConstraints(fill(log(0.05), 4), fill(log(200.0), 4)),
            start, Evolutionary.CMAES(μ=4, λ=8, sigma0=0.2), Evolutionary.Options(iterations=150, rng=Xoshiro(1)))
        open(file, "w") do io
            println(io, "# 4-state heteroclinic GLV (Horchler construction, β = ν = 1), α fitted for additive noise $noise on u (FittedModels.horchler_dwell_fit).")
            println(io, "# Euler–Maruyama dt = $dt s, sampled at $rate Hz; seeds $seeds, each $duration s after $transient s.")
            println(io, "# Objective Σ log(mean dwell / DATA_DWELL.mean)² = $(Evolutionary.minimum(cma)), " *
                        "$(Evolutionary.iterations(cma)) generations of CMA-ES; fitted $(Dates.today()).")
            println(io, "# α: Forward, Pause 1, Reverse, Pause 2")
            println(io, join(exp.(Evolutionary.minimizer(cma)), " "))
        end
    end
    reached = match(r"Objective .* = ([0-9.eE+-]+),", read(file, String))
    if !isnothing(reached) && parse(Float64, reached[1]) > miss
        @warn "The heteroclinic network's fit for noise $noise misses Roberts et al.'s dwell times: " *
              "objective $(reached[1]) > $miss ($(basename(file)))."
    end
    return read_horchler_fit(file)
end

"""
Roberts et al. (2016)'s stochastic neuronal model with their Table 1 weights: each weight the
mean of its fits to the five wild-type cohorts (`rates and weights fits.xlsx`, sheet
"Tables 1&2", column B), and `A` the rate scale of those fits (the sheet's cell A1).
"""
const ROBERTS_MODEL = (; A=0.40215852749999997, hF=1.007, hR=1.082, wFF=-0.213, wRR=1.914, wFR=-5.403, wRF=-0.812)

# Which units are up in each state, (F, R): Forward, Pause 1 (X), Reverse, Pause 2 (Y).
const ROBERTS_UNITS = ((true, false), (false, false), (false, true), (true, true))

"""
    roberts_rates(weights = ROBERTS_MODEL)

The transition rates (Hz) of Roberts et al. (2016)'s stochastic neuronal model, as a 4 × 4
matrix from state to state in the order Forward, Pause 1 (X), Reverse, Pause 2 (Y). Two units,
F and R, are each up or down: F is up in Forward and Pause 2, R in Reverse and Pause 2. A unit
that is down goes up at rate A e^{xᵢ} and one that is up goes down at rate A e^{−xᵢ}, with input
x_F = h_F + w_FF s_F + w_RF s_R and x_R = h_R + w_RR s_R + w_FR s_F (sⱼ = 1 when unit j is up).
These are their eight rates (aXF = A e^{h_F}, …) under their 6-weight constraints
aYF = aRX aXR / aFY and aYR = aFX aXF / aRY (`wormtracks.ipf`); a cohort's fitted weights give
back its fitted rates to 1e-10.
"""
function roberts_rates(weights=ROBERTS_MODEL)
    (; A, hF, hR, wFF, wRR, wFR, wRF) = weights
    Q = zeros(4, 4)
    for (i, (f, r)) in enumerate(ROBERTS_UNITS)
        xF, xR = hF + wFF * f + wRF * r, hR + wRR * r + wFR * f
        Q[i, findfirst(==((!f, r)), ROBERTS_UNITS)] = A * exp(f ? -xF : xF)
        Q[i, findfirst(==((f, !r)), ROBERTS_UNITS)] = A * exp(r ? -xR : xR)
    end
    return Q
end

"""
    roberts_model_run(Q, seed; rate, transient, duration)

A run of the Markov chain with rates `Q` (`roberts_rates`), simulated exactly (exponential
waits, Gillespie), from a state drawn uniformly with `seed`: the state at every sample,
`transient` dropped, then `duration` sampled `rate` times per unit time.
"""
function roberts_model_run(Q, seed; rate, transient, duration)
    rng = Xoshiro(seed)
    leave = vec(sum(Q; dims=2))
    state = rand(rng, 1:4)
    next = randexp(rng) / leave[state]
    samples = transient .+ (0:round(Int, duration * rate)) ./ rate
    labels = Vector{Int}(undef, length(samples))
    for (k, t) in enumerate(samples)
        while next <= t
            u, j = rand(rng) * leave[state], 1
            while j < 4 && u > Q[state, j]
                u -= Q[state, j]
                j += 1
            end
            state = j
            next += randexp(rng) / leave[state]
        end
        labels[k] = state
    end
    return labels
end

# ── Bouts ───────────────────────────────────────────────────────────────────────────────

"""
    bout_lengths(label_series, n; rate)

The bouts of labels 1:n over several label series, one label per sample at `rate` samples per
unit time: n vectors of bout lengths, every bout but each series' first and last (cut short by
the ends of the series).
"""
function bout_lengths(label_series, n; rate)
    runs = [bouts(labels)[2:end-1] for labels in label_series]
    return [[b.length / rate for r in runs for b in r if b.label == i] for i in 1:n]
end

# ══ wormwideweb.org recordings ═════════════════════════════════════════════════════════

# Colours. Forward/Pause/Reverse follow the CTRNN notebooks; FALLBACK_COLORS are the
# remaining categorical slots (validated light-mode set), handed out in order to classes
# without an assigned colour. Orange is left out so it only ever means Pause.
const FORWARD = "#2a78d6"
const PAUSE = "#eb6834"
const REVERSE = "#1baf7a"
const INK = "#0b0b0b"
const INK_SECONDARY = "#52514e"

# ── Data ──────────────────────────────────────────────────────────────────────────────────

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
