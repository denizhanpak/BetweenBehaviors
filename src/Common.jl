"""
    Common

What all three notebooks share:

- model-independent tools on BifurcationKit, the SciML stack and DynamicalSystems.jl: runs,
  equilibria, invariant manifolds, limit cycles, two-parameter bifurcation charts, complex
  "ghost" equilibria, and their plots;
- the pieces of the models: the logistic σ and the CTRNN field, the data's dwell times and the
  fitness the fits score against them, and Horchler et al.'s 4-state heteroclinic network
  (Forward → Pause 1 → Reverse → Pause 2).
"""
module Common

using BifurcationKit
const BK = BifurcationKit
using SciMLBase: ODEProblem, SDEProblem, NonlinearProblem, NonlinearFunction, DiscreteCallback,
    VectorContinuousCallback, CallbackSet, terminate!, successful_retcode, solve
using OrdinaryDiffEqTsit5: Tsit5
using StochasticDiffEq: EM
using DiffEqNoiseProcess: WienerProcess
using StaticArrays: SVector, SMatrix
using DynamicalSystemsBase: CoupledODEs
using ChaosTools: fixedpoints, interval, estimate_period
using StatsBase: rle
using NonlinearSolve: NewtonRaphson
using Plots
using LinearAlgebra
using Base.CoreLogging: with_logger, NullLogger
using Random

export VectorField, ODE_TOL, trajectory, sde_trajectory, crossings, bouts
export find_equilibria, classify, saddle_value, is_symmetric
export saddle_manifolds, manifold_surface, limit_cycle, speed_minima
export continue_equilibria, bifurcation_chart, complex_equilibrium, cyclic_complex_equilibria
export STABILITY_MARKERS, CYCLE_COLOR, draw_equilibria!, draw_path!, mesh_lines, phase_figure
export σ, dσ, ctrnn_system
export DATA_DWELL, mean_bout_lengths, dwell_scores, dwell_fitness
export squared_glv_rates, horchler_parameters, HORCHLER_FIT

# ══ Dynamical-systems tools ════════════════════════════════════════════════════════════

const ODE_TOL = (abstol=1e-10, reltol=1e-10)

"""Run `f()` with BifurcationKit's log messages, prints and displays suppressed."""
function quietly(f)
    pushdisplay(TextDisplay(devnull))
    try
        return with_logger(NullLogger()) do
            redirect_stdout(f, devnull)
        end
    finally
        popdisplay()
    end
end

# ── System and runs ─────────────────────────────────────────────────────────────────────

"""
    VectorField(f!, jacobian, lower, upper)

ẏ = f(y, p): the in-place field `f!(dy, y, p)`, its Jacobian `jacobian(y, p)` and the
state-space box [`lower`, `upper`].
"""
struct VectorField{F,J}
    f!::F
    jacobian::J
    lower::Vector{Float64}
    upper::Vector{Float64}
end

dimension(vf::VectorField) = length(vf.lower)
vector_field(vf::VectorField, y, p) = (dy = similar(y); vf.f!(dy, y, p); dy)
ode_problem(vf::VectorField, y0, tspan, p) = ODEProblem((dy, y, p, t) -> vf.f!(dy, y, p), y0, tspan, p)

"""Stop a run once it leaves the box padded by 1."""
leave_box(vf::VectorField) = DiscreteCallback((y, t, integrator) -> any(y .< vf.lower .- 1) || any(y .> vf.upper .+ 1), terminate!)

"""
    trajectory(vf, p, y0, duration; saveat)

The run from `y0` for `duration` (backward if negative), `(; t, Y)` with a column of `Y` per
time; it stops early if it leaves the box.
"""
function trajectory(vf::VectorField, p, y0, duration; saveat=abs(duration) / 1000)
    sol = solve(ode_problem(vf, y0, (0.0, duration), p), Tsit5(); saveat, callback=leave_box(vf), ODE_TOL...)
    return (; t=sol.t, Y=reduce(hcat, sol.u))
end

"""
    sde_trajectory(rates, p, y0, duration; noise, dt, seed=nothing, rng=Xoshiro(seed), saveat=dt, transient=0.0)

dy = rates(y, p) dt + noise dW by StochasticDiffEq's Euler–Maruyama (`EM()`, step `dt`), noise
from `rng`. `rates` takes and returns a tuple. `(; t, Y)` sampled every `saveat` from `transient`.
"""
function sde_trajectory(rates, p, y0, duration; noise, dt, seed=nothing, rng=Xoshiro(seed), saveat=dt, transient=0.0)
    u0 = SVector(y0...)
    σ = noise isa Number ? zero(u0) .+ noise : SVector(noise...)
    W = WienerProcess(0.0, zero(u0), nothing; rng, reseed=false)
    problem = SDEProblem((y, p, t) -> SVector(rates(Tuple(y), p)), (y, p, t) -> σ, u0, (0.0, duration), p; noise=W)
    sol = solve(problem, EM(); dt, saveat=transient:saveat:duration)
    return (; t=sol.t, Y=reduce(hcat, sol.u))
end

"""
    crossings(vf, p, y0, duration; levels, tol=ODE_TOL)

Every time a variable of the run from `y0` crosses its level, `(; t, i, up)` in time order:
y[i] crosses `levels[i]` at `t`, rising if `up`.
"""
function crossings(vf::VectorField, p, y0, duration; levels, tol=ODE_TOL)
    found = NamedTuple{(:t, :i, :up),Tuple{Float64,Int,Bool}}[]
    record!(integrator, flags) = foreach(i -> push!(found, (t=integrator.t, i, up=flags[i] > 0)), findall(!iszero, flags))
    events = VectorContinuousCallback((out, y, t, integrator) -> (out .= y .- levels), record!, dimension(vf);
        save_positions=(false, false))
    solve(ode_problem(vf, y0, (0.0, duration), p), Tsit5(); callback=CallbackSet(events, leave_box(vf)),
        save_everystep=false, tol...)
    return found
end

"""Run-length encoding as one `(; label, length)` per run of equal labels (the dwell bouts)."""
bouts(labels) = [(; label, length) for (label, length) in zip(rle(labels)...)]

# ── Equilibria ──────────────────────────────────────────────────────────────────────────

"""
    classify(vf, y, p)

The equilibrium `y` with its `stability` (`:stable`, `:unstable`, `:saddle` or
`:nonhyperbolic`), `eigenvalues` and `eigenvectors`.
"""
function classify(vf::VectorField, y, p)
    F = eigen(vf.jacobian(y, p))
    re = real.(F.values)
    stability = any(abs.(re) .< 1e-8) ? :nonhyperbolic : all(re .< 0) ? :stable : all(re .> 0) ? :unstable : :saddle
    return (; y, stability, eigenvalues=F.values, eigenvectors=F.vectors)
end

"""Every equilibrium in the box (ChaosTools' interval root finding), each [`classify`](@ref)'d."""
function find_equilibria(vf::VectorField, p)
    n = dimension(vf)
    rule(y, p, t) = (dy = Vector{eltype(y)}(undef, n); vf.f!(dy, collect(y), p); SVector{n}(dy))
    jacobian(y, p, t) = SMatrix{n,n}(vf.jacobian(collect(y), p))
    system = CoupledODEs(rule, SVector{n}((vf.lower .+ vf.upper) ./ 2), p)
    roots, _, _ = fixedpoints(system, interval.(vf.lower, vf.upper), jacobian)
    return [classify(vf, collect(y), p) for y in roots]
end

"""Largest negative over largest positive real part of the eigenvalues; NaN unless a saddle."""
function saddle_value(eq)
    re = real.(eq.eigenvalues)
    pos, neg = filter(>(0), re), filter(<(0), re)
    return isempty(pos) || isempty(neg) ? NaN : maximum(neg) / maximum(pos)
end

"""True for a point on the diagonal (all variables equal)."""
is_symmetric(y; tol=1e-6) = maximum(y) - minimum(y) < tol

# ── Invariant manifolds ─────────────────────────────────────────────────────────────────

"""
    saddle_manifolds(vf, eq, p; duration=75.0, h=1e-4)

The 1-D manifolds of a saddle: runs from y* ± h v forward along a lone unstable eigenvector and
backward along a lone stable one, `(; direction, t, Y)` each.
"""
function saddle_manifolds(vf::VectorField, eq, p; duration=75.0, h=1e-4)
    manifolds = []
    for (direction, sign) in ((:unstable, 1), (:stable, -1))
        selected = findall(l -> sign * real(l) > 0, eq.eigenvalues)
        length(selected) == 1 || continue
        v = real.(eq.eigenvectors[:, only(selected)])
        for s in (1, -1)
            push!(manifolds, (; direction, trajectory(vf, p, eq.y .+ s * h .* v, sign * duration)...))
        end
    end
    return manifolds
end

"""
    manifold_surface(vf, eq, p; direction=:stable, n=48, radius=1e-3, duration=50.0, samples=120, starts=nothing)

A 2-D manifold of `eq` as one matrix per variable (row: a run, column: a time; NaN after a run
leaves the box), for [`mesh_lines`](@ref). The runs start on a closed curve of `radius` around
`eq` in the eigenspace (a circle for a complex pair, a square in eigenvector coordinates for two
real eigenvalues), backward for the stable manifold. With `starts`, points on the manifold, they
run from those toward `eq` instead and are cut at their closest approach; NaN starts break rings.
"""
function manifold_surface(vf::VectorField, eq, p; direction=:stable, n=48, radius=1e-3, duration=50.0, samples=120,
        starts=nothing)
    trace(y0, tend) = (Y = trajectory(vf, p, y0, tend; saveat=duration / (samples - 1)).Y[:, 1:min(end, samples)];
        [Y fill(NaN, size(Y, 1), samples - size(Y, 2))])
    as_matrices(rows) = [permutedims(reduce(hcat, [r[i, :] for r in rows])) for i in 1:dimension(vf)]
    function trace_to_eq(y0)
        any(isnan, y0) && return fill(NaN, dimension(vf), samples)
        Y = trace(y0, direction == :stable ? duration : -duration)
        distance = [any(isnan, c) ? Inf : norm(c - eq.y) for c in eachcol(Y)]
        Y[:, argmin(distance)+1:end] .= NaN
        return Y
    end
    isnothing(starts) || return as_matrices(trace_to_eq.(starts))

    λ, V = eq.eigenvalues, eq.eigenvectors
    selected = direction == :stable ? findall(l -> real(l) < 0, λ) : findall(l -> real(l) > 0, λ)
    length(selected) == 2 || error("the $direction manifold is $(length(selected))-dimensional, not 2-D")
    v = V[:, selected[1]]
    complex_pair = abs(imag(λ[selected[1]])) > 1e-10
    a, b = normalize.(complex_pair ? (real.(v), imag.(v)) : (real.(v), real.(V[:, selected[2]])))

    function around(s)   # coefficients of (a, b) around the closed curve, s ∈ [0, 1)
        complex_pair && return (cos(2π * s), sin(2π * s))
        side, u = divrem(4s, 1)
        u = 2u - 1
        return side == 0 ? (1.0, u) : side == 1 ? (-u, 1.0) : side == 2 ? (-1.0, -u) : (u, -1.0)
    end
    tend = direction == :stable ? -duration : duration
    return as_matrices([trace(eq.y .+ radius .* (c[1] .* a .+ c[2] .* b), tend)
                        for c in around.(mod.(range(0, 1; length=n + 1), 1))])
end

# ── Limit cycles ────────────────────────────────────────────────────────────────────────

"""
    limit_cycle(vf, p, y_guess; settle=150.0, n_samples=1000)

Settle onto the cycle by a run, then solve for it by collocation (BifurcationKit), the period
guessed from midline crossings (ChaosTools' `estimate_period`). `(; p, T, Y)` with `Y` sampled
uniformly in time over one period.
"""
function limit_cycle(vf::VectorField, p, y_guess; settle=150.0, n_samples=1000)
    y_on = solve(ode_problem(vf, y_guess, (0.0, settle), p), Tsit5(); ODE_TOL...).u[end]
    run = trajectory(vf, p, y_on, 200.0; saveat=200.0 / 2e4)
    x = run.Y[1, :]
    T_guess = estimate_period(x, :zc, run.t; line=(maximum(x) + minimum(x)) / 2)
    guess = solve(ode_problem(vf, y_on, (0.0, 1.3T_guess), p), Tsit5(); ODE_TOL...)
    coll, x_guess = generate_ci_problem(Collocation(40, 4; jacobian=BK.DenseAnalyticalInplace()),
        bifurcation_problem(vf, p, y_on, PropertyLens{first(keys(p))}()), guess, T_guess)
    po = BK.newton(coll, x_guess, NewtonPar(tol=1e-11, max_iterations=30))
    po.converged || error("the limit cycle did not converge")
    T = getperiod(coll, po.u, p)
    x0 = BK.get_periodic_orbit(coll, po.u, p)[:, 1]
    samples = solve(ode_problem(vf, x0, (0.0, T), p), Tsit5(); saveat=range(0, T; length=n_samples + 1), ODE_TOL...).u
    return (; p, T, Y=reduce(hcat, samples[1:end-1]))
end

"""
    speed_minima(vf, lc; observe=identity, scale=y -> one.(y))

Bottlenecks of a cycle: for each variable, the deepest local minimum of the speed ‖scale(y) ⊙ ẏ‖
where that variable dominates `observe(y)`. Rotated so phase 0 is variable 1's bottleneck:
`(; O, speed, phase, minima)`, each minimum `(; node, phase, o, speed)`.
"""
function speed_minima(vf::VectorField, lc; observe=identity, scale=y -> one.(y))
    speed = [norm(scale(y) .* vector_field(vf, y, lc.p)) for y in eachcol(lc.Y)]
    n = length(speed)
    deepest = Dict{Int,Int}()
    for k in 1:n
        speed[k] < speed[mod1(k - 1, n)] && speed[k] < speed[mod1(k + 1, n)] || continue
        node = argmax(observe(lc.Y[:, k]))
        (!haskey(deepest, node) || speed[k] < speed[deepest[node]]) && (deepest[node] = k)
    end
    shift = deepest[1] - 1
    speed = circshift(speed, -shift)
    phase = 2π .* (0:n-1) ./ n
    O = reduce(hcat, observe.(eachcol(circshift(lc.Y, (0, -shift)))))
    at(k) = mod1(k - shift, n)
    minima = sort([(; node, phase=phase[at(k)], o=O[:, at(k)], speed=speed[at(k)]) for (node, k) in deepest]; by=m -> m.node)
    return (; O, speed, phase, minima)
end

# ── Continuation (BifurcationKit) ───────────────────────────────────────────────────────

bifurcation_problem(vf::VectorField, p, y, lens) =
    BifurcationProblem((dy, y, p) -> vf.f!(dy, y, p), y, p, lens; J=(y, p) -> vf.jacobian(y, p))

"""Continue the equilibrium `y` in `lens` over [p_min, p_max] both ways, detecting folds, branch and Hopf points."""
function continue_equilibria(vf::VectorField, p, y, lens; p_min, p_max)
    opts = ContinuationPar(; p_min, p_max, ds=0.01, dsmax=0.05, detect_bifurcation=3, nev=dimension(vf),
        n_inversion=8, max_steps=3000, detect_loop=true)
    return quietly(() -> continuation(bifurcation_problem(vf, p, y, lens), PALC(), opts; bothside=true, normC=norminf))
end

"""Continue the fold, branch or Hopf point `index` of `br` in `lens2` (bounded to [p_min, p_max])."""
function codim2_curve(br, index, lens2; p_min, p_max)
    opts = ContinuationPar(br.contparams; detect_bifurcation=1, max_steps=1500, p_min, p_max, detect_loop=false)
    return continuation(br, index, lens2, opts; bothside=true, normC=norminf, detect_codim2_bifurcation=2,
        update_minaug_every_step=1)
end

lens_name(::PropertyLens{name}) where {name} = name

"""
    bifurcation_chart(vf, p, lens1, lens2; range1, range2, starts1, starts2)

At each value of one parameter in `starts1` (for `lens1`) or `starts2`, continue every
equilibrium at the middle of the other's range across it, and every fold, branch and Hopf point
found into a curve in both parameters (once per curve). `(; curves, range1, range2)`, each curve
`(; type, x, y)` with `type` `:hopf` or `:fold`.
"""
function bifurcation_chart(vf::VectorField, p, lens1, lens2; range1, range2, starts1, starts2)
    curves = []
    on_curve(type, a, b) = any(c -> c.type == type && minimum(hypot.(c.x .- a, c.y .- b)) < 0.02, curves)
    sweeps = ((lens=lens1, other=lens2, range=range1, other_range=range2, fixed=starts2),
        (lens=lens2, other=lens1, range=range2, other_range=range1, fixed=starts1))
    quietly() do
        for s in sweeps, value in s.fixed
            q = set(set(p, s.other, value), s.lens, sum(s.range) / 2)
            for eq in find_equilibria(vf, q)
                br = continue_equilibria(vf, q, eq.y, s.lens; p_min=s.range[1], p_max=s.range[2])
                for (index, sp) in enumerate(br.specialpoint)
                    sp.type in (:fold, :bp, :hopf) || continue
                    type = sp.type == :hopf ? :hopf : :fold
                    a, b = s.lens === lens1 ? (sp.param, value) : (value, sp.param)
                    on_curve(type, a, b) && continue
                    curve = try
                        codim2_curve(br, index, s.other; p_min=s.other_range[1], p_max=s.other_range[2])
                    catch
                        continue   # a start Newton cannot follow
                    end
                    push!(curves, (; type, x=getproperty(curve.branch, lens_name(lens1)),
                        y=getproperty(curve.branch, lens_name(lens2))))
                end
            end
        end
    end
    return (; curves, range1, range2)
end

# ── Complex equilibria (ghosts) ─────────────────────────────────────────────────────────

"""Equilibrium off the real axis by complex Newton from `guess`; errors if it fails or lands on a real one."""
function complex_equilibrium(vf::VectorField, p, guess)
    f = NonlinearFunction((dy, y, p) -> vf.f!(dy, y, p); jac=(J, y, p) -> (J .= vf.jacobian(y, p)))
    root = solve(NonlinearProblem(f, ComplexF64.(guess), p), NewtonRaphson(); abstol=1e-13)
    successful_retcode(root) || error("complex equilibrium solve failed")
    norm(imag.(root.u)) > 1e-8 || error("converged to a real equilibrium")
    return root.u
end

"""A complex equilibrium and its cyclic images, `(; node, y, o)` sorted by the `node` dominating Re observe(y)."""
function cyclic_complex_equilibria(vf::VectorField, p, guess; observe=identity)
    root = complex_equilibrium(vf, p, guess)
    images = [(o = observe(y); (node=argmax(real.(o)), y, o)) for y in (circshift(root, k) for k in 0:length(root)-1)]
    return sort(images; by=g -> g.node)
end

# ── Plotting ────────────────────────────────────────────────────────────────────────────

const STABILITY_MARKERS = (
    stable=(shape=:circle, color=:black, label="stable"),
    unstable=(shape=:circle, color=:white, label="unstable"),
    saddle=(shape=:diamond, color="#9a9a9a", label="saddle"),
    nonhyperbolic=(shape=:star5, color=:black, label="nonhyperbolic"),
)
const PHASE_TICKS = ([0, π/2, π, 3π/2, 2π], ["0", "π/2", "π", "3π/2", "2π"])
const CYCLE_COLOR = "#4a3aa7"
const UNSTABLE_MANIFOLD_COLOR = "#e34948"
const STABLE_MANIFOLD_COLOR = "#2a78d6"

"""Plot the columns of `Y`, mapped through `coords`, as a path."""
function draw_path!(plt, Y; coords=identity, kwargs...)
    C = reduce(hcat, coords.(eachcol(Y)))
    return plot!(plt, (C[i, :] for i in axes(C, 1))...; kwargs...)
end

"""Equilibria as markers, shape and fill by stability (`fill(e)` overrides the fill)."""
function draw_equilibria!(plt, eqs; coords=identity, fill=nothing)
    seen = Set{Symbol}()
    for e in eqs
        m = STABILITY_MARKERS[e.stability]
        c = coords(e.y)
        scatter!(plt, ([c[i]] for i in eachindex(c))...; markershape=m.shape, markercolor=isnothing(fill) ? m.color : fill(e),
            markerstrokecolor=:black, markersize=6, label=(e.stability in seen ? false : m.label))
        push!(seen, e.stability)
    end
    return plt
end

"""A surface's coordinate matrices as NaN-separated lines (GR ignores surface transparency): every row and every `ring_every`-th column."""
function mesh_lines(xyz, ring_every)
    lines = [[[M[k, :] for k in axes(M, 1)]; [M[:, k] for k in 1:ring_every:size(M, 2)]] for M in xyz]
    return [reduce(vcat, [[l; NaN] for l in L]) for L in lines]
end

"""
    phase_figure(cyc, distances, labels, colors; ylabel, title)

The speed around a cycle ([`speed_minima`](@ref)) with its bottlenecks in `colors[node]`, the
`distances` from reference points against phase, and the speed as a colour strip.
"""
function phase_figure(cyc, distances, labels, colors; ylabel, title)
    fig = plot(layout=grid(3, 1; heights=[0.32, 0.53, 0.15]), size=(850, 700), link=:x, xlims=(0, 2π))
    blank = (PHASE_TICKS[1], fill("", 5))
    plot!(fig[1], cyc.phase, cyc.speed; color=:black, linewidth=2, label=false, ylabel="speed", title, xticks=blank,
        yscale=:identity)
    for m in cyc.minima
        scatter!(fig[1], [m.phase], [m.speed]; color=colors[m.node], markerstrokecolor=:black, markersize=6, label=false)
    end
    for (d, label, color) in zip(distances, labels, colors)
        plot!(fig[2], cyc.phase, d; color, linewidth=2, label)
    end
    d_min, d_max = extrema(reduce(vcat, distances))
    plot!(fig[2]; ylabel, ylims=(d_min - 0.02, d_max + 0.3 * (d_max - d_min)), legend=:top,
        legend_columns=length(labels), xticks=blank)
    heatmap!(fig[3], cyc.phase, [0, 1], repeat(cyc.speed', 2); color=cgrad(["#ece9f8", CYCLE_COLOR, "#1f1752"]),
        colorbar=false, yticks=([0.5], ["speed"]), xticks=PHASE_TICKS, xlabel="phase")
    return fig
end

# ══ Model pieces: σ, the CTRNN field, dwell times and Horchler's network ═══════════════

"""Logistic σ(x) = 1 / (1 + e⁻ˣ); `dσ` is its derivative σ(1 − σ)."""
σ(x) = 1 / (1 + exp(-x))
dσ(x) = σ(x) * (1 - σ(x))

"""
    ctrnn_system(weights; bias=p -> 0, tau=p -> 1)

The CTRNN τ ⊙ ẏ = −y + W σ(y + θ) and its Jacobian, as the `(f!, jacobian)` of a `VectorField`;
`weights(p)`, `bias(p)` and `tau(p)` give W (W[i, j]: j → i), θ and τ. States may be complex.
"""
function ctrnn_system(weights; bias=p -> 0, tau=p -> 1)
    f!(dy, y, p) = (dy .= (.-y .+ weights(p) * σ.(y .+ bias(p))) ./ tau(p))
    jacobian(y, p) = (-I + weights(p) * Diagonal(dσ.(y .+ bias(p)))) ./ tau(p)
    return f!, jacobian
end

"""The data's mean dwell time in each behaviour (s), in the order of the locomotion cycle."""
const DATA_DWELL = (names=["Forward", "Pause 1", "Reverse", "Pause 2"], mean=[5.329, 0.441, 1.945, 0.208])

"""Mean length of the bouts `runs` (`(; label, length)`) of each label 1:n; 0 for a label that never occurs."""
mean_bout_lengths(runs, n) = [
    (lengths = [r.length for r in runs if r.label == i]; isempty(lengths) ? 0.0 : sum(lengths) / length(lengths))
    for i in 1:n
]

"""
    dwell_scores(runs, target; timing)

How well bouts `runs` follow the cycle 1 → 2 → … → n → 1 with mean dwell times `target`: the
fraction of hand-overs to the next label (`order`), `timing(mean_bout, target)` and the fraction
of labels that occur (`balance`).
"""
function dwell_scores(runs, target; timing)
    n = length(target)
    handovers = zip(runs[1:end-1], runs[2:end])
    order = isempty(handovers) ? 0.0 : count(((a, b),) -> b.label == mod1(a.label + 1, n), handovers) / length(handovers)
    mean_bout = mean_bout_lengths(runs, n)
    return (; order, timing=timing(mean_bout, target), balance=count(>(0), mean_bout) / n)
end

"""The fitness the fits maximise: 2 order + 5 timing + 0.5 balance, at most 7.5."""
dwell_fitness(s) = 2 * s.order + 5 * s.timing + 0.5 * s.balance

"""
    horchler_parameters(α)

The parameters `(; α, ρ)` of Horchler et al.'s (2015) heteroclinic cycle 1 → 2 → … → n → 1 with
β = ν = 1: ρᵢᵢ = αᵢ, ρᵢⱼ = αᵢ − αⱼ for i = j + 1 (mod n) and αᵢ + αⱼ otherwise, so the saddle of
state j is unstable toward j + 1 only, with eigenvalue αⱼ.
"""
function horchler_parameters(α)
    n = length(α)
    return (; α, ρ=[i == j ? α[i] : i == mod1(j + 1, n) ? α[i] - α[j] : α[i] + α[j] for i in 1:n, j in 1:n])
end

"""u̇ = u ⊙ (α − ρ u²), Horchler's network in u (activities x = u²), at a tuple and as a tuple."""
squared_glv_rates(u::NTuple{N}, q) where {N} =
    ntuple(i -> u[i] * (q.α[i] - sum(q.ρ[i, j] * u[j]^2 for j in 1:N)), Val(N))

"""The rates α fitted for noise 0.01 (`HorchlerParameters4StateGLV.jl`), the start of the refit."""
const HORCHLER_FIT = (α=[0.937797, 14.4316, 2.85567, 32.5532], noise=0.01)

end # module
