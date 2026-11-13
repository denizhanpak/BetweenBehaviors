"""
    SimpleModels

The 3-node rock–paper–scissors rings of the paper. Node i receives the weight `p` from its
predecessor and `n` from its successor; for p > n the cycle runs Rock → Paper → Scissors.

- `glv3`: ẋᵢ = xᵢ (1 − xᵢ + a_n xᵢ₊₁ + a_p xᵢ₋₁) + ε, parameters `(; ap, an)`.
- `ctrnn3`: ẏᵢ = −yᵢ + σ(yᵢ) + w_n σ(yᵢ₊₁) + w_p σ(yᵢ₋₁), parameters `(; wp, wn)`; outputs σ(y).

Also the metastability score of the paper's heatmaps (`heatmapsV2.py`) and the figures.
"""
module SimpleModels

using LinearAlgebra
using Random
using Statistics
using Plots
using StatsBase: rle
using OrdinaryDiffEqTsit5: Tsit5
using DynamicalSystemsBase: CoupledODEs, StateSpaceSet
using Attractors: BasinMapProximity, basins_of_attraction
using Meshing: isosurface, MarchingTetrahedra
using ..Common

export SIMPLE_NAMES, RPS_COLORS, glv3, glv3_squared_rates, ctrnn3, ctrnn3_rates
export GLV_SCAN, CTRNN_SCAN, metastability_scan, ends_on_cycle
export numbers_text, weights_text, time_series_plot
export STAR_COLORS, draw_stars!, metastability_heatmap, chart_axes, label_regions!, OVERLAY_COLORS, chart_overlay!
export basin_grid, basin_flows, basin_portrait, manifold_portrait, saddle_value_portrait

const INK = "#0b0b0b"
const INK_SECONDARY = "#52514e"
const SIMPLE_NAMES = ["Rock", "Paper", "Scissors"]
const RPS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

"""3 × 3 ring matrix (M[i, j]: j → i): `self` on the diagonal, `predecessor` from i − 1, `successor` from i + 1."""
function ring3(self, predecessor, successor)
    T = promote_type(typeof(self), typeof(predecessor), typeof(successor))
    return T[i == j ? self : j == mod1(i - 1, 3) ? predecessor : successor for i in 1:3, j in 1:3]
end

# ── Models ──────────────────────────────────────────────────────────────────────────────

# Immigration keeps the orthant invariant and turns the transcritical lines into folds.
const IMMIGRATION = 1e-6

"""The GLV field at x as a tuple, parameters `(; ap, an)`."""
function glv3_rates(x, q)
    f, p, r = x
    (; ap, an) = q
    return (f * (1 - f + an * p + ap * r) + IMMIGRATION,
        p * (1 - p + an * r + ap * f) + IMMIGRATION,
        r * (1 - r + an * f + ap * p) + IMMIGRATION)
end

"""
The GLV in u with x = u², without immigration: u̇ᵢ = (uᵢ/2)(1 − xᵢ + a_n xᵢ₊₁ + a_p xᵢ₋₁). The
same flow, but x stays non-negative under noise on u.
"""
function glv3_squared_rates(u, q)
    f, p, r = u .^ 2
    (; ap, an) = q
    return (u[1] / 2 * (1 - f + an * p + ap * r), u[2] / 2 * (1 - p + an * r + ap * f), u[3] / 2 * (1 - r + an * f + ap * p))
end

"""The 3-node GLV (Jacobian diag(1 + Mx) + diag(x) M, M = ring3(−1, a_p, a_n))."""
const glv3 = VectorField((dx, x, q) -> (dx .= glv3_rates(x, q)),
    (x, q) -> (M = ring3(-1, q.ap, q.an); Diagonal(1 .+ M * x) + Diagonal(x) * M), fill(-4.1, 3), fill(4.1, 3))

"""The CTRNN field at y as a tuple, parameters `(; wp, wn)`."""
function ctrnn3_rates(y, q)
    f, p, r = y
    (; wp, wn) = q
    sf, sp, sr = σ(f), σ(p), σ(r)
    return (-f + sf + wn * sp + wp * sr, -p + sp + wn * sr + wp * sf, -r + sr + wn * sf + wp * sp)
end

"""The 3-node CTRNN, W = ring3(1, w_p, w_n). Equilibria have yᵢ ∈ [w_p + w_n, 1], inside the box for w ∈ [−20, 0]."""
const ctrnn3 = VectorField(ctrnn_system(q -> ring3(1, q.wp, q.wn))..., fill(-41.0, 3), fill(2.0, 3))

# ── Metastability (heatmapsV2.py) ───────────────────────────────────────────────────────

"""
Per cell, `runs` Euler–Maruyama runs of `duration` (step `dt`) sampled every `every` from
`transient`. `heatmapsV2.py` sampled every 1; 0.1 keeps quick hand-overs from merging bouts.
"""
const SCAN = (runs=20, duration=500.0, transient=200.0, dt=0.1, every=0.1)

"""
How each model is scanned: the field, a random start, the outputs scored, the weight range and
the parameters at (p, n). The GLV runs in u = √x with the noise on u (`heatmapsV2.py` puts it on
x and reflects at 0); starts are x uniform on [0, 1]³. CTRNN starts are N(0.5, 1) per node.
"""
const GLV_SCAN = (rates=glv3_squared_rates, start=rng -> sqrt.(rand(rng, 3)), output=u -> u^2,
    range=range(-3.0, 0.0; length=100), params=(p, n) -> (ap=p, an=n))
const CTRNN_SCAN = (rates=ctrnn3_rates, start=rng -> randn(rng, 3) .+ 0.5, output=σ,
    range=range(-20.0, 0.0; length=100), params=(p, n) -> (wp=p, wn=n))

"""
The test of `heatmapsV2.py` on outputs `O` (3 × samples): a sample is dominant when one node
exceeds the other two together; the run passes with at least 5 dominant and 5 transition bouts
and a mean dominant bout at least twice the mean transition bout.
"""
function is_metastable(O)
    any(isnan, O) && return false
    dominant, lengths = rle([2 * maximum(o) > sum(o) for o in eachcol(O)])
    held, between = lengths[dominant], lengths[.!dominant]
    return length(held) >= 5 && length(between) >= 5 && mean(held) >= 2 * mean(between)
end

"""
    metastability_scan(model; noise, seed=1, cells=length(model.range))

The fraction of `SCAN.runs` noisy runs that are metastable at each of `cells` × `cells` weights,
threaded, each cell with its own seeded stream: `(; p, n, score)`, `score[i, j]` at `p[j]`, `n[i]`.
"""
function metastability_scan(model; noise, seed=1, cells=length(model.range))
    values = range(first(model.range), last(model.range); length=cells)
    score = zeros(cells, cells)
    Threads.@threads for cell in CartesianIndices(score)
        i, j = Tuple(cell)
        q, rng = model.params(values[j], values[i]), Xoshiro(hash((seed, i, j)))
        score[i, j] = count(1:SCAN.runs) do _
            run = sde_trajectory(model.rates, q, model.start(rng), SCAN.duration; noise, dt=SCAN.dt, rng,
                saveat=SCAN.every, transient=SCAN.transient)
            is_metastable(model.output.(run.Y))
        end / SCAN.runs
    end
    return (; p=collect(values), n=collect(values), score)
end

"""
    ends_on_cycle(vf, q; output=identity, duration=3000.0)

Whether the run from just off the symmetric equilibrium (the one with the largest outputs) is
still moving over its second half and ends on no stable equilibrium.
"""
function ends_on_cycle(vf, q; output=identity, duration=3000.0)
    eqs = find_equilibria(vf, q)
    stable = [e.y for e in eqs if e.stability == :stable]
    symmetric = argmax(e -> sum(output.(e.y)), filter(e -> is_symmetric(e.y), eqs))
    run = trajectory(vf, q, symmetric.y .+ [0.01, 0.0, -0.01], duration; saveat=0.1)
    second_half = run.Y[:, end÷2:end]
    moving = maximum(maximum(second_half; dims=2) .- minimum(second_half; dims=2)) > 1e-3
    return moving && !any(y -> norm(run.Y[:, end] - y) < 1e-3, stable)
end

# ── Text and time series ────────────────────────────────────────────────────────────────

"""A number as text, with a true minus sign and without a trailing `.0`."""
number_text(x) = replace(string(isinteger(x) ? Int(x) : x), "-" => "−")
numbers_text(v) = "(" * join(number_text.(v), ", ") * ")"

"""The two weights of `q` as text, e.g. `(aₚ, aₙ) = (−1.1, −1.7)`."""
function weights_text(q)
    letter = haskey(q, :ap) ? "a" : "w"
    p, n = haskey(q, :ap) ? (q.ap, q.an) : (q.wp, q.wn)
    return "($(letter)ₚ, $(letter)ₙ) = " * numbers_text((p, n))
end

"""
The outputs `O` (3 × samples) against `t`, shaded by the node holding the run: a node takes over
when it exceeds the other two together and holds until another does.
"""
function time_series_plot(t, O; title, ylabel)
    plt = plot(; title, ylabel, xlabel="time (model units)", size=PLANE_SIZE, xlims=extrema(t), titlefontsize=10,
        legend=:outertop, legend_columns=2, legendfontsize=8, left_margin=4Plots.mm, bottom_margin=4Plots.mm,
        top_margin=3Plots.mm)
    half = (t[2] - t[1]) / 2
    held = accumulate((held, o) -> 2 * maximum(o) > sum(o) ? argmax(o) : held, eachcol(O); init=0)
    states, lengths = rle(held)
    for (state, n, stop) in zip(states, lengths, cumsum(lengths))
        state > 0 && vspan!(plt, [t[stop-n+1] - half, t[stop] + half]; color=RPS_COLORS[state],
            alpha=0.15, linewidth=0, linealpha=0, label=false)
    end
    for i in 1:3
        plot!(plt, t, O[i, :]; color=RPS_COLORS[i], linewidth=2, label=SIMPLE_NAMES[i])
    end
    vspan!(plt, [NaN, NaN]; color=:gray, alpha=0.15, linewidth=0, linealpha=0, label="shading: dominant node")
    return plt
end

# ── Parameter plane: heatmap and chart on the same axes ─────────────────────────────────

const PLANE_SIZE = (640, 560)
const STAR_COLORS = (excitable="#ffffff", cycle="#e87ba4")

"""A star at each setting in `stars`, `(; p, n, label, color)`."""
function draw_stars!(plt, stars)
    for s in stars
        scatter!(plt, [s.p], [s.n]; markershape=:star5, markersize=14, markercolor=s.color,
            markerstrokecolor=INK, markerstrokewidth=1.2, label=s.label)
    end
    return plt
end

"""The score of a [`metastability_scan`](@ref) over the (p, n) plane in the colour map `color`."""
function metastability_heatmap(scan; xlabel, ylabel, title="", legend=:topright, color=:hot)
    return heatmap(scan.p, scan.n, scan.score; color, clims=(0, 1), colorbar_title="fraction of $(SCAN.runs) runs metastable",
        xlabel, ylabel, title, aspect_ratio=:equal, xlims=extrema(scan.p), ylims=extrema(scan.n), size=PLANE_SIZE,
        legend, legendfontsize=7, titlefontsize=10, left_margin=2Plots.mm)
end

"""The chart's curves and `extra` ones, with points outside the chart's box set to NaN."""
function clipped(chart, extra=[])
    inside(a, b) = chart.range1[1] <= a <= chart.range1[2] && chart.range2[1] <= b <= chart.range2[2]
    return [(; c.type, x=[inside(a, b) ? a : NaN for (a, b) in zip(c.x, c.y)], c.y) for c in [chart.curves; extra]]
end

"""A [`bifurcation_chart`](@ref) on axes that line up with [`metastability_heatmap`](@ref) (room for its colorbar)."""
function chart_axes(chart; xlabel, ylabel, title="")
    colorbar_room = 0.1 * maximum(PLANE_SIZE) * Plots.px + 4Plots.mm   # GR's colorbar width and title
    plt = plot(; aspect_ratio=:equal, size=PLANE_SIZE, legend=:topright, legendfontsize=7, title,
        titlefontsize=10, left_margin=2Plots.mm, right_margin=colorbar_room)
    seen = Set{Symbol}()
    for c in clipped(chart)
        plot!(plt, c.x, c.y; color=(c.type == :hopf ? CYCLE_COLOR : :black), linewidth=2,
            label=(c.type in seen ? false : (c.type == :hopf ? "Hopf" : "fold")))
        push!(seen, c.type)
    end
    return plot!(plt; xlims=chart.range1, ylims=chart.range2, xlabel, ylabel)
end

"""Write each region's name: `regions` holds `(; x, y, text)`, optionally `rotation` and `halign`."""
function label_regions!(plt, regions)
    for r in regions
        annotate!(plt, r.x, r.y, Plots.text(r.text, 8, INK_SECONDARY, get(r, :halign, :center);
            rotation=get(r, :rotation, 0)))
    end
    return plt
end

"""Colour-vision-safe overlay colours: cividis map; Okabe–Ito vermillion, blue and bluish green curves on white halos."""
const OVERLAY_COLORS = (map=:cividis, hopf="#d55e00", fold="#0072b2", heteroclinic="#009e73", halo="#ffffff")

"""
    chart_overlay!(plt, chart; scan, labels=[], extra=[])

The chart's curves, and `extra` ones `(; type=:heteroclinic, x, y)`, over the heatmap of `scan`:
Hopf solid, fold dashed, heteroclinic dotted, on halos. Each label `(; type, near)` names the
nearest curve of that kind along it, on `near`'s side, in ink or white by the cells beneath.
"""
function chart_overlay!(plt, chart; scan, labels=[], extra=[])
    colors = OVERLAY_COLORS
    curves = clipped(chart, extra)
    names = Dict(:hopf => "Hopf", :fold => "fold", :heteroclinic => "heteroclinic")
    styles = Dict(:hopf => :solid, :fold => :dash, :heteroclinic => :dot)
    for c in curves   # halos first, so no halo covers another curve
        plot!(plt, c.x, c.y; color=colors.halo, linewidth=6, label=false)
    end
    for type in (:hopf, :fold, :heteroclinic), (k, c) in enumerate(filter(c -> c.type == type, curves))
        plot!(plt, c.x, c.y; color=colors[type], linewidth=2.5, linestyle=styles[type], label=(k == 1 ? names[type] : false))
    end

    span = chart.range1[2] - chart.range1[1]
    window = max(1, length(scan.p) ÷ 60)   # cells either side averaged for the text colour
    for l in labels
        points = [(c, k) for c in curves if c.type == l.type for k in eachindex(c.x) if !isnan(c.x[k])]
        isempty(points) && error("no $(l.type) curve inside the chart for the label near $(l.near)")
        c, k = argmin(((c, k),) -> hypot(c.x[k] - l.near[1], c.y[k] - l.near[2]), points)
        a, b = max(k - 1, 1), min(k + 1, length(c.x))
        dx, dy = c.x[b] - c.x[a], c.y[b] - c.y[a]
        rotation = 90 - mod(90 - atand(dy, dx), 180)   # upright
        rotation < -80 && (rotation += 180)            # a steep label reads upward
        normal = [-dy, dx] ./ hypot(dx, dy)
        side = sign(dot(normal, [l.near[1] - c.x[k], l.near[2] - c.y[k]]))
        x, y = [c.x[k], c.y[k]] .+ (side == 0 ? 1 : side) * 0.045 * span .* normal
        i, j = argmin(abs.(scan.n .- y)), argmin(abs.(scan.p .- x))
        cells = scan.score[max(i - window, 1):min(i + window, end), max(j - window, 1):min(j + window, end)]
        annotate!(plt, x, y, Plots.text(names[l.type], 10, mean(cells) >= 0.5 ? INK : "#ffffff", :center; rotation))
    end
    return plt
end

# ── Basins in 3-D ───────────────────────────────────────────────────────────────────────

"""
    basin_grid(vf, q, attractors; values, inverse=identity, cycle=nothing, ε=0.01)

The basins on the grid `values`³ (Attractors.jl, `BasinMapProximity`): `(; values, labels)`,
labels 1–3 for `attractors`, 4 for the limit cycle `cycle` (one column per sample) and 0 for
none. Grid and attractors are in display coordinates; `inverse` maps them to states coordinate
by coordinate.
"""
function basin_grid(vf, q, attractors; values, inverse=identity, cycle=nothing, ε=0.01)
    system = CoupledODEs((dy, y, p, t) -> vf.f!(dy, y, p), inverse(collect(attractors[1])), q;
        diffeq=(alg=Tsit5(), ODE_TOL...))
    sets = Dict(k => StateSpaceSet([inverse(collect(a))]) for (k, a) in enumerate(attractors))
    isnothing(cycle) || (sets[4] = StateSpaceSet([inverse(collect(c)) for c in eachcol(cycle)]))
    axis = inverse(collect(values))
    found = basins_of_attraction(BasinMapProximity(system, sets; ε), (axis, axis, axis); show_progress=false)
    return (; values, labels=max.(found.basins, 0))
end

"""Mean of each entry of `F` and its (up to 26) neighbours."""
smooth3d(F) = [mean(F[max(i - 1, 1):min(i + 1, end), max(j - 1, 1):min(j + 1, end), max(k - 1, 1):min(k + 1, end)])
               for i in axes(F, 1), j in axes(F, 2), k in axes(F, 3)]

const BASIN_COLORS = [RPS_COLORS; CYCLE_COLOR]
attractor_name(k) = k == 4 ? "the limit cycle" : SIMPLE_NAMES[k]

"""Runs of `duration` from every `every`-th grid point of a [`basin_grid`](@ref) that reaches an attractor: `(; Y, label)`, Y in display coordinates."""
function basin_flows(vf, q, b; every=10, duration=60.0, samples=300, coords=identity, inverse=identity)
    picks = [I for I in CartesianIndices(b.labels)[1:every:end, 1:every:end, 1:every:end] if b.labels[I] > 0]
    flows = Vector{Any}(undef, length(picks))
    Threads.@threads for m in eachindex(picks)
        I = picks[m]
        run = trajectory(vf, q, inverse(b.values[collect(Tuple(I))]), duration; saveat=duration / samples)
        flows[m] = (; Y=reduce(hcat, coords.(eachcol(run.Y))), label=Int(b.labels[I]))
    end
    return flows
end

"""
    basin_portrait(b, eqs; coords=identity, lims, flows=[], boundary_color=nothing, path=nothing,
                   path_label="trajectory", cycle=nothing, title="", camera=(120, 30), axis_note="")

The basins of `b` in 3-D: each of Rock's, Paper's and Scissors' basin boundaries as a translucent
surface (where the neighbour-averaged indicator crosses ½, by Meshing.jl's marching tetrahedra), in its colour or all in
`boundary_color`, shaded by facing; the `flows` in the colour they reach; the equilibria `eqs`;
the `cycle` and the run `path` (display coordinates) in the cube `lims`³.
"""
function basin_portrait(b, eqs; coords=identity, lims, flows=[], boundary_color=nothing,
        path=nothing, path_label="trajectory", cycle=nothing, title="", camera=(120, 30), axis_note="")
    plt = plot(; xlabel=SIMPLE_NAMES[1] * axis_note, ylabel=SIMPLE_NAMES[2] * axis_note,
        zlabel=SIMPLE_NAMES[3] * axis_note, xlims=lims, ylims=lims, zlims=lims, camera, title,
        titlefontsize=10, legend=:topright, legendfontsize=7, size=(760, 700))

    light = normalize([0.4, -0.5, 0.75])   # triangles shaded in 6 steps by how squarely they face it
    for k in 1:3
        vertices, triangles = isosurface(smooth3d(Float64.(b.labels .== k)), MarchingTetrahedra(iso=0.5), b.values, b.values, b.values)
        isempty(triangles) && continue
        P = reduce(hcat, collect.(vertices))
        shade(t) = (n = cross(P[:, t[2]] - P[:, t[1]], P[:, t[3]] - P[:, t[1]]); norm(n) == 0 ? 1.0 : abs(dot(n, light)) / norm(n))
        steps = min.(floor.(Int, shade.(triangles) .* 6), 5)
        base = Plots.plot_color(something(boundary_color, BASIN_COLORS[k]))
        for s in 0:5
            group = triangles[steps .== s]
            isempty(group) && continue
            f = 0.55 + 0.45 * (s + 0.5) / 6
            mesh3d!(plt, P[1, :], P[2, :], P[3, :]; connections=group, fillalpha=0.3, linewidth=0,
                fillcolor=Plots.RGB(f * base.r, f * base.g, f * base.b), label=false)
        end
        isnothing(boundary_color) && scatter!(plt, [NaN], [NaN], [NaN]; markershape=:square, markersize=8,
            markercolor=BASIN_COLORS[k], markeralpha=0.6, markerstrokewidth=0, label="boundary of $(attractor_name(k))'s basin")
    end
    isnothing(boundary_color) || scatter!(plt, [NaN], [NaN], [NaN]; markershape=:square, markersize=8,
        markercolor=boundary_color, markeralpha=0.6, markerstrokewidth=0, label="basin boundary")

    for k in sort(unique(f.label for f in flows)), (m, f) in enumerate(filter(f -> f.label == k, flows))
        draw_path!(plt, f.Y; color=BASIN_COLORS[k], linewidth=1, alpha=0.6, label=(m == 1 ? "runs to $(attractor_name(k))" : false))
    end
    isnothing(cycle) || draw_path!(plt, [cycle cycle[:, 1]]; color=CYCLE_COLOR, linewidth=3, label="limit cycle")
    if !isnothing(path)
        draw_path!(plt, path; color=INK, linewidth=2, label=path_label)
        scatter!(plt, [path[1, 1]], [path[2, 1]], [path[3, 1]]; markershape=:rect, markercolor=:white,
            markerstrokecolor=INK, markersize=5, label="its start")
    end
    in_view(e) = all(lims[1] - 0.01 .<= coords(e.y) .<= lims[2] + 0.01)
    draw_equilibria!(plt, filter(e -> e.stability != :stable && in_view(e), eqs); coords)
    for e in filter(e -> e.stability == :stable && in_view(e), eqs)
        c = coords(e.y)
        scatter!(plt, [c[1]], [c[2]], [c[3]]; markershape=:circle, markercolor=:black, markersize=8,
            markerstrokewidth=3, markerstrokecolor=is_symmetric(e.y) ? "#9a9a9a" : RPS_COLORS[argmax(c)], label=false)
    end
    scatter!(plt, [NaN], [NaN], [NaN]; markershape=:circle, markercolor=:black, markerstrokecolor=:black,
        markersize=6, label="stable equilibrium")
    return plt
end

# ── 3-D portrait with the saddles' manifolds ────────────────────────────────────────────

"""
    manifold_portrait(vf, q, eqs; lims, path, path_label, title, camera, duration, stable_surface,
                      unstable_surface, surface_alpha, ring_every, symmetric_name)

3-D portrait in the cube `lims` (widened by 1 %, as GR does not clip 3-D lines): the equilibria
`eqs`, the run `path`, and every saddle's manifolds in the colour of its node (grey for the
symmetric saddle, `symmetric_name`): 1-D ones as lines (`saddle_manifolds` over `duration`;
unstable solid, stable dashed), 2-D ones as translucent meshes (`manifold_surface` with the
keywords `stable_surface` or `unstable_surface`, or a function of the saddle returning them).
"""
function manifold_portrait(vf, q, eqs; lims, path, path_label, title, camera, duration, stable_surface,
        unstable_surface, surface_alpha, ring_every, symmetric_name)
    slack = 0.01 * (lims[2] - lims[1])   # GLV equilibria sit up to ~1e-5 outside the orthant
    shown(c) = all(lims[1] - slack .<= c .<= lims[2] + slack) ? c : fill(NaN, length(c))
    color_of(e) = is_symmetric(e.y) ? "#9a9a9a" : RPS_COLORS[argmax(e.y)]
    shown_lims = (lims[1] - slack, lims[2] + slack)   # GR drops markers beyond the limits
    plt = plot(; xlabel=SIMPLE_NAMES[1], ylabel=SIMPLE_NAMES[2], zlabel=SIMPLE_NAMES[3], xlims=shown_lims,
        ylims=shown_lims, zlims=shown_lims, xticks=:auto, yticks=:auto, zticks=:auto, camera, title,
        titlefontsize=10, legend=:topright, legendfontsize=7, size=(760, 700))

    # Legend entries name whose manifolds they are, with swatches darker than the meshes.
    whose(e) = is_symmetric(e.y) ? "the $symmetric_name" : "the " * join(SIMPLE_NAMES, ", ", " and ") * " saddles"
    swatch(e) = is_symmetric(e.y) ? "#9a9a9a" : INK_SECONDARY
    entries = Dict{Tuple{Int,Symbol,String},Any}()

    saddles = filter(e -> e.stability == :saddle, eqs)
    for e in saddles
        unstable = count(l -> real(l) > 0, e.eigenvalues)
        for direction in (:stable, :unstable)
            (direction == :unstable ? unstable : 3 - unstable) == 2 || continue
            options = direction == :stable ? stable_surface : unstable_surface
            xyz = manifold_surface(vf, e, q; direction, (options isa Function ? options(e) : options)...)
            C = [shown([xyz[1][k, j], xyz[2][k, j], xyz[3][k, j]]) for k in axes(xyz[1], 1), j in axes(xyz[1], 2)]
            plot!(plt, mesh_lines([getindex.(C, i) for i in 1:3], ring_every)...; color=color_of(e), alpha=surface_alpha,
                linewidth=0.6, label=false)
            entries[(2, direction, whose(e))] = (; color=swatch(e), linestyle=:solid, linewidth=2, alpha=0.8)
        end
    end

    # The path first, thin and light, so what is drawn after it stays visible.
    draw_path!(plt, path; coords=shown, color=INK_SECONDARY, linewidth=1, alpha=0.6, linestyle=:dot, label=path_label)
    for e in saddles, m in saddle_manifolds(vf, e, q; duration)
        style = m.direction == :unstable ? (; linewidth=2.5, linestyle=:solid) : (; linewidth=1.5, linestyle=:dash)
        draw_path!(plt, m.Y; coords=shown, color=color_of(e), style..., label=false)
        entries[(1, m.direction, whose(e))] = (; color=swatch(e), style..., alpha=1.0)
    end
    for key in sort(collect(keys(entries)); by=k -> (-k[1], k[2] == :stable, k[3]))
        dimension, direction, owner = key
        plural = startswith(owner, "the $symmetric_name") ? "manifold" : "manifolds"
        plot!(plt, [NaN], [NaN], [NaN]; entries[key]...,
            label="$dimension-D $direction $plural of $owner$(dimension == 2 ? " (mesh)" : "")")
    end
    draw_equilibria!(plt, filter(e -> !any(isnan, shown(e.y)), eqs);
        fill=e -> e.stability == :saddle ? color_of(e) : STABILITY_MARKERS[e.stability].color)
    return plt
end

# ── Saddle values: the planar saddle of GLV_Simplee_Saddle.nb ───────────────────────────

"""ẋ = μx − x³, ẏ = −y, parameters `(; μ)`: the origin is a saddle with saddle value −1/μ."""
const toy_saddle = VectorField((dz, z, q) -> (dz[1] = q.μ * z[1] - z[1]^3; dz[2] = -z[2]; dz),
    (z, q) -> [q.μ-3z[1]^2 0.0; 0.0 -1.0], fill(-0.5, 2), fill(0.5, 2))

"""Shades of the runs, light to dark as the saddle value falls."""
const SADDLE_VALUE_GREYS = ["#a3a3a3", "#8c8c8c", "#767676", "#616161", "#4c4c4c", "#383838", "#252525", "#111111"]

"""An arrowhead on the path (X, Y) at `fraction` of its length, along it (against it if `reverse`)."""
function arrow_along!(plt, X, Y, fraction; reverse=false, kwargs...)
    s = [0.0; cumsum(hypot.(diff(X), diff(Y)))]
    i = clamp(searchsortedfirst(s, fraction * s[end]), 2, length(s))
    from, to = reverse ? (i, i - 1) : (i - 1, i)
    return plot!(plt, [X[from], X[to]], [Y[from], Y[to]]; arrow=Plots.arrow(:closed, :head), label=false, kwargs...)
end

"""
    saddle_value_portrait(; title="")

`Saddle.png` of `GLV_Simplee_Saddle.nb`: the toy saddle's manifolds (at μ = 2.5) and, for each
μ in 0.1:0.4:2.9, the run from (0.03, 0.5) labelled by its saddle value and shaded darker the
lower it is, each with an arrowhead.
"""
function saddle_value_portrait(; title="")
    μs, start, duration = 0.1:0.4:2.9, [0.03, 0.5], 100.0
    lims = (-0.5, 0.5)
    inside(Y) = [all(lims[1] .<= y .<= lims[2]) ? y : [NaN, NaN] for y in eachcol(Y)]
    plt = plot(; aspect_ratio=:equal, xlims=lims, ylims=lims, size=PLANE_SIZE, xlabel="x", ylabel="y", title,
        titlefontsize=10, framestyle=:box, legend=:bottomleft, legendfontsize=8, left_margin=2Plots.mm)

    q = (; μ=2.5)
    seen = Set{Symbol}()
    for m in saddle_manifolds(toy_saddle, classify(toy_saddle, [0.0, 0.0], q), q; duration=20.0)
        unstable = m.direction == :unstable
        color = unstable ? Common.UNSTABLE_MANIFOLD_COLOR : Common.STABLE_MANIFOLD_COLOR
        P = inside(m.Y)
        X, Y = first.(P), last.(P)
        plot!(plt, X, Y; color, linewidth=1.5, linestyle=unstable ? :solid : :dash,
            label=(m.direction in seen ? false : "$(m.direction) manifold"))
        push!(seen, m.direction)
        kept = .!isnan.(X)
        # Flow runs out along the unstable manifold and in along the stable one (traced backward).
        arrow_along!(plt, X[kept], Y[kept], 0.55; reverse=!unstable, color, linewidth=1.5)
    end

    shades = Dict(zip(sort(collect(μs); rev=true), SADDLE_VALUE_GREYS))
    for μ in μs
        V = saddle_value(classify(toy_saddle, [0.0, 0.0], (; μ)))
        P = inside(trajectory(toy_saddle, (; μ), start, duration; saveat=duration / 4000).Y)
        X, Y = first.(P), last.(P)
        kept = .!isnan.(X)
        plot!(plt, X, Y; color=shades[μ], linewidth=2, label="μ = $(number_text(μ)): V = $(number_text(round(V; sigdigits=3)))")
        arrow_along!(plt, X[kept], Y[kept], 0.6; color=shades[μ], linewidth=2)
    end
    scatter!(plt, [0.0], [0.0]; markershape=STABILITY_MARKERS.saddle.shape, markercolor=STABILITY_MARKERS.saddle.color,
        markerstrokecolor=:black, markersize=8, label="saddle (origin)")
    return plt
end

end # module
