"""
    SimpleModels

The 3-node rock–paper–scissors rings of the paper. Node i is inhibited with strength `p` by its
predecessor and `n` by its successor, both positive as in the paper's equations; for p < n the
cycle runs Rock → Paper → Scissors.

- `glv3`: the paper's eq. (5) in amplitudes cᵢ, ċᵢ = cᵢ (1 − cᵢ² − a_n cᵢ₊₁² − a_p cᵢ₋₁²) + √(2D) Yᵢ(t)
  (`glv3_squared_rates`), and in activities xᵢ = cᵢ², ẋᵢ = 2xᵢ (1 − xᵢ − a_n xᵢ₊₁ − a_p xᵢ₋₁) + ε for the
  deterministic analysis (`glv3`), parameters `(; ap, an)`.
- `ctrnn3`: ẏᵢ = −yᵢ + σ(yᵢ) − w_n σ(yᵢ₊₁) − w_p σ(yᵢ₋₁), parameters `(; wp, wn)`; outputs σ(y).

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

export SIMPLE_NAMES, RPS_COLORS, OFF_DIAGONAL, glv3, glv3_squared_rates, ctrnn3, ctrnn3_rates
export GLV_SCAN, CTRNN_SCAN, metastability_scan, ends_on_cycle
export numbers_text, weights_text, time_series_plot
export setting_stars, draw_stars!, metastability_heatmap, chart_axes, label_regions!, OVERLAY_COLORS, chart_overlay!
export scan_panel, cube_axes, basin_grid, basin_flows, basin_portrait, manifold_portrait, saddle_value_portrait

const INK_SECONDARY = "#52514e"
const SIMPLE_NAMES = ["Rock", "Paper", "Scissors"]
const RPS_COLORS = [PALETTE.blue, PALETTE.orange, PALETTE.green]

"""3 × 3 ring matrix (M[i, j]: j → i): `self` on the diagonal, `predecessor` from i − 1, `successor` from i + 1."""
function ring3(self, predecessor, successor)
    T = promote_type(typeof(self), typeof(predecessor), typeof(successor))
    return T[i == j ? self : j == mod1(i - 1, 3) ? predecessor : successor for i in 1:3, j in 1:3]
end

# ── Models ──────────────────────────────────────────────────────────────────────────────

"""A small step off the diagonal, to leave the symmetric equilibrium."""
const OFF_DIAGONAL = [0.01, 0.0, -0.01]

# Immigration keeps the orthant invariant and turns the transcritical lines into folds.
const IMMIGRATION = 1e-6

"""
The GLV field in activities x = c² as a tuple, parameters `(; ap, an)`: eq. (5) without noise,
ẋᵢ = 2xᵢ (1 − xᵢ − a_n xᵢ₊₁ − a_p xᵢ₋₁), plus immigration.
"""
function glv3_rates(x, q)
    f, p, r = x
    (; ap, an) = q
    return (2f * (1 - f - an * p - ap * r) + IMMIGRATION,
        2p * (1 - p - an * r - ap * f) + IMMIGRATION,
        2r * (1 - r - an * f - ap * p) + IMMIGRATION)
end

"""
The GLV in amplitudes u = c (activities x = u²), the drift of the paper's eq. (5) without
immigration: u̇ᵢ = uᵢ (1 − xᵢ − a_n xᵢ₊₁ − a_p xᵢ₋₁). The same flow as [`glv3`](@ref), but x stays
non-negative under the noise √(2D) Yᵢ(t) on u.
"""
function glv3_squared_rates(u, q)
    f, p, r = u .^ 2
    (; ap, an) = q
    return (u[1] * (1 - f - an * p - ap * r), u[2] * (1 - p - an * r - ap * f), u[3] * (1 - r - an * f - ap * p))
end

"""The 3-node GLV in activities (Jacobian 2 (diag(1 + Mx) + diag(x) M), M = −ring3(1, a_p, a_n))."""
const glv3 = VectorField((dx, x, q) -> (dx .= glv3_rates(x, q)),
    (x, q) -> (M = -ring3(1, q.ap, q.an); 2 * (Diagonal(1 .+ M * x) + Diagonal(x) * M)), fill(-4.1, 3), fill(4.1, 3))

"""The CTRNN field at y as a tuple, parameters `(; wp, wn)`."""
function ctrnn3_rates(y, q)
    f, p, r = y
    (; wp, wn) = q
    sf, sp, sr = σ(f), σ(p), σ(r)
    return (-f + sf - wn * sp - wp * sr, -p + sp - wn * sr - wp * sf, -r + sr - wn * sf - wp * sp)
end

"""The 3-node CTRNN, W = ring3(1, −w_p, −w_n). Equilibria have yᵢ ∈ [−(w_p + w_n), 1], inside the box for w ∈ [0, 20]."""
const ctrnn3 = VectorField(ctrnn_system(q -> ring3(1, -q.wp, -q.wn))..., fill(-41.0, 3), fill(2.0, 3))

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
    range=range(0.0, 3.0; length=100), params=(p, n) -> (ap=p, an=n))
const CTRNN_SCAN = (rates=ctrnn3_rates, start=rng -> randn(rng, 3) .+ 0.5, output=σ,
    range=range(0.0, 20.0; length=100), params=(p, n) -> (wp=p, wn=n))

"""Whether one node of the outputs `o` is dominant: it exceeds the other two together (`heatmapsV2.py`)."""
is_dominant(o) = 2 * maximum(o) > sum(o)

"""
The test of `heatmapsV2.py` on outputs `O` (3 × samples): the run passes with at least 5 dominant
(`is_dominant`) and 5 transition bouts and a mean dominant bout at least twice the mean
transition bout.
"""
function is_metastable(O)
    any(isnan, O) && return false
    dominant, lengths = rle([is_dominant(o) for o in eachcol(O)])
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
    run = trajectory(vf, q, symmetric.y .+ OFF_DIAGONAL, duration; saveat=0.1)
    second_half = run.Y[:, end÷2:end]
    moving = maximum(maximum(second_half; dims=2) .- minimum(second_half; dims=2)) > 1e-3
    return moving && !any(y -> norm(run.Y[:, end] - y) < 1e-3, stable)
end

# ── Text and time series ────────────────────────────────────────────────────────────────

"""A number as text, with a true minus sign and without a trailing `.0`."""
number_text(x) = replace(string(isinteger(x) ? Int(x) : x), "-" => "−")
numbers_text(v) = "(" * join(number_text.(v), ", ") * ")"

"""The two weights of `q` as text, e.g. `(aₚ, aₙ) = (1.1, 1.7)`."""
function weights_text(q)
    letter = haskey(q, :ap) ? "a" : "w"
    p, n = haskey(q, :ap) ? (q.ap, q.an) : (q.wp, q.wn)
    return "($(letter)ₚ, $(letter)ₙ) = " * numbers_text((p, n))
end

"""The legend entry of the shading of [`time_series_plot`](@ref)."""
shading_key!(plt) = vspan!(plt, [NaN, NaN]; color=:gray, alpha=0.15, linewidth=0, linealpha=0, label="shading: dominant node")

"""
The outputs `O` (3 × samples) against `t`, shaded by the node holding the run: a node takes over
when it is dominant (`is_dominant`) and holds until another is.
"""
function time_series_plot(t, O; title, ylabel, legendfontsize=8)
    plt = plot(; title, ylabel, xlabel="time (s)", size=PLANE_SIZE, xlims=extrema(t), titlefontsize=10,
        legend=:outertop, legend_columns=2, legendfontsize, left_margin=4Plots.mm, bottom_margin=4Plots.mm,
        top_margin=3Plots.mm)
    held = accumulate((held, o) -> is_dominant(o) ? argmax(o) : held, eachcol(O); init=0)
    shade_spans!(plt, t, held, node -> node > 0 ? RPS_COLORS[node] : nothing; alpha=0.15)
    for i in 1:3
        plot!(plt, t, O[i, :]; color=RPS_COLORS[i], linewidth=2, label=SIMPLE_NAMES[i])
    end
    return shading_key!(plt)
end

# ── Parameter plane: heatmap and chart on the same axes ─────────────────────────────────

const PLANE_SIZE = (640, 560)
const STAR_COLORS = (excitable="#ffffff", cycle=PALETTE.pink)

"""The stars `(; p, n, label, color)` of a notebook's two settings: the multistable one white, the cycling one pink."""
setting_stars(multistable, cycling) = [(p=q[1], n=q[2], label=weights_text(q), color)
                                       for (q, color) in ((multistable, STAR_COLORS.excitable), (cycling, STAR_COLORS.cycle))]

"""A star at each setting in `stars` ([`setting_stars`](@ref))."""
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

"""Write each region's name: `regions` holds `(; x, y, text)`, optionally `rotation`."""
function label_regions!(plt, regions)
    for r in regions
        annotate!(plt, r.x, r.y, Plots.text(r.text, 8, INK_SECONDARY, :center; rotation=get(r, :rotation, 0)))
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

"""
    scan_panel(overlay, top, bottom; titlefontsize=14)

The heatmap with the bifurcation curves, `overlay`, beside the noisy runs `top` and `bottom` of its
two starred settings (`time_series_plot`), under one legend for both runs.
"""
function scan_panel(overlay, top, bottom; titlefontsize=14)
    # Copies: combining plots re-parents their subplots, and the originals are shown and saved on their own.
    heat = plot!(deepcopy(overlay); colorbar_title="quasi-stability score")
    top = plot!(deepcopy(top); xlabel="", legend=false)
    bottom = plot!(deepcopy(bottom); legend=false)
    # One legend for both runs, in its own row above them.
    key = plot(; framestyle=:none, legend=:top, legend_columns=2, foreground_color_legend=nothing)
    for i in 1:3
        plot!(key, [NaN], [NaN]; color=RPS_COLORS[i], linewidth=2, label=SIMPLE_NAMES[i])
    end
    shading_key!(key)
    panel = plot(heat, key, top, bottom; layout=@layout([a{0.6w} [k{0.07h}; grid(2, 1)]]), size=(1450, 720),
        left_margin=6Plots.mm, right_margin=4Plots.mm, bottom_margin=6Plots.mm)
    # Larger text throughout, the curve labels (annotations) included.
    plot!(panel; titlefontsize, guidefontsize=14, tickfontsize=11, legendfontsize=12,
        colorbar_titlefontsize=13, colorbar_tickfontsize=11)
    plot!(panel[2]; top_margin=0Plots.mm, bottom_margin=0Plots.mm)
    for sp in panel.subplots, (_, _, label) in sp[:annotations]
        label.font.pointsize = round(Int, 1.4 * label.font.pointsize)
    end
    return panel
end

# ── Basins in 3-D ───────────────────────────────────────────────────────────────────────

"""The axes of a 3-D plot of the three nodes in the cube `lims`³."""
cube_axes(lims) = (xlabel=SIMPLE_NAMES[1], ylabel=SIMPLE_NAMES[2], zlabel=SIMPLE_NAMES[3], xlims=lims, ylims=lims, zlims=lims)

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
attractor_name(k) = k == 4 ? "limit cycle" : SIMPLE_NAMES[k]
# Two bars joined along the bottom edge, in GR's legend marker units (the fill swatch is 15.55 × 3.13).
const STRIPED_SWATCH = Shape(3.11 .* [-1.5, -0.5, -0.5, -1.5, -1.5, 0.5, 1.5, 1.5, 0.5, 0.5, -1.5],
    1.565 .* [-1.0, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1])

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
    banded(P, triangles, s, width)

Cut a triangle mesh (vertices the columns of `P`) along the levels of the per-vertex scalar `s`
at the multiples of `width`: `(; P, triangles, band)`, the triangles of each piece and the
`band = floor(s / width)` it lies in.
"""
function banded(P, triangles, s, width)
    points, pieces, band = Vector{Float64}[], NTuple{3,Int}[], Int[]
    # The part of a polygon of (vertex, value) pairs with sign * (value − level) ≥ 0 (Sutherland–Hodgman).
    function clip(polygon, level, sign)
        kept = empty(polygon)
        for (m, (p, v)) in enumerate(polygon)
            q, w = polygon[mod1(m + 1, length(polygon))]
            inside, next_inside = sign * (v - level) >= 0, sign * (w - level) >= 0
            inside && push!(kept, (p, v))
            inside == next_inside || push!(kept, (p + (level - v) / (w - v) * (q - p), level))
        end
        return kept
    end
    for t in triangles
        corners = [(P[:, i], s[i]) for i in t]
        for k in floor(Int, minimum(last, corners) / width):floor(Int, maximum(last, corners) / width)
            polygon = clip(clip(corners, k * width, 1), (k + 1) * width, -1)
            length(polygon) < 3 && continue
            n = length(points)
            append!(points, first.(polygon))
            for m in 2:length(polygon)-1
                push!(pieces, (n + 1, n + m, n + m + 1))
                push!(band, k)
            end
        end
    end
    return (; P=reduce(hcat, points; init=zeros(3, 0)), triangles=pieces, band)
end

"""
    basin_portrait(b, eqs; coords=identity, lims, flows=[], stripe_width=nothing, cycle=nothing, title="",
                   camera=(120, 30))

The basins of `b` in 3-D: each of Rock's, Paper's and Scissors' basin boundaries as a translucent
surface (where the neighbour-averaged indicator crosses ½, by Meshing.jl's marching tetrahedra),
shaded by facing, in its basin's colour (one grey "basin boundary" legend entry for all three) or,
with a `stripe_width`, in stripes alternating between the colours of the two basins it divides
(bands of that width in the sum of the coordinates); the `flows` in the colour they reach; the
equilibria `eqs`; and the `cycle` (display coordinates) in the cube `lims`³.
"""
function basin_portrait(b, eqs; coords=identity, lims, flows=[], stripe_width=nothing, cycle=nothing, title="",
        camera=(120, 30))
    plt = plot(; cube_axes(lims)..., camera, title, titlefontsize=10, legend=:topright, legendfontsize=7, size=(760, 700))

    light = normalize([0.4, -0.5, 0.75])   # triangles shaded in 6 steps by how squarely they face it
    fillalpha = isnothing(stripe_width) ? 0.3 : 0.45   # stripes need more colour to read
    function draw_surface!(P, triangles, color)
        shade(t) = (n = cross(P[:, t[2]] - P[:, t[1]], P[:, t[3]] - P[:, t[1]]); norm(n) == 0 ? 1.0 : abs(dot(n, light)) / norm(n))
        steps = min.(floor.(Int, shade.(triangles) .* 6), 5)
        base = Plots.plot_color(color)
        for s in 0:5
            group = triangles[steps .== s]
            isempty(group) && continue
            f = 0.55 + 0.45 * (s + 0.5) / 6
            mesh3d!(plt, P[1, :], P[2, :], P[3, :]; connections=group, fillalpha, linewidth=0,
                fillcolor=Plots.RGB(f * base.r, f * base.g, f * base.b), label=false)
        end
    end
    swatch!(color, label) = scatter!(plt, [NaN], [NaN], [NaN]; markershape=:square, markersize=8,
        markercolor=color, markeralpha=0.6, markerstrokewidth=0, label)

    indicator = Dict(k => smooth3d(Float64.(b.labels .== k)) for k in unique(b.labels) if k > 0)
    nearest(c) = CartesianIndex((argmin(abs.(b.values .- x)) for x in c)...)
    for k in 1:3
        vertices, triangles = isosurface(indicator[k], MarchingTetrahedra(iso=0.5), b.values, b.values, b.values)
        isempty(triangles) && continue
        P = reduce(hcat, collect.(vertices))
        if isnothing(stripe_width)
            draw_surface!(P, triangles, BASIN_COLORS[k])
            continue
        end
        # The basin across each triangle, the one most present at the grid point nearest its centre;
        # a boundary between two basins is drawn once, from the lower-numbered side.
        others = [j for j in keys(indicator) if j != k]
        across = map(triangles) do t
            I = nearest(mean(P[:, i] for i in t))
            j = argmax(j -> indicator[j][I], others)
            indicator[j][I] > 0.2 ? j : 0
        end
        for j in sort(filter(>(k), others))
            cut = banded(P, triangles[across .== j], vec(sum(P; dims=1)), stripe_width)
            isempty(cut.triangles) && continue
            for (parity, color) in enumerate((BASIN_COLORS[k], BASIN_COLORS[j]))
                group = cut.triangles[mod.(cut.band, 2) .== parity - 1]
                isempty(group) || draw_surface!(cut.P, group, color)
            end
        end
    end
    # One legend entry for the striped boundaries: a light grey block (the fill swatch) with two
    # dark grey bars (the marker, a shape sized to the block's height and a fifth of its width each).
    # Without stripes, one grey swatch for all the boundaries.
    isnothing(stripe_width) ? swatch!(GREY, "basin boundary") :
        plot!(plt, [NaN], [NaN], [NaN]; fillrange=0, fillcolor="#c8c8c8", fillalpha=1, linealpha=0,
            markershape=STRIPED_SWATCH, markercolor="#707070", markerstrokewidth=0, markerstrokealpha=0, label="state boundary")

    for k in sort(unique(f.label for f in flows)), (m, f) in enumerate(filter(f -> f.label == k, flows))
        draw_path!(plt, f.Y; color=BASIN_COLORS[k], linewidth=1, alpha=0.6, label=(m == 1 && (k != 4 || isnothing(cycle)) ? attractor_name(k) : false))
    end
    isnothing(cycle) || draw_path!(plt, [cycle cycle[:, 1]]; color=CYCLE_COLOR, linewidth=3, label="limit cycle")
    in_view(e) = all(lims[1] - 0.01 .<= coords(e.y) .<= lims[2] + 0.01)
    draw_equilibria!(plt, filter(e -> e.stability != :stable && in_view(e), eqs); coords)
    for e in filter(e -> e.stability == :stable && in_view(e), eqs)
        c = coords(e.y)
        scatter!(plt, [c[1]], [c[2]], [c[3]]; markershape=:circle, markercolor=:black, markersize=8,
            markerstrokewidth=3, markerstrokecolor=is_symmetric(e.y) ? GREY : RPS_COLORS[argmax(c)], label=false)
    end
    scatter!(plt, [NaN], [NaN], [NaN]; markershape=:circle, markercolor=:black, markerstrokecolor=:black,
        markersize=6, label="stable equilibrium")
    return plt
end

# ── 3-D portrait with the saddles' manifolds ────────────────────────────────────────────

"""
    manifold_portrait(vf, q, eqs; lims, title, camera, duration, symmetric_name)

3-D portrait in the cube `lims` (widened by 1 %, as GR does not clip 3-D lines): the equilibria
`eqs` and every saddle's 1-D manifolds (`saddle_manifolds` over `duration`; unstable solid, stable
dashed) in the colour of its node, grey for the symmetric saddle, `symmetric_name`. An unstable
manifold ends where it reaches another saddle.
"""
function manifold_portrait(vf, q, eqs; lims, title, camera, duration, symmetric_name)
    slack = 0.01 * (lims[2] - lims[1])   # GLV equilibria sit up to ~1e-5 outside the orthant
    shown(c) = all(lims[1] - slack .<= c .<= lims[2] + slack) ? c : fill(NaN, length(c))
    color_of(e) = is_symmetric(e.y) ? GREY : RPS_COLORS[argmax(e.y)]
    # GR drops markers beyond the limits, so the axes are widened too.
    plt = plot(; cube_axes((lims[1] - slack, lims[2] + slack))..., xticks=:auto, yticks=:auto, zticks=:auto, camera,
        title, titlefontsize=10, legend=:topright, legendfontsize=7, size=(760, 700))

    reach = 0.02 * (lims[2] - lims[1])   # how near another saddle a manifold has reached it
    saddles = sort(filter(e -> e.stability == :saddle, eqs); by=e -> is_symmetric(e.y))   # the symmetric saddle last
    # One legend entry per direction, for the node saddles and for the symmetric one, darker than the lines.
    entries = Dict{Tuple{Symbol,Bool},Any}()
    for e in saddles
        others = [s.y for s in saddles if s !== e]
        for m in saddle_manifolds(vf, e, q; duration)
            unstable = m.direction == :unstable
            # An unstable manifold stops where it reaches another saddle: run on, it follows the
            # next connection too and hides that saddle's own manifold under it.
            arrival = unstable ? findfirst(j -> any(o -> norm(m.Y[:, j] - o) < reach, others), axes(m.Y, 2)) : nothing
            style = unstable ? (; linewidth=4, linestyle=:solid) : (; linewidth=2.5, linestyle=:dash)
            draw_path!(plt, isnothing(arrival) ? m.Y : m.Y[:, 1:arrival]; coords=shown, color=color_of(e), style..., label=false)
            entries[(m.direction, is_symmetric(e.y))] = (; color=is_symmetric(e.y) ? GREY : INK_SECONDARY, style..., alpha=1.0)
        end
    end
    for (direction, symmetric) in sort(collect(keys(entries)); by=k -> (k[1] == :stable, k[2]))
        # The node saddles' entries need no owner; the symmetric saddle's says whose it is.
        plot!(plt, [NaN], [NaN], [NaN]; entries[(direction, symmetric)]...,
            label="1-D $direction manifold" * (symmetric ? " ($symmetric_name)" : ""))
    end
    draw_equilibria!(plt, filter(e -> !any(isnan, shown(e.y)), eqs);
        fill=e -> e.stability == :saddle ? color_of(e) : STABILITY_MARKERS[e.stability].color)
    return plt
end

# ── Saddle values: the planar saddle of GLV_Simplee_Saddle.nb ───────────────────────────

"""ẋ = μx − x³, ẏ = −y, parameters `(; μ)`: the origin is a saddle with saddle value 1/μ (contraction rate over expansion rate, the paper's S)."""
const toy_saddle = VectorField((dz, z, q) -> (dz[1] = q.μ * z[1] - z[1]^3; dz[2] = -z[2]; dz),
    (z, q) -> [q.μ-3z[1]^2 0.0; 0.0 -1.0], fill(-0.5, 2), fill(0.5, 2))

"""Shades of the runs, light to dark as the saddle value rises."""
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
μ in 0.1:0.4:2.9, the run from (0.03, 0.5) labelled by its saddle value S and shaded darker the
higher it is, each with an arrowhead.
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
        S = -saddle_value(classify(toy_saddle, [0.0, 0.0], (; μ)))  # the paper's S = −λ_min / λ_max > 0
        P = inside(trajectory(toy_saddle, (; μ), start, duration; saveat=duration / 4000).Y)
        X, Y = first.(P), last.(P)
        kept = .!isnan.(X)
        plot!(plt, X, Y; color=shades[μ], linewidth=2, label="μ = $(number_text(μ)): S = $(number_text(round(S; sigdigits=3)))")
        arrow_along!(plt, X[kept], Y[kept], 0.6; color=shades[μ], linewidth=2)
    end
    scatter!(plt, [0.0], [0.0]; markershape=STABILITY_MARKERS.saddle.shape, markercolor=STABILITY_MARKERS.saddle.color,
        markerstrokecolor=:black, markersize=8, label="saddle (origin)")
    return plt
end

end # module
