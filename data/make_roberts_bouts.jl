# Writes data/roberts2016_bouts.csv, every wild-type bout (its length in video frames, 29.97 Hz) of Roberts et al. (2016), from their
# worm-tracking archive on Dryad (doi:10.5061/dryad.35qv6, CC0). Dryad blocks automated
# downloads: download "wormtracks archive.zip" by hand, unzip it, and run
#
#     julia data/make_roberts_bouts.jl path/to/wormtracks
#
# The files are Igor Pro binary waves (.ibw). Each of the five wild-type cohorts (C15, C21,
# C57, C162, C186: Table 1's n = 5) has `Wild type/C<n>/worm.ibw` (the worm number at every
# video frame, 29.97 Hz) and `IBW files/C<n>_Viterbi.ibw`, the most likely sequence of the four
# HMM states under the cohort's best-fit rates. Roberts' code (`wormtracks.ipf`) drops 2 frames
# at each end of every worm before decoding, so worm k covers (its frame count − 4) Viterbi
# frames; it labels the states 0 = X (both units off, the pause after Forward), 1 = R, 2 = F,
# 3 = Y (both on, the pause after Reverse).

"""The wild-type cohorts of Roberts et al. (2016) Table 1."""
const ROBERTS_WILD_TYPE = ["C15", "C21", "C57", "C162", "C186"]

# Roberts' Viterbi codes → the cycle order Forward, Pause 1 (X), Reverse, Pause 2 (Y).
const VITERBI_ORDER = Dict(2 => 1, 0 => 2, 1 => 3, 3 => 4)

const IBW_TYPES = Dict(2 => Float32, 4 => Float64, 8 => Int8, 0x10 => Int16, 0x20 => Int32,
    0x48 => UInt8, 0x50 => UInt16, 0x60 => UInt32)

"""
    read_ibw(path)

One Igor binary wave (version 2 or 5, little-endian, 1-D, real): `(; dx, data)` with `dx` the
x step (s per frame here).
"""
function read_ibw(path)
    bytes = read(path)
    get(n, T, at) = ltoh(reinterpret(T, bytes[at+1:at+n*sizeof(T)])[1])
    version = get(1, Int16, 0)
    if version == 2
        type, npnts, dx, start = get(1, Int16, 16), get(1, Int32, 16 + 42), get(1, Float64, 16 + 48), 16 + 110
    elseif version == 5
        type, npnts, dx, start = get(1, Int16, 64 + 16), get(1, Int32, 64 + 12), get(1, Float64, 64 + 84), 64 + 320
    else
        error("$path: not a little-endian version 2 or 5 Igor binary wave")
    end
    T = IBW_TYPES[type]
    data = ltoh.(reinterpret(T, bytes[start+1:start+npnts*sizeof(T)]))
    return (; dx, data=collect(data))
end

"""Lengths of the runs of equal values in `x`, with their values: `[(value, length), …]`."""
function runs(x)
    starts = [1; findall(x[2:end] .!= x[1:end-1]) .+ 1]
    return [(x[s], e - s + 1) for (s, e) in zip(starts, [starts[2:end] .- 1; length(x)])]
end

"""
    roberts_worm_bouts(dir; cohorts = ROBERTS_WILD_TYPE)

Every Viterbi-decoded bout of the wild-type worms, in seconds, worm by worm (each worm's first
and last bout dropped, cut short by the recording): one `(; cohort, worm, seconds, lengths)` per tracked
worm, with `seconds` the length of its decoded track and `lengths` its bouts per state.
"""
function roberts_worm_bouts(dir; cohorts=ROBERTS_WILD_TYPE)
    tracks = NamedTuple{(:cohort, :worm, :seconds, :lengths),Tuple{String,Int,Float64,Vector{Vector{Float64}}}}[]
    for c in cohorts
        viterbi = read_ibw(joinpath(dir, "IBW files", "$(c)_Viterbi.ibw"))
        worms = runs(read_ibw(joinpath(dir, "Wild type", c, "worm.ibw")).data)
        frames = [n - 4 for (_, n) in worms]
        sum(frames) == length(viterbi.data) || error("$c: worm frames do not match the Viterbi states")
        stop = cumsum(frames)
        for (k, n) in enumerate(frames)
            lengths = [Float64[] for _ in 1:4]
            for (state, m) in runs(viterbi.data[stop[k]-n+1:stop[k]])[2:end-1]
                push!(lengths[VITERBI_ORDER[state]], m * viterbi.dx)
            end
            push!(tracks, (; cohort=c, worm=Int(first(worms[k])), seconds=n * viterbi.dx, lengths))
        end
    end
    return tracks
end

const STATE_NAMES = ["Forward", "Pause 1", "Reverse", "Pause 2"]

dir = isempty(ARGS) ? joinpath(@__DIR__, "roberts2016", "wormtracks") : ARGS[1]
open(joinpath(@__DIR__, "roberts2016_bouts.csv"), "w") do io
    println(io, "cohort,worm,state,frames")   # frames at 29.97 Hz: seconds = frames / 29.97
    for w in roberts_worm_bouts(dir), (i, lengths) in enumerate(w.lengths), s in lengths
        println(io, w.cohort, ",", w.worm, ",", STATE_NAMES[i], ",", round(Int, s * 29.97))
    end
end
