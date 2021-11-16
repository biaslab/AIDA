# load helper functions
include("../agent/utils.jl")

CONTEXTS = ["train", "babble"]

mutable struct ContextMemory
    name    :: String
    params  :: Tuple
    dataset :: Dict
end

mutable struct EFEAgent
    cmems  :: Vector{ContextMemory}
    
    grid

    function EFEAgent(names::Vector{String}, nsteps::T, ndims::T, npoints::T) where T<:Int64
        params = (0.2, 0.5)
        cmems = Vector{ContextMemory}()
        for name in names
            push!(cmems, ContextMemory(name, params, Dict("X" => [rand(ndims, npoints)], "y" => [[.0]])))
        end
        grid = Iterators.product(LinRange(0, 2, nsteps), LinRange(0, 2, nsteps))
        new(cmems, grid)
    end
end