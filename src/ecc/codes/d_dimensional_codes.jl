"""Implemented in a package extension with `Oscar`. Check the docs for the [Oscar extension](http://qc.quantumsavory.org/stable/ECC_API/#Implemented-in-an-extension-requiring-Oscar.jl)"""
function boundary_maps end

"""D-Dimensional Surface codes ([Berthusen_2024](@cite), [Zeng_2019](@cite))

Implemented as a package extension with `Oscar`. Check the [QuantumClifford documentation](http://qc.quantumsavory.org/stable/ECC_API/) for more details on that extension."""
function DDimensionalSurfaceCode(args...; kwargs...)
    ext = Base.get_extension(QuantumClifford, :QuantumCliffordOscarExt)
    if isnothing(ext)
        throw("The `DDimensionalSurfaceCode` depends on the package `Oscar` but you have not installed or imported it yet. Immediately after you import `Oscar`, the `DDimensionalSurfaceCode` will be available.")
    end
    return ext.DDimensionalSurfaceCode(args...; kwargs...)
end

"""D-Dimensional Toric codes ([Berthusen_2024](@cite), [Zeng_2019](@cite))

Implemented as a package extension with `Oscar`. Check the [QuantumClifford documentation](http://qc.quantumsavory.org/stable/ECC_API/) for more details on that extension."""
function DDimensionalToricCode(args...; kwargs...)
    ext = Base.get_extension(QuantumClifford, :QuantumCliffordOscarExt)
    if isnothing(ext)
        throw("The `DDimensionalToricCode` depends on the package `Oscar` but you have not installed or imported it yet. Immediately after you import `Oscar`, the `DDimensionalToricCode` will be available.")
    end
    return ext.DDimensionalToricCode(args...; kwargs...)
end
