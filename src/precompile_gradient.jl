import PrecompileTools
import Random
import Zygote
import JLD2

using Dates
using DeepCompartmentModels

include("lib/data.jl");
include("lib/model.jl");

@info "Precompiling gradient call..."
PrecompileTools.@setup_workload begin
    rng = Random.Xoshiro()
    I = Float32[0 1000 60 * 1000 1/60]
    callback = generate_dosing_callback(I, Float32)

    test_indv = Individual(
        Float32[70, 150], # x
        rand(Float32, 10), # t
        rand(Float32, 10), # y
        callback; initial = zeros(Float32, 2), id = 1
    )
    population = Population([test_indv])
    dcm = load_model()
    ps, st = setup(rng, dcm, population)

    PrecompileTools.@compile_workload begin 
        grad = gradient(dcm, population, ps, st)
    end
end
println("Done!")