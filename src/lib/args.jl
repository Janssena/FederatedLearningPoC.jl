# TODO: Check for more elegant argument handling package
function handle_args(args::AbstractVector{<:String})
    println("Received following arguments: $args")
    data_path_idx = findfirst(Base.Fix2(startswith, "--data="), args)
    if isnothing(data_path_idx)
        data_path_idx = findfirst(endswith(".csv"), args)
        if isnothing(data_path_idx)
            throw(ErrorException(
                "Data path not supplied to script. Specify data location via --data argument: `julia --project=. --data=/my/path/data.csv`"
            ))
        end
    end
    
    data_path = split(args[data_path_idx], "--data=")[end]
    @info "Using data from: $(data_path)"

    @assert endswith(lowercase(data_path), ".csv") "data file is not a CSV file"
    @assert isfile(data_path) "data path does not link to existing file"
    
    output_path_idx = findfirst(Base.Fix2(startswith, "--out="), args)
    if isnothing(output_path_idx)
        throw(ErrorException(
            "Output path not supplied to script. Specify output file location via --out argument: `julia --project=. --out=/my/output.json`"
        ))
    end
    output_path = split(args[output_path_idx], "--out=")[end]
    @assert endswith(lowercase(output_path), ".json") || endswith(lowercase(output_path), "jld2") "output path does not point to JSON or JLD2 file"

    if !(any(startswith.(args, "--ps=")))
        return (data = data_path, output = output_path)
    else
        ps_path_idx = findfirst(Base.Fix2(startswith, "--ps="), args)
        ps_path = split(args[ps_path_idx], "--ps=")[end]

        @assert endswith(lowercase(ps_path), ".jld2") "parameter file is not a JLD2 file"
        @assert isfile(ps_path) "parameter path does not link to existing file"

        return (data = data_path, output = output_path, parameters = ps_path)
    end
end

function parse_parameters(ps_path, ::Type{T}=Float32) where T
    obj = JLD2.load(ps_path)
    return fmap(Base.Fix1(_convert, T), obj["ps"]), fmap(Base.Fix1(_convert, T), obj["st"])
end