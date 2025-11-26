f(x::AbstractVector, θ) = exp(θ[1] + max((x[1] / 45), 40 / 45) * θ[2]) * (0.7009 ^ x[2])
f(x::AbstractMatrix, θ) = exp.(θ[1] .+ max.((x[:, 1] ./ 45.), 40 / 45) .* θ[2]) .* (0.7009 .^ x[:, 2])

function sample_vwf(rng, age, bgo)
  X = LogNormal(0.1578878428424249, 0.3478189783243864)
  b = Bijectors.Scale(f([age, bgo], [4.11, 0.644 * 0.8]))
  Y = transformed(X, b)
  return rand(rng, Y)
end

# Note: height in nhanes data set is in cm
bmi(weight, height) = weight / (height / 100)^2

function ffm_males(weight, height, age)
    _bmi = bmi(weight, height)
    first_term = 0.88 + (1 - 0.88) / (1 + (age / 13.4)^1.27)
    second_term = 9270 * weight / (6680 + (216 * _bmi))
    return first_term * second_term
end