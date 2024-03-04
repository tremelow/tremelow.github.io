using QuadGK, NLsolve
using PythonPlot

f(ρ, φ) = (1.0 - ρ * cos(φ))^2 - (1.0 + ρ^2 - 2ρ * cos(φ))^3
r(φ) = nlsolve(ρ -> f(ρ[1], φ), [1.0]).zero[1]

phi = range(0.0, π/2, 101)
# plot(phi, r.(phi))

# since the plot seems okay
quadgk(φ -> r(φ)*sin(φ)*cos(φ), 0, pi/2)