using PythonPlot

fig = figure(figsize=(7.0, 3.0))
PythonPlot.pyplot.style.use("seaborn-v0_8")

f(ρ, φ) = (1.0 - ρ * cos(φ))^2 - (1.0 + ρ^2 - 2ρ * cos(φ))^3
rho = range(0.0, 1.1, 701)
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for a in alphas
    phi = 0.5 * pi * (1.0 - a)
    min_b, p = 0.2, 3
    b = ((1.0 - min_b^(1/p)) * a + min_b^(1/p))^p
	plot(rho, f.(rho, phi), alpha=b, c="C1")
end
ylim(-0.1, 0.42)
xlabel("\$\\rho\$"); ylabel("\$f(\\rho, \\varphi)\$")
tight_layout()
# savefig("dewdrop-planet-radius.svg")