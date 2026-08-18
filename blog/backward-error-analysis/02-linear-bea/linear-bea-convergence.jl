using PythonPlot

default_figsize = (8.0, 3.8)
PythonPlot.pyplot.style.use("seaborn-v0_8")

all_nτ = ceil.(Int, 10.0 .^ range(1, 5, 17))
all_dτ = 1.0 ./ all_nτ
h1, h2, h3 = all_dτ[5], all_dτ[9], all_dτ[13]

function error_correc(h, nτ)
	mu_h = (exp(h) - 1.0) / h
	dτ = 1.0 / nτ
	sch_sol = (1.0 + dτ*mu_h) ^ nτ
	return abs(exp(1.0) - sch_sol)
end

err1 = map(nτ -> error_correc(h1,nτ), all_nτ)
err2 = map(nτ -> error_correc(h2,nτ), all_nτ)
err3 = map(nτ -> error_correc(h3,nτ), all_nτ)

figure(figsize=default_figsize); xscale("log"); yscale("log");
plot(all_dτ, err1, label="h = \$10^{-2}\$", marker="o")
plot(all_dτ, err2, label="h = \$10^{-3}\$", marker="*", markersize=10)
plot(all_dτ, err3, label="h = \$10^{-4}\$", marker="D")
plot(all_dτ, all_dτ, label="order 1", linestyle="dashed")
legend()
ylabel("err"); xlabel("\$\\tau\$")
tight_layout()