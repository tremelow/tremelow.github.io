using PythonPlot
using StaticArrays
using LinearAlgebra

figure(figsize=(9,3.7))
PythonPlot.pyplot.style.use("seaborn-v0_8")

function simulate(scheme, u0, nt)
    u = [copy(u0) for _ in 0:nt]
    for n in 1:nt
        u[n+1] = scheme * u[n]
    end
    return u
end
function fuse_sol(un)
    nt, dim = length(un), length(un[1])
    sol = Matrix{Float64}(undef, nt, dim)
    for n in 1:nt
        sol[n, :] .= un[n]
    end
    return sol
end

J = Float64.(@SMatrix [0 -1; 1 0])
I2 = @SMatrix [1.0 0.0; 0.0 1.0]
u0 = @SVector [1.0, 0.0]
h, nt = 0.4, 7

n_substeps = 100
dt = h / n_substeps

phi_dt = exp(dt*J)
exact_ut = fuse_sol(simulate(phi_dt, u0, nt*n_substeps))
plot(exact_ut[:,1], exact_ut[:,2])


euler_sch = I + h*J
modif_un = simulate(I + h*J, u0, nt)
correc_J = (exp(h*J) - I) / h
correc_un = simulate(I + h*correc_J, u0, nt)

plt_modif_un, plt_correc_un = fuse_sol(modif_un), fuse_sol(correc_un)
plt_modif, = plot(plt_modif_un[:,1], plt_modif_un[:,2], marker="o", linestyle="none")
plt_correc, = plot(plt_correc_un[:,1], plt_correc_un[:,2], marker="*", linestyle="none")

modif_phi_dt = exp(dt * log(I2 + h*J) / h)
modif_ut = fuse_sol(simulate(modif_phi_dt, u0, n_substeps*nt))
plot(modif_ut[:,1], modif_ut[:,2], linestyle="dashed", color=plt_modif.get_color())

correc_phi_dt = exp(dt * correc_J)
correc_phi_t = accumulate((Sn,n) -> Sn * correc_phi_dt, 1:n_substeps, init=I2)
for un in correc_un
	correc_ut = fuse_sol([un, map(phi_t -> phi_t * un, correc_phi_t)...])
	plot(correc_ut[:,1], correc_ut[:,2], linestyle="dashdot", color=plt_correc.get_color())
end

tight_layout()
savefig("linear-bea-harmonic.svg")