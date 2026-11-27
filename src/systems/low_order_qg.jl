module LowOrderQG
using LinearAlgebra, Random, Statistics, TOML
export load_model, rhs!, jacobian!, rk4!, trajectory, csvmatrix

csvmatrix(path)=reduce(vcat,[permutedims(parse.(Float64,split(line,','))) for line in readlines(path) if !isempty(line)])
function load_model(root)
    meta=TOML.parsefile(joinpath(root,"metadata.toml"))
    rows=[split(line,',') for line in readlines(joinpath(root,"tensor.csv"))[2:end]]
    terms=[(i=parse(Int,r[1]),j=parse(Int,r[2]),k=parse(Int,r[3]),v=parse(Float64,r[4])) for r in rows if parse(Int,r[1])>0]
    (dimension=meta["dimension"],terms=terms,metadata=meta)
end
function rhs!(du,x,m)
    fill!(du,0)
    @inbounds for t in m.terms
        a=t.j==0 ? one(eltype(x)) : x[t.j]
        b=t.k==0 ? one(eltype(x)) : x[t.k]
        du[t.i]+=t.v*a*b
    end
    du
end
function jacobian!(J,x,m)
    fill!(J,0)
    @inbounds for t in m.terms
        t.j>0 && (J[t.i,t.j]+=t.v*(t.k==0 ? 1 : x[t.k]))
        t.k>0 && (J[t.i,t.k]+=t.v*(t.j==0 ? 1 : x[t.j]))
    end
    J
end
function rk4!(x,h,m,k1,k2,k3,k4,tmp)
    rhs!(k1,x,m);@. tmp=x+h/2*k1
    rhs!(k2,tmp,m);@. tmp=x+h/2*k2
    rhs!(k3,tmp,m);@. tmp=x+h*k3
    rhs!(k4,tmp,m);@. x=x+h/6*(k1+2k2+2k3+k4)
    x
end
function trajectory(m;n,dt,h,transient,seed,lyap_steps=min(n,20000))
    sub=round(Int,dt/h);isapprox(sub*h,dt)||error("Integrator step must divide save step")
    rng=MersenneTwister(seed);x=0.1 .* rand(rng,m.dimension);buf=[zeros(m.dimension) for _ in 1:5]
    for _ in 1:round(Int,transient/h);rk4!(x,h,m,buf...);end
    all(isfinite,x)||error("Nonfinite transient")
    X=Matrix{Float64}(undef,m.dimension,n);d0=1e-8
    y=x.+d0.*normalize(randn(rng,m.dimension));yb=[zeros(m.dimension) for _ in 1:5];logs=Float64[]
    for t in 1:n
        for _ in 1:sub
            rk4!(x,h,m,buf...)
            t<=lyap_steps && rk4!(y,h,m,yb...)
        end
        X[:,t].=x
        if t<=lyap_steps
            delta=y.-x;distance=norm(delta)
            distance>0&&isfinite(distance)||error("Degenerate Lyapunov separation")
            push!(logs,log(distance/d0));y.=x.+(d0/distance).*delta
        end
    end
    all(isfinite,X)||error("Nonfinite saved trajectory")
    (traj=X,lyap=sum(logs)/(length(logs)*dt),lyap_loggrowth=logs,final_state=x)
end
end
