# Lorenz2005 Model II follows the NCAR DART comp_dt formulation.
using LinearAlgebra, Statistics, Random

function qhd_workspace(N,K=1)
    H=K÷2;offsets=collect(-H:H)
    weights=[iseven(K)&&abs(j)==H ? .5/K : 1/K for j in offsets]
    indices=[mod1(i+j,N) for i in 1:N,j in -3K:3K]
    (N=N,K=K,H=H,offsets=offsets,weights=weights,indices=indices,W=zeros(N),
     a=zeros(N),b=zeros(N),c=zeros(N),d=zeros(N),tmp=zeros(N))
end

@inline qhd_index(w,i,j)=w.indices[i,j+3w.K+1]

function qhd_rhs!(du,x,name,p,w)
    N=length(x)
    if startswith(name,"thomas")
        @inbounds for i in 1:N;du[i]=sin(x[i==N ? 1 : i+1])-p*x[i];end
    elseif name=="lorenz96_40_f8"
        @inbounds for i in 1:N
            du[i]=(x[mod1(i+1,N)]-x[mod1(i-2,N)])*x[mod1(i-1,N)]-x[i]+p
        end
    elseif name=="lorenz2005_ii"
        K=w.K
        @inbounds for i in 1:N
            v=0.
            for (j,a) in zip(w.offsets,w.weights);v+=a*x[qhd_index(w,i,-j)];end
            w.W[i]=v
        end
        @inbounds for i in 1:N
            v=-w.W[qhd_index(w,i,-2K)]*w.W[qhd_index(w,i,-K)]
            for (j,a) in zip(w.offsets,w.weights)
                v+=a*w.W[qhd_index(w,i,-K+j)]*x[qhd_index(w,i,K+j)]
            end
            du[i]=v-x[i]+p
        end
    else
        error("Unknown quick candidate: $name")
    end
    du
end

function qhd_step!(x,h,name,p,w)
    qhd_rhs!(w.a,x,name,p,w);@. w.tmp=x+h/2*w.a
    qhd_rhs!(w.b,w.tmp,name,p,w);@. w.tmp=x+h/2*w.b
    qhd_rhs!(w.c,w.tmp,name,p,w);@. w.tmp=x+h*w.c
    qhd_rhs!(w.d,w.tmp,name,p,w);@. x=x+h/6*(w.a+2w.b+2w.c+w.d)
    x
end

function qhd_data(s,n;half=false,transient=true,lyap_samples=2000)
    N=s["dimension"];name=s["name"];p=s["parameter"];h=s["internal_dt"]/(half ? 2 : 1)
    sub=round(Int,s["dt"]/h);isapprox(sub*h,s["dt"];atol=1e-12)||error("Nonintegral output interval")
    rng=MersenneTwister(20261121);x=startswith(name,"thomas") ? randn(rng,N)./sqrt(2p) : p.+.1randn(rng,N)
    w=qhd_workspace(N,get(s,"K",1))
    if transient
        for _ in 1:round(Int,s["transient_time"]/h);qhd_step!(x,h,name,p,w);end
    end
    y=x.+1e-7.*normalize(randn(rng,N));wy=qhd_workspace(N,get(s,"K",1));acc=0.;L=min(n,lyap_samples)
    A=zeros(N,n)
    for t in 1:n
        for _ in 1:sub
            qhd_step!(x,h,name,p,w)
            t<=L&&qhd_step!(y,h,name,p,wy)
        end
        if t<=L
            v=y.-x;distance=norm(v);isfinite(distance)&&distance>0||error("Invalid physical perturbation")
            acc+=log(distance/1e-7);y.=x.+v.*(1e-7/distance)
        end
        A[:,t].=x
    end
    all(isfinite,A)||error("Nonfinite reference trajectory")
    (traj=A,largest_lyapunov=L==0 ? NaN : acc/(L*s["dt"]))
end
