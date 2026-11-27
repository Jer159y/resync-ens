# Exact sorted empirical-CDF CRPS, extracted from the original auditor.
function prepare_distribution(v)
 s=sort(Float64.(v));n=length(s);p=vcat(0.,cumsum(s))
 dispersion=sum((2i-n-1)*s[i] for i in 1:n)/n^2
 (;s,p,n,dispersion)
end
function score_distribution(f,y)
 r=searchsortedlast(f.s,y)
 (y*r-f.p[r+1]+f.p[end]-f.p[r+1]-y*(f.n-r))/f.n-f.dispersion
end
function empirical(v,y)
 mean(abs.(v.-y))-sum(abs(x-z) for x in v,z in v)/(2length(v)^2)
end
