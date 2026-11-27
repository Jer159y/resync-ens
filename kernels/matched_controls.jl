# Extracted without numerical edits from tools/run_boundary_history_controls_20261123.jl; see provenance/source_export.json.
using DelimitedFiles
control_num(x)=x isa Number ? Float64(x) : parse(Float64,string(x))
function control_csv(path)
 A,h=readdlm(path,',',Any,'\n';header=true)
 [Dict(Symbol(h[1,j])=>A[i,j] for j in axes(A,2)) for i in axes(A,1)]
end
control_hash(A)=bytes2hex(sha256(reinterpret(UInt8,vec(A))))
function control_replay!(states,ctx,hist,src,j,s,tau)
 Y=assemble_history(hist,src,j,s,tau);ps,_,esn=ctx.ensemble[j]
 states[j]=warmup_full_history(ps,ctx.init_states[j],esn,Y;x0=ctx.x0)
end
function control_temporal(A,y,base,s,label)
 D,T,N=size(A);rr=NamedTuple[]
 for lag in unique([1,max(1,base÷10),max(1,base÷2)])
  values=Float64[];truths=Float64[];excluded=0
  for d in 1:D
   for i in 1:N
    a=view(A,d,1:T-lag,i);b=view(A,d,1+lag:T,i)
    if all(isfinite,a) && all(isfinite,b) && std(a)>1e-10 && std(b)>1e-10;push!(values,cor(a,b));else;excluded+=1;end
   end
   a=view(y,d,1:T-lag);b=view(y,d,1+lag:T)
   all(isfinite,a) && all(isfinite,b) && std(a)>1e-10 && std(b)>1e-10 && push!(truths,cor(a,b))
  end
  push!(rr,(condition=label,segment=s,lag_steps=lag,mean_member_coordinate_acf=isempty(values) ? NaN : mean(values),
   mean_truth_coordinate_acf=length(truths)==D ? mean(truths) : NaN,n_eligible=length(values),n_excluded=excluded,
   scope="exploratory_within_segment_no_boundary_concatenation"))
 end
 rr
end
function control_forks(ctx,s,endstates,cands,replayed,donorinputs,bad,base,mu,sd,target,numerics)
 N=ctx.n;D=length(ctx.x0);tau=ctx.τ;T=ctx.n_segments;rows=NamedTuple[];expected=""
 for r in 0:1,b in 0:1
  carried=Any[deepcopy(v) for v in endstates];xin=[copy(v[:,end]) for v in cands]
  for j in bad
   r==1 && (carried[j]=deepcopy(replayed[j]));b==1 && (xin[j]=copy(donorinputs[j]))
  end
  out=Vector{Matrix{Float64}}(undef,N);newstates=Vector{Any}(undef,N);geom=fill(NaN,N,2)
  generate_candidates!(out,newstates,geom,ctx,carried,xin,trues(N),false)
  A=cat(out...;dims=3);r==1 && b==1 && (expected=control_hash(A))
  Z=(A.*reshape(ctx.norm.sd,D,1,1).+reshape(ctx.norm.mu,D,1,1).-reshape(mu,D,1,1))./reshape(sd,D,1,1)
  y=(ctx.truth[:,s*tau+1:(s+1)*tau].*ctx.norm.sd.+ctx.norm.mu.-mu)./sd
  valid=all(isfinite,Z);pm=ResyncV4Metrics.probability_metrics(Z,y;stride=max(1,base÷20))
  push!(rows,(event=s,boundary_input=b,history_replay=r,forecast_start_step=s*tau+1,forecast_stop_step=(s+1)*tau,
   window_base_intervals=tau÷base,n_members=N,n_recipients=length(bad),valid=valid,
   crps=pm.crps,spread=valid ? ResyncV4Metrics.window_spread(Z) : NaN,
   segment_marginal_w1=valid ? ResyncV4Metrics.marginal_w1(Z,target) : NaN,
   scope="conditional_next_own_segment_under_common_parent_history"))
 end
 rows,expected
end
function control_save(e,out,scope)
 scalarrows=[(; (k=>v for (k,v) in pairs(r) if !(v isa AbstractArray))...) for r in e.base_rows]
 paper1_csv(joinpath(out,scope*"_base_windows.csv"),scalarrows)
 paper1_csv(joinpath(out,scope*"_distribution_windows.csv"),e.distribution_rows)
end
function control_climatology(target)
 [begin
   x=sort(collect(view(target,d,:)));n=length(x)
   (x=x,p=vcat(0.,cumsum(x)),n=n,disp=sum((2i-n-1)*x[i] for i in 1:n)/n^2)
  end for d in axes(target,1)]
end
function control_climscore(f,y)
 r=searchsortedlast(f.x,y)
 (y*r-f.p[r+1]+f.p[end]-f.p[r+1]-y*(f.n-r))/f.n-f.disp
end
function control_leads(A,B,y,base,offset,label,clim)
 D,T,N=size(A);rows=NamedTuple[]
 for t in 1:max(1,base÷20):T
  oc=mean(ResyncV4Metrics.empirical_crps(view(A,d,t,:),y[d,t]) for d in 1:D)
  bc=mean(ResyncV4Metrics.empirical_crps(view(B,d,t,:),y[d,t]) for d in 1:D)
  cc=mean(control_climscore(clim[d],y[d,t]) for d in 1:D)
  rmse=sqrt(mean((mean(view(A,d,t,:))-y[d,t])^2 for d in 1:D))
  spread=sqrt(mean(var(view(A,d,t,:)) for d in 1:D))
  push!(rows,(condition=label,lead_steps=offset+t,lead_base_intervals=(offset+t)/base,own_crps=oc,
   bookkeeping_crps=bc,calibration_cdf_crps=cc,expected_iid300_crps=cc+mean(f.disp for f in clim)/N,
   own_mean_normalized_rmse=rmse,own_snapshot_spread=spread))
 end
 rows
end
function control_rollout(ctx,inputs,p,out;k,m,boundary,replay,forks=false,original,case,bases)
 done=joinpath(out,"complete.toml");isfile(done) && return TOML.parsefile(done)
 mkpath(out);N=ctx.n;D=length(ctx.x0);tau=ctx.τ;T=ctx.predict_len;base=inputs.manifest["base_tau"]
 num=p["numerics"];kwargs=(probability_stride=max(1,base÷20),dimension_samples=num["effective_dimension_sample_times"],constant_tol=num["constant_amplitude_tolerance"])
 own=WindowEvaluator(base,N,bases,inputs.target,inputs.reference["spread_reference"];kwargs...)
 eff=WindowEvaluator(base,N,bases,inputs.target,inputs.reference["spread_reference"];kwargs...)
 states=Any[deepcopy(v) for v in ctx.init_states];xin=[copy(ctx.x0) for _ in 1:N]
 hist=replay ? [Matrix{Float64}(undef,D,T) for _ in 1:N] : Matrix{Float64}[];src=zeros(Int,N,ctx.n_segments)
 opts=ResyncOptions(p=k/N,bad_selector=:feature_distance_from_centroid,donor=:uniform)
 cands=Vector{Matrix{Float64}}(undef,N);endstates=Vector{Any}(undef,N);geom=fill(NaN,N,2)
 events=NamedTuple[];conditional=NamedTuple[];temporal=NamedTuple[];leads=NamedTuple[];clim=control_climatology(inputs.target);expected="";firsthash="";label="b$(Int(boundary))_r$(Int(replay))"
 try
 for s in 1:ctx.n_segments
  generate_candidates!(cands,endstates,geom,ctx,states,xin,trues(N),false);A=cat(cands...;dims=3)
  if s==1;firsthash=control_hash(Array(view(A,:,1:base,:)));end
  !isempty(expected) && control_hash(A)!=expected && error("Fork11 does not reproduce parent next candidate")
  fin=[all(isfinite,c) for c in cands];sc=criterion_scores(:feature_distance_from_centroid,SegmentPool(cands,fin,geom,s,opts.random_seed),ctx)
  bad=select_bad(sc,fin,trues(N),k);badset=Set(bad);good=[i for i in 1:N if !(i in badset)]
  isempty(good) && throw(V4NumericalForecastError("No nonselected donor"));donors=assign_donors(bad,good,opts,s)
  states=Any[deepcopy(v) for v in endstates];xin=[copy(v[:,end]) for v in cands];B=copy(A)
  if replay
   cols=seg_range(s,tau);for i in 1:N;hist[i][:,cols].=cands[i];src[i,s]=i;end
   before=copy(src[:,1:s])
   for j in bad
    d=donors[j];hist[j][:,cols].=cands[d];src[j,1:s-1].=view(before,d,1:s-1);src[j,s]=j
   end
   Threads.@threads for j in bad;control_replay!(states,ctx,hist,src,j,s,tau);end
  end
  for j in bad
   d=donors[j];B[:,:,j].=cands[d]
   boundary && (xin[j]=copy(cands[d][:,end]))
   push!(events,(segment=s,recipient=j,donor=d,boundary_input=Int(boundary),history_replay=Int(replay)))
  end
  AZ=(A.*reshape(ctx.norm.sd,D,1,1).+reshape(ctx.norm.mu,D,1,1).-reshape(inputs.mu,D,1,1))./reshape(inputs.sd,D,1,1)
  BZ=(B.*reshape(ctx.norm.sd,D,1,1).+reshape(ctx.norm.mu,D,1,1).-reshape(inputs.mu,D,1,1))./reshape(inputs.sd,D,1,1)
  y=(ctx.truth[:,(s-1)*tau+1:s*tau].*ctx.norm.sd.+ctx.norm.mu.-inputs.mu)./inputs.sd
  if !all(isfinite,AZ) || !all(isfinite,BZ)
   throw(V4NumericalForecastError("Nonfinite own or bookkeeping record in mechanism control; no finite-subset replacement"))
  end
  for lo in 1:base:tau
   ii=lo:lo+base-1;push_base!(own,view(AZ,:,ii,:),view(y,:,ii));push_base!(eff,view(BZ,:,ii,:),view(y,:,ii))
  end
  ll=control_leads(AZ,BZ,y,base,(s-1)*tau,label,clim)
  abs(mean(x.own_crps for x in ll)-mean(x.crps for x in own.base_rows[end-m+1:end]))<1e-12 || error("Lead CRPS aggregation differs from window score")
  append!(leads,ll)
  append!(temporal,control_temporal(AZ,y,base,s,label))
  if forks && s<ctx.n_segments
   rr,expected=control_forks(ctx,s,endstates,cands,states,xin,bad,base,inputs.mu,inputs.sd,inputs.target,num);append!(conditional,rr)
   paper1_csv(joinpath(out,"conditional_forks.csv"),conditional)
  end
  if s%10==0
   paper1_csv(joinpath(out,"events.csv"),events);paper1_csv(joinpath(out,"temporal_acf.csv"),temporal);paper1_csv(joinpath(out,"lead_scores.csv"),leads)
   control_save(own,out,"own");control_save(eff,out,"bookkeeping")
  end
  scheme_toml(joinpath(dirname(out),"progress.toml"),Dict("case"=>case,"condition"=>label,"segment"=>s,"segments"=>ctx.n_segments,"forks"=>forks))
  println("CONTROL $case $label segment=$s/$(ctx.n_segments)");flush(stdout)
 end
 catch err
  err isa V4NumericalForecastError || rethrow()
  record=Dict("status"=>"invalid_forecast","condition"=>label,"reason"=>err.reason,"complete_base_windows"=>own.n_bases)
  scheme_toml(done,record);return record
 end
 control_save(own,out,"own");control_save(eff,out,"bookkeeping");paper1_csv(joinpath(out,"events.csv"),events);paper1_csv(joinpath(out,"temporal_acf.csv"),temporal);paper1_csv(joinpath(out,"lead_scores.csv"),leads)
 os=finish(own);es=finish(eff);paper1_csv(joinpath(out,"own_summary.csv"),[os]);paper1_csv(joinpath(out,"bookkeeping_summary.csv"),[es])
 maxdiff=0.;verified=false
 if (!boundary && !replay) || (boundary && replay)
  legacy=joinpath(original,"test",!boundary && !replay ? "k0_m1" : "k$(k)_m$(m)")
  firsthash==TOML.parsefile(joinpath(legacy,"complete.toml"))["first_own_sha256"] || error("Initial forecasts differ from frozen test")
  for (scope,cols,file) in [("own",[:spread,:crps,:spread_loss],"base_windows.csv"),("own",[:w1],"distribution_windows.csv")]
   a=control_csv(joinpath(out,scope*"_"*file));b=control_csv(joinpath(legacy,scope*"_"*file))
   for i in eachindex(a),v in cols;maxdiff=max(maxdiff,abs(control_num(a[i][v])-control_num(b[i][v])));end
  end
  if boundary && replay
   for (file,cols) in [("base_windows.csv",[:crps,:spread,:spread_loss]),("distribution_windows.csv",[:w1])]
    a=control_csv(joinpath(out,"bookkeeping_"*file));b=control_csv(joinpath(legacy,"effective_"*file))
    for i in eachindex(a),v in cols;maxdiff=max(maxdiff,abs(control_num(a[i][v])-control_num(b[i][v])));end
   end
  end
  maxdiff<1e-9 || error("Control anchor differs from completed test: $maxdiff");verified=true
 end
 result=Dict("status"=>"complete","condition"=>label,"n_base_windows"=>bases,"first_own_sha256"=>firsthash,"anchor_checked"=>verified,"max_anchor_difference"=>maxdiff,"conditional_events"=>forks ? ctx.n_segments-1 : 0,"raw_saved"=>false)
 scheme_toml(done,result);result
end
