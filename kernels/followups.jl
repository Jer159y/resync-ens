# Extracted without numerical edits from tools/run_resync_revision_20261124.jl; see provenance/source_export.json.
function revision_state(history,src,j,s,tau,ctx,W)
 ps,_,esn=ctx.ensemble[j];T=s*tau
 if W===nothing || W>=T
  Y=assemble_history(history,src,j,s,tau)
  return warmup_full_history(ps,ctx.init_states[j],esn,Y;x0=ctx.x0)
 end
 # Inputs y(T-W),...,y(T-1), matching warmup_recent exactly without full assembly.
 Y=Matrix{Float64}(undef,length(ctx.x0),W)
 for (a,t) in enumerate(T-W:T-1)
  seg=cld(t,tau);Y[:,a].=view(history[src[j,seg]],:,t)
 end
 TestUtils.warmup_teacher_forcing(ps,deepcopy(ctx.init_states[j]),esn,Y)
end
function revision_projections(target,D)
 rng=MersenneTwister(REVP["slice_seed"]);P=randn(rng,REVP["slice_directions"],D)
 P./=sqrt.(sum(abs2,P;dims=2));n=min(REVP["slice_samples"],size(target,2))
 ids=randperm(rng,size(target,2))[1:n]
 Q=sort(P*target[:,ids];dims=2)
 (;P,Q,n)
end
function revision_sliced(A,ref,seed)
 D,T,N=size(A);X=reshape(A,D,T*N);rng=MersenneTwister(seed)
 ids=randperm(rng,T*N)[1:ref.n]
 mean(abs.(sort(ref.P*X[:,ids];dims=2).-ref.Q))
end
function revision_flags(A,tol)
 D,T,N=size(A);idx=unique(round.(Int,range(1,T;length=min(T,64))))
 [begin
  X=Matrix(view(A,:,idx,j));finite=all(isfinite,view(A,:,:,j))
  amplitude=finite ? sqrt(mean(abs2,X.-mean(X;dims=2))) : NaN
  (constant=finite && amplitude<=tol,nonfinite=!finite)
 end for j in 1:N]
end
function revision_rollout(ctx,inputs,p,out;policy,bases,base,anchor=nothing)
 done=joinpath(out,"complete.toml");isfile(done)&&return TOML.parsefile(done)
 mkpath(out);N=ctx.n;D=length(ctx.x0);tau=ctx.τ;T=ctx.predict_len;num=p["numerics"]
 kwargs=(probability_stride=max(1,base÷20),dimension_samples=num["effective_dimension_sample_times"],constant_tol=num["constant_amplitude_tolerance"])
 oe=WindowEvaluator(base,N,bases,inputs.target,inputs.reference["spread_reference"];kwargs...)
 re=WindowEvaluator(base,N,bases,inputs.target,inputs.reference["spread_reference"];kwargs...)
 k=REVCASE=="duffing" ? 30 : 60
 W=policy=="recent_quarter" ? max(1,round(Int,base/4)) : policy=="recent_one" ? base : policy=="recent_four" ? 4base : nothing
 replay=policy in ["recent_quarter","recent_one","recent_four","full"]
 select=policy!="none";boundary=policy in ["boundary","recent_quarter","recent_one","recent_four","full"]
 states=Any[deepcopy(x) for x in ctx.init_states];xin=[copy(ctx.x0) for _ in 1:N]
 hist=replay ? [Matrix{Float64}(undef,D,T) for _ in 1:N] : Matrix{Float64}[];src=zeros(Int,N,ctx.n_segments)
 opts=ResyncOptions(p=k/N,bad_selector=:feature_distance_from_centroid,donor=:uniform)
 cands=Vector{Matrix{Float64}}(undef,N);ends=Vector{Any}(undef,N);geom=fill(NaN,N,2)
 flags=NamedTuple[];flagwindows=NamedTuple[];slices=NamedTuple[];temporal=NamedTuple[];jumps=NamedTuple[]
 seen=Dict(scope=>falses(N) for scope in ["own","returned"])
 streak=Dict(scope=>zeros(Int,N) for scope in ["own","returned"])
 longest=Dict(scope=>zeros(Int,N) for scope in ["own","returned"])
 ref=revision_projections(inputs.target,D);roll_seconds=0.;firsthash="";tstart=time()
 previous_own=nothing;previous_ret=nothing
 for s in 1:ctx.n_segments
  tick=time();generate_candidates!(cands,ends,geom,ctx,states,xin,trues(N),false)
  A=cat(cands...;dims=3);s==1 && (firsthash=control_hash(A[:,1:base,:]))
  fin=[all(isfinite,x) for x in cands];bad=Int[];donors=Dict{Int,Int}()
  if select
   scores=criterion_scores(:feature_distance_from_centroid,SegmentPool(cands,fin,geom,s,opts.random_seed),ctx)
   bad=select_bad(scores,fin,trues(N),k);badset=Set(bad);good=[i for i in 1:N if !(i in badset)]
   isempty(good)&&error("No donor available");donors=assign_donors(bad,good,opts,s)
  end
  states=Any[deepcopy(x) for x in ends];xin=[copy(x[:,end]) for x in cands];B=copy(A)
  if replay
   cols=seg_range(s,tau);for j in 1:N;hist[j][:,cols].=cands[j];src[j,s]=j;end
   before=copy(src[:,1:s])
   for j in bad
    d=donors[j];hist[j][:,cols].=cands[d];src[j,1:s-1].=view(before,d,1:s-1);src[j,s]=j
   end
   Threads.@threads for j in bad;states[j]=revision_state(hist,src,j,s,tau,ctx,W);end
  end
  for j in bad
   d=donors[j];B[:,:,j].=cands[d];boundary && (xin[j]=copy(cands[d][:,end]))
   push!(jumps,(scope="intervention",kind="input_transfer",segment=s,recipient=j,donor=d,normalized_boundary_input_change=sqrt(mean(abs2,(cands[d][:,end].-cands[j][:,end]).*ctx.norm.sd./inputs.sd))))
  end
  roll_seconds+=time()-tick
  AZ=(A.*reshape(ctx.norm.sd,D,1,1).+reshape(ctx.norm.mu,D,1,1).-reshape(inputs.mu,D,1,1))./reshape(inputs.sd,D,1,1)
  BZ=(B.*reshape(ctx.norm.sd,D,1,1).+reshape(ctx.norm.mu,D,1,1).-reshape(inputs.mu,D,1,1))./reshape(inputs.sd,D,1,1)
  y=(ctx.truth[:,(s-1)*tau+1:s*tau].*ctx.norm.sd.+ctx.norm.mu.-inputs.mu)./inputs.sd
  if !all(isfinite,AZ) || !all(isfinite,BZ)
   record=Dict("status"=>"invalid_nonfinite_own_or_returned","policy"=>policy,"segment"=>s,"nonfinite_own_values"=>count(x->!isfinite(x),AZ),"nonfinite_returned_values"=>count(x->!isfinite(x),BZ))
   scheme_toml(done,record);return record
  end
  for (scope,Z,eval) in [("own",AZ,oe),("returned",BZ,re)]
   for lo in 1:base:tau
    ii=lo:lo+base-1;V=view(Z,:,ii,:);push_base!(eval,V,view(y,:,ii));b=eval.n_bases
    ff=revision_flags(V,REVP["constant_tolerance"])
    for j in 1:N
     flagged=ff[j].constant||ff[j].nonfinite;seen[scope][j]|=flagged
     streak[scope][j]=flagged ? streak[scope][j]+1 : 0
     longest[scope][j]=max(longest[scope][j],streak[scope][j])
     flagged && push!(flags,(scope=scope,base_index=b,slot=j,constant=ff[j].constant,nonfinite=ff[j].nonfinite))
    end
    push!(flagwindows,(scope=scope,base_index=b,n_members=N,n_constant=count(x->x.constant,ff),n_nonfinite=count(x->x.nonfinite,ff),n_flagged=count(x->x.constant||x.nonfinite,ff)))
    if eval.filled==0
     push!(slices,(scope=scope,stop_base=b,n_directions=size(ref.P,1),n_samples=ref.n,sliced_w1=revision_sliced(eval.pending,ref,REVP["slice_seed"]+b),sampling="fixed_seed_equal_count_without_replacement"))
    end
   end
  end
  append!(temporal,control_temporal(AZ,y,base,s,"own"));append!(temporal,control_temporal(BZ,y,base,s,"returned"))
  # Returned path jumps and own continuation increments, kept separate from input transfer.
  for (scope,Z,previous) in [("own",AZ,previous_own),("returned",BZ,previous_ret)]
   if previous!==nothing
    dif=Z[:,1,:].-previous
    push!(jumps,(scope=scope,kind="path_boundary_increment",segment=s,recipient=0,donor=0,normalized_boundary_input_change=sqrt(mean(abs2,dif))))
   end
  end
  previous_own=copy(AZ[:,end,:]);previous_ret=copy(BZ[:,end,:])
  if s%10==0
   control_save(oe,out,"own");control_save(re,out,"returned")
   paper1_csv(joinpath(out,"flag_windows.csv"),flagwindows);paper1_csv(joinpath(out,"sliced_windows.csv"),slices)
   scheme_toml(joinpath(dirname(out),"progress.toml"),Dict("case"=>REVCASE,"policy"=>policy,"segment"=>s,"segments"=>ctx.n_segments))
  end
  println("REVISION $REVCASE $policy $s/$(ctx.n_segments)");flush(stdout)
 end
 for (name,rows) in [("member_flag_events",flags),("flag_windows",flagwindows),("sliced_windows",slices),("temporal_acf",temporal),("boundary_inputs",jumps)]
  !isempty(rows)&&paper1_csv(joinpath(out,name*".csv"),rows)
 end
 paper1_csv(joinpath(out,"flag_member_summary.csv"),[(scope=scope,slot=j,ever_flagged=seen[scope][j],longest_streak_base_windows=longest[scope][j]) for scope in ["own","returned"] for j in 1:N])
 control_save(oe,out,"own");control_save(re,out,"returned")
 paper1_csv(joinpath(out,"own_summary.csv"),[finish(oe)]);paper1_csv(joinpath(out,"returned_summary.csv"),[finish(re)])
 anchorerr=0.
 if anchor!==nothing && bases==200 && policy in ["none","full"]
  source=joinpath(anchor,policy=="none" ? "b0_r0" : "b1_r1")
  for scope in ["own","returned"]
   oldscope=scope=="own" ? "own" : "bookkeeping"
   # b00 bookkeeping is record-only, so its returned record is intentionally not a no-correction anchor.
   policy=="none" && scope=="returned" && continue
   aa=control_csv(joinpath(out,scope*"_base_windows.csv"));bb=control_csv(joinpath(source,oldscope*"_base_windows.csv"))
   for i in eachindex(aa),metric in [:spread,:crps,:spread_loss]
    anchorerr=max(anchorerr,abs(control_num(aa[i][metric])-control_num(bb[i][metric])))
   end
  end
  anchorerr<1e-9||error("Original test replay does not match anchor: $anchorerr")
 end
 result=Dict("status"=>"complete","policy"=>policy,"base_intervals"=>bases,"first_own_sha256"=>firsthash,"forecast_seconds"=>roll_seconds,"diagnostics_and_forecast_seconds"=>time()-tstart,"max_anchor_difference"=>anchorerr,"raw_saved"=>false)
 scheme_toml(done,result);result
end
function revision_newinputs(inputs,bank,out)
 (;inputs...,bank=bank,out=out)
end
