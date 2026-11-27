# Mechanism overview: input histories, recipient-specific states and autonomous feedback.
# All curves and activations are schematic; this file does not simulate a recovery experiment.
using CairoMakie, Random, SHA, TOML, LinearAlgebra
const INTRO_OUT=@__DIR__
const INTRO_BLUE="#2C6A83"
const INTRO_ORANGE="#B36145"
const INTRO_GRAY="#A1ABB2"
const INTRO_CONTEXT="#678B9B" # midpoint of donor blue and the previous context gray
const INTRO_INK="#28333D"
function intro_arrow(a,p,q;color=INTRO_INK,width=1.5,head=.9)
    p=Point2f(p);q=Point2f(q)
    lines!(a,Point2f[p,q];color=color,linewidth=width)
    u=Point2f(q-p);u/=norm(u);v=Point2f(-u[2],u[1])
    poly!(a,Point2f[q,q-head*u+.4head*v,q-head*u-.4head*v];color=color)
end
function intro_canvas()
    f=Figure(size=(1050,245),figure_padding=(9,9,5,5))
    a=Axis(f[1,1];limits=(0,100,0,24))
    hidedecorations!(a);hidespines!(a);f,a
end
intro_donor(t)=sin(2pi*t/1.6)+.19sin(2pi*t/.49)
intro_math(s)=CairoMakie.Makie.LaTeXStrings.latexstring(s)
function intro_id(a,x,y,s;color=INTRO_INK,size=20)
    text!(a,x,y;text=intro_math(s),align=(:center,:center),fontsize=size,color=color)
end
function intro_network(a,x,y,stage;scale=1.,color=INTRO_ORANGE,seed=31)
    rng=MersenneTwister(seed)
    theta=range(0,2pi;length=160)
    poly!(a,Point2f[Point2f(x+3.9scale*cos(t),y+3.9scale*sin(t)) for t in theta];
        color=(color,.025),strokecolor=(color,.85),strokewidth=1.)
    positions=vcat([Point2f(x+2.8scale*cos(t),y+2.8scale*sin(t)) for t in range(.2,2pi+.2;length=9)[1:8]],
        [Point2f(x+1.3scale*cos(t),y+1.3scale*sin(t)) for t in range(.5,2pi+.5;length=7)[1:6]])
    for i in eachindex(positions),j in i+1:length(positions)
        rand(rng)<.22&&lines!(a,Point2f[positions[i],positions[j]];color=(INTRO_INK,.32),linewidth=.8)
    end
    for (i,j) in ((1,4),(4,7),(7,1))
        p=positions[i];q=positions[j]
        intro_arrow(a,p,.85q+.15p;color=(INTRO_INK,.6),width=.8,head=.28scale)
    end
    for (i,p) in enumerate(positions)
        activation=.2+.72*(sin(.8i+1.1stage)+1)/2
        scatter!(a,[p];color=(color,activation),markersize=7.5sqrt(scale),
            strokecolor=color,strokewidth=.65)
    end
end
function intro_save(f,name)
    for ext in ("pdf","svg");save(joinpath(INTRO_OUT,name*"."*ext),f;pt_per_unit=.55);end
    save(joinpath(INTRO_OUT,name*".png"),f;px_per_unit=1.5)
end
function make_resync_intuition()
    mkpath(INTRO_OUT)
    f,a=intro_canvas();t=range(0,4;length=600);xx=4 .+59 .*t./4
    # Autonomous ensemble candidates and correction boundaries.
    for x in range(4,63;length=5)
        lines!(a,[x,x],[3.,21.8];color=(INTRO_GRAY,.28),linewidth=.7)
    end
    lines!(a,[4,64],[3.,3.];color=INTRO_GRAY,linewidth=.8)
    for j in 1:14
        yy=12 .+3.4 .*[intro_donor(v+.025j)+.035sin(3v+.7j) for v in t]
        lines!(a,xx,yy;color=(INTRO_CONTEXT,.7),linewidth=.85)
    end
    donor=12 .+3.4 .*intro_donor.(t)
    lines!(a,xx,donor;color=INTRO_BLUE,linewidth=2.)
    for (j,drift) in enumerate((2.4,1.7,-2.1))
        yy=12 .+3.4 .*[intro_donor(v+.025j) for v in t].+drift.*max.(t.-2,0).^1.4
        lines!(a,xx,yy;color=(INTRO_ORANGE,j==1 ? 1. : .65),linewidth=j==1 ? 2.3 : 1.45)
        scatter!(a,[last(xx)],[last(yy)];color=INTRO_ORANGE,markersize=6)
    end
    # Member identities are carried into panels b/c; these are leader lines, not inputs.
    selected_y=12+3.4intro_donor(4.025)+2.4*2^1.4
    theta=range(0,2pi;length=90)
    lines!(a,63 .+1.0 .*cos.(theta),selected_y .+1.0 .*sin.(theta);color=INTRO_ORANGE,linewidth=1.2)
    lines!(a,Point2f[(64.4,selected_y),(72.,18.),(80.5,18.)];color=INTRO_ORANGE,linewidth=1.4)
    lines!(a,Point2f[(64.,last(donor)),(72.,7.),(80.5,7.)];color=INTRO_BLUE,linewidth=1.4)
    intro_network(a,85.,18.,1;scale=.9)
    intro_network(a,85.,7.,2;scale=.9,color=INTRO_BLUE,seed=47)
    intro_id(a,93.,18.,"i";color=INTRO_ORANGE)
    intro_id(a,93.,7.,"d";color=INTRO_BLUE)
    intro_save(f,"a_departure")

    f,a=intro_canvas();xx=range(3,33;length=600);time=range(0,4;length=600)
    lines!(a,xx,11 .+2.6 .*intro_donor.(time);color=INTRO_BLUE,linewidth=2.1)
    for x in range(3,33;length=5)
        lines!(a,[x,x],[6.7,15.1];color=(INTRO_BLUE,.28),linewidth=.7)
    end
    intro_arrow(a,(5.,4.3),(31.,4.3);color=INTRO_BLUE,width=1.,head=.65)
    intro_arrow(a,(35.,11.),(44.,11.);color=INTRO_BLUE)
    intro_network(a,6.,20.5,2;scale=.55,color=INTRO_BLUE,seed=47)
    intro_id(a,11.,20.5,"d";color=INTRO_BLUE,size=18)
    intro_arrow(a,(6.,17.8),(6.,15.2);color=INTRO_BLUE,width=1.,head=.55)
    # One recipient reservoir, rather than a chain that could imply different models.
    intro_network(a,53.,11.,4;scale=2.05)
    intro_id(a,53.,1.4,"i";color=INTRO_ORANGE,size=19)
    # Restore the stored initial state before replaying the complete input history.
    theta=range(0,2pi;length=80)
    poly!(a,Point2f[Point2f(39+1.15cos(t),20+1.15sin(t)) for t in theta];
        color=(INTRO_ORANGE,.18),strokecolor=INTRO_ORANGE,strokewidth=1.)
    intro_arrow(a,(40.5,19.8),(47.4,16.6);color=INTRO_ORANGE,width=1.1,head=.65)
    intro_id(a,39.,22.5,raw"r_i(0)";size=19)
    # The observable driven response replaces an abstract latent-state trajectory.
    # This constructed transient is an illustration, not a simulated convergence result.
    tt=range(0,4;length=500);sx=range(73.,98.;length=length(tt))
    input=11 .+2.6 .*intro_donor.(tt)
    response=input.+4.4 .*exp.(-1.3 .*tt).*cos.(.7 .*tt)
    intro_arrow(a,(62.5,11.),(71.,11.);color=INTRO_ORANGE,width=1.4)
    lines!(a,sx,input;color=INTRO_BLUE,linewidth=1.1,linestyle=:dash)
    lines!(a,sx,response;color=INTRO_ORANGE,linewidth=2.0)
    lines!(a,[73.,98.],[6.7,6.7];color=INTRO_GRAY,linewidth=.8)
    intro_id(a,85.,18.5,raw"y_i^{\mathrm{driven}}(t)";color=INTRO_ORANGE,size=18)
    intro_id(a,18.,16.5,raw"y_d(t)";color=INTRO_BLUE,size=18)
    # Terminal state is only a handoff marker, rather than another abstract plot.
    scatter!(a,[64.],[21.];color=INTRO_ORANGE,marker=:diamond,markersize=7)
    lines!(a,[59.,64.],[17.3,20.4];color=INTRO_ORANGE,linewidth=.9)
    intro_id(a,73.,22.,raw"r_i(T_s)";size=18)
    intro_save(f,"b_history_warmup")

    f,a=intro_canvas();intro_network(a,14.,13.,4;scale=1.1)
    intro_id(a,21.,17.5,"i";color=INTRO_ORANGE,size=19)
    # The same terminal-state diamond links b to the initial state used in c.
    scatter!(a,[14.],[19.5];color=INTRO_ORANGE,marker=:diamond,markersize=7)
    lines!(a,[14.,14.],[18.9,17.5];color=INTRO_ORANGE,linewidth=1.)
    intro_id(a,16.,22.5,raw"r_i(T_s)";size=18)
    # One terminal donor sample starts the first autonomous update; later input is own feedback.
    scatter!(a,[3.],[19.5];color=INTRO_BLUE,markersize=6)
    intro_id(a,3.7,22.5,raw"y_d(T_s)";color=INTRO_BLUE,size=17)
    intro_arrow(a,(4.5,18.5),(10.5,15.6);color=INTRO_BLUE,width=1.2,head=.6)
    # Contrast the archived departing candidate with a new post-replay own forecast.
    # No line interpolates or rewrites the pre-replay candidate across the boundary.
    lines!(a,[27.,98.],[7.,7.];color=INTRO_GRAY,linewidth=.8)
    lines!(a,[52.,52.],[7.,22.];color=(INTRO_INK,.55),linewidth=.9,linestyle=:dash)
    intro_id(a,52.,4.7,raw"T_s";size=17)
    past=range(2.,4.;length=300);xp=range(27.,51.5;length=length(past))
    future=range(4.02,7.5;length=500);xx=range(53.5,97;length=length(future))
    for j in 1:12
        yp=13 .+2.7 .*[intro_donor(v+.02j)+.04sin(v*3+j) for v in past]
        yy=13 .+2.7 .*[intro_donor(v+.02j)+.04sin(v*3+j) for v in future]
        lines!(a,xp,yp;color=(INTRO_CONTEXT,.7),linewidth=.8)
        lines!(a,xx,yy;color=(INTRO_CONTEXT,.7),linewidth=.8)
    end
    previous=13 .+2.7 .*intro_donor.(past .+ .025).+2.4 .*max.(past.-2,0).^1.4
    lines!(a,xp,previous;color=INTRO_ORANGE,linewidth=2.1,linestyle=:dash)
    scatter!(a,[last(xp)],[last(previous)];color=INTRO_ORANGE,markersize=6)
    own(t)=sin(2pi*t/1.64+.13)+.2sin(2pi*t/.51+.2)
    yy=13 .+2.7 .*own.(future)
    lines!(a,xx,yy;color=INTRO_ORANGE,linewidth=2.4)
    scatter!(a,[first(xx)],[first(yy)];color=:white,strokecolor=INTRO_ORANGE,strokewidth=1.3,markersize=6.5)
    # The reconstructed reservoir produces only the new forecast, not its archived past.
    lines!(a,Point2f[(20.,13.),(23.,13.),(23.,6.3),(53.5,6.3)];color=INTRO_ORANGE,linewidth=1.4)
    intro_arrow(a,(53.5,6.3),(53.5,first(yy)-.6);color=INTRO_ORANGE,width=1.4,head=.65)
    intro_arrow(a,(31.,6.3),(40.,6.3);color=INTRO_ORANGE,width=1.4,head=.65)
    # Feedback uses the new recipient output, rather than another donor input stream.
    index=410;x=xx[index];y=yy[index]
    lines!(a,Point2f[(x,y),(x,3.0),(14.,3.0),(14.,8.0)];color=INTRO_ORANGE,linewidth=1.45)
    intro_arrow(a,(14.,8.),(14.,9.0);color=INTRO_ORANGE,head=.7)
    intro_arrow(a,(64.,3.),(52.,3.);color=INTRO_ORANGE,head=.8)
    intro_id(a,49.,1.3,raw"y_i(t)";color=INTRO_ORANGE,size=18)
    intro_save(f,"c_autonomous_return")

    f=Figure(size=(1050,34),figure_padding=0)
    Legend(f[1,1],[LineElement(color=INTRO_CONTEXT,linewidth=1.5),LineElement(color=INTRO_BLUE,linewidth=2),
        LineElement(color=INTRO_ORANGE,linewidth=2)],
        ["Other ensemble members","Donor d / generated history","Recipient i / output response"];
        orientation=:horizontal,framevisible=false,labelsize=16,tellwidth=false)
    save(joinpath(INTRO_OUT,"legend.pdf"),f;pt_per_unit=.55)
    open(io->TOML.print(io,Dict("generator_sha256"=>bytes2hex(open(sha256,@__FILE__)),
        "scope"=>"Panel a geometry preserved; context traces use the midpoint blue-gray. Panel b contrasts donor history with a constructed recipient driven response, not a latent-state plot. Panel c separates an archived pre-replay departure from a constructed new autonomous forecast. Neither transient decay nor recovery is measured or guaranteed. Terminal donor sample and own-output feedback remain distinct.");sorted=true),joinpath(INTRO_OUT,"provenance.toml"),"w")
end
if abspath(PROGRAM_FILE)==@__FILE__;make_resync_intuition();end
