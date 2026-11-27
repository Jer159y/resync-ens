# ============================================================
# robust_run.jl
#
# 원격/무인 실행(tmux + SSH)을 위한 최소한의 관측성 유틸리티.
# 세션이 죽어도 "어디까지 진행됐는지"를 SSH로만 확인할 수 있게
# 로그 파일과 heartbeat 상태 파일을 남긴다.
#
# - log 파일: append-only, 매 줄마다 flush → 프로세스가 죽는 순간까지의
#   기록이 항상 디스크에 남아 있다.
# - status 파일: 매번 tmp에 쓰고 rename으로 교체 (원자적 덮어쓰기) →
#   읽는 시점에 half-written 파일을 볼 일이 없다.
#
# JSON 계열 패키지가 Project.toml에 없으므로 `key=value` 줄글 텍스트로 쓴다.
# ============================================================

using Dates

struct RunLog
    log_path::String
    status_path::String
    start_time::DateTime
end

"""
    init_run_log(output_dir, name) -> RunLog

`output_dir`에 `<name>_log.txt`(누적 로그)와 `<name>_status.txt`(최신 상태
스냅샷) 경로를 준비하고 시작 메시지를 기록한다.
"""
function init_run_log(output_dir::AbstractString, name::AbstractString)
    mkpath(output_dir)
    log_path = joinpath(output_dir, "$(name)_log.txt")
    status_path = joinpath(output_dir, "$(name)_status.txt")
    rl = RunLog(log_path, status_path, now())

    open(log_path, "a") do io
        println(io, "="^60)
        println(
            io,
            "[$(now())] run started  pid=$(getpid())  host=$(gethostname())  " *
            "julia=$(VERSION)  threads=$(Threads.nthreads())",
        )
        flush(io)
    end

    heartbeat!(rl, "started")
    return rl
end

"""
    log_line!(rl, msg)

stdout에 출력하고, 같은 줄을 로그 파일에 append + flush한다.
프로세스가 이 함수 호출 직후 죽더라도 이 줄까지는 디스크에 남는다.

`on_progress` 콜백에서 흔히 `@showprogress`/`Progress` 진행 바가 떠 있는 도중에
호출되는데, ProgressMeter는 프레임 사이에 개행 없이 "\\r" + 줄 끝까지 지우기로만
다시 그린다. 이 함수가 그냥 `println`만 하면 그 텍스트가 진행 바의 마지막 프레임
뒤에 그대로 이어붙어 나온다(줄바꿈 없이 같은 줄에 얹힘) — 그래서 출력 전에 먼저
현재 줄을 지운다.
"""
function log_line!(rl::RunLog, msg::AbstractString)
    stamped = "[$(now())] $msg"
    print(stdout, "\r\u1b[K")
    println(stamped)
    flush(stdout)
    open(rl.log_path, "a") do io
        println(io, stamped)
        flush(io)
    end
    return nothing
end

"""
    heartbeat!(rl, stage; extra=Dict())

현재 단계/경과 시간/PID/peak RSS를 `status.txt`에 기록한다. tmp 파일에 쓴 뒤
`mv`로 교체하므로 SSH로 아무 때나 읽어도 항상 완전한 내용만 보인다.

PID가 살아있는지는 `ps -p <pid>`로, 언제 마지막으로 갱신됐는지는
`updated_at`으로 SSH에서 바로 확인할 수 있다.
"""
function heartbeat!(rl::RunLog, stage::AbstractString; extra::AbstractDict=Dict{String,Any}())
    tmp_path = rl.status_path * ".tmp"
    elapsed_sec = round(Dates.value(now() - rl.start_time) / 1000; digits=1)
    rss_mb = round(Sys.maxrss() / 1024^2; digits=1)

    open(tmp_path, "w") do io
        println(io, "stage=$stage")
        println(io, "updated_at=$(now())")
        println(io, "elapsed_sec=$elapsed_sec")
        println(io, "pid=$(getpid())")
        println(io, "peak_rss_mb=$rss_mb")
        for (k, v) in extra
            println(io, "$k=$v")
        end
        flush(io)
    end
    mv(tmp_path, rl.status_path; force=true)
    return nothing
end
