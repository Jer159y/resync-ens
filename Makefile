.PHONY: verify paper figures test
verify:
	python3 scripts/verify.py
paper:
	python3 scripts/build.py
figures:
	python3 scripts/build.py --redraw
test:
	julia --project=. --startup-file=no tests/runtests.jl
