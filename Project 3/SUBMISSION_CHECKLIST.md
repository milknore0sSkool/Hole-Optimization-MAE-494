# Submission checklist

## Already completed in this package

- [x] Minimal PDE formulation
- [x] Difficulty knob
- [x] Conventional baseline
- [x] P2 individual loss histories
- [x] P3 error vs difficulty
- [x] P4 pointwise error fields
- [x] P4 Fourier error spectrum
- [x] P5 training/inference cost
- [x] Path B controlled diagnosis: supervised capacity test
- [x] Path C targeted optimizer remedy
- [x] Fixed seed and runnable code
- [x] Python data loader/download path
- [x] MATLAB loader
- [x] Report draft
- [x] Recommendation memo draft
- [x] Presentation outline

## Must be completed by the team from a real researcher interaction

- [ ] Reserve/contact researcher according to course rules
- [ ] Interview researcher
- [ ] Fill outreach log
- [ ] Replace generic problem brief with their scientific problem and value
- [ ] Record their current solver + one-run cost + number of solves
- [ ] Record useful accuracy threshold
- [ ] Explain why Burgers preserves a real difficulty in their problem, or change the minimal case if it does not
- [ ] Send researcher-checked problem brief
- [ ] Adapt recommendation memo to their application
- [ ] Send final memo to researcher
- [ ] Add response if received

## Before GitHub submission

- [ ] Run `python src/download_data.py`
- [ ] Run `python run_all.py` on the final machine
- [ ] Check `results/external_validation.json`
- [ ] Confirm all figures render in `PROJECT_REPORT.md`
- [ ] Add team names and researcher details
- [ ] Push repository publicly only after removing any confidential material
