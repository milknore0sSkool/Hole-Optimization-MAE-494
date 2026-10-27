# Outreach and interview — fill from the real conversation

The course requires a real researcher interview. This file is intentionally not fabricated.

## Outreach log

| Date | Person | Lab / department | Why their PDE work fits | Outcome |
|---|---|---|---|---|
| [date] | [name] | [lab] | [specific PDE/application] | [reply / meeting / no response] |

## Researcher-checked problem brief

**Researcher:** [name and role]  
**Scientific question:** [what they are trying to learn/design/predict]  
**Real-world value:** [why it matters]  
**Governing equations:** [their PDEs]  
**Domain / BC / IC:** [details]  
**Current solver:** [software/method]  
**One-run cost:** [wall time + hardware]  
**How many solves they need:** [design sweep / UQ / calibration / control]  
**Main computational bottleneck:** [geometry / mesh / stiffness / sharp gradients / inverse data / etc.]  
**Available measurements:** [type, density, noise]  
**Useful accuracy threshold:** [number supplied by researcher]  
**Prior PINN/PINO attempt:** [yes/no; what happened]  
**What minimal result would change their next decision:** [answer]

## Mapping this repository to the interview

Use the Burgers experiment only if the researcher's real problem contains a comparable **convection/sharp-gradient optimization difficulty**. If their bottleneck is instead inverse coefficient identification, high-dimensional inputs, or repeated parametric solves, change the minimal case rather than forcing this one onto their problem.

## Required close-the-loop items

- [ ] Send the researcher a half-page problem brief and ask for corrections.
- [ ] Record whether they approved the summary.
- [ ] Adapt `RECOMMENDATION_MEMO.md` to their actual problem.
- [ ] Send the final memo.
- [ ] Record their response if any.
