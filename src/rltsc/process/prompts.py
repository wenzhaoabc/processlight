PROMPTS: dict[str, str] = {}


PROMPTS["SYSTEM"] = """You are an expert traffic signal control agent operating under a strict reasoning framework.
Your task is to select exactly one legal phase for the current intersection based on the provided traffic facts.
"""


PROMPTS["BACKGROUND"] = """## Background Context
You are a traffic signal controller at a four-way intersection with 12 movements: [NL, NT, NR, SL, ST, SR, EL, ET, ER, WL, WT, WR] and 4 phases [ETWT, NTST, ELWL, NLSL]. Your task is to derive the optimal phase through step-by-step reasoning.

Notes:
- `E`, `W`, `N`, `S`: the incoming road direction of vehicles, E for east, W for west, N for north, S for south.
- `T`, `L`, `R`: through, left-turn, and right-turn. Right turns are always allowed and are not selected as signal phases.
- Example: `ET` means vehicles enter from the east and go through.
- Every intersection has four phases [`ETWT`, `NTST`, `ELWL`, `NLSL`]. A phase releases two compatible movements. Example: `ETWT` releases `ET` and `WT`.
- Legal phases for this decision: [{LEGAL_PHASES}].
"""


PROMPTS["RULES"] = """## Decision Rules
Please refer to the following instructions during reasoning:
1. Intersection congestion is mainly dominated by early queued vehicles. Traffic control shall prioritize lanes with long queues and vehicles near the stop line at the local intersection. Vehicles in remote road sections are unlikely to arrive soon and do not need priority consideration, so as to relieve local intersection traffic pressure first.
2. Assess downstream traffic conditions before releasing any traffic movement, and obtain the lane occupancy of downstream intersections through the Adjacent Intersection Context table to prevent traffic spillover congestion. Meanwhile, monitor network boundary pressure to restrict excessive upstream traffic inflow.
3. Avoid long-time waiting of vehicles in a single traffic direction. A waiting duration of 5 minutes is close to severe delay. Timely grant green light release when downstream traffic conditions permit.
4. Unified lane congestion judgment standard based on occupancy: 0-0.2 means free flow, 0.2-0.4 means basically smooth, above 0.4 is mild congestion, and above 0.6 is severe congestion.
"""


PROMPTS["LOCAL"] = """## Current Intersection State
The table below describes the traffic state for each controlled movement at the current intersection.

- Occupancy refers to the instantaneous spatial occupancy of vehicles in the incoming lane.
- Pressure is computed from local queue and downstream queue (Early Queue Length - Lanes Average Downstream Queue Length). Under comparable conditions, the phase that releases maximum pressure can effectively alleviate traffic congestion.
- Segment Veh Occupancy denotes moving vehicle space occupancy on the incoming lane, sorted from near the intersection to far upstream.

{LOCAL_STATE_TABLE}
"""


PROMPTS["NEIGHBOR"] = """## Adjacent Intersection Context
This table shows traffic data of adjacent intersections (queue length, occupancy) of each its phases. The last column `Related Movements` denotes inflow movements to the intersection your are controlling. eg: `WT,NL -> WT,WL` means the traffic flow from the WT/NL lanes of this upstream intersection will flow into the WT/WL lanes of the controlled intersection, which represents the traffic state information of your upstream lanes.

{NEIGHBOR_STATE_TABLE}
"""


PROMPTS["HISTORY"] = """## Recent History
The following history indicators are computed from recent decision steps.
Use them to judge trend, starvation, repeated service, and short-term future impact.

**Recent Phase Decisions:**
- History window: `{HISTORY_WINDOW}` decision steps
- Recent phases: `{RECENT_PHASES}`
- Last phase: `{LAST_PHASE}`

**Phase State Changes**

{PHASE_HISTORY_TABLE}
"""


PROMPTS["FORMAT"] = """## Output Prompt
Reason step-by-step via 7 fixed Level-2 heading sections. The first 6 sections contain reasoning text and no more than 4 atomic <claim>; use empty <claim></claim> if the step is irrelevant. The Final Answer section must contain <phase></phase> only and must not contain <claim>.

A <claim> must be factual, atomic, and directly verifiable from the provided tables or precomputed indicators.

Allowed claim syntax:
- Cell value: `[Row][Column]=value`
- Cell comparison: `[Row][Column]>[Other Row][Column]`, `[Row][Column]>=value`, `[Row][Column]<=value`
- Ranking/extreme: `RANK[Column][Group]=row1 > row2`, `MAX[Column][Group]=row`, `MIN[Column][Group]=row`
- Swapped ranking/extreme: `RANK[Group][Column]=row1 > row2`, `MAX[Group][Column]=row`, `MIN[Group][Column]=row`
- Adjacent context: `ADJ[Direction][Column]=value`, `ADJ[Direction][Row][Column]=value`
- Trend: `TREND[Phase][Indicator]=stable|increasing|decreasing|value`; value may include unit and a short parenthesized note.
- Quantifier: `All [Group][Column]<=value`, `No [Group] has [Column]>=value`

Valid groups include `Movement`, `Phase`, `All Movements`, `Phase Simulation`, and `phase_results`.

Forbidden inside <claim>:
- Do not use `Both`.
- Do not write subjective/free-form claims such as `NTST is the best phase`.
- Do not mix metric types; occupancy thresholds 0.2/0.4/0.6 apply only to occupancy, not queue length.

Examples:
<claim>MAX[Pressure][Phase]=NLSL</claim>
<claim>RANK[Pressure][Phase]=NLSL > ELWL > ETWT > NTST</claim>
<claim>ADJ[South][Queue Length][ST]=2</claim>
<claim>TREND[NLSL][Service Gap]=7 steps </claim>
<claim>[NLSL][Pressure]<[ELWL][Pressure]</claim>

**Seven Standard Reasoning Steps:**

## Phase Mapping
Map valid signal phases to traffic movements and output factual claims.
## Local Demand
Compare queue length, pressure, waiting time and near-stop vehicles.
## Downstream
Evaluate downstream bearing capacity and prevent overflow.
## Neighbor Coordination
Evaluate adjacent intersection impacts via ADJ[] references.
## Flow Trend
Analyze recent trend and service gap via TREND[] references.
## Fairness
Check excessive waiting risk with the 5-minute threshold.
## Final Answer
Synthesize the claims in plain text, select one legal phase, and enclose it with <phase></phase>. Do not write <claim> in this section.
"""
