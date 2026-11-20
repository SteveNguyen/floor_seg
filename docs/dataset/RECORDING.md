# Recording protocol: real home floors

One page for colleagues recording their home with the Reachy Mini on the Poussah base. Goal: real floors the
floor model gets wrong, seen from the robot's own camera (about 0.35 m above the floor, head pitched 15 deg down).

## Before

- Read and sign the consent form ([CONSENT.md](CONSENT.md)) if you own or rent the home; otherwise ask the owner.
- **Keep people out of view**: ask housemates to leave the room you record, move family photos and documents out
  of the floor's way if you can. Pets are fine.
- Charge the base; check the robot stands and drives on the phone.

## Start / stop

*To be completed: the exact Navette commands to start the runtime with recording on, open the phone teleop page,
stop, and find the run folder.*
- The run folder (`navette-YYYYMMDD-HHMMSS`) stays on the laptop; import it with
  `python -m floor_seg.dataset.import_run RUN_DIR STAGING --session homeNN` (one session id per recording, no
  names in it).

## What to drive over and near

Drive slowly, mostly forward, with some turns in place; get close to things (stop ~10 cm before them), then back
off and approach from another angle.

- Rugs: thin and thick, their edges and fringes; doormats, bath mats.
- Door thresholds and floor transitions (tiles to wood, wood to carpet).
- Shiny floors (tiles, varnished wood), sunlit patches and strong shadows, dark floors.
- Cables lying on the floor, power strips.
- Chair and table legs, under tables and beds (as far as the robot fits), sofa skirts.
- Pet bowls, shoes, toys, bags, boxes: everyday clutter. Do not tidy up.

## How much

- About **10 minutes per room**, every room you are happy to share.
- **Two times of day** (daylight and evening), and lights on and off where it makes a difference.
- One recording = one session; several sessions per home are fine.

## After

Run the privacy pass and the review with the home owner (`floor_seg.dataset.privacy`, `floor_seg.dataset.review`),
record the consent (`floor_seg.dataset.consent`). Nothing leaves the laptop before these steps.
