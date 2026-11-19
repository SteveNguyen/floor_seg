---
license: cc-by-4.0
pretty_name: Reachy Mini real home floors
task_categories:
  - image-segmentation
tags:
  - robotics
  - reachy-mini
  - floor-segmentation
configs:
  - config_name: default
    data_files:
      - split: train
        path: train/**
      - split: validation
        path: validation/**
---

# Reachy Mini real home floors

**Stub, to be completed before any release.** Camera frames recorded in real homes by a Reachy Mini on its
balancing wheeled base, driven by phone teleoperation, close to the floor. Made to evaluate (and later train)
[pollen-robotics/reachy-mini-floor-seg](https://huggingface.co/pollen-robotics/reachy-mini-floor-seg) on real
floors: rugs, doormats, thresholds, shiny, sunlit and dark floors, cables, chair legs, clutter.

Every home in this dataset was recorded with the written consent of its owner or tenant, who reviewed every frame
before release. No floor labels yet (TODO: labelling, same rule as the synthetic set: anything more than 1 cm above
the floor is an obstacle; cables, door thresholds, rug edges and rug surfaces are floor).

## Content

<!-- SESSIONS -->

A session is one recording in one home; a home is in one split only.

`<split>/metadata.jsonl` columns (ImageFolder layout: `file_name` is the image):

| Column | Meaning |
|---|---|
| `file_name` | `images/<session>_<k>.jpg`, RGB 640 x 360 JPEG, through the camera's wide-angle lens (distorted) |
| `session` | session id (`home03`: no address, no names) |
| `calibration` | `calibration/<session>.json`: `image_resolution`, `K` (3 x 3), `D` (OpenCV distortion coefficients) of the frames |
| `t_s` | seconds since the start of the session (no date or time of day) |
| `cam_h_m` | camera height above the floor, m (from the IMU 'up' and the head pose, about 0.35-0.40) |
| `pitch_down_deg` | optical axis angle below the horizon, deg (head pitched about 15 deg down + the base's tilt) |
| `roll_deg` | camera roll: not recorded yet (null) |
| `x_m`, `y_m`, `heading_deg` | wheel odometry pose of the camera in the session's start frame (drifts) |
| `v`, `w` | commanded linear (m/s) and angular (rad/s) speed at that frame |
| `motion` | `still`, `moving` or `turning` |
| `nav_state`, `ctrl` | navigation runtime states (behaviour, base controller) |

## How it was recorded

Reachy Mini on the Poussah balancing base, head camera about 0.35 m above the floor, head pitched 15 deg down,
driven by phone teleoperation with the Navette navigation runtime (protocol: `docs/dataset/RECORDING.md` in the
floor_seg repo). Frames were kept every 0.10 m of travel, 10 deg of turn or 2 s, whichever came first
(`floor_seg.dataset.import_run`).

## Privacy

Before anything left the recording laptop:

1. Wall-clock times, run folder names and file metadata were dropped; sessions are named `homeNN`.
2. Faces were detected (YuNet, OpenCV Zoo, MIT) and blurred (strong Gaussian on the face box enlarged by 30 %);
   every image was re-encoded without EXIF (`floor_seg.dataset.privacy`).
3. The home owner reviewed every frame on a local page and deleted any they did not want published
   (`floor_seg.dataset.review`).
4. Only sessions with written consent were exported (`floor_seg.dataset.consent`, `floor_seg.dataset.export`).

To have a frame or a whole home removed, open an issue or contact Pollen Robotics (TODO: contact address).

## License

CC BY 4.0. Attribution: Pollen Robotics (TODO: final wording, citation).
