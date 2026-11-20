# Consent form: Reachy Mini real home floors dataset

*Template. One form per home and recording session; keep the signed form privately (it is never published).
Record it with `python -m floor_seg.dataset.consent` (see that module); only "consent given", the date and the
signer's role (owner / tenant) go with the data, never your name.*

## What is recorded

A small robot (Reachy Mini on a wheeled base, about 40 cm tall) is driven by remote control through the rooms you
choose, for about 10 minutes per room. Its camera, about 35 cm above the floor and pointed slightly down, takes
pictures of the floor and of what stands on it (furniture legs, rugs, cables, objects...). The robot also logs its
own motion (distance travelled, turns) and camera angle.

**Not recorded:** sound, location or address, your name, the date or time of day of the recording.

## What is removed before publication

- Faces are detected automatically and blurred.
- Every picture's file metadata (EXIF: camera, date...) is removed.
- **You review every picture** on a page on the recording laptop and delete any you do not want published
  (people, documents, photos on the walls, anything personal). Nothing is published before your review.
- The recording is identified only by a code such as `home03`.

## How it is published

The pictures and motion data are published as an open dataset, under the **Creative Commons Attribution 4.0
(CC BY 4.0)** licence, on the Hugging Face Hub by Pollen Robotics, to develop and evaluate floor detection for
robots. Under this licence anyone may copy, use and modify the data, including commercially, with attribution.

## Your right to withdraw

You may withdraw your consent at any time, without giving a reason, by contacting
*[contact address, to be completed before use]*. The recording is then deleted from our computers and removed from the next version
of the dataset. Copies that others have already downloaded from a published version cannot be recalled.

## Consent

I own or rent the home recorded in this session and agree that the pictures and robot data recorded there, after
the face blurring and my own review described above, are published under CC BY 4.0.

- Session code: ____________
- I am the: [ ] owner  [ ] tenant
- Consent given: [ ] yes  [ ] no
- Date: ____________
- Signature: ____________
