"""Floor segmentation on the live camera, ON THE ROBOT (CM4). Does not move the robot. Timing / CPU benchmark.

    /venvs/apps_venv/bin/python floor_live.py --model floor.onnx --duration 30 --rate 10 --threads 1

(README numbers: --threads 1, ORT spinning off = the default here.) Also the reference for robot-side
inference without torch: the preprocessing and the logit-difference threshold below match floor_seg/model.py.

Per processed frame: resize (camera frame -> 320x192) + normalize, ONNX Runtime fp32 inference, floor mask
(softmax > --thr at model resolution). --rate caps processing (newest frame is taken, others dropped),
0 = as fast as possible. --busy N starts N busy-loop processes to stand in for other apps loading the CPU.
Reports per-stage median / p90 / p99 ms, achieved Hz, this process's CPU (% of one core), whole-machine
CPU % (from /proc/stat) and load average. --save DIR writes an overlay JPEG every --save-every frames.
Standalone (no torch, no floor_seg import): needs only numpy, cv2, onnxruntime and reachy_mini.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from reachy_mini import ReachyMini

IN_W, IN_H = 320, 192  # floor_seg/model.py
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def busy():
    while True:
        pass


def cpu_times():
    """(busy, total) jiffies over all cores, from /proc/stat."""
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    idle = v[3] + v[4]  # idle + iowait
    return sum(v) - idle, sum(v)


def stats(xs):
    if not xs:  # --no-infer
        return None
    return {"med": round(float(np.median(xs)), 1), "p90": round(float(np.percentile(xs, 90)), 1),
            "p99": round(float(np.percentile(xs, 99)), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--rate", type=float, default=10, help="max processed Hz, 0 = unlimited")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--busy", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--save", help="folder for overlay JPEGs")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--spin", action="store_true", help="let ORT threads busy-wait (ORT default)")
    ap.add_argument("--no-infer", action="store_true", help="camera stream only: cost of receiving frames")
    ap.add_argument("--baseline", action="store_true", help="only measure machine CPU %% for --duration, no camera")
    args = ap.parse_args()
    if args.baseline:  # what the daemon (and anything else running) uses without us
        c0, w0 = cpu_times(), time.monotonic()
        time.sleep(args.duration)
        c1 = cpu_times()
        print(json.dumps({"baseline_s": round(time.monotonic() - w0),
                          "machine_cpu_pct": round(100 * (c1[0] - c0[0]) / (c1[1] - c0[1])),
                          "loadavg": os.getloadavg()}))
        return

    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = args.threads, 1
    if not args.spin:  # ORT worker threads busy-wait between runs by default: wasted CPU on a shared machine
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    hogs = [mp.Process(target=busy, daemon=True) for _ in range(args.busy)]
    for h in hogs:
        h.start()
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    t_prep, t_inf, t_post, t_all, n_drop = [], [], [], [], 0
    with ReachyMini(connection_mode="localhost_only") as mini:
        while mini.media.get_frame() is None:  # wait for the stream
            pass
        for _ in range(3):  # warm-up
            sess.run(None, {"input": np.zeros((1, 3, IN_H, IN_W), np.float32)})
        c0, w0, p0 = cpu_times(), time.monotonic(), time.process_time()
        next_t = w0
        n = 0
        while time.monotonic() - w0 < args.duration:
            frame = mini.media.get_frame()  # blocks up to ~20 ms, None if no new frame
            if frame is None:
                continue
            shape = frame.shape[:2]
            if args.no_infer:
                n += 1
                continue
            if args.rate and time.monotonic() < next_t:  # over the rate cap: drop this frame
                n_drop += 1
                continue
            next_t = max(next_t + 1 / args.rate, time.monotonic()) if args.rate else 0
            t0 = time.perf_counter()
            x = cv2.resize(frame, (IN_W, IN_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32)
            x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].copy()
            t1 = time.perf_counter()
            z = sess.run(None, {"input": x})[0][0]
            t2 = time.perf_counter()
            floor = (z[1] - z[0]) > np.log(args.thr / (1 - args.thr))  # softmax(z)[1] > thr
            t3 = time.perf_counter()
            t_prep.append((t1 - t0) * 1e3)
            t_inf.append((t2 - t1) * 1e3)
            t_post.append((t3 - t2) * 1e3)
            t_all.append((t3 - t0) * 1e3)
            if args.save and n % args.save_every == 0:
                small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                m = cv2.resize(floor.astype(np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST) > 0
                small[m] = (0.5 * small[m] + [0, 127, 0]).astype(np.uint8)
                cv2.imwrite(os.path.join(args.save, f"{n:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            n += 1
        wall = time.monotonic() - w0
        proc = time.process_time() - p0
        c1 = cpu_times()
    for h in hogs:
        h.terminate()
    print(json.dumps({
        "threads": args.threads, "busy": args.busy, "rate_cap": args.rate, "frame": list(shape),
        "hz": round(n / wall, 1), "dropped": n_drop,
        "prep_ms": stats(t_prep), "infer_ms": stats(t_inf), "post_ms": stats(t_post), "total_ms": stats(t_all),
        "proc_cpu_pct_of_core": round(100 * proc / wall), "machine_cpu_pct": round(100 * (c1[0] - c0[0]) / (c1[1] - c0[1])),
        "loadavg": os.getloadavg()}))


if __name__ == "__main__":
    main()
