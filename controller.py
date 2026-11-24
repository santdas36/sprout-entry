from pathlib import Path

import numpy as np

from sprout_challenge import Controller

HERE = Path(__file__).resolve().parent


class Sprout(Controller):
    v_fast = 2.0        # m/s commanded on smooth ground
    v_edge = 1.9        # m/s near sharp edges (set lower to slow for hurdles and stairs)
    v_bar = 1.7         # m/s approaching a barrier
    v_hurdle = 1.5      # m/s near hurdles (thin ridges in the scan)
    hurdle_thr = 0.05   # m: a ridge must stand this far above both neighbours
    edge_thr = 0.065    # m: height jump between neighbouring scan rows that counts as an edge
    slow_back = 0.6     # m behind the torso an edge still slows us
    accel, decel = 1.0, 3.0
    jump = True         # go over barriers (False: walk around through the bypass lane)

    def reset(self, info):
        z = np.load(HERE / "policy.npz")
        self.mean, self.std = z["mean"], z["std"]
        n = len([k for k in z.files if k.startswith("W")])
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
        self.nin = self.layers[0][0].shape[0]
        self.pose = np.asarray(info["default_pose"], float)
        self.last = np.zeros(12)
        self.barriers = sorted(info.get("barriers", []), key=lambda b: b["x0"])
        self.dt = info["dt"]
        self.sx = info["scan_points"][:, 0, 0]
        self.edges = []
        self.hurdles = []
        self.v = 0.6

    # ---------------------------------------------------------------- planning
    def bar_inputs(self, x):
        for b in self.barriers:
            if b["x1"] + 0.8 > x:
                d = b["x0"] - x
                if d < 5.0:
                    return np.array([max(d, 0.0) / 5.0, b["height"]])
                break
        return np.array([1.0, 0.0])

    def on_bridge(self, obs):
        h = obs.height_scan
        near = (self.sx > -0.6) & (self.sx < 1.0)
        centre = h[near][:, 2:5].mean()
        sides = np.r_[h[near][:, 0], h[near][:, 6]].min()
        return centre - sides > 0.08

    def lane(self, x, obs):
        if self.jump or self.on_bridge(obs):
            return 0.0
        for b in self.barriers:
            if b["x0"] - 8.0 <= x <= b["x1"] + 1.2:
                return b["bypass_y"]
        return 0.0

    def see(self, obs):
        h = obs.height_scan[:, 2:5].mean(axis=1)
        jump = np.abs(np.diff(h))
        x0, c = obs.position[0], np.cos(obs.yaw)
        for i in np.nonzero(jump > self.edge_thr)[0]:
            xe = x0 + c * 0.5 * (self.sx[i] + self.sx[i + 1])
            if not any(abs(xe - e) < 0.15 for e in self.edges):
                self.edges.append(xe)
        # hurdles: a thin ridge, up then back down within one or two scan rows (stairs and ramps stay up)
        for i in range(1, len(h) - 1):
            back = h[i] - h[i - 1]
            fwd = max(h[i] - h[i + 1], h[i] - h[i + 2] if i + 2 < len(h) else 0.0)
            if back > self.hurdle_thr and fwd > self.hurdle_thr:
                xh = x0 + c * self.sx[i]
                if not any(abs(xh - e) < 0.3 for e in self.hurdles) and not self.near_barrier(xh):
                    self.hurdles.append(xh)

    def near_barrier(self, x):
        return any(b["x0"] - 0.5 <= x <= b["x1"] + 0.5 for b in self.barriers)

    def target_speed(self, x):
        v = self.v_fast
        for b in self.barriers:
            if b["x0"] - 4.0 <= x <= b["x1"] + 1.0:
                return self.v_bar if self.jump else v
        for e in self.edges:
            if x - self.slow_back <= e <= x + 1.6:
                v = min(v, self.v_edge)
        for e in self.hurdles:
            if x - self.slow_back <= e <= x + 1.6:
                v = min(v, self.v_hurdle)
        return v

    # ---------------------------------------------------------------- control
    def act(self, obs):
        self.see(obs)
        x = obs.position[0]
        vt = self.target_speed(x)
        self.v = min(vt, self.v + self.accel * self.dt) if vt > self.v else max(vt, self.v - self.decel * self.dt)
        y, yaw = obs.position[1] - self.lane(x, obs), obs.yaw
        want = np.clip(-0.4 * y, -0.45, 0.45)
        vy = np.clip(-0.5 * y, -0.2, 0.2) * np.cos(yaw)
        wz = np.clip(-1.5 * (yaw - want), -0.6, 0.6)
        scan = np.clip(-obs.height_scan - 0.85, -1.0, 1.0).ravel()
        parts = [obs.base_lin_vel, obs.gyro, obs.gravity, obs.joint_pos - self.pose, obs.joint_vel,
                 self.last, [self.v, vy, wz], scan]
        if self.nin == 141:
            parts.append(self.bar_inputs(x))
        h = (np.hstack(parts) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = h / (1.0 + np.exp(-h))
        self.last = np.tanh(h[:12])
        return self.pose + 0.5 * self.last
