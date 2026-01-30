import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import uuid
import random
import winsound


class SumoTrafficEnv(gym.Env):
    """
    Stable SUMO Gymnasium env (single intersection) with:
    - PPO control
    - Ambulance priority override
    - Safe TraCI label connection (prevents peer shutdown issues)
    """

    def __init__(
        self,
        sumo_cfg="config.sumocfg",
        sumo_binary="sumo-gui",
        max_steps=1000,
        min_green=10,
        render_mode="none",
    ):
        super().__init__()

        self.sumo_cfg = sumo_cfg
        self.sumo_binary = sumo_binary
        self.max_steps = max_steps
        self.min_green = min_green
        self.render_mode = render_mode

        self.ambulance_prob = 0.02  # chance per step
        self.ambulance_min_gap = 50  # minimum steps between ambulances
        self.last_ambulance_step = -9999
        self.ambulance_count = 0
        self.episode_id = 0

        # Observation: [NS_queue, EW_queue, phase_dir, amb_ns, amb_ew]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([500, 500, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: 0=keep, 1=switch
        self.action_space = spaces.Discrete(2)

        self.step_count = 0
        self.last_switch_step = 0
        self.tl_id = None

        self.ns_incoming_edges = ["N2TL", "S2TL"]
        self.ew_incoming_edges = ["E2TL", "W2TL"]

        # unique label for TraCI connection
        self.label = f"env_{uuid.uuid4().hex[:8]}"
        self.conn = None

        self.flash_interval = 2  # change to 3 for faster flashing
        self.flash_state = False  # False=red, True=white

        self.siren_playing = False
        self.siren_file = "siren.wav"

    # -------------------------
    # SUMO START / STOP (SAFE)
    # -------------------------
    def _start_sumo(self):
        binary = "sumo-gui" if self.render_mode == "human" else self.sumo_binary

        cmd = [
            binary,
            "-c",
            self.sumo_cfg,
            "--no-step-log",
            "true",
            "--quit-on-end",
            "false",
            "--start",
            "false",
            "--gui-settings-file",
            "my_gui_settings.xml",
        ]

        if self.render_mode == "human":
            cmd += ["--delay", "100"]

        # IMPORTANT: use label to avoid conflicts
        traci.start(cmd, label=self.label)
        self.conn = traci.getConnection(self.label)

        self.tl_id = self.conn.trafficlight.getIDList()[0]

    def _close_sumo(self):
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        self.conn = None

        # also close default traci (if any)
        try:
            traci.close(False)
        except Exception:
            pass

    # -------------------------
    # HELPERS
    # -------------------------
    def _edge_queue(self, edge_id: str) -> int:
        q = 0
        n_lanes = self.conn.edge.getLaneNumber(edge_id)
        for i in range(n_lanes):
            lane_id = f"{edge_id}_{i}"
            q += self.conn.lane.getLastStepHaltingNumber(lane_id)
        return q

    def _get_queues(self):
        ns = sum(self._edge_queue(e) for e in self.ns_incoming_edges)
        ew = sum(self._edge_queue(e) for e in self.ew_incoming_edges)
        return ns, ew

    def _get_phase_direction(self):
        phase = self.conn.trafficlight.getPhase(self.tl_id)
        return 0 if phase in [0, 1] else 1  # 0=NS, 1=EW

    def _set_green_direction(self, direction: int):
        if direction == 0:
            self.conn.trafficlight.setPhase(self.tl_id, 0)  # NS green
        else:
            self.conn.trafficlight.setPhase(self.tl_id, 2)  # EW green

    def _detect_ambulance(self):
        amb_ns = 0
        amb_ew = 0

        for vid in self.conn.vehicle.getIDList():
            vtype = self.conn.vehicle.getTypeID(vid)

            if "ambulance" not in vtype.lower() and "amb" not in vid.lower():
                continue

            road = self.conn.vehicle.getRoadID(vid)

            if road in self.ns_incoming_edges:
                amb_ns = 1
            elif road in self.ew_incoming_edges:
                amb_ew = 1

        return amb_ns, amb_ew

    def maybe_spawn_ambulance(self):
        # avoid too many ambulances too close
        if (self.step_count - self.last_ambulance_step) < self.ambulance_min_gap:
            return

        # random spawn
        if random.random() > self.ambulance_prob:
            return

        # pick a random direction
        route_choices = ["N_S", "S_N", "E_W", "W_E"]
        route_id = random.choice(route_choices)

        amb_id = f"AMB_{self.episode_id}_{self.ambulance_count}"
        self.ambulance_count += 1
        self.last_ambulance_step = self.step_count

        try:
            self.conn.vehicle.add(
                vehID=amb_id,
                routeID=route_id,
                typeID="ambulance",
                depart=str(self.conn.simulation.getTime()),
            )

            self.conn.vehicle.setColor(amb_id, (255, 0, 0, 255))

        except Exception:
            # if spawn fails (lane blocked etc), ignore
            return

    def flash_ambulances(self):
        # flash only every few steps
        if self.step_count % self.flash_interval != 0:
            return

        self.flash_state = not self.flash_state

        # choose color
        if self.flash_state:
            color = (255, 255, 255, 255)  # white
        else:
            color = (255, 0, 0, 255)  # red

        # set color for all ambulances currently in simulation
        for vid in self.conn.vehicle.getIDList():
            try:
                if self.conn.vehicle.getTypeID(vid) == "ambulance":
                    self.conn.vehicle.setColor(vid, color)
            except Exception:
                pass

    def handle_siren_sound(self, ambulance_present: bool):
        if ambulance_present and not self.siren_playing:
            self.siren_playing = True
            winsound.PlaySound(
                self.siren_file,
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
            )

        if not ambulance_present and self.siren_playing:
            self.siren_playing = False
            winsound.PlaySound(None, winsound.SND_ASYNC)

    # -------------------------
    # GYM API
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # close old sim if still open
        self._close_sumo()

        # start fresh
        self._start_sumo()

        self.step_count = 0
        self.last_switch_step = 0

        self.last_ambulance_step = -9999
        self.ambulance_count = 0

        self.episode_id += 1

        # step once
        self.conn.simulationStep()

        ns, ew = self._get_queues()
        phase_dir = self._get_phase_direction()
        amb_ns, amb_ew = self._detect_ambulance()

        obs = np.array([ns, ew, phase_dir, amb_ns, amb_ew], dtype=np.float32)
        info = {"ns_queue": ns, "ew_queue": ew, "amb_ns": amb_ns, "amb_ew": amb_ew}
        return obs, info

    def step(self, action):
        self.step_count += 1

        current_dir = self._get_phase_direction()
        amb_ns, amb_ew = self._detect_ambulance()

        # 🚑 Emergency preemption (override PPO)
        if amb_ns == 1 and amb_ew == 0:
            self._set_green_direction(0)
            self.last_switch_step = self.step_count

        elif amb_ew == 1 and amb_ns == 0:
            self._set_green_direction(1)
            self.last_switch_step = self.step_count

        else:
            # PPO control
            if action == 1:
                if (self.step_count - self.last_switch_step) >= self.min_green:
                    self._set_green_direction(1 - current_dir)
                    self.last_switch_step = self.step_count

        # advance sim
        self.maybe_spawn_ambulance()

        # flash lights
        self.flash_ambulances()

        # 🔊 siren sound (start/stop)
        ambulance_present = amb_ns == 1 or amb_ew == 1
        self.handle_siren_sound(ambulance_present)

        self.conn.simulationStep()

        ns, ew = self._get_queues()
        phase_dir = self._get_phase_direction()
        amb_ns, amb_ew = self._detect_ambulance()

        reward = -(ns + ew)

        terminated = False
        truncated = self.step_count >= self.max_steps

        # stop if SUMO ended (no vehicles left)
        if self.conn.simulation.getMinExpectedNumber() == 0:
            truncated = True

        obs = np.array([ns, ew, phase_dir, amb_ns, amb_ew], dtype=np.float32)
        info = {
            "ns_queue": ns,
            "ew_queue": ew,
            "total_queue": ns + ew,
            "amb_ns": amb_ns,
            "amb_ew": amb_ew,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        self._close_sumo()
