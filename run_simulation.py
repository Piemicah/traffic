import traci
import subprocess
import time
import matplotlib.pyplot as plt

PORT = 8813


class TrafficLightAgent:
    def __init__(self, tl_id):
        self.tl_id = tl_id
        self.last_switch = 0
        self.min_green = 10
        self.switch_threshold = 2

        self.ns_incoming_edges = ["N2TL", "S2TL"]
        self.ew_incoming_edges = ["E2TL", "W2TL"]

    def get_edge_queue(self, edge_id: str) -> int:
        queue = 0
        n_lanes = traci.edge.getLaneNumber(edge_id)
        for lane_index in range(n_lanes):
            lane = f"{edge_id}_{lane_index}"
            queue += traci.lane.getLastStepHaltingNumber(lane)
        return queue

    def observe(self):
        ns_queue = sum(self.get_edge_queue(edge) for edge in self.ns_incoming_edges)
        ew_queue = sum(self.get_edge_queue(edge) for edge in self.ew_incoming_edges)
        return ns_queue, ew_queue

    def set_main_green(self, direction: str):
        # Most common generated phases:
        # 0/1 = NS green/yellow, 2/3 = EW green/yellow
        if direction == "NS":
            traci.trafficlight.setPhase(self.tl_id, 0)
        elif direction == "EW":
            traci.trafficlight.setPhase(self.tl_id, 2)

    def act(self, step, ns_queue, ew_queue):
        if step - self.last_switch < self.min_green:
            return

        current_phase = traci.trafficlight.getPhase(self.tl_id)
        current_direction = "NS" if current_phase in [0, 1] else "EW"

        if ns_queue > ew_queue + self.switch_threshold:
            preferred = "NS"
        elif ew_queue > ns_queue + self.switch_threshold:
            preferred = "EW"
        else:
            preferred = current_direction

        if preferred != current_direction:
            self.set_main_green(preferred)
            self.last_switch = step


def compute_waiting_metrics():
    """
    Returns:
      avg_wait_time: average waiting time of vehicles currently in sim
      total_wait_time: sum waiting time of vehicles currently in sim
      veh_count: vehicles currently in sim
    """
    vehicle_ids = traci.vehicle.getIDList()
    if not vehicle_ids:
        return 0.0, 0.0, 0

    waits = [traci.vehicle.getWaitingTime(v) for v in vehicle_ids]
    total_wait = sum(waits)
    avg_wait = total_wait / len(waits)
    return avg_wait, total_wait, len(vehicle_ids)


def run(sim_steps=3600, log_every=5):
    # Start SUMO with TraCI
    subprocess.Popen(
        [
            "sumo-gui",
            "-c",
            "config.sumocfg",
            "--remote-port",
            str(PORT),
            "--delay",
            "100",
        ]
    )
    time.sleep(2)

    traci.init(PORT)

    tl_id = traci.trafficlight.getIDList()[0]
    print("Traffic Light:", tl_id)

    # Phase debug (useful once)
    logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
    print("Number of phases:", len(logic.phases))
    for i, p in enumerate(logic.phases):
        print(i, "duration:", p.duration, "state:", p.state)

    agent = TrafficLightAgent(tl_id)

    # ----- METRIC STORAGE -----
    times = []
    ns_queues = []
    ew_queues = []
    total_queues = []
    avg_waits = []
    total_waits = []
    throughput = []  # arrived vehicles cumulative

    arrived_cumulative = 0

    # ----- MAIN LOOP -----
    for step in range(sim_steps):
        traci.simulationStep()

        # Arrived vehicles in this step
        arrived_now = len(traci.simulation.getArrivedIDList())
        arrived_cumulative += arrived_now

        # Agent observation + action
        ns_queue, ew_queue = agent.observe()
        agent.act(step, ns_queue, ew_queue)

        # Collect metrics periodically
        if step % log_every == 0:
            avg_wait, total_wait, veh_count = compute_waiting_metrics()

            times.append(step)
            ns_queues.append(ns_queue)
            ew_queues.append(ew_queue)
            total_queues.append(ns_queue + ew_queue)
            avg_waits.append(avg_wait)
            total_waits.append(total_wait)
            throughput.append(arrived_cumulative)

            print(
                f"t={step:4d}s | NS={ns_queue:3d} EW={ew_queue:3d} | "
                f"TotalQ={ns_queue+ew_queue:3d} | "
                f"AvgWait={avg_wait:6.2f}s | InSim={veh_count:3d} | Arrived={arrived_cumulative}"
            )

    traci.close()

    # ----- PLOTS -----
    plt.figure()
    plt.plot(times, ns_queues, label="NS Queue")
    plt.plot(times, ew_queues, label="EW Queue")
    plt.plot(times, total_queues, label="Total Queue")
    plt.xlabel("Time (s)")
    plt.ylabel("Queue (halted vehicles)")
    plt.title("Queue Length Over Time")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(times, avg_waits, label="Avg Waiting Time (s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Seconds")
    plt.title("Average Waiting Time Over Time")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(times, throughput, label="Throughput (Arrived Vehicles)")
    plt.xlabel("Time (s)")
    plt.ylabel("Vehicles")
    plt.title("Throughput Over Time")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    run(sim_steps=3600, log_every=5)
