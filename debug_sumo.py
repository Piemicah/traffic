import traci

traci.start(["sumo-gui", "-c", "config.sumocfg", "--delay", "100"])
conn = traci.getConnection()

for i in range(200):
    conn.simulationStep()
    print("step", i, "vehicles:", conn.vehicle.getIDCount())

conn.close()
