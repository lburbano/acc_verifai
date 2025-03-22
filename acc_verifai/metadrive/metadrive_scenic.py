import scenic
from scenic.simulators.metadrive import MetaDriveSimulator


# %%
scenario = scenic.scenarioFromFile(
    "../car.scenic",
    model="scenic.simulators.metadrive.model",
    mode2D=True,
)

# %%
scene, _ = scenario.generate()
simulator = MetaDriveSimulator()
simulation = simulator.simulate(scene, maxSteps=10)

# %%
if simulation:  # `simulate` can return None if simulation fails
    result = simulation.result
    for i, state in enumerate(result.trajectory):
        ego_pos = state
        print(f"Time step {i}: ego at {ego_pos}")

# %%
