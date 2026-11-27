# Public food information

The new optional `Config(public_food_menu=True)` observation supplies a global menu for all three physical food venues: current price, portion lower bound, mean, upper bound and service time. Prices and durations are normalized only for network input. The original global service locations, route distances and local lidar remain available. Deterministic meals, wider distributions and physical service-identity swaps use their corresponding menu terms. The reduced-quote condition continues to hide prices.

The actual next portion is not revealed. The menu describes raw restoration; usable restoration is limited by the satiety ceiling, and hunger continues during travel and service. 

For checkpoint compatibility, the option defaults to false. Existing trained policies retain 179 service-interface features. A future menu-enabled policy receives 194 features and requires a new training artifact and explicit configuration. Do not load older weights into that larger schema or resume the frozen four-model study after changing its sources. Training remains stopped.

[Implementation](../src/env/resource_taxi.py), [Transaction and observation check](../experiments/check_public_menu.py) and [Verification results](../experiments/public_menu_verification.json)
