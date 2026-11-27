"""Export inspectable scientific design figures, separate from learned results."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from env import city


def effective_uniform(low, high, capacity):
    c = np.asarray(capacity)
    return np.where(c <= low, c, np.where(c >= high, (low+high)/2,
                                         c-(c-low)**2/(2*(high-low))))


def main():
    target = ROOT/"media/figures"
    target.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    colors = ["#2a9d8f", "#2a9d8f", "#5b77a5", "#5b77a5", "#b276b2", "#b276b2",
              "#d9a441", "#e67b58", "#e67b58", "#e67b58"]
    for x, y in city.WALLS:
        ax.add_patch(Circle((x, y), 1, color="#73808c"))
    for name, (x, y), color in zip(city.NAMES, city.ZONES, colors):
        ax.add_patch(Circle((x, y), 1.55, color=color, alpha=.16))
        ax.plot(x, y, ".", color=color)
        ax.annotate(name, (x, y), xytext=(0, 15), textcoords="offset points", ha="center", fontsize=9)
    for (x, y), (vx, vy), (lo, hi) in zip(city.TRAFFIC, city.TRAFFIC_VELOCITY, city.TRAFFIC_LIMITS):
        ax.plot([lo, hi], [y, y], "--", color="#c95b55", alpha=.6)
        ax.add_patch(Circle((x, y), 1, fill=False, color="#c95b55", linewidth=2))
    ax.add_patch(Circle(city.START, 1, color="#efc04d", alpha=.9))
    ax.set(xlim=(0, 39), ylim=(0, 29), xlabel="Map x", ylabel="Map y", title="Fixed task geography and unit-circle obstacles")
    ax.set_aspect("equal")
    ax.grid(alpha=.15)
    fig.savefig(target/"environment_map.png", dpi=200)
    plt.close(fig)
    capacity = np.linspace(0, 1, 501)
    menus = (("Budget diner", .04, .06, 5), ("Cafe", .12, .18, 15), ("Restaurant", .55, .75, 35))
    values = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for name, lo, hi, price in menus:
        effective = effective_uniform(lo, hi, capacity)
        values[name] = effective.tolist()
        axes[0].plot(capacity, effective, label=name)
        axes[1].plot(capacity, effective/price, label=name)
    axes[0].set(xlabel="Remaining satiety capacity", ylabel="Expected usable portion", title="Clipping changes the value of a meal")
    axes[1].set(xlabel="Remaining satiety capacity", ylabel="Expected usable satiety / mean price ($)", title="A benchmark excluding travel and service costs")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(target/"food_clipping.png", dpi=200)
    plt.close(fig)
    ticks = np.arange(1, 4651)
    old = -.1*(1-.999**ticks)-.999**(ticks-1)
    new = np.full(ticks.shape, -.1)
    success = -.1+1.1*.999**(ticks-1)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(ticks/155, old, label="Earlier failure reward (saturated debt)")
    ax.plot(ticks/155, new, label="Current unsuccessful base return")
    ax.plot(ticks/155, success, label="Current ownership base return (no reserve)")
    ax.set(xlabel="Terminal time (game days)", ylabel="Discounted base return", title="Reward audit: failure delay and successful completion")
    ax.text(.99, .98, "Design calculations; financial increments excluded", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(target/"reward_timing.png", dpi=200)
    plt.close(fig)
    (ROOT/"reports/design_calculations.json").write_text(json.dumps({"capacity": capacity.tolist(),
        "effective_uniform_portions": values, "meal_scope": "Ratios to mean quoted prices, not expected inverse-price ratios. Travel, service hunger, fuel and missed work are excluded.",
        "terminal_ticks": ticks.tolist(), "old_failure": old.tolist(), "current_failure": new.tolist(),
        "current_success_no_reserve": success.tolist()}), encoding="utf-8")
    print("Wrote map, food-clipping and reward-timing figures.")


if __name__ == "__main__":
    main()
