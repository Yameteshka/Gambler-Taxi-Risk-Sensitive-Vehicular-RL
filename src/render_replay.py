"""Render recorded numeric trajectories; commentary never changes a simulation."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from env import city


def load_episode(directory, scenario_id):
    rows, repayments = [], {}
    for file in sorted(Path(directory).glob("*trace*.npz")):
        with np.load(file, allow_pickle=False) as data:
            mask = (data["scenario_id"] == scenario_id) & data["active"]
            for tick, lane in np.argwhere(mask):
                row = {k: data[k][tick, lane] if data[k].ndim > 1 else data[k][tick] for k in data.files if k not in
                    ("metadata", "loan_receipt_location", "loan_receipt_id", "loan_receipt_amount")}
                rows.append(row)
            if "loan_receipt_location" in data:
                for (tick, lane), loan_id, amount in zip(data["loan_receipt_location"],
                        data["loan_receipt_id"], data["loan_receipt_amount"]):
                    if mask[tick, lane]:
                        key = int(data["time"][tick, lane])
                        repayments.setdefault(key, []).append((int(loan_id), float(amount)))
    if not rows:
        raise ValueError("No active trajectory found for this scenario")
    return rows, repayments


def commentary(row):
    events = row["events"]
    lines = []
    if events[0]: lines.append("Shift completed: wages received.")
    if events[1]: lines.append("Work window missed.")
    if events[2]: lines.append(f"Microloan #{int(row['loan_opened'])} opened; cash and liabilities increased.")
    if events[3]: lines.append(f"Car payment: ${float(row['car_payment']):.2f}.")
    if events[4]: lines.append(f"Personal debt payment: ${float(row['micro_payment']):.2f}.")
    if row["game_resolved"]:
        lines.append(f"Game settled: net ${float(row['gambling_net']):+.2f}.")
    if events[10]: lines.append("Sleep completed.")
    if events[11]: lines.append("Meal delivered; satiety restored by the recorded portion.")
    if events[12]: lines.append("Refueling completed.")
    if events[13]: lines.append("Scheduled-payment shortfall: late fee charged.")
    if row["outcome"]: lines.append(("", "Ownership completed", "Collision", "Deadline reached", "Starvation")[int(row["outcome"])])
    return lines


def render(directory, scenario_id, output, stride=4):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    import imageio_ffmpeg
    pygame.init()
    screen = pygame.Surface((1280, 800))
    font = pygame.font.SysFont("DejaVu Sans", 17)
    small = pygame.font.SysFont("DejaVu Sans", 13)
    title = pygame.font.SysFont("DejaVu Sans", 28, bold=True)
    rows, receipts = load_episode(directory, scenario_id)
    directory = Path(directory)
    metadata = json.loads((directory/"evaluation.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    role = "Learned service policy + shared driving" if metadata.get("interface") == "strategy" else (
        "Scripted reference" if metadata.get("kind") == "reference" else "Recorded vehicle policy")
    macro_names = metadata.get("macro_names", [])
    final_balances = None
    if (directory/"endpoints.npz").exists():
        with np.load(directory/"endpoints.npz",allow_pickle=False) as endpoint:
            selected = np.flatnonzero(endpoint["scenario_id"] == scenario_id)
            if selected.size:
                lane = int(selected[0])
                final_balances = dict(zip(map(int,endpoint["loan_ids"][lane]),map(float,endpoint["loan_balance"][lane])))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), (1280, 800), fps=20, codec="libx264", quality=8)
    writer.send(None)
    ledger, notes, event_log = {}, [], []
    previous_intention = None
    if config.get("starting_microloan", 0):
        ledger[1] = {"principal": config["starting_microloan"], "balance": config["starting_microloan"], "opened": 0, "payments": []}
    borrowed = config.get("starting_microloan", 0)

    def point(pos):
        return int(25+20*pos[0]), int(95+20*(city.HEIGHT-1-pos[1]))

    def text(message, x, y, color=(224, 230, 238), face=None):
        screen.blit((face or font).render(str(message), True, color), (x, y))

    try:
        for index, row in enumerate(rows):
            tick = int(row["time"])
            if row["loan_opened"]:
                amount = float(row["borrowed_total"])-borrowed
                ledger[int(row["loan_opened"])] = {"principal": amount, "balance": amount, "opened": tick-1, "payments": []}
            borrowed = float(row["borrowed_total"])
            for loan_id, amount in receipts.get(tick, []):
                ledger[loan_id]["balance"] = max(0., ledger[loan_id]["balance"]-amount)
                ledger[loan_id]["payments"].append({"tick": tick-1, "amount": amount})
            for loan in ledger.values():
                loan["balance"] *= (1+config["micro_daily_rate"])**(1/config["day"])
            if index == len(rows)-1 and final_balances is not None:
                for loan_id,loan in ledger.items():loan["balance"] = final_balances.get(loan_id,0.)
            messages = commentary(row)
            if macro_names and bool(row.get("decision_start",False)):
                intention = int(row["macro_executed"])
                if intention != previous_intention:
                    messages.insert(0,"Intention selected: "+macro_names[intention]+".")
                    previous_intention = intention
            if index == len(rows)-1 and int(row["outcome"]) == 0:
                messages.append("Observation window ended; campaign unfinished.")
            if messages:
                notes = (notes+messages)[-5:]
                event_log.extend({"tick": tick, "text": m} for m in messages)
            if index % stride and index != len(rows)-1:
                continue
            screen.fill((15, 20, 29))
            text("GAMBLER'S TAXI", 25, 18, face=title)
            text(f"{role} | Scenario {scenario_id} | Day {1+(tick-1)//config['day']} | Tick {tick}", 25, 57)
            if macro_names:
                text("Intention: "+macro_names[int(row["macro_executed"])],850,57,face=small)
            pygame.draw.rect(screen, (25, 34, 45), (15, 85, 805, 605), border_radius=8)
            for wall in city.WALLS:
                pygame.draw.circle(screen, (86, 96, 111), point(wall), 20)
            for i, zone in enumerate(city.ZONES):
                color = (65, 160, 140) if i in (city.HOME, city.WORK) else (97, 119, 184)
                pygame.draw.circle(screen, color, point(zone), int(config["zone_radius"]*20), 2)
                x, y = point(zone)
                name = city.NAMES[i]
                if config.get("food_swap") and i in (city.CAFE, city.RESTAURANT):
                    name = city.NAMES[city.CAFE+city.RESTAURANT-i]
                text(name, x-25, y-12, face=small)
            for pos in row["traffic"]:
                pygame.draw.circle(screen, (208, 100, 87), point(pos), 20)
            pos, forward = row["position"], row["forward"]
            pygame.draw.circle(screen, (250, 202, 87), point(pos), 20)
            pygame.draw.line(screen, (255, 246, 193), point(pos), point(pos+forward*.85), 3)
            text("Resources and accounts", 850, 95, face=title)
            fields = [("Living cash", f"${float(row['cash']):.2f}"), ("Car loan", f"${float(row['car_debt']):.2f}"),
                ("Personal debt", f"${float(row['micro_debt']):.2f}"), ("Active microloans", int(row["loan_count"])),
                ("Fuel", f"{float(row['fuel']):.2f}"), ("Satiety", f"{100*float(row['satiety']):.0f}%"),
                ("Fatigue", f"{100*float(row['fatigue']):.0f}%"), ("Speed", f"{float(row['speed']):.2f}")]
            for n, (label, value) in enumerate(fields):
                text(label, 850, 145+29*n)
                text(value, 1090, 145+29*n, color=(250, 202, 87))
            pages = max(1, (len(ledger)+7)//8)
            page = (index//120) % pages
            text(f"Individual microloans (page {page+1}/{pages})", 850, 400)
            for n, (loan_id, loan) in enumerate(list(ledger.items())[8*page:8*page+8]):
                text(f"#{loan_id}  ${loan['principal']:.0f} -> ${loan['balance']:.2f}  {100*config['micro_daily_rate']:.0f}%/day", 850, 435+23*n, face=small)
            text("Recorded events", 25, 712)
            for n, message in enumerate(notes[-3:]):
                text(message, 25, 741+17*n, face=small)
            writer.send(pygame.surfarray.array3d(screen).transpose((1, 0, 2)).copy())
    finally:
        writer.close()
        pygame.image.save(screen, str(output.with_suffix(".png")))
        pygame.quit()
    output.with_suffix(".events.json").write_text(json.dumps(event_log, indent=2), encoding="utf-8")
    output.with_suffix(".loans.json").write_text(json.dumps({"daily_rate": config["micro_daily_rate"],
        "loans": ledger}, indent=2), encoding="utf-8")
    print(str(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--scenario-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    render(args.evaluation, args.scenario_id, args.output, args.stride)
