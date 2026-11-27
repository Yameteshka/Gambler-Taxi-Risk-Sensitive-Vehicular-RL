"""Record headless episodes, then render video from the saved numeric log."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from env import city


STRATEGIES = ("work_sleep", "conditional_gambling", "slots_first", "blackjack_first", "bank_and_gamble")
CHASE_LOSSES = "chase_losses"
FAST_CASH = "fast_cash"


def record(args):
    from dataclasses import replace
    import equinox as eqx
    import jax
    from algorithms import hybrid_dsac as dsac
    from env import gamblers_taxi as taxi
    import scenarios
    from train import evaluate_policy, keys, load_actor

    env = taxi.GamblersTaxiEnv
    actor, config, meta = load_actor(args.checkpoint) if args.checkpoint else (None, taxi.TaxiConfig(), {})
    if args.strategy == CHASE_LOSSES and not actor:
        config = replace(config, enable_dependence=True)
    if args.dependence:
        config = replace(config, enable_dependence=True)
    params, _ = env.init_params(jax.random.key(args.seed), config)
    if args.strategy == CHASE_LOSSES:
        strategy = scenarios.CHASE_LOSSES
    elif args.strategy == FAST_CASH:
        strategy = scenarios.FAST_CASH
    else:
        strategy = STRATEGIES.index(args.strategy)
    seed = args.seed
    count = args.find_success or args.find_impulse
    if count:
        states = eqx.filter_jit(evaluate_policy)(
            actor, params, keys(seed, count), strategy=-1 if actor else strategy,
            deterministic=not args.sample_actions,
        )
        outcomes = np.asarray(states.outcome)
        successful = (outcomes == taxi.SUCCESS) if args.find_success else (
            np.asarray(states.counters[:, taxi.IMPULSES]) > 0
        )
        if args.require_gamble:
            successful &= np.asarray(states.counters[:, taxi.SPINS] + states.counters[:, taxi.HANDS]) > 0
        if args.require_repayment:
            successful &= np.asarray(states.counters[:, taxi.REPAYMENTS]) > 0
        successes = np.flatnonzero(successful)
        if not len(successes):
            raise RuntimeError(f"No matching episode among seeds {seed}-{seed + count - 1}")
        seed += int(successes[0])

    actions = None
    if actor:
        @jax.jit
        def actions(key, obs):
            return taxi.TaxiAction(*dsac.sample(actor, key, obs.features, obs.action_mask, not args.sample_actions))

    episode = scenarios.rollout(params, seed, strategy, actions)
    frames = []

    def frame(state, obs, events, choice, requested, controls, reward, base, shaping,
              daily_cost=0., gambling_net=0., game_resolved=0):
        return (
            np.asarray(state.time), np.asarray(state.agent_pos), np.asarray(state.agent_forward_dir),
            np.asarray(state.kinematic_obstacles), np.asarray(obs.collision_rays),
            np.asarray([state.speed, state.cash, state.debt, state.stake, state.fatigue,
                        state.mood, state.tilt, state.employed, state.activity, state.remaining]),
            np.asarray(events), np.asarray(choice), np.asarray(requested), np.asarray(controls),
            np.asarray([reward, base, shaping]), np.asarray(state.hand), np.asarray(state.dealer[0]),
            np.asarray(state.outcome),
            np.asarray([state.habit, taxi._urge(state, params)]),
            np.asarray(state.anxiety),
            np.asarray(daily_cost), np.asarray(gambling_net), np.asarray(game_resolved),
        )

    for obs, state, action, next_obs, new, reward, done, info in episode:
        if not frames:
            frames.append(frame(state, obs, np.zeros_like(info["events"]), taxi.DRIVE, taxi.DRIVE,
                                np.zeros(2), 0., 0., 0.))
        frames.append(frame(new, next_obs, info["events"], info["choice"], info["requested_choice"], action.control, reward,
                            info["base_reward"], info["shaping"], info["daily_cost"], info["gambling_net"],
                            info["game_resolved"]))
    if args.find_impulse and not any(row[6][taxi.IMPULSES] > 0 for row in frames):
        raise RuntimeError("The scalar replay did not reproduce the impulse detected by batched evaluation")
    if args.require_repayment and not any(row[6][taxi.REPAYMENTS] > 0 for row in frames):
        raise RuntimeError("The scalar replay did not reproduce a bank repayment")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    fields = ("tick", "position", "forward", "traffic", "rays", "status", "events", "choice", "requested_choice",
              "control", "reward", "hand", "dealer_upcard", "outcome", "dependence", "anxiety",
              "daily_cost", "gambling_net", "game_resolved")
    arrays = {name: np.stack(values) for name, values in zip(fields, zip(*frames))}
    header = {"seed": seed, "policy": str(args.checkpoint.stem if actor else args.strategy),
              "checkpoint_step": meta.get("stage"), "zone_radius": config.zone_radius, "buyout": config.buyout,
              "dt": config.dt, "horizon": config.horizon, "day": config.day,
              "shift_window": config.shift_window, "actions": taxi.ACTION_NAMES, "events": taxi.COUNTER_NAMES,
              "dependence": config.enable_dependence, "slot_stake": config.slot_stake,
              "blackjack_stake": config.blackjack_stake, "casino_open": config.casino_open,
              "casino_close": config.casino_close, "casino_round_interval": config.casino_round_interval,
              "operating_cost": config.operating_cost, "starting_debt": config.starting_debt,
              "salary": config.salary, "slot_payout": config.slot_payout, "slot_probability": config.slot_probability,
              "blackjack_win_commission": config.blackjack_win_commission}
    np.savez_compressed(args.log, header=json.dumps(header), **arrays)
    print(f"{args.log} seed={seed} frames={len(frames)} outcome={taxi.OUTCOME_NAMES[int(new.outcome)]}")


def dialogue(event, cash, debt, fatigue, tilt, rng, bank_required):
    if event == "wins":
        opening = ("That one paid off.", "The table finally turned.", "Good. I can breathe for a second.")
        thought = ("One more spin would be pushing it.", "Check the balance before deciding what comes next.")
    elif event == "losses":
        opening = ("There goes the stake.", "That hand was expensive.", "Another miss.")
        if cash < 8.:
            thought = ("I need a shift just to get cash again.", "I'm too tired to think. The next shift better pay.")
        elif tilt > 0.35:
            thought = ("I keep thinking one more round will fix it.", "I want another round, but the clock is running.")
        else:
            thought = ("Time to see what is still open.", "The next stop needs to count.")
    elif event == "paid":
        opening = ("Shift done.", "The wages came through.", "That was a long shift.")
        thought = ("Could use a few hours at home.", "Another shift might cover the rest.") if fatigue > 0.5 else (
            "Check what is left before heading out.", "The buyout is getting closer.")
    elif event == "sleeps":
        opening = ("Finally got some sleep.", "My hands feel steadier.", "I can focus on the road again.")
        thought = ("Let's see how much time is left.", "I should check the next shift.")
    elif event == "missed":
        opening = ("The shift is gone.", "They closed the shift before I got there.")
        thought = ("I needed that pay.", "I'll have to wait for another opening.")
    elif event == "fired":
        opening = ("They took me off the roster.", "I missed one shift too many.", "The job is gone.")
        thought = ("The job centre might have something.", "I need to decide where to go now.")
    elif event == "loans":
        opening = ("My hands are tight on the wheel.", "The loan landed. My stomach dropped.",
                   "I keep doing the math in my head.")
        thought = ("The meter is running on what I owe.",)
        if bank_required:
            thought += ("I have to clear it at the bank before I can buy the taxi.",)
    elif event == "repayments":
        opening = ("I made a payment.", "I put some money back into the loan.")
        thought = ("There's still more to pay.",) if debt > 0 else ("The loan is cleared. I can finally breathe.",)
    elif event == "impulses":
        opening = ("I was about to leave.", "I had my foot on the gas.", "I said that was the last one.")
        thought = ("Then I put another stake on the table.", "Somehow I'm still here.")
    else:
        return ""
    balance = f"Debt is ${debt:.0f}." if debt > 1. else f"${cash:.0f} in hand."
    return f"{rng.choice(opening)} {balance} {rng.choice(thought)}"


def render(args):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    import imageio_ffmpeg

    with np.load(args.log, allow_pickle=False) as file:
        header = json.loads(str(file["header"]))
        log = {name: file[name] for name in file.files if name != "header"}
    pygame.init()
    screen = pygame.Surface((1100, 720))
    fonts = [pygame.font.SysFont("dejavusans", size, bold=(index in (1, 2)))
             for index, size in enumerate((15, 18, 27, 12))]
    scale = 24
    map_x, map_y = 24, 79
    panel_x = 775
    palettes = ("#81c7a0", "#6fc1dd", "#b5bdf3", "#ead18f", "#e486b0", "#d4a4f4", "#f0dc6f")
    ink, muted, accent = "#f3eedf", "#aaaeb5", "#82d6bb"

    def px(coord):
        x, y = coord
        return (round(map_x + (x + 0.5)*scale), round(map_y + (city.HEIGHT - y - 0.5)*scale))

    def text(value, x, y, color=ink, font=0):
        screen.blit(fonts[font].render(str(value), True, color), (x, y))

    def bar(label, value, x, y, color):
        text(label, x, y)
        pygame.draw.rect(screen, "#343e49", (x, y + 25, 266, 6), border_radius=3)
        pygame.draw.rect(screen, color, (x, y + 25, int(266*np.clip(value, 0., 1.)), 6), border_radius=3)

    base = pygame.Surface(screen.get_size())
    base.fill("#151b22")
    pygame.draw.rect(base, "#202b33", (map_x, map_y, city.WIDTH*scale, city.HEIGHT*scale), border_radius=6)
    for x in range(1, city.WIDTH):
        pygame.draw.line(base, "#293640", (map_x+x*scale, map_y), (map_x+x*scale, map_y+city.HEIGHT*scale))
    for y in range(1, city.HEIGHT):
        pygame.draw.line(base, "#293640", (map_x, map_y+y*scale), (map_x+city.WIDTH*scale, map_y+y*scale))
    for wall in city.WALLS:
        pygame.draw.circle(base, "#3f4f5c", px(wall), scale)
    for zone, name, color in zip(city.ZONES, city.ZONE_NAMES, palettes):
        pygame.draw.circle(base, color, px(zone), round(header["zone_radius"]*scale), 2)
        image = fonts[3].render(name, True, color, "#202b33")
        base.blit(image, image.get_rect(center=(px(zone)[0], px(zone)[1] - 5)))
    pygame.draw.rect(base, "#232d38", (panel_x-16, 79, 317, 480), border_radius=8)
    rng = np.random.default_rng(header["seed"])
    args.video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1100x720", "-framerate", str(args.fps),
           "-i", "-", "-an", "-c:v", "libx264", "-crf", "19", "-preset", "fast", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(args.video)]
    final = len(log["tick"]) - 1
    line, line_until = "", -1
    result_text, result_until = "", -1
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        for i in range(final + 1):
            screen.blit(base, (0, 0))
            time = log["tick"][i]
            pos, forward = log["position"][i], log["forward"][i]
            speed, cash, debt, stake, fatigue, mood, tilt, employed, activity, remaining = log["status"][i]
            text("Gambler's Taxi", map_x, 18, ink, 2)
            text(f"Day {int(time)//header['day']+1}   |   Shift {'open' if time%header['day'] < header['shift_window'] else 'closed'}",
                 440, 30, muted, 0)
            if "casino_open" in header:
                casino_open = header["casino_open"] <= time%header["day"] < header["casino_close"]
                text("Casino open" if casino_open else "Casino closed", 620, 30, muted, 0)
            text(f"Seed {header['seed']}   |   {header['policy']}", panel_x, 30, muted, 0)
            for obstacle in log["traffic"][i]:
                pygame.draw.circle(screen, "#e28d71", px(obstacle), scale - 2)
                pygame.draw.circle(screen, "#f5ba9e", px(obstacle), scale - 2, 1)
            for ray in log["rays"][i]:
                pygame.draw.line(screen, "#507267", px(pos), px(pos + ray), 1)
            if i > 0:
                pygame.draw.lines(screen, "#c19d55", False, [px(x) for x in log["position"][max(0, i-65):i+1]], 2)
            f = np.asarray(forward)
            side = np.array((-f[1], f[0]))
            body = [px(pos + a*f + b*side) for a, b in ((0.65, 0.), (-0.42, 0.38), (-0.24, 0.), (-0.42, -0.38))]
            pygame.draw.polygon(screen, "#ffdc83", body)
            purchased = int(log["outcome"][i]) == 1
            text("Net capital before purchase" if purchased else "Net capital", panel_x, 100, muted)
            capital = cash + stake - debt + header["buyout"]*purchased
            text(f"${capital:,.2f}  /  ${header['buyout']:.0f}", panel_x, 125, accent, 2)
            text(f"Cash ${cash:,.2f}   Debt ${debt:,.2f}", panel_x, 166)
            if stake > 0:
                text(f"Stake in play ${stake:,.2f}", panel_x, 191, muted)
            text(f"Speed {speed:.1f}   |   {'Employed' if employed else 'Unemployed'}", panel_x, 218)
            bar(f"Fatigue  {fatigue:.0%}", fatigue, panel_x, 250, "#efad73")
            bar(f"Mood  {mood:+.2f}", (mood+1.)/2, panel_x, 297, "#82d6bb")
            bar(f"Tilt  {tilt:.0%}", tilt, panel_x, 344, "#d292d8")
            if "anxiety" in log:
                anxiety = log["anxiety"][i]
                bar(f"Anxiety  {anxiety:.0%}", anxiety, panel_x, 391, "#f29a80")
                text(f"Time  {int(time)} / {header['horizon']}", panel_x, 437, muted, 3)
            else:
                bar(f"Time  {int(time)} / {header['horizon']}", time/header["horizon"], panel_x, 391, "#8da9df")
            label = header["actions"][int(log["choice"][i])]
            if remaining > 0 and int(activity) != 0:
                label = f"{header['actions'][int(activity)]}, {int(remaining)} ticks"
            text(label, panel_x, 462, ink, 1)
            if "anxiety" in log and debt > 0 and int(activity) != 7:
                text("Garage locked until debt is paid", panel_x, 494, "#f29a80", 3)
            if header.get("starting_debt"):
                offer = f"Shift +${header['salary']:.0f}  |  Slots +${header['slot_payout']-header['slot_stake']:.0f} at {header['slot_probability']:.0%}"
                text(offer, 730, 599, muted, 3)
                blackjack_profit = header["blackjack_stake"] * (1. - header.get("blackjack_win_commission", 0.10))
                text(f"Blackjack +${blackjack_profit:.0f} on a regular win", 730, 651, muted, 3)
            if header.get("dependence"):
                habit, urge = log["dependence"][i]
                text(f"Habit {habit:.0%}   |   Urge {urge:.0%}", panel_x, 524, "#efad73")
            if int(activity) == 7:
                cards = " ".join(str(card) for card in log["hand"][i] if card)
                text(f"Hand {cards}   |   Dealer {log['dealer_upcard'][i]}", panel_x, 487, muted)
            if "daily_cost" in log and log["daily_cost"][i]:
                result_text, result_until = f"Daily costs  -${log['daily_cost'][i]:.2f}", i + 38
            if "game_resolved" in log and log["game_resolved"][i]:
                game = "Slots" if int(log["game_resolved"][i]) == 6 else "Blackjack"
                result_text, result_until = f"{game}  {log['gambling_net'][i]:+.2f} USD", i + 38
            for name, count in zip(header["events"], log["events"][i]):
                if count:
                    message = dialogue(name, cash, debt, fatigue, tilt, rng, bank_required="anxiety" in log)
                    if message:
                        line, line_until = message, i + 57
            if i <= result_until:
                text(result_text, 485, 599, "#efad73", 3)
            outcome = int(log["outcome"][i])
            if outcome:
                line = ("", "Taxi bought out. The garage is reached.", "The taxi crashed.", "The deadline passed.")[outcome]
                line_until = final + args.fps
            pygame.draw.rect(screen, "#232d38", (map_x, 581, 1052, 111), border_radius=7)
            text("Driver", 44, 599, accent, 3)
            text(line if i <= line_until else label, 44, 626, ink, 1)
            image = pygame.image.tobytes(screen, "RGB")
            encoder.stdin.write(image)
            if i == final:
                for _ in range(2*args.fps):
                    encoder.stdin.write(image)
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("FFmpeg video encoding failed")
    pygame.quit()
    print(args.video)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("record")
    capture.add_argument("--log", type=Path, required=True)
    capture.add_argument("--checkpoint", type=Path)
    capture.add_argument("--sample-actions", action="store_true", help="sample rather than take the actor's modal actions")
    capture.add_argument("--strategy", choices=(*STRATEGIES, CHASE_LOSSES, FAST_CASH), default="work_sleep")
    capture.add_argument("--seed", type=int, default=20000)
    search = capture.add_mutually_exclusive_group()
    search.add_argument("--find-success", type=int, default=0)
    search.add_argument("--find-impulse", type=int, default=0,
                        help="select an episode with an impulsive bet, regardless of its outcome")
    capture.add_argument("--require-gamble", action="store_true",
                         help="when searching, select a successful episode that played a game")
    capture.add_argument("--require-repayment", action="store_true")
    capture.add_argument("--dependence", action="store_true")
    video = sub.add_parser("render")
    video.add_argument("--log", type=Path, required=True)
    video.add_argument("--video", type=Path, required=True)
    video.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    (record if args.command == "record" else render)(args)


if __name__ == "__main__":
    main()
