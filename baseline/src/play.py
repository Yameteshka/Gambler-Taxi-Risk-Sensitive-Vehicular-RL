"""Drive the new environment with the course Pygame frontend."""

import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp

from env import city
from env import gamblers_taxi as taxi
from utils.renderer import PygameFrontend, pygame


class TaxiFrontend(PygameFrontend):
    def __init__(self, env, params, state):
        super().__init__(env, params, state)
        self.CELL_SIZE = 30
        self.screen = pygame.display.set_mode((city.WIDTH*self.CELL_SIZE, city.HEIGHT*self.CELL_SIZE))
        pygame.display.set_caption("Gambler's Taxi")
        self.radius = round(self.CELL_SIZE*params.config.zone_radius)
        self.font = pygame.font.SysFont("Arial", 17)
        self.step = eqx.filter_jit(env.step)
        self.action = taxi.DRIVE

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        throttle = int(keys[pygame.K_UP] or keys[pygame.K_w]) - int(keys[pygame.K_DOWN] or keys[pygame.K_s])
        steer = int(keys[pygame.K_LEFT] or keys[pygame.K_a]) - int(keys[pygame.K_RIGHT] or keys[pygame.K_d])
        return taxi.TaxiAction(jnp.array((throttle, steer), dtype=jnp.float32), jnp.int32(self.action))

    def draw(self):
        super().draw()
        color = (65, 90, 110)
        for name, (x, y) in zip(city.ZONE_NAMES, city.ZONES):
            pos = (round((x + 0.5)*self.CELL_SIZE), round((city.HEIGHT-y-0.5)*self.CELL_SIZE))
            pygame.draw.circle(self.screen, color, pos, self.radius, 2)
            label = self.font.render(name, True, color, (255, 255, 255))
            self.screen.blit(label, label.get_rect(center=pos))
        pygame.display.flip()

    def run(self):
        bindings = {
            pygame.K_1: taxi.SHIFT, pygame.K_2: taxi.SLEEP, pygame.K_3: taxi.APPLY,
            pygame.K_4: taxi.BORROW, pygame.K_5: taxi.REPAY, pygame.K_6: taxi.SPIN,
            pygame.K_7: taxi.DEAL, pygame.K_8: taxi.HIT, pygame.K_9: taxi.STAND, pygame.K_0: taxi.BUY,
        }
        while self.running:
            self.action = taxi.DRIVE
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.key, reset_key = jax.random.split(self.key)
                        self.obs, self.state = self.env.reset(reset_key, self.params)
                    self.action = bindings.get(event.key, self.action)
            if int(self.state.outcome) == taxi.ACTIVE:
                self.key, key = jax.random.split(self.key)
                self.obs, self.state, _, _, _ = self.step(key, self.state, self.handle_keys(), self.params)
            state = self.state
            self.info = {
                "Time": f"{int(state.time)} / {self.params.config.horizon}",
                "Cash / debt": f"${float(state.cash):.2f} / ${float(state.debt):.2f}",
                "Fatigue": f"{float(state.fatigue):.0%}",
                "Anxiety": f"{float(state.anxiety):.0%}",
                "Outcome": taxi.OUTCOME_NAMES[int(state.outcome)],
                "Available": ", ".join(
                    f"{index % 10} {taxi.ACTION_NAMES[index]}" for index, valid in enumerate(self.obs.action_mask)
                    if bool(valid) and 0 < index < taxi.WAIT
                ) or "drive / wait",
            }
            if self.params.config.enable_dependence:
                self.info["Habit / urge"] = f"{float(state.habit):.0%} / {float(taxi._urge(state, self.params)):.0%}"
            self.draw()
            self.clock.tick(round(1/self.params.config.dt))
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependence", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    key = jax.random.key(0)
    params, state = taxi.GamblersTaxiEnv.init_params(key, taxi.TaxiConfig(enable_dependence=args.dependence))
    TaxiFrontend(taxi.GamblersTaxiEnv, params, state).run()
