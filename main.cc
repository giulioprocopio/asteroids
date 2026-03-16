#include <chrono>
#include <iostream>

#include "asteroids.hh"

void add_default_asteroids(Game &game) {
  game.add_asteroid(
      {.pos = {220.0, 120.0}, .vel = {-2.5, -1.0}, .mass = 2000.0});
  game.add_asteroid({.pos = {-280.0, -40.0}, .vel = {1.8, 2.1}, .mass = 180.0});
  game.add_asteroid({.pos = {60.0, -260.0}, .vel = {-1.2, 2.8}, .mass = 130.0});
  game.add_asteroid({.pos = {-140.0, 240.0}, .vel = {2.2, -1.6}, .mass = 95.0});
}

int main() {
  Game game;
  game.set_ship({.pos = {0.0, 0.0}, .vel = {0.0, 0.0}, .angle = 0.0});
  add_default_asteroids(game);

  constexpr double fixed_dt = 1.0 / 120.0;

#ifdef AST_USE_SDL2
  Renderer renderer(1280, 720);
  using Clock = std::chrono::steady_clock;

  auto last_time = Clock::now();
  double accumulator = 0.0;

  while (!renderer.should_quit()) {
    const auto now = Clock::now();
    double frame_dt = std::chrono::duration<double>(now - last_time).count();
    last_time = now;

    // Avoid spiraling after debugger pauses or tab stalls.
    frame_dt = std::min(frame_dt, 0.25);
    accumulator += frame_dt;

    game.handle_input(renderer.poll_input());
    while (accumulator >= fixed_dt) {
      game.update(fixed_dt);
      accumulator -= fixed_dt;
    }

    renderer.render(game);
  }
#endif

  return 0;
}