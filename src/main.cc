#include <chrono>
#include <iostream>

#include "asteroids.hh"

void add_default_asteroids(Game &game) {
  game.add_asteroid(
      {.pos = {220.0, 120.0}, .vel = {-2.5, -1.0}, .mass = 100000.0});
  game.add_asteroid({.pos = {-280.0, -40.0}, .vel = {1.8, 2.1}, .mass = 180.0});
  game.add_asteroid(
      {.pos = {60.0, -260.0}, .vel = {-1.2, 2.8}, .mass = 1300.0});
  game.add_asteroid(
      {.pos = {-140.0, 240.0}, .vel = {2.2, -1.6}, .mass = 10000.0});
}

int main() {
  Game game;
  // add_default_asteroids(game);
  game.generate_asteroid_field(5e-6, 100.0, 10000.0, 0.0, 20000.0);
  game.set_ship({.pos = {0.0, 0.0}, .vel = {0.0, 0.0}, .angle = 0.0});
  game.remove_asteroids({0.0, 0.0}, 100.0);

#ifdef AST_USE_SDL2
  Renderer renderer(asteroids_config.window.width,
                    asteroids_config.window.height);
  using Clock = std::chrono::steady_clock;

  auto last_time = Clock::now();
  double accumulator = 0.0;

  while (!renderer.should_quit()) {
    const auto now = Clock::now();
    double frame_dt = std::chrono::duration<double>(now - last_time).count();
    last_time = now;

    // Avoid spiraling after debugger pauses or tab stalls.
    frame_dt = std::min(frame_dt, asteroids_config.timing.max_frame_dt);
    accumulator += frame_dt;

    game.handle_input(renderer.poll_input());
    while (accumulator >= asteroids_config.timing.fixed_dt) {
      game.update(asteroids_config.timing.fixed_dt);
      accumulator -= asteroids_config.timing.fixed_dt;
    }

    renderer.render(game);
  }
#endif

  return 0;
}