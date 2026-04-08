#include <chrono>
#include <iostream>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "asteroids.hh"

void add_default_asteroids(Game &game) {
  game.add_asteroid(
      {.pos = {220.0, 120.0}, .vel = {-2.5, -1.0}, .mass = 50000.0});
  game.add_asteroid({.pos = {-280.0, -40.0}, .vel = {1.8, 2.1}, .mass = 180.0});
  game.add_asteroid(
      {.pos = {60.0, -260.0}, .vel = {-1.2, 2.8}, .mass = 1300.0});
  game.add_asteroid(
      {.pos = {-140.0, 240.0}, .vel = {2.2, -1.6}, .mass = 10000.0});
}

int main() {
  Game game;
  add_default_asteroids(game);
  game.generate_rand_world_asteroids(2e-6, {100.0, 10000.0}, {0.0, 20000.0},
                                     {-50.0, 0.0});
  game.generate_rand_asteroid_cluster({3000.0, 0.0}, 150.0, 5e-4,
                                      {100.0, 1000.0}, {0.0, 2000.0}, {PI, PI},
                                      {-100.0, 0.0});

  game.set_ship();
  game.clear_ship_vicinity();

#ifdef AST_USE_SDL2
  Renderer renderer(asteroids_config.window.width,
                    asteroids_config.window.height);
  using Clock = std::chrono::steady_clock;

  struct LoopContext {
    Game* game;
    Renderer* renderer;
    Clock::time_point last_time;
    double accumulator;
    double border_spawn_rate;
  };

  LoopContext ctx = {
    &game,
    &renderer,
    Clock::now(),
    0.0,
    7.5e-4
  };

  auto loop_iteration = [](void* arg) {
    auto* c = static_cast<LoopContext*>(arg);
    const auto now = Clock::now();
    double frame_dt = std::chrono::duration<double>(now - c->last_time).count();
    c->last_time = now;

    // Avoid spiraling after debugger pauses or tab stalls.
    frame_dt = std::min(frame_dt, asteroids_config.timing.max_frame_dt);
    c->accumulator += frame_dt;

    c->game->handle_input(c->renderer->poll_input());
    while (c->accumulator >= asteroids_config.timing.fixed_dt) {
      const double step_density =
          c->border_spawn_rate * asteroids_config.timing.fixed_dt;
      c->game->generate_rand_incoming_asteroids(step_density, {100.0, 1000.0},
                                            {15000.0, 40000.0}, {0.0, 0.0});
      c->game->update(asteroids_config.timing.fixed_dt);
      c->accumulator -= asteroids_config.timing.fixed_dt;
    }

    c->renderer->render(*c->game);
    
#ifndef __EMSCRIPTEN__
    if (c->renderer->should_quit()) {
      exit(0);
    }
#endif
  };

#ifdef __EMSCRIPTEN__
  emscripten_set_main_loop_arg(loop_iteration, &ctx, 0, 1);
#else
  while (!renderer.should_quit()) {
    loop_iteration(&ctx);
  }
#endif
#endif

  return 0;
}