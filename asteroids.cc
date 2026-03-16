#include "asteroids.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

uint32_t hash_u32(uint32_t value) {
  value ^= value >> 17;
  value *= 0xed5ad4bbU;
  value ^= value >> 11;
  value *= 0xac4c1b51U;
  value ^= value >> 15;
  value *= 0x31848babU;
  value ^= value >> 14;
  return value;
}

double random_unit_from_seed(uint32_t seed) {
  return static_cast<double>(hash_u32(seed)) /
         static_cast<double>(std::numeric_limits<uint32_t>::max());
}

Asteroid normalize_asteroid(Asteroid a) {
  a.radius = std::sqrt(a.mass) * AST_RADIUS_PER_SQRT_MASS;
  return a;
}

// Pairwise gravity with softening; returns one acceleration per asteroid.
std::vector<Vec2> compute_asteroid_accelerations(
    std::span<const Asteroid> asteroids) {
  std::vector<Vec2> acc(asteroids.size(), {0.0, 0.0});
  for (size_t i = 0; i < asteroids.size(); ++i) {
    for (size_t j = i + 1; j < asteroids.size(); ++j) {
      const Vec2 d = asteroids[j].pos - asteroids[i].pos;
      const double d2 = dot(d, d) + AST_EPS * AST_EPS;
      const double inv_d = 1.0 / std::sqrt(d2);
      const double inv_d3 = inv_d * inv_d * inv_d;
      const double si = AST_G * asteroids[j].mass * inv_d3;
      const double sj = AST_G * asteroids[i].mass * inv_d3;
      acc[i] += d * si;
      acc[j] -= d * sj;
    }
  }
  return acc;
}

// Gravitational acceleration on a single point from all asteroids.
Vec2 compute_ship_acceleration(const Vec2 &pos,
                               std::span<const Asteroid> asteroids) {
  if constexpr (AST_SHIP_GRAVITY) {
    Vec2 acc{0.0, 0.0};
    for (const auto &a : asteroids) {
      const Vec2 d = a.pos - pos;
      const double d2 = dot(d, d) + AST_EPS * AST_EPS;
      const double inv_d = 1.0 / std::sqrt(d2);
      const double inv_d3 = inv_d * inv_d * inv_d;
      acc += d * (AST_G * a.mass * inv_d3);
    }
    return acc;
  }
}

}  // namespace

void Space::add_asteroid(Asteroid a) {
  a = normalize_asteroid(a);
  if (a.id <= 0) {
    a.id = next_asteroid_id_++;
  }
  asteroids_.push_back(a);
}

void Space::set_ship(Ship s) { ship_ = s; }

void Space::set_input(const InputState &input) {
  // Edge-trigger fire: queue a shot only on the transition from low to high
  if (input.fire && !input_.fire) {
    fire_pending_ = true;
  }
  input_ = input;
}

std::span<const Asteroid> Space::asteroids() const { return asteroids_; }

std::span<const Bullet> Space::bullets() const { return bullets_; }

std::span<const Explosion> Space::explosions() const { return explosions_; }

const Ship &Space::ship() const { return ship_; }

void Space::step(double dt) {
  // Remove expired explosions
  for (auto &e : explosions_) {
    e.age += dt;
  }
  std::erase_if(explosions_, [](const Explosion &e) {
    return e.age > AST_ESPLOSION_LIFETIME;
  });

  // Rotate ship
  if (input_.rotate_left) {
    ship_.angle += AST_SHIP_ROTATION_SPEED * dt;
  }
  if (input_.rotate_right) {
    ship_.angle -= AST_SHIP_ROTATION_SPEED * dt;
  }
  ship_.angle = std::fmod(ship_.angle, TWO_PI);

  const Vec2 thrust_dir{std::cos(ship_.angle), std::sin(ship_.angle)};

  // First half-kick
  {
    auto acc_ast = compute_asteroid_accelerations(asteroids_);
    for (size_t i = 0; i < asteroids_.size(); ++i) {
      asteroids_[i].vel += acc_ast[i] * (0.5 * dt);
    }

    Vec2 acc_ship = compute_ship_acceleration(ship_.pos, asteroids_);
    if (input_.thrust_forward) {
      acc_ship += thrust_dir * AST_SHIP_THRUST_FORWARD;
    }
    if (input_.thrust_backward) {
      acc_ship -= thrust_dir * AST_SHIP_THRUST_BACKWARD;
    }
    ship_.vel += acc_ship * (0.5 * dt);
  }

  // Drift
  for (auto &a : asteroids_) {
    a.pos += a.vel * dt;
  }
  ship_.pos += ship_.vel * dt;

  constexpr double bound_width = AST_WORLD_HALF_WIDTH - AST_SHIP_RADIUS;
  constexpr double bound_height = AST_WORLD_HALF_HEIGHT - AST_SHIP_RADIUS;

  // Clamp ship to world boundary; zero velocity component into the wall
  if (ship_.pos.x < -bound_width) {
    ship_.pos.x = -bound_width;
    if (ship_.vel.x < 0) {
      ship_.vel.x = 0;
    }
  }
  if (ship_.pos.x > bound_width) {
    ship_.pos.x = bound_width;
    if (ship_.vel.x > 0) {
      ship_.vel.x = 0;
    }
  }
  if (ship_.pos.y < -bound_height) {
    ship_.pos.y = -bound_height;
    if (ship_.vel.y < 0) {
      ship_.vel.y = 0;
    }
  }
  if (ship_.pos.y > bound_height) {
    ship_.pos.y = bound_height;
    if (ship_.vel.y > 0) {
      ship_.vel.y = 0;
    }
  }

  // Second half-kick at new positions
  {
    auto acc_ast = compute_asteroid_accelerations(asteroids_);
    for (size_t i = 0; i < asteroids_.size(); ++i) {
      asteroids_[i].vel += acc_ast[i] * (0.5 * dt);
    }

    Vec2 acc_ship = compute_ship_acceleration(ship_.pos, asteroids_);
    if (input_.thrust_forward) {
      acc_ship += thrust_dir * AST_SHIP_THRUST_FORWARD;
    }
    if (input_.thrust_backward) {
      acc_ship -= thrust_dir * AST_SHIP_THRUST_BACKWARD;
    }
    ship_.vel += acc_ship * (0.5 * dt);
  }

  // Despawn asteroids that have left the finite world
  std::erase_if(asteroids_, [](const Asteroid &a) {
    return std::abs(a.pos.x) > AST_WORLD_HALF_WIDTH ||
           std::abs(a.pos.y) > AST_WORLD_HALF_HEIGHT;
  });

  // Spawn a bullet if a fire was triggered this frame
  if (fire_pending_) {
    bullets_.push_back({ship_.pos + thrust_dir * AST_SHIP_RADIUS,
                        ship_.vel + thrust_dir * AST_BULLET_SPEED,
                        AST_BULLET_LIFETIME});
    fire_pending_ = false;
  }

  // Advance bullets removing expired or out-of-bounds ones
  for (auto &b : bullets_) {
    b.pos += b.vel * dt;
    b.lifetime -= dt;
  }
  std::erase_if(bullets_, [](const Bullet &b) {
    return b.lifetime <= 0.0 || std::abs(b.pos.x) > AST_WORLD_HALF_WIDTH ||
           std::abs(b.pos.y) > AST_WORLD_HALF_HEIGHT;
  });

  // Bullet-asteroid collision
  std::vector<bool> bullet_hit(bullets_.size(), false);
  std::uniform_int_distribution<int> seed_dist(1,
                                               std::numeric_limits<int>::max());

  for (size_t bi = 0; bi < bullets_.size(); ++bi) {
    for (size_t ai = 0; ai < asteroids_.size(); ++ai) {
      const Vec2 d = bullets_[bi].pos - asteroids_[ai].pos;
      const double r = asteroids_[ai].radius;
      if (dot(d, d) > r * r) {
        continue;
      }

      bullet_hit[bi] = true;

      // Surface impact point in the direction of the incoming bullet
      const double dn = std::max(norm(d), 1e-9);
      const Vec2 surf = asteroids_[ai].pos + (d / dn) * r;
      explosions_.push_back({surf, r * 0.5, 0.0, seed_dist(random_engine_)});
      break;  // One bullet hits one asteroid
    }
  }

  // Remove hit bullets
  {
    std::vector<Bullet> surviving;
    surviving.reserve(bullets_.size());
    for (size_t i = 0; i < bullets_.size(); ++i) {
      if (!bullet_hit[i]) {
        surviving.push_back(bullets_[i]);
      }
    }
    bullets_ = std::move(surviving);
  }

  // Other collisions
}

const Space &Game::space() const { return space_; }

// Zoom not implemented yet, return default zoom for now
CameraState Game::camera() const { return {space_.ship().pos, 1.0}; }

void Game::add_asteroid(Asteroid a) { space_.add_asteroid(std::move(a)); }

void Game::set_ship(Ship s) { space_.set_ship(s); }

void Game::handle_input(const InputState &input) { space_.set_input(input); }

void Game::update(double dt) { space_.step(dt); }

#ifdef AST_USE_SDL2

struct Renderer::State {
  SDL_Window *window = nullptr;
  int window_width = 0;
  int window_height = 0;
  SDL_Renderer *renderer = nullptr;
  bool quit = false;
  std::unordered_map<int, std::vector<Vec2>> asteroid_shapes;
};

namespace {

double polygon_area(const std::vector<Vec2> &polygon) {
  if (polygon.size() < 3) {
    return 0.0;
  }

  // Compute area using the shoelace formula
  double area = 0.0;
  for (size_t i = 0; i < polygon.size(); ++i) {
    const Vec2 &a = polygon[i];
    const Vec2 &b = polygon[(i + 1) % polygon.size()];
    area += a.x * b.y - b.x * a.y;
  }
  return std::abs(0.5 * area);
}

void draw_circle_outline(SDL_Renderer *renderer, int cx, int cy, int r) {
  if (r <= 0) {
    SDL_RenderDrawPoint(renderer, cx, cy);
    return;
  }

  constexpr int segments = 24;
  for (int i = 0; i < segments; ++i) {
    const double a0 = (TWO_PI * i) / segments;
    const double a1 = (TWO_PI * (i + 1)) / segments;
    const int x0 = cx + static_cast<int>(std::lround(std::cos(a0) * r));
    const int y0 = cy + static_cast<int>(std::lround(std::sin(a0) * r));
    const int x1 = cx + static_cast<int>(std::lround(std::cos(a1) * r));
    const int y1 = cy + static_cast<int>(std::lround(std::sin(a1) * r));
    SDL_RenderDrawLine(renderer, x0, y0, x1, y1);
  }
}

const std::vector<Vec2> &get_or_create_asteroid_shape(
    std::unordered_map<int, std::vector<Vec2>> &shapes, int id) {
  const int stable_id = std::max(1, id);
  auto existing = shapes.find(stable_id);
  if (existing != shapes.end()) {
    return existing->second;
  }

  const uint32_t base_seed =
      hash_u32(static_cast<uint32_t>(stable_id) * 2654435761U);
  const int vertex_count = 8 + static_cast<int>(base_seed % 5U);

  std::vector<Vec2> polygon;
  polygon.reserve(vertex_count);

  for (int i = 0; i < vertex_count; ++i) {
    const double base_angle = (TWO_PI * i) / vertex_count;
    const uint32_t vertex_seed =
        hash_u32(base_seed + static_cast<uint32_t>(i) * 977U);
    const double angle_jitter = (random_unit_from_seed(vertex_seed) - 0.5) *
                                (TWO_PI / vertex_count) * 0.35;
    const double radial =
        0.72 + 0.55 * random_unit_from_seed(vertex_seed + 19937U);
    const double angle = base_angle + angle_jitter;
    polygon.push_back({std::cos(angle) * radial, std::sin(angle) * radial});
  }

  const double area = polygon_area(polygon);
  if (area > 1e-9) {
    const double scale = std::sqrt(PI / area);
    for (Vec2 &point : polygon) {
      point.x *= scale;
      point.y *= scale;
    }
  }

  auto [inserted, _] = shapes.emplace(stable_id, std::move(polygon));
  return inserted->second;
}

void draw_asteroid_polygon(SDL_Renderer *renderer, int cx, int cy, int r,
                           int id,
                           std::unordered_map<int, std::vector<Vec2>> &shapes) {
  const std::vector<Vec2> &polygon = get_or_create_asteroid_shape(shapes, id);
  if (polygon.size() < 2) {
    return;
  }

  for (size_t i = 0; i < polygon.size(); ++i) {
    const Vec2 &a = polygon[i];
    const Vec2 &b = polygon[(i + 1) % polygon.size()];
    const int x0 = cx + static_cast<int>(std::lround(a.x * r));
    const int y0 = cy + static_cast<int>(std::lround(a.y * r));
    const int x1 = cx + static_cast<int>(std::lround(b.x * r));
    const int y1 = cy + static_cast<int>(std::lround(b.y * r));
    SDL_RenderDrawLine(renderer, x0, y0, x1, y1);
  }
}

void draw_ship(SDL_Renderer *renderer, int cx, int cy, double angle,
               double radius) {
  const double ac = std::cos(angle), as = std::sin(angle);  // Axis unit vector
  const double wc = std::cos(angle + 0.75 * PI),
               ws = std::sin(angle + 0.75 * PI);  // Left wing unit vector

  const int fx = cx + static_cast<int>(std::lround(ac * radius));
  const int fy = cy - static_cast<int>(std::lround(as * radius));
  const int bx = cx - static_cast<int>(std::lround(ac * radius * 0.5));
  const int by = cy + static_cast<int>(std::lround(as * radius * 0.5));
  const int lx = cx + static_cast<int>(std::lround(wc * radius));
  const int ly = cy - static_cast<int>(std::lround(ws * radius));
  const int rx = cx - static_cast<int>(std::lround(ws * radius));
  const int ry = cy - static_cast<int>(std::lround(wc * radius));

  SDL_RenderDrawLine(renderer, fx, fy, lx, ly);
  SDL_RenderDrawLine(renderer, lx, ly, bx, by);
  SDL_RenderDrawLine(renderer, bx, by, rx, ry);
  SDL_RenderDrawLine(renderer, rx, ry, fx, fy);
}

}  // namespace

Renderer::Renderer(int window_width, int window_height) {
  state_ = new State();
  state_->window_width = window_width;
  state_->window_height = window_height;
  SDL_Init(SDL_INIT_VIDEO);
  state_->window = SDL_CreateWindow(AST_WINDOW_TITLE, SDL_WINDOWPOS_CENTERED,
                                    SDL_WINDOWPOS_CENTERED, window_width,
                                    window_height, SDL_WINDOW_SHOWN);
  state_->renderer = SDL_CreateRenderer(
      state_->window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
}

Renderer::~Renderer() {
  if (!state_) {
    return;
  }
  if (state_->renderer) {
    SDL_DestroyRenderer(state_->renderer);
  }
  if (state_->window) {
    SDL_DestroyWindow(state_->window);
  }
  SDL_Quit();
  delete state_;
}

bool Renderer::should_quit() const { return state_ && state_->quit; }

InputState Renderer::poll_input() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT) {
      state_->quit = true;
    }
    if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
      state_->quit = true;
    }
  }

  const Uint8 *keys = SDL_GetKeyboardState(nullptr);
  return {
      .thrust_forward =
          static_cast<bool>(keys[SDL_SCANCODE_UP] || keys[SDL_SCANCODE_W]),
      .thrust_backward =
          static_cast<bool>(keys[SDL_SCANCODE_DOWN] || keys[SDL_SCANCODE_S]),
      .rotate_left =
          static_cast<bool>(keys[SDL_SCANCODE_LEFT] || keys[SDL_SCANCODE_A]),
      .rotate_right =
          static_cast<bool>(keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D]),
      .fire = static_cast<bool>(keys[SDL_SCANCODE_SPACE]),
  };
}

void Renderer::render(const Game &game) {
  const Space &space = game.space();
  const Ship &ship = space.ship();
  const CameraState camera = game.camera();

  int width = state_->window_width;
  int height = state_->window_height;
  SDL_GetRendererOutputSize(state_->renderer, &width, &height);
  const double scale = std::min(static_cast<double>(width) / AST_WINDOW_UNITS,
                                static_cast<double>(height) / AST_WINDOW_UNITS);

  auto to_screen = [&](const Vec2 &world) {
    const double x = width * 0.5 + (world.x - camera.pos.x) * scale;
    const double y = height * 0.5 - (world.y - camera.pos.y) * scale;
    return std::pair<int, int>{static_cast<int>(std::lround(x)),
                               static_cast<int>(std::lround(y))};
  };

  SDL_SetRenderDrawColor(state_->renderer, 0, 0, 0, 255);
  SDL_RenderClear(state_->renderer);

  // World bound rectangle
  {
    const auto [tlx, tly] =
        to_screen({-AST_WORLD_HALF_WIDTH, AST_WORLD_HALF_HEIGHT});
    const auto [brx, bry] =
        to_screen({AST_WORLD_HALF_WIDTH, -AST_WORLD_HALF_HEIGHT});
    SDL_Rect bounds{
        .x = std::min(tlx, brx),
        .y = std::min(tly, bry),
        .w = std::abs(brx - tlx),
        .h = std::abs(bry - tly),
    };
    SDL_SetRenderDrawColor(state_->renderer, 255, 255, 255, 255);
    SDL_RenderDrawRect(state_->renderer, &bounds);
  }

  // Asteroids
  SDL_SetRenderDrawColor(state_->renderer, 255, 255, 255, 255);
  for (const Asteroid &asteroid : space.asteroids()) {
    const auto [x, y] = to_screen(asteroid.pos);
    const int r =
        std::max(2, static_cast<int>(std::lround(asteroid.radius * scale)));
    draw_asteroid_polygon(state_->renderer, x, y, r, asteroid.id,
                          state_->asteroid_shapes);
  }

  // Bullets
  SDL_SetRenderDrawColor(state_->renderer, 255, 255, 255, 255);
  for (const Bullet &bullet : space.bullets()) {
    const auto [x, y] = to_screen(bullet.pos);
    SDL_Rect rect{x - 1, y - 1, 3, 3};
    SDL_RenderFillRect(state_->renderer, &rect);
  }

  // Explosions
  SDL_SetRenderDrawColor(state_->renderer, 255, 255, 255, 255);
  for (const Explosion &explosion : space.explosions()) {
    const auto [x, y] = to_screen(explosion.pos);
    const int r =
        std::max(2, static_cast<int>(std::lround(explosion.radius * scale)));
    draw_circle_outline(state_->renderer, x, y, r);
  }

  // Ship
  {
    const auto [x, y] = to_screen(ship.pos);
    SDL_SetRenderDrawColor(state_->renderer, 255, 255, 255, 255);
    draw_ship(state_->renderer, x, y, ship.angle, AST_SHIP_RADIUS * scale);
  }

  SDL_RenderPresent(state_->renderer);
}

#endif  // AST_USE_SDL2