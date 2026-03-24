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

AsteroidsConfig asteroids_config{};

namespace {

const AsteroidsConfig &cfg() { return asteroids_config; }

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
  a.radius = std::sqrt(a.mass) * cfg().asteroid.radius_per_sqrt_mass;
  return a;
}

// Pairwise gravity with softening; returns one acceleration per asteroid.
std::vector<Vec2> compute_asteroid_accelerations(
    std::span<const Asteroid> asteroids) {
  std::vector<Vec2> acc(asteroids.size(), {0.0, 0.0});
  for (size_t i = 0; i < asteroids.size(); ++i) {
    for (size_t j = i + 1; j < asteroids.size(); ++j) {
      const Vec2 d = asteroids[j].pos - asteroids[i].pos;
      const double d2 =
          dot(d, d) + cfg().physics.softening * cfg().physics.softening;
      const double inv_d = 1.0 / std::sqrt(d2);
      const double inv_d3 = inv_d * inv_d * inv_d;
      const double si = cfg().physics.gravity * asteroids[j].mass * inv_d3;
      const double sj = cfg().physics.gravity * asteroids[i].mass * inv_d3;
      acc[i] += d * si;
      acc[j] -= d * sj;
    }
  }
  return acc;
}

// Gravitational acceleration on a single point from all asteroids.
Vec2 compute_ship_acceleration(const Vec2 &pos,
                               std::span<const Asteroid> asteroids) {
  Vec2 acc{0.0, 0.0};
  if (cfg().ship.gravity) {
    for (const auto &a : asteroids) {
      const Vec2 d = a.pos - pos;
      const double d2 =
          dot(d, d) + cfg().physics.softening * cfg().physics.softening;
      const double inv_d = 1.0 / std::sqrt(d2);
      const double inv_d3 = inv_d * inv_d * inv_d;
      acc += d * (cfg().physics.gravity * a.mass * inv_d3);
    }
  }
  return acc;
}

std::vector<Asteroid> fragment_asteroid(const Asteroid &a, std::mt19937 &rng) {
  std::vector<Asteroid> fragments;

  std::discrete_distribution<int> count_dist({50, 30, 15, 5});
  int count = count_dist(rng) + 2;

  std::vector<double> props(count);
  std::uniform_real_distribution<double> prop_dist(0.3, 1.0);
  double total_prop = 0.0;
  for (int i = 0; i < count; ++i) {
    props[i] = prop_dist(rng);
    total_prop += props[i];
  }

  std::vector<double> masses(count);
  for (int i = 0; i < count; ++i) {
    masses[i] = a.mass * (props[i] / total_prop);
  }

  std::uniform_real_distribution<double> angle_dist(0.0, TWO_PI);
  double base_angle = angle_dist(rng);

  // Add jitter so the explosion angles don't look artificially perfect
  constexpr double max_jitter = PI / 4.0;
  std::uniform_real_distribution<double> jitter_dist(-max_jitter, max_jitter);

  // Fracture releases a specific amount of internal kinetic energy
  double released_energy = a.mass * cfg().asteroid.fracture_energy_per_mass;
  // Distribute fracture energy equally among fragments (regardless of mass)
  double energy_per_frag = released_energy / count;

  std::vector<Vec2> velocities(count);
  std::vector<Vec2> offsets(count);
  Vec2 momentum = {0.0, 0.0};
  Vec2 cm = {0.0, 0.0};

  for (int i = 0; i < count; ++i) {
    double angle = base_angle + i * (TWO_PI / count) + jitter_dist(rng);

    double speed = std::sqrt(2.0 * energy_per_frag / masses[i]);
    velocities[i] = {std::cos(angle) * speed, std::sin(angle) * speed};
    momentum += velocities[i] * masses[i];

    // Push the fragments outward along the fracture lines so they don't
    // instantly violently overlap. The offset ensures they spawn mostly clear
    // of each other.
    double frag_radius =
        std::sqrt(masses[i]) * cfg().asteroid.radius_per_sqrt_mass;
    double push_dist = a.radius * 0.5 + frag_radius * 0.5;
    offsets[i] = {std::cos(angle) * push_dist, std::sin(angle) * push_dist};
    cm += offsets[i] * masses[i];
  }

  // The net momentum of the outward pop must be zero and the center of mass
  // must coincide with the original asteroid
  Vec2 vel_corr = momentum / a.mass;
  Vec2 pos_corr = cm / a.mass;

  for (int i = 0; i < count; ++i) {
    if (masses[i] < cfg().asteroid.min_mass) {
      continue;  // Disintegrate into dust
    }

    Asteroid frag;
    frag.mass = masses[i];
    frag.pos = a.pos + offsets[i] - pos_corr;
    frag.vel = a.vel + velocities[i] - vel_corr;
    frag.stress = 0.0;
    fragments.push_back(normalize_asteroid(frag));
  }

  return fragments;
}

bool should_merge(const Asteroid &a, const Asteroid &b, std::mt19937 &rng) {
  double rel_vel = norm(a.vel - b.vel);
  double min_mass = std::min(a.mass, b.mass),
         max_mass = std::max(a.mass, b.mass);
  double mass_ratio = min_mass / max_mass;

  double p = 1.0;
  p -= 1 - std::exp(-rel_vel / cfg().asteroid.merge_speed_threshold);
  p -= mass_ratio * 0.3;
  p -= (a.stress + b.stress) / 2.0;
  p = std::clamp(p, 0.0, 1.0);

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng) < p;
}

bool should_split(const Asteroid &a, double impulse, std::mt19937 &rng) {
  double p = a.stress;
  p += 1 - std::exp(-impulse / a.mass * cfg().asteroid.split_impulse_scale);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng) < p;
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
    return e.age > cfg().explosion.lifetime;
  });

  // Rotate ship
  if (input_.rotate_left) {
    ship_.angle += cfg().ship.rotation_speed * dt;
  }
  if (input_.rotate_right) {
    ship_.angle -= cfg().ship.rotation_speed * dt;
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
      acc_ship += thrust_dir * cfg().ship.thrust_forward;
    }
    if (input_.thrust_backward) {
      acc_ship -= thrust_dir * cfg().ship.thrust_backward;
    }
    ship_.vel += acc_ship * (0.5 * dt);
  }

  // Drift
  for (auto &a : asteroids_) {
    a.pos += a.vel * dt;
  }
  ship_.pos += ship_.vel * dt;

  const double bound_width = cfg().world.half_width - cfg().ship.radius;
  const double bound_height = cfg().world.half_height - cfg().ship.radius;

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
      acc_ship += thrust_dir * cfg().ship.thrust_forward;
    }
    if (input_.thrust_backward) {
      acc_ship -= thrust_dir * cfg().ship.thrust_backward;
    }
    ship_.vel += acc_ship * (0.5 * dt);
  }

  // Despawn asteroids that have left the finite world
  std::erase_if(asteroids_, [](const Asteroid &a) {
    return std::abs(a.pos.x) > cfg().world.half_width + cfg().world.padding ||
           std::abs(a.pos.y) > cfg().world.half_height + cfg().world.padding;
  });

  // Spawn a bullet if a fire was triggered this frame
  if (fire_pending_) {
    bullets_.push_back({ship_.pos + thrust_dir * cfg().ship.radius,
                        ship_.vel + thrust_dir * cfg().bullet.speed,
                        cfg().bullet.lifetime});
    fire_pending_ = false;
  }

  // Advance bullets removing expired or out-of-bounds ones
  for (auto &b : bullets_) {
    b.pos += b.vel * dt;
    b.lifetime -= dt;
  }
  std::erase_if(bullets_, [](const Bullet &b) {
    return b.lifetime <= 0.0 ||
           std::abs(b.pos.x) > cfg().world.half_width + cfg().world.padding ||
           std::abs(b.pos.y) > cfg().world.half_height + cfg().world.padding;
  });

  std::uniform_int_distribution<int> seed_dist(1,
                                               std::numeric_limits<int>::max());

  // Bullet-asteroid collision
  std::vector<bool> bullet_hit(bullets_.size(), false);
  std::vector<bool> asteroid_destroyed(asteroids_.size(), false);
  std::vector<Asteroid> new_asteroids;

  for (size_t bi = 0; bi < bullets_.size(); ++bi) {
    for (size_t ai = 0; ai < asteroids_.size(); ++ai) {
      if (asteroid_destroyed[ai]) continue;

      // Check if bullet is within asteroid radius
      const Vec2 d = bullets_[bi].pos - asteroids_[ai].pos;
      const double r = asteroids_[ai].radius;
      if (dot(d, d) > r * r) continue;
      bullet_hit[bi] = true;

      // Surface impact point in the direction of the incoming bullet
      const double dn = std::max(norm(d), EPS);
      const Vec2 surf = asteroids_[ai].pos + (d / dn) * r;
      explosions_.push_back(
          {surf, r * cfg().explosion.scale, 0.0, seed_dist(random_engine_)});

      // Apply bullet physics (inelastic momentum transfer)
      Vec2 old_vel = asteroids_[ai].vel;
      asteroids_[ai].vel +=
          bullets_[bi].vel * (cfg().bullet.mass / asteroids_[ai].mass);
      double impulse = norm(asteroids_[ai].vel - old_vel) * asteroids_[ai].mass;

      asteroids_[ai].stress += cfg().bullet.stress_on_hit;

      if (should_split(asteroids_[ai], impulse, random_engine_)) {
        std::vector<Asteroid> frags =
            fragment_asteroid(asteroids_[ai], random_engine_);
        new_asteroids.insert(new_asteroids.end(), frags.begin(), frags.end());
        asteroid_destroyed[ai] = true;
      }

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

  // Asteroid-asteroid collision
  for (size_t i = 0; i < asteroids_.size(); ++i) {
    if (asteroid_destroyed[i]) continue;
    for (size_t j = i + 1; j < asteroids_.size(); ++j) {
      if (asteroid_destroyed[j]) continue;

      Vec2 r = asteroids_[j].pos - asteroids_[i].pos;
      double dist2 = dot(r, r);
      double rad_sum = asteroids_[i].radius + asteroids_[j].radius;
      if (dist2 > rad_sum * rad_sum) continue;

      double dist = std::sqrt(dist2);
      dist = std::max(dist, EPS);  // Avoid division by zero
      Vec2 normal = r / dist;

      Vec2 rel_vel = asteroids_[j].vel - asteroids_[i].vel;
      double vel_along_normal = dot(rel_vel, normal);

      // Do not resolve if velocities are separating
      if (vel_along_normal > 0) continue;

      if (should_merge(asteroids_[i], asteroids_[j], random_engine_)) {
        Asteroid merged;
        merged.mass = asteroids_[i].mass + asteroids_[j].mass;
        merged.vel = (asteroids_[i].vel * asteroids_[i].mass +
                      asteroids_[j].vel * asteroids_[j].mass) /
                     merged.mass;
        merged.pos = (asteroids_[i].pos * asteroids_[i].mass +
                      asteroids_[j].pos * asteroids_[j].mass) /
                     merged.mass;
        // Inherit the ID of the larger one to keep shape continuity
        merged.id = (asteroids_[i].mass > asteroids_[j].mass)
                        ? asteroids_[i].id
                        : asteroids_[j].id;
        merged.stress = (asteroids_[i].stress * asteroids_[i].mass +
                         asteroids_[j].stress * asteroids_[j].mass) /
                        merged.mass;

        merged = normalize_asteroid(merged);

        asteroid_destroyed[i] = true;
        asteroid_destroyed[j] = true;
        new_asteroids.push_back(merged);
        break;  // `i`-th asteroid is destroyed, break out of `j` loop
      } else {
        // Elastic-like bounce
        double j_impulse =
            -(1 + cfg().asteroid.elastic_restitution) * vel_along_normal;
        j_impulse /= (1.0 / asteroids_[i].mass + 1.0 / asteroids_[j].mass);

        Vec2 impulse = normal * j_impulse;
        asteroids_[i].vel -= impulse / asteroids_[i].mass;
        asteroids_[j].vel += impulse / asteroids_[j].mass;

        asteroids_[i].stress +=
            1 - std::exp(-std::abs(j_impulse) / asteroids_[i].mass *
                         cfg().asteroid.split_impulse_scale);
        asteroids_[j].stress +=
            1 - std::exp(-std::abs(j_impulse) / asteroids_[j].mass *
                         cfg().asteroid.split_impulse_scale);

        // Separate them so they don't stick
        double overlap = rad_sum - dist;
        asteroids_[i].pos -= normal * (overlap * 0.5);
        asteroids_[j].pos += normal * (overlap * 0.5);

        if (should_split(asteroids_[i], std::abs(j_impulse), random_engine_)) {
          std::vector<Asteroid> frags =
              fragment_asteroid(asteroids_[i], random_engine_);
          new_asteroids.insert(new_asteroids.end(), frags.begin(), frags.end());
          asteroid_destroyed[i] = true;
        }
        if (should_split(asteroids_[j], std::abs(j_impulse), random_engine_)) {
          std::vector<Asteroid> frags =
              fragment_asteroid(asteroids_[j], random_engine_);
          new_asteroids.insert(new_asteroids.end(), frags.begin(), frags.end());
          asteroid_destroyed[j] = true;
        }

        if (asteroid_destroyed[i]) {
          break;  // `i` is destroyed, break out of `j` loop
        }
      }
    }
  }

  // Update asteroid list once per frame (after all collisions are processed)
  {
    std::vector<Asteroid> surviving;
    surviving.reserve(asteroids_.size() + new_asteroids.size());
    for (size_t i = 0; i < asteroids_.size(); ++i) {
      if (!asteroid_destroyed[i]) {
        // Natural stress healing over time
        asteroids_[i].stress = std::max(
            0.0, asteroids_[i].stress - cfg().asteroid.stress_decay * dt);
        surviving.push_back(asteroids_[i]);
      } else {
        // Spawn an explosion for newly destroyed asteroids
        explosions_.push_back({asteroids_[i].pos, asteroids_[i].radius, 0.0,
                               seed_dist(random_engine_)});
      }
    }

    asteroids_ = std::move(surviving);
    for (const auto &a : new_asteroids) {
      add_asteroid(a);
    }
  }
}

const Space &Game::space() const { return space_; }

// Zoom not implemented yet, return default zoom for now
CameraState Game::camera() const { return {space_.ship().pos, 1.0}; }

void Game::set_ship(Ship s) { space_.set_ship(s); }

void Game::handle_input(const InputState &input) { space_.set_input(input); }

void Game::update(double dt) { space_.step(dt); }

void Game::add_asteroid(Asteroid a) { space_.add_asteroid(std::move(a)); }

void Game::generate_asteroid(const Vec2 &pos, double min_mass, double max_mass,
                             double min_momentum, double max_momentum) {
  const double mass_lo = std::max(min_mass, cfg().asteroid.min_mass);
  const double mass_hi = std::max(max_mass, mass_lo);
  const double momentum_lo = std::max(min_momentum, 0.0);
  const double momentum_hi = std::max(max_momentum, momentum_lo);

  std::uniform_real_distribution<double> mass_dist(mass_lo, mass_hi);
  std::uniform_real_distribution<double> momentum_dist(momentum_lo,
                                                       momentum_hi);
  std::uniform_real_distribution<double> angle_dist(0.0, TWO_PI);

  const double mass = mass_dist(random_engine_);

  // Skip asteroids that would spawn inside another one to avoid instant
  // high impulse collisions.
  bool collides = false;
  for (const auto &a : space_.asteroids()) {
    const Vec2 d = a.pos - pos;
    const double r =
        a.radius + std::sqrt(mass) * cfg().asteroid.radius_per_sqrt_mass;
    if (dot(d, d) < r * r) {
      collides = true;
      break;
    }
  }
  if (collides) {
    return;
  }

  const double angle_vel = angle_dist(random_engine_);
  const double speed = momentum_dist(random_engine_) / mass;
  const Vec2 vel{std::cos(angle_vel) * speed, std::sin(angle_vel) * speed};

  add_asteroid({.pos = pos, .vel = vel, .mass = mass});
}

void Game::generate_asteroid_field(const Vec2 &center, double radius,
                                   double density, double min_mass,
                                   double max_mass, double min_momentum,
                                   double max_momentum) {
  if (radius <= 0.0 || density <= 0.0) {
    return;
  }

  const double area = PI * radius * radius;
  const int count = std::max(1, static_cast<int>(std::lround(area * density)));

  std::uniform_real_distribution<double> angle_dist(0.0, TWO_PI);
  std::uniform_real_distribution<double> radial_dist(0.0, 1.0);

  for (int i = 0; i < count; ++i) {
    // Square root produces uniform point density over the disk area
    const double angle_pos = angle_dist(random_engine_);
    const double dist = std::sqrt(radial_dist(random_engine_)) * radius;
    const Vec2 pos{center.x + std::cos(angle_pos) * dist,
                   center.y + std::sin(angle_pos) * dist};

    generate_asteroid(pos, min_mass, max_mass, min_momentum, max_momentum);
  }
}

void Game::remove_asteroids(const Vec2 &center, double radius) {
  if (radius <= 0.0) {
    return;
  }

  std::vector<Asteroid> surviving;
  for (const auto &a : space_.asteroids()) {
    Vec2 d = a.pos - center;
    if (dot(d, d) > radius * radius) {
      surviving.push_back(a);
    }
  }
  space_.asteroids() = std::move(surviving);
}

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

  const int segments = std::max(6, cfg().render.circle_segments);
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
  if (area > EPS) {
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
  const double ac = std::cos(angle),
               as = std::sin(angle);  // Axis unit vector
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
  state_->window = SDL_CreateWindow(cfg().window.title, SDL_WINDOWPOS_CENTERED,
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
  const double scale =
      std::min(static_cast<double>(width) / cfg().render.window_units,
               static_cast<double>(height) / cfg().render.window_units);

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
        to_screen({-cfg().world.half_width, cfg().world.half_height});
    const auto [brx, bry] =
        to_screen({cfg().world.half_width, -cfg().world.half_height});
    SDL_Rect bounds{
        .x = std::min(tlx, brx),
        .y = std::min(tly, bry),
        .w = std::abs(brx - tlx),
        .h = std::abs(bry - tly),
    };
    SDL_SetRenderDrawColor(state_->renderer, 255, 255, 255, 255);

    const int step = cfg().render.bound_dash + cfg().render.bound_gap;

    for (int x = bounds.x; x < bounds.x + bounds.w; x += step) {
      const int x2 = std::min(x + cfg().render.bound_dash, bounds.x + bounds.w);
      SDL_RenderDrawLine(state_->renderer, x, bounds.y, x2, bounds.y);
      SDL_RenderDrawLine(state_->renderer, x, bounds.y + bounds.h, x2,
                         bounds.y + bounds.h);
    }

    for (int y = bounds.y; y < bounds.y + bounds.h; y += step) {
      const int y2 = std::min(y + cfg().render.bound_dash, bounds.y + bounds.h);
      SDL_RenderDrawLine(state_->renderer, bounds.x, y, bounds.x, y2);
      SDL_RenderDrawLine(state_->renderer, bounds.x + bounds.w, y,
                         bounds.x + bounds.w, y2);
    }
  }

  // Asteroids
  for (const Asteroid &asteroid : space.asteroids()) {
#if AST_DEBUG_STRESS_GRADIENT
    const double stress = std::clamp(asteroid.stress, 0.0, 1.0);
    const Uint8 value = static_cast<Uint8>(std::lround(80.0 + stress * 175.0));
    SDL_SetRenderDrawColor(state_->renderer, value, value, value, 255);
#else
    SDL_SetRenderDrawColor(state_->renderer, 255, 255, 255, 255);
#endif
    const auto [x, y] = to_screen(asteroid.pos);
    const int r =
        std::max(cfg().render.min_draw_radius_px,
                 static_cast<int>(std::lround(asteroid.radius * scale)));
    draw_asteroid_polygon(state_->renderer, x, y, r, asteroid.id,
                          state_->asteroid_shapes);
  }

  // Bullets
  SDL_SetRenderDrawColor(state_->renderer, 255, 255, 255, 255);
  for (const Bullet &bullet : space.bullets()) {
    const auto [x, y] = to_screen(bullet.pos);
    SDL_Rect rect{x - cfg().render.bullet_half_px,
                  y - cfg().render.bullet_half_px, cfg().render.bullet_size_px,
                  cfg().render.bullet_size_px};
    SDL_RenderFillRect(state_->renderer, &rect);
  }

  // Explosions
  SDL_SetRenderDrawColor(state_->renderer, 255, 255, 255, 255);
  for (const Explosion &explosion : space.explosions()) {
    const auto [x, y] = to_screen(explosion.pos);
    const int r =
        std::max(cfg().render.min_draw_radius_px,
                 static_cast<int>(std::lround(explosion.radius * scale)));
    draw_circle_outline(state_->renderer, x, y, r);
  }

  // Ship
  {
    const auto [x, y] = to_screen(ship.pos);
    SDL_SetRenderDrawColor(state_->renderer, 255, 255, 255, 255);
    draw_ship(state_->renderer, x, y, ship.angle, cfg().ship.radius * scale);
  }

  SDL_RenderPresent(state_->renderer);
}

#endif  // AST_USE_SDL2