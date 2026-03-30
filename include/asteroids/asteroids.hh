#ifndef ASTEROIDS_HH
#define ASTEROIDS_HH

#include <cmath>
#include <random>
#include <span>
#include <vector>

#ifdef AST_USE_SDL2
#if __has_include(<SDL2/SDL.h>)
#include <SDL2/SDL.h>
#elif __has_include(<SDL.h>)
#include <SDL.h>
#else
#error "SDL2 headers not found"
#endif
#endif

// Config and constants

#define AST_TITLE "Asteroids"
#define AST_WINDOW_WIDTH 1280
#define AST_WINDOW_HEIGHT 720
#define AST_DEBUG 0

struct WindowConfig {
  const char *title = AST_TITLE;
  int width = AST_WINDOW_WIDTH;
  int height = AST_WINDOW_HEIGHT;
};

struct WorldConfig {
  double half_width = 4000.0;
  double half_height = 4000.0;
  double padding = 1000.0;
};

struct PhysicsConfig {
  double gravity = 20.0;
  double softening = 1.0;
};

struct ShipConfig {
  double radius = 20.0;
  double thrust_forward = 75.0;
  double thrust_backward = 40.0;
  double rotation_speed = 3.0;
#if AST_DEBUG
  bool gravity = false;
#else
  bool gravity = true;
#endif
};

struct BulletConfig {
  double speed = 200.0;
  double lifetime = 3.0;
  double mass = 20.0;
  double stress_on_hit = 0.2;
#if AST_DEBUG
  bool gravity = false;
#else
  bool gravity = true;
#endif
};

struct AsteroidConfig {
  double radius_per_sqrt_mass = 0.5;
  double min_mass = 100.0;
  double fracture_energy_per_mass = 1000.0;
  double elastic_restitution = 0.85;
  double split_impulse_scale = 0.1;
  double merge_speed_threshold = 75.0;
  double stress_decay = 0.05;
#if AST_DEBUG
  double active_radius = 200.0;
  double passive_radius = 200.0;
#else
  double active_radius = 1200.0;
  double passive_radius = 1600.0;
#endif
  int passive_update_stride = 4;
};

struct ExplosionConfig {
  double scale = 0.5;
  double lifetime = 0.35;
};

struct RenderConfig {
  double window_units =
      1000.0;  // Number of world units that fit in the smaller window dimension
  int bound_dash = 20;
  int bound_gap = 20;
  int min_draw_radius_px = 2;
  int circle_segments = 24;
  int bullet_size_px = 3;
  int bullet_half_px = 1;
};

struct TimingConfig {
  double fixed_dt = 1.0 / 60.0;
  double max_frame_dt = 0.25;
};

struct AsteroidsConfig {
  WindowConfig window{};
  WorldConfig world{};
  PhysicsConfig physics{};
  ShipConfig ship{};
  BulletConfig bullet{};
  AsteroidConfig asteroid{};
  ExplosionConfig explosion{};
  RenderConfig render{};
  TimingConfig timing{};
};

extern AsteroidsConfig asteroids_config;

// Math utilities

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define PI M_PI
#define TWO_PI (2.0 * M_PI)

#define EPS 1e-6

struct Vec2 {
  double x, y;
};

inline Vec2 operator+(const Vec2 &a, const Vec2 &b) {
  return {a.x + b.x, a.y + b.y};
}
inline Vec2 operator-(const Vec2 &a, const Vec2 &b) {
  return {a.x - b.x, a.y - b.y};
}
inline Vec2 operator-(const Vec2 &v) { return {-v.x, -v.y}; }
inline Vec2 operator*(const Vec2 &v, double s) { return {v.x * s, v.y * s}; }
inline Vec2 operator*(double s, const Vec2 &v) { return {v.x * s, v.y * s}; }
inline Vec2 operator/(const Vec2 &v, double s) { return {v.x / s, v.y / s}; }

inline Vec2 &operator+=(Vec2 &a, const Vec2 &b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
inline Vec2 &operator-=(Vec2 &a, const Vec2 &b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

inline double dot(const Vec2 &a, const Vec2 &b) {
  return a.x * b.x + a.y * b.y;
}
inline double norm(const Vec2 &v) { return std::sqrt(dot(v, v)); }

// Utilities

template <typename T>
struct Range {
  T min, max;
};

// Entity definitions

struct Asteroid {
  Vec2 pos, vel;
  double mass;
  double radius = 0.0;
  int id = 0;
  double stress = 0.0;
  bool active = true;
};

struct Ship {
  Vec2 pos, vel;
  double angle = 0.0;  // Radians, counterclockwise from positive x-axis
};

struct Bullet {
  Vec2 pos, vel;
  double lifetime = 3.0;  // Seconds remaining before despawn
};

struct Explosion {
  Vec2 pos;
  double radius;
  double age = 0.0;
  int seed = 0;
};

// Input / camera

struct InputState {
  bool thrust_forward = false;
  bool thrust_backward = false;
  bool rotate_left = false;
  bool rotate_right = false;
  bool fire = false;
};

struct CameraState {
  Vec2 pos;  // World-space centre of the view (tracks ship)
  double zoom = 1.0;
};

// Worker classes

class Space {
 public:
  std::span<const Asteroid> asteroids() const;
  std::span<const Bullet> bullets() const;
  std::span<const Explosion> explosions() const;
  const Ship &ship() const;

  void add_asteroid(Asteroid a);
  void set_ship(Ship s);
  void set_input(const InputState &input);

  // Advance the simulation by `dt` seconds (symplectic kick-drift-kick
  // integrator)
  void step(double dt);

 private:
  Ship ship_{};
  std::vector<Asteroid> asteroids_;
  std::vector<Bullet> bullets_;
  std::vector<Explosion> explosions_;

  int next_asteroid_id_ = 1;

  InputState input_{};
  bool fire_pending_ = false;  // Edge-triggered fire flag
  unsigned int step_counter_ = 0;

  std::mt19937 random_engine_{std::random_device{}()};
};

class Game {
 public:
  const Space &space() const;
  CameraState camera() const;

  void set_ship(Ship s = {});

  void handle_input(const InputState &input);
  void update(double dt);

  void add_asteroid(Asteroid a = {});
  void generate_rand_asteroid(const Vec2 &pos, Range<double> mass,
                              Range<double> momentum,
                              Range<double> angle = {0.0, TWO_PI},
                              Vec2 vel_bias = {0.0, 0.0});
  void generate_rand_asteroid_cluster(const Vec2 &center, double radius,
                                      double density, Range<double> mass,
                                      Range<double> momentum,
                                      Range<double> angle = {0.0, TWO_PI},
                                      Vec2 vel_bias = {0.0, 0.0});
  void generate_rand_world_asteroids(double density, Range<double> mass,
                                     Range<double> momentum,
                                     Range<double> angle = {0.0, TWO_PI},
                                     Vec2 vel_bias = {0.0, 0.0});
  void clear_ship_vicinity(double radius = 50.0);
  void generate_rand_incoming_asteroids(double density, Range<double> mass,
                                        Range<double> momentum,
                                        Vec2 vel_bias = {0.0, 0.0});

 private:
  Space space_;
  std::mt19937 random_engine_{std::random_device{}()};
};

#ifdef AST_USE_SDL2

class Renderer {
 public:
  explicit Renderer(int window_width = AST_WINDOW_WIDTH,
                    int window_height = AST_WINDOW_HEIGHT);
  ~Renderer();

  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;

  // Consume SDL events and return the current keyboard input state
  InputState poll_input();

  // Draw one frame using the game's current state and camera
  void render(const Game &game);

  bool should_quit() const;

 private:
  struct State;
  State *state_ = nullptr;
};

#endif  // AST_USE_SDL2

#endif  // ASTEROIDS_HH
