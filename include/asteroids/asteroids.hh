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

#define AST_WINDOW_TITLE "Asteroids"
#define AST_WINDOW_WIDTH 1280
#define AST_WINDOW_HEIGHT 720
#define AST_WINDOW_UNITS \
  1000.0  // World units corresponding to the smaller window dimension (with
          // default zoom)

// Times are in seconds, angles in radians, distances and masses in arbitrary
// units.
#define AST_WORLD_HALF_WIDTH 2000.0
#define AST_WORLD_HALF_HEIGHT 2000.0
#define AST_G 20.0
#define AST_EPS 1.0  // Softening to prevent singularities
#define AST_ESPLOSION_LIFETIME 0.35
#define AST_SHIP_THRUST_FORWARD 75.0
#define AST_SHIP_THRUST_BACKWARD 40.0
#define AST_SHIP_ROTATION_SPEED 3.0
#define AST_BULLET_SPEED 150.0
#define AST_BULLET_LIFETIME 3.0
#define AST_BULLET_MASS 20.0
#define AST_BULLET_STRESS 0.2
#define AST_RADIUS_PER_SQRT_MASS 0.5  // Radius per square root of mass
#define AST_SHIP_RADIUS 20.0
#define AST_SHIP_GRAVITY true  // Ship is affected by asteroid gravity
#define AST_MIN_ASTEROID_MASS 100.0
#define AST_ASTEROID_FRACTURE_ENERGY_PER_MASS 1000.0
#define AST_ASTEROID_MERGE_SPEED_THRESHOLD 75.0
#define AST_ASTEROID_SPLIT_IMPULSE_SCALE 0.2
#define AST_EXPLOSION_SCALE 0.5
#define AST_ASTEROID_ELASTIC_RESTITUTION 0.5
#define AST_ASTEROID_STRESS_DECAY 0.05  // Stress healing per second

#define AST_WORLD_WIDTH (2.0 * AST_WORLD_HALF_WIDTH)
#define AST_WORLD_HEIGHT (2.0 * AST_WORLD_HALF_HEIGHT)

// Math utilities

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define PI M_PI
#define TWO_PI (2.0 * M_PI)

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

// Entity definitions

struct Asteroid {
  Vec2 pos, vel;
  double mass;
  double radius = 0.0;
  int id = 0;
  double stress = 0.0;
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

  std::mt19937 random_engine_{std::random_device{}()};
};

class Game {
 public:
  const Space &space() const;
  CameraState camera() const;

  void add_asteroid(Asteroid a);
  void set_ship(Ship s);
  void handle_input(const InputState &input);
  void update(double dt);

 private:
  Space space_;
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
