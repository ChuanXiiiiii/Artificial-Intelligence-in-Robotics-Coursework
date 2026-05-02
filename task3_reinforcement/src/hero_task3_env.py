from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common import resolve_project_root
from entity_inference import BRIBABLE_CLASSES, HOSTILE_CLASSES, Task1EntityClassifier, resolve_task1_model_path
from sensor_adapter import SENSOR_COLUMNS, SensorFeatureAdapter


BASELINE_ROOT = resolve_project_root() / "SEMTM0016_DungeonMazeWorld-main"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.append(str(BASELINE_ROOT))

from envs.simple_dungeonworld_env import Actions, DungeonMazeEnv  # noqa: E402


@dataclass
class EntityState:
    entity_id: int
    species: str
    pos: Tuple[int, int]
    entity_type: str
    direction: Tuple[int, int]
    wait_timer: int = 0
    in_wall: bool = False
    alive: bool = True


class HeroTask3Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 16,
        max_steps: int = 256,
        render_mode: str | None = None,
        reward_scheme: str = "A",
        n_virtual_entities: int = 12,
        target_braid_loops: int = 5,
        task1_model_type: str = "hog_svm",
        task1_seed: int = 42,
        task2_cluster_seed: int = 42,
        task2_fixed_k: int = 6,
        hostile_collision_penalty_b: float = -8.0,
        bribable_contact_bonus: float = 2.0,
        goal_reward: float = 100.0,
        distance_scale_b: float = 1.0,
        approach_bribable_scale_b: float = 0.2,
        potential_scale_c: float = 0.2,
        path_scale_e: float = 1.0,
        step_penalty_e: float = -0.01,
        wall_hit_penalty_c: float = -0.3,
        dead_loop_penalty_c: float = -0.1,
        stagnation_penalty_c: float = 0.0,
        hostile_collision_penalty_g: float = -6.0,
        kill_zone_penalty_h: float = -15.0,
        wingedrat_kill_bonus_h: float = 1.0,
        stealth_wait_steps_h: int = 3,
        bribe_cost_min_g: float = 0.08,
        bribe_cost_max_g: float = 0.35,
        include_astar_hint: bool = False,
        include_astar_ego_hint: bool = False,
        prefer_mps_for_task1: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.reward_scheme = reward_scheme.upper()
        self.n_virtual_entities = int(n_virtual_entities)
        self.target_braid_loops = max(0, int(target_braid_loops))
        self.hostile_collision_penalty_b = float(hostile_collision_penalty_b)
        self.bribable_contact_bonus = float(bribable_contact_bonus)
        self.goal_reward = float(goal_reward)
        self.distance_scale_b = float(distance_scale_b)
        self.approach_bribable_scale_b = float(approach_bribable_scale_b)
        self.potential_scale_c = float(potential_scale_c)
        self.path_scale_e = float(path_scale_e)
        self.step_penalty_e = float(step_penalty_e)
        self.potential_scale_multiplier = 1.0
        self.wall_hit_penalty_c = float(wall_hit_penalty_c)
        self.dead_loop_penalty_c = float(dead_loop_penalty_c)
        self.stagnation_penalty_c = float(stagnation_penalty_c)
        self.hostile_collision_penalty_g = float(hostile_collision_penalty_g)
        self.kill_zone_penalty_h = float(kill_zone_penalty_h)
        self.wingedrat_kill_bonus_h = float(wingedrat_kill_bonus_h)
        self.stealth_wait_steps_h = max(1, int(stealth_wait_steps_h))
        self.bribe_cost_min_g = float(bribe_cost_min_g)
        self.bribe_cost_max_g = float(bribe_cost_max_g)
        self.include_astar_hint = bool(include_astar_hint)
        self.include_astar_ego_hint = bool(include_astar_ego_hint)
        self.render_mode = render_mode

        self.base_env = DungeonMazeEnv(render_mode=render_mode, grid_size=self.grid_size)
        self.action_space = self.base_env.action_space

        task3_root = Path(__file__).resolve().parents[1]
        device = "mps" if prefer_mps_for_task1 else "cpu"
        task1_model_path = resolve_task1_model_path(task3_root, task1_model_type, task1_seed)
        self.entity_classifier = Task1EntityClassifier(
            model_type=task1_model_type,
            model_path=task1_model_path,
            device=device,
        )
        self.sensor_adapter = SensorFeatureAdapter(
            project_root=resolve_project_root(),
            fixed_cluster_k=task2_fixed_k,
            seed=task2_cluster_seed,
        )
        self.hostile_cluster_ids = {
            self.sensor_adapter.species_cluster_id(species)
            for species in HOSTILE_CLASSES
            if species in self.sensor_adapter.species_prototypes
        }
        self.bribable_cluster_ids = {
            self.sensor_adapter.species_cluster_id(species)
            for species in BRIBABLE_CLASSES
            if species in self.sensor_adapter.species_prototypes
        }

        obs_channels = 5 if self._is_stealth_scheme() else 4
        if self._is_stealth_scheme():
            obs_high = float(max(1, self.sensor_adapter.cluster_count))
            self.observation_space = spaces.Box(
                low=0.0,
                high=obs_high,
                shape=(obs_channels, 7, 7),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(obs_channels, 7, 7), dtype=np.uint8)

        self.rng = np.random.default_rng(seed=seed)
        self.entity_map: Dict[Tuple[int, int], EntityState] = {}
        self.entities: List[EntityState] = []
        self.next_entity_id = 1
        self.has_weapon = False

        self.step_count = 0
        self.episode_return = 0.0
        self.wall_collision_count = 0
        self.dead_loop_events = 0
        self.hostile_collision_count = 0
        self.bribable_contact_count = 0
        self.bribe_cost_total = 0.0
        self.stagnation_events = 0
        self.trajectory: List[Tuple[int, int]] = []
        self.position_window: deque[Tuple[int, int]] = deque(maxlen=24)
        self.short_position_window: deque[Tuple[int, int]] = deque(maxlen=5)
        self.true_distance_map: Dict[Tuple[int, int], int] = {}
        self.prev_target_dist = 0.0
        self.prev_true_target_dist = float(self.grid_size * self.grid_size)
        self.prev_bribable_dist = float(self.grid_size)

    def _is_final_reward_scheme(self) -> bool:
        return self.reward_scheme in {"G", "H"}

    def _is_stealth_scheme(self) -> bool:
        return self.reward_scheme == "H"

    def _uses_path_shaping(self) -> bool:
        return self.reward_scheme in {"E", "F", "F2", "G", "H"}

    def _uses_loop_penalties(self) -> bool:
        return self.reward_scheme in {"C", "E", "F", "F2", "G", "H"}

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _pos_tuple(pos: np.ndarray) -> Tuple[int, int]:
        return int(pos[0]), int(pos[1])

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _is_passable_cell(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return False
        cell = self.base_env.maze.get_cell_item(x, y)
        if cell is None:
            return True
        return str(getattr(cell, "type", "")) == "target"

    def _is_passable_for_path(self, x: int, y: int, target: Tuple[int, int]) -> bool:
        if x <= 0 or y <= 0 or x >= self.grid_size - 1 or y >= self.grid_size - 1:
            return False
        if (x, y) == target:
            return True
        cell = self.base_env.maze.get_cell_item(x, y)
        if cell is None:
            return True
        return str(getattr(cell, "type", "")) == "target"

    def _build_true_distance_map(self, target: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        q: deque[Tuple[int, int]] = deque([target])
        dist: Dict[Tuple[int, int], int] = {target: 0}
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

        while q:
            x, y = q.popleft()
            base_d = dist[(x, y)]
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if nxt in dist:
                    continue
                if not self._is_passable_for_path(nx, ny, target):
                    continue
                dist[nxt] = base_d + 1
                q.append(nxt)

        return dist

    def _true_distance_to_target(self, pos: Tuple[int, int], target: Tuple[int, int]) -> float:
        if not self.true_distance_map:
            self.true_distance_map = self._build_true_distance_map(target)
        d = self.true_distance_map.get(pos)
        if d is None:
            return float(self.grid_size * self.grid_size)
        return float(d)

    def _next_astar_step(self, pos: Tuple[int, int], target: Tuple[int, int]) -> Tuple[float, float]:
        """Return the discrete (dx, dy) move in {-1,0,1} that steps along the shortest valid path to target.
        Returns (0.0, 0.0) if already at target or unreachable."""
        if not self.true_distance_map:
            self.true_distance_map = self._build_true_distance_map(target)
        cur_d = self.true_distance_map.get(pos)
        if cur_d is None or cur_d == 0:
            return 0.0, 0.0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (pos[0] + dx, pos[1] + dy)
            nd = self.true_distance_map.get(nb)
            if nd is not None and nd < cur_d:
                return float(dx), float(dy)
        return 0.0, 0.0

    def _reset_episode_stats(self) -> None:
        self.step_count = 0
        self.episode_return = 0.0
        self.wall_collision_count = 0
        self.dead_loop_events = 0
        self.hostile_collision_count = 0
        self.bribable_contact_count = 0
        self.bribe_cost_total = 0.0
        self.stagnation_events = 0
        self.trajectory = []
        self.position_window.clear()
        self.short_position_window.clear()
        self.has_weapon = False
        self.global_step_counter = 0

    def _spawn_protection_cells(self) -> set[Tuple[int, int]]:
        rx, ry = self._pos_tuple(self.base_env.robot_position)
        direction = int(self.base_env.robot_direction)
        forward_vectors = {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0),
        }
        dx, dy = forward_vectors.get(direction, (0, -1))

        protected: set[Tuple[int, int]] = set()
        x, y = rx, ry
        while self._in_bounds((x, y)):
            protected.add((x, y))
            nx, ny = x + dx, y + dy
            if not self._in_bounds((nx, ny)):
                break
            cell = self.base_env.maze.get_cell_item(nx, ny)
            if cell is not None and str(getattr(cell, "type", "")) == "wall":
                break
            x, y = nx, ny

        return protected

    def _entity_type(self, species: str) -> str:
        if species in {"human", "halfling"}:
            return "neutral"
        if species == "orc":
            return "ground"
        if species in {"wingedrat", "lizard"}:
            return "flying"
        return "neutral"

    def _create_entity(self, species: str, pos: Tuple[int, int]) -> EntityState:
        entity_type = self._entity_type(species)
        direction = (0, 0)
        if entity_type == "ground":
            candidates = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if self._is_passable_cell(pos[0] + dx, pos[1] + dy):
                    candidates.append((dx, dy))
            if candidates:
                direction = candidates[int(self.rng.integers(0, len(candidates)))]
            else:
                direction = (1, 0) if bool(self.rng.integers(0, 2)) else (0, 1)
        elif entity_type == "flying":
            if bool(self.rng.integers(0, 2)):
                direction = (1, 0) if bool(self.rng.integers(0, 2)) else (-1, 0)
            else:
                direction = (0, 1) if bool(self.rng.integers(0, 2)) else (0, -1)

        entity = EntityState(
            entity_id=self.next_entity_id,
            species=species,
            pos=pos,
            entity_type=entity_type,
            direction=direction,
        )
        self.next_entity_id += 1
        return entity

    def _rebuild_entity_map(self) -> None:
        self.entity_map = {ent.pos: ent for ent in self.entities if ent.alive}

    def _remove_entity_at(self, pos: Tuple[int, int]) -> None:
        ent = self.entity_map.pop(pos, None)
        if ent is not None:
            ent.alive = False

    def _spawn_virtual_entities(self) -> None:
        if self._is_stealth_scheme():
            self._spawn_stealth_entities()
            return

        candidates: List[Tuple[int, int]] = []
        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                if (x, y) == (1, 1):
                    continue
                if (x, y) == (self.grid_size - 2, self.grid_size - 2):
                    continue
                cell = self.base_env.maze.get_cell_item(x, y)
                if cell is None:
                    candidates.append((x, y))

        if not candidates:
            self.entity_map = {}
            self.entities = []
            return

        count = min(self.n_virtual_entities, len(candidates))
        chosen = self.rng.choice(len(candidates), size=count, replace=False)
        species_pool = np.array(["halfling", "human", "lizard", "orc", "wingedrat"], dtype=object)

        self.entities = []
        entity_map: Dict[Tuple[int, int], EntityState] = {}
        for idx in np.asarray(chosen).tolist():
            pos = candidates[int(idx)]
            species = str(self.rng.choice(species_pool))
            ent = self._create_entity(species, pos)
            self.entities.append(ent)
            entity_map[pos] = ent
        self.entity_map = entity_map

    def _spawn_stealth_entities(self) -> None:
        roster = ["orc", "orc", "wingedrat", "wingedrat", "lizard", "lizard", "human", "halfling"]
        safe_cells = self._spawn_protection_cells()
        start = (1, 1)
        target = (self.grid_size - 2, self.grid_size - 2)

        candidates: List[Tuple[int, int]] = []
        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                pos = (x, y)
                if pos == start or pos == target or pos in safe_cells:
                    continue
                cell = self.base_env.maze.get_cell_item(x, y)
                if cell is None:
                    candidates.append(pos)

        self.rng.shuffle(candidates)
        self.entities = []
        for species, pos in zip(roster, candidates):
            ent = self._create_entity(species, pos)
            self.entities.append(ent)

        self._rebuild_entity_map()

    def _maze_cycle_rank(self, target: Tuple[int, int]) -> int:
        """Return cycle rank (E - V + C) of the current passable-cell graph."""
        passable: set[Tuple[int, int]] = set()
        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                if self._is_passable_for_path(x, y, target):
                    passable.add((x, y))

        if not passable:
            return 0

        edges = 0
        for x, y in passable:
            if (x + 1, y) in passable:
                edges += 1
            if (x, y + 1) in passable:
                edges += 1

        visited: set[Tuple[int, int]] = set()
        components = 0
        for node in passable:
            if node in visited:
                continue
            components += 1
            q: deque[Tuple[int, int]] = deque([node])
            visited.add(node)
            while q:
                cx, cy = q.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (cx + dx, cy + dy)
                    if nb in passable and nb not in visited:
                        visited.add(nb)
                        q.append(nb)

        return max(0, edges - len(passable) + components)

    def _carve_braid_loops(self, target_loops: int = 5) -> Tuple[int, int]:
        """Open interior walls until the maze gains the requested number of graph cycles."""
        start = (1, 1)
        target = (self.grid_size - 2, self.grid_size - 2)
        desired_loops = max(0, int(target_loops))
        wall_candidates: List[Tuple[int, int]] = []

        if desired_loops <= 0:
            return 0, 0

        # Only consider non-boundary interior walls to preserve physical borders.
        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                pos = (x, y)
                if pos == start or pos == target:
                    continue
                cell = self.base_env.maze.get_cell_item(x, y)
                if cell is not None and str(getattr(cell, "type", "")) == "wall":
                    wall_candidates.append(pos)

        if not wall_candidates:
            return 0, 0

        candidate_order = list(wall_candidates)
        self.rng.shuffle(candidate_order)

        opened_walls = 0
        loops_added = 0
        current_cycle_rank = self._maze_cycle_rank(target)

        # Pass 1: prefer candidates that add exactly one loop, improving map readability.
        for x, y in candidate_order:
            if loops_added >= desired_loops:
                break
            wall_obj = self.base_env.maze.get_cell_item(x, y)
            if wall_obj is None or str(getattr(wall_obj, "type", "")) != "wall":
                continue

            self.base_env.maze.add_cell_item(x, y, None)
            new_cycle_rank = self._maze_cycle_rank(target)
            cycle_gain = new_cycle_rank - current_cycle_rank
            if cycle_gain == 1:
                opened_walls += 1
                loops_added += 1
                current_cycle_rank = new_cycle_rank
            else:
                self.base_env.maze.add_cell_item(x, y, wall_obj)

        # Pass 2 fallback: allow multi-loop gains if still below target.
        if loops_added < desired_loops:
            for x, y in candidate_order:
                if loops_added >= desired_loops:
                    break
                wall_obj = self.base_env.maze.get_cell_item(x, y)
                if wall_obj is None or str(getattr(wall_obj, "type", "")) != "wall":
                    continue

                remaining = desired_loops - loops_added
                self.base_env.maze.add_cell_item(x, y, None)
                new_cycle_rank = self._maze_cycle_rank(target)
                cycle_gain = new_cycle_rank - current_cycle_rank
                if cycle_gain > 0 and cycle_gain <= remaining:
                    opened_walls += 1
                    loops_added += int(cycle_gain)
                    current_cycle_rank = new_cycle_rank
                else:
                    self.base_env.maze.add_cell_item(x, y, wall_obj)

        return opened_walls, loops_added

    def _entity_cluster_id(self, species: str) -> int:
        cluster_id = self.sensor_adapter.species_cluster_id(species)
        return int(np.clip(cluster_id, 0, max(0, self.sensor_adapter.cluster_count - 1)))

    def _entity_bribe_cost(self, species: str) -> float:
        return self.sensor_adapter.species_bribe_cost(
            species,
            min_cost=self.bribe_cost_min_g,
            max_cost=self.bribe_cost_max_g,
        )

    def _entity_faction(self, species: str) -> str:
        cluster_id = self._entity_cluster_id(species)
        is_hostile_cluster = cluster_id in self.hostile_cluster_ids
        is_bribable_cluster = cluster_id in self.bribable_cluster_ids
        if is_hostile_cluster and not is_bribable_cluster:
            return "hostile"
        if is_bribable_cluster and not is_hostile_cluster:
            return "bribable"
        if species in HOSTILE_CLASSES:
            return "hostile"
        if species in BRIBABLE_CLASSES:
            return "bribable"
        return "neutral"

    def _ego_offset_to_world(self, rel_right: int, rel_forward: int) -> Tuple[int, int]:
        direction = int(self.base_env.robot_direction)
        forward_vectors = {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0),
        }
        right_vectors = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1),
        }
        fx, fy = forward_vectors[direction]
        rx_vec, ry_vec = right_vectors[direction]
        robot_x, robot_y = map(int, self.base_env.robot_position.tolist())
        world_x = robot_x + rx_vec * rel_right + fx * rel_forward
        world_y = robot_y + ry_vec * rel_right + fy * rel_forward
        return int(world_x), int(world_y)

    def _nearest_entity_distance(self, pos: Tuple[int, int], class_set: set[str]) -> float:
        dists: List[int] = []
        for (x, y), ent in self.entity_map.items():
            if ent.species in class_set:
                dists.append(self._manhattan(pos, (x, y)))
        if not dists:
            return float(self.grid_size)
        return float(min(dists))

    def _advance_entity(self, ent: EntityState) -> Tuple[int, int]:
        pos = ent.pos
        if ent.entity_type in {"neutral", "static"}:
            return pos

        if ent.entity_type == "ground" and ent.in_wall and ent.wait_timer == 0:
            ent.wait_timer = 1
            return pos

        if ent.wait_timer > 0:
            if ent.wait_timer < self.stealth_wait_steps_h:
                ent.wait_timer += 1
                return pos

            ent.wait_timer = 0
            ent.direction = (-ent.direction[0], -ent.direction[1])
            if ent.entity_type == "ground":
                ent.in_wall = False
            next_pos = (pos[0] + ent.direction[0], pos[1] + ent.direction[1])
            if self._in_bounds(next_pos):
                return next_pos
            return pos

        dx, dy = ent.direction
        next_pos = (pos[0] + dx, pos[1] + dy)
        if ent.entity_type == "flying":
            if self._in_bounds(next_pos):
                return next_pos
            ent.wait_timer = 1
            return pos

        if self._is_passable_cell(next_pos[0], next_pos[1]):
            return next_pos
        if self._in_bounds(next_pos):
            ent.in_wall = True
            return next_pos

        ent.wait_timer = 1
        return pos

    def _step_entities(self) -> None:
        if not self._is_stealth_scheme():
            return

        occupied = {ent.pos for ent in self.entities if ent.alive}
        new_map: Dict[Tuple[int, int], EntityState] = {}
        for ent in self.entities:
            if not ent.alive:
                continue
            proposed = self._advance_entity(ent)
            if proposed != ent.pos and proposed in occupied:
                proposed = ent.pos
            ent.pos = proposed
            new_map[ent.pos] = ent
        self.entity_map = new_map

    def _apply_kill_zone(self, hero_pos: Tuple[int, int]) -> Tuple[bool, float, str]:
        if not self._is_stealth_scheme():
            return False, 0.0, ""

        reward_delta = 0.0
        killed_by = ""
        terminated = False

        for ent in self.entities:
            if not ent.alive:
                continue
            if self._entity_faction(ent.species) != "hostile":
                continue

            # Simplified kill zone: only exact coordinate overlap triggers death.
            if hero_pos != ent.pos:
                continue

            if ent.species == "wingedrat" and self.has_weapon:
                reward_delta += self.wingedrat_kill_bonus_h
                self._remove_entity_at(ent.pos)
                continue

            terminated = True
            reward_delta += self.kill_zone_penalty_h
            killed_by = ent.species
            self.hostile_collision_count += 1
            break

        return terminated, reward_delta, killed_by

        return terminated, reward_delta, killed_by

    def _is_dead_loop(self) -> bool:
        maxlen = int(self.position_window.maxlen or 0)
        if len(self.position_window) < maxlen:
            return False
        unique = len(set(self.position_window))
        return unique <= 5

    def _is_short_stagnation(self) -> bool:
        maxlen = int(self.short_position_window.maxlen or 0)
        if len(self.short_position_window) < maxlen:
            return False
        # "No significant movement" in recent 5 steps: only 1-2 distinct coordinates.
        unique = len(set(self.short_position_window))
        return unique <= 2

    def _front_flags(self) -> Tuple[float, float]:
        front = self.base_env.get_robot_front_pos()
        front_cell = self.base_env.maze.get_cell_item(int(front[0]), int(front[1]))
        is_wall = float(front_cell is not None and getattr(front_cell, "type", "") == "wall")
        is_target = float(front_cell is not None and getattr(front_cell, "type", "") == "target")
        return is_wall, is_target

    def _build_observation(self) -> np.ndarray:
        channels = 5 if self._is_stealth_scheme() else 4
        obs_dtype = np.float32 if self._is_stealth_scheme() else np.uint8
        obs_matrix = np.zeros((channels, 7, 7), dtype=obs_dtype)

        target = self._pos_tuple(self.base_env.target_position)
        rx, ry = self._pos_tuple(self.base_env.robot_position)
        robot_t = (rx, ry)

        if not self.true_distance_map:
            self.true_distance_map = self._build_true_distance_map(target)
        robot_dist = self.true_distance_map.get(robot_t)

        for iy in range(7):
            for ix in range(7):
                rel_right = ix - 3
                rel_forward = 3 - iy
                mx, my = self._ego_offset_to_world(rel_right=rel_right, rel_forward=rel_forward)

                # Channel 0: terrain wall map with out-of-bounds padded as wall.
                if mx < 0 or my < 0 or mx >= self.grid_size or my >= self.grid_size:
                    obs_matrix[0, iy, ix] = 1
                else:
                    cell = self.base_env.maze.get_cell_item(mx, my)
                    is_wall = cell is not None and str(getattr(cell, "type", "")) == "wall"
                    obs_matrix[0, iy, ix] = 1 if is_wall else 0

                # Channel 1: entity radar layer encoded as Task2 cluster ID [1..6].
                ent = self.entity_map.get((mx, my))
                if ent is not None:
                    obs_matrix[1, iy, ix] = float(self._entity_cluster_id(ent.species) + 1)
                else:
                    obs_matrix[1, iy, ix] = 0

                # Channel 2: A* downhill guidance mask toward target.
                dist = self.true_distance_map.get((mx, my))
                if robot_dist is not None and dist is not None and dist <= robot_dist:
                    obs_matrix[2, iy, ix] = 1
                else:
                    obs_matrix[2, iy, ix] = 0

                # Channel 3: breadcrumb memory from recent positions.
                obs_matrix[3, iy, ix] = 1 if (mx, my) in self.position_window else 0

                if channels > 4:
                    obs_matrix[4, iy, ix] = 1.0 if self.has_weapon else 0.0

        return obs_matrix.astype(obs_dtype, copy=False)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_episode_stats()
        self.potential_scale_multiplier = 1.0
        self.base_env.reset(seed=seed, options=options)
        opened_walls, braid_loops_added = self._carve_braid_loops(target_loops=self.target_braid_loops)
        # Keep front-view features consistent after topology post-processing.
        self.base_env.robot_camera_view = self.base_env.get_robot_camera_view()
        self._spawn_virtual_entities()

        robot = self._pos_tuple(self.base_env.robot_position)
        target = self._pos_tuple(self.base_env.target_position)
        self.true_distance_map = self._build_true_distance_map(target)
        self.prev_target_dist = float(self._manhattan(robot, target))
        self.prev_true_target_dist = self._true_distance_to_target(robot, target)
        self.prev_bribable_dist = self._nearest_entity_distance(robot, BRIBABLE_CLASSES)

        self.trajectory.append(robot)
        self.position_window.append(robot)
        self.short_position_window.append(robot)

        return self._build_observation(), {
            "reward_scheme": self.reward_scheme,
            "entity_count": len(self.entity_map),
            "has_weapon": float(1.0 if self.has_weapon else 0.0),
            "braid_opened_walls": float(opened_walls),
            "braid_loops_added": float(braid_loops_added),
            "braid_loop_target": float(self.target_braid_loops),
        }

    def step(self, action: int):
        prev_pos = self._pos_tuple(self.base_env.robot_position)
        target_pos = self._pos_tuple(self.base_env.target_position)

        _, base_reward, terminated, truncated, _ = self.base_env.step(action)
        reached_goal = bool(terminated)
        curr_pos = self._pos_tuple(self.base_env.robot_position)

        total_reward = float(base_reward)
        hit_wall = bool(int(action) == int(Actions.move_forwards) and curr_pos == prev_pos and float(base_reward) == 0.0)
        if hit_wall:
            self.wall_collision_count += 1
            if self._uses_loop_penalties():
                total_reward += self.wall_hit_penalty_c

        killed_by = ""
        if self._is_stealth_scheme() and not reached_goal:
            killed, kill_bonus, killed_by = self._apply_kill_zone(curr_pos)
            total_reward += kill_bonus
            if killed:
                terminated = True

        ent_at_pos = self.entity_map.get(curr_pos)
        if ent_at_pos is not None and not terminated:
            species = ent_at_pos.species
            faction = self._entity_faction(species)
            if self._is_stealth_scheme():
                if faction == "bribable":
                    self.bribable_contact_count += 1
                    bribe_cost = self._entity_bribe_cost(species)
                    self.bribe_cost_total += bribe_cost
                    total_reward -= bribe_cost
                    if not self.has_weapon:
                        total_reward += self.bribable_contact_bonus
                    self.has_weapon = True
                    self._remove_entity_at(curr_pos)
                elif faction == "hostile" and species == "wingedrat" and self.has_weapon:
                    total_reward += self.wingedrat_kill_bonus_h
                    self._remove_entity_at(curr_pos)
            else:
                if faction == "hostile":
                    self.hostile_collision_count += 1
                    if self.reward_scheme == "B":
                        total_reward += self.hostile_collision_penalty_b
                    elif self._is_final_reward_scheme():
                        total_reward += self.hostile_collision_penalty_g
                    self._remove_entity_at(curr_pos)
                elif faction == "bribable":
                    self.bribable_contact_count += 1
                    if self._is_final_reward_scheme():
                        bribe_cost = self._entity_bribe_cost(species)
                        self.bribe_cost_total += bribe_cost
                        total_reward -= bribe_cost
                    else:
                        total_reward += self.bribable_contact_bonus
                    self._remove_entity_at(curr_pos)

        if self._is_stealth_scheme() and not terminated and not reached_goal:
            self.global_step_counter += 1
            # Time dilation: entities only move every other step (half speed).
            if self.global_step_counter % 2 == 0:
                self._step_entities()
            # Collision check runs every tick regardless, since hero may walk into a static entity.
            killed, kill_bonus, killed_by = self._apply_kill_zone(curr_pos)
            total_reward += kill_bonus
            if killed:
                terminated = True

        curr_target_dist = float(self._manhattan(curr_pos, target_pos))
        curr_true_target_dist = self._true_distance_to_target(curr_pos, target_pos)
        curr_bribable_dist = self._nearest_entity_distance(curr_pos, BRIBABLE_CLASSES)

        if self.reward_scheme == "B":
            total_reward += self.distance_scale_b * (self.prev_target_dist - curr_target_dist)
            total_reward += self.approach_bribable_scale_b * (self.prev_bribable_dist - curr_bribable_dist)
        elif self.reward_scheme == "C":
            # Potential-based shaping: closer to goal gets positive reward, farther gets negative reward.
            total_reward += (self.potential_scale_c * self.potential_scale_multiplier) * (
                self.prev_target_dist - curr_target_dist
            )
        elif self._uses_path_shaping():
            # True-path shaping: reward progress on shortest valid path (wall-aware), not Manhattan shortcut.
            # Scheme F additionally surfaces A* next-step direction in the observation vector.
            total_reward += self.path_scale_e * (self.prev_true_target_dist - curr_true_target_dist)
            total_reward += self.step_penalty_e

        self.prev_target_dist = curr_target_dist
        self.prev_true_target_dist = curr_true_target_dist
        self.prev_bribable_dist = curr_bribable_dist

        if reached_goal:
            total_reward += self.goal_reward

        self.step_count += 1
        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        self.trajectory.append(curr_pos)
        self.position_window.append(curr_pos)
        self.short_position_window.append(curr_pos)

        if self._uses_loop_penalties() and self._is_short_stagnation():
            self.stagnation_events += 1
            total_reward += self.stagnation_penalty_c

        dead_loop = self._is_dead_loop()
        if dead_loop:
            self.dead_loop_events += 1
            if self._uses_loop_penalties():
                total_reward += self.dead_loop_penalty_c

        self.episode_return += total_reward
        done = bool(terminated or truncated)

        info: Dict[str, Any] = {
            "is_success": float(1.0 if reached_goal else 0.0),
            "dead_loop_events": float(self.dead_loop_events),
            "stagnation_events": float(self.stagnation_events),
            "wall_collision_count": float(self.wall_collision_count),
            "bribe_cost_total": float(self.bribe_cost_total),
            "true_target_distance": float(curr_true_target_dist),
            "hit_wall": float(1.0 if hit_wall else 0.0),
            "episode_step": float(self.step_count),
        }
        if done:
            info.update(
                {
                    "episode_return": float(self.episode_return),
                    "episode_length": float(self.step_count),
                    "hostile_collision_count": float(self.hostile_collision_count),
                    "bribable_contact_count": float(self.bribable_contact_count),
                    "is_killed": float(1.0 if terminated and not reached_goal else 0.0),
                    "has_weapon": float(1.0 if self.has_weapon else 0.0),
                    "killed_by": str(killed_by),
                    "trajectory": "|".join([f"{x}_{y}" for x, y in self.trajectory[:512]]),
                }
            )

        return self._build_observation(), float(total_reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
