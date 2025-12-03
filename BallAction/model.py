import json
import cv2
from collections import defaultdict, Counter, deque
from .visual import StatHud

pass_actions = ["PASS", "HIGH PASS", "CROSS", "THROW IN", "HEADER", "FREE KICK"]

class BallActionModel():
    def __init__(self, ballaction_path: str, ballaction_conf: float, fps: int = 25):
        with open(ballaction_path, 'r') as f:
            results = json.load(f)['predictions']

        self.result_filtered = {
            int(v['position'] / 1000 * 25): (v['label'], v['confidence'])
            for v in results if v['confidence'] >= ballaction_conf
        }
        self.label_conf = ("No Action", 1.0)
        self.label_dict = {
            "team1": defaultdict(lambda: {p: 0 for p in ['p', 'p_c']}),
            "team2": defaultdict(lambda: {p: 0 for p in ['p', 'p_c']})
        }
        self.prev_idx = -1
        self.prev_action = None
        self.prev_action_team = None
        self.prev_action_player = None

        self.prev_possession_team = deque(maxlen=int(fps//2))
        self.prev_possession_player = deque(maxlen=int(fps//2))

    def update_action(self, action):
        def weighted_count(items: deque):
            items_filtered = defaultdict(int)
            weight = [i / sum(range(1, len(items) + 1)) for i in range(1, len(items) + 1)]

            for w, v in zip(weight, items):
                if v is not None:
                    items_filtered[v] += w

            if len(items_filtered) == 0:
                return None

            return max(items_filtered, key=items_filtered.get)
        
        possession_team = weighted_count(self.prev_possession_team)
        possession_player = weighted_count(self.prev_possession_player)

        if possession_team:
            if action in pass_actions:
                self.label_dict[possession_team][possession_team]['p'] += 1

                if possession_player:
                    self.label_dict[possession_team][possession_player]['p'] += 1

            if self.prev_action in pass_actions and possession_team == self.prev_action_team:
                self.label_dict[possession_team][self.prev_action_team]['p_c'] += 1

                if self.prev_action_player:
                    self.label_dict[possession_team][self.prev_action_player]['p_c'] += 1

        self.prev_action = action
        self.prev_action_team = possession_team
        self.prev_action_player = possession_player

    def visualize_frame(self, frame, frame_idx, posession_team, possession_player):
        if frame_idx in self.result_filtered:
            self.label_conf = tuple(self.result_filtered[frame_idx])
            self.prev_idx = frame_idx
            self.update_action(self.label_conf[0])
        
        self.prev_possession_player.append(possession_player)
        self.prev_possession_team.append(posession_team)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 1
        padding = 5

        header = f"Previous action (at {int(self.prev_idx//25)}s): {self.label_conf[0]} ({self.label_conf[1]:.2f})\n"
        header += "(p: pass, p_c: pass completed)\n"
        # header += f"team_possesion: " + ", ".join([i if i is not None else "" for i in self.prev_possession_team]) + "\n"
        # header += f"player_possesion: " + ", ".join([i if i is not None else "" for i in self.prev_possession_player]) + "\n"

        stat_hud = StatHud(header, font, font_scale, thickness, padding)
        frame = stat_hud.visualize_stat(self.label_dict, frame)

        return frame