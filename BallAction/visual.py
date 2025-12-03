import cv2

class StatHud:
    def __init__(self, header: str, font, font_scale: float, thickness: int, padding: int):
        self.header = header

        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        self.padding = padding

        self.team_spacing = 4 * padding

    def horizontal(self, frame, texts, origin=(0, 100), calculate_only=False):
        curr_height = self.padding
        width = 0

        for text in texts:
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]

            if not calculate_only:
                cv2.putText(frame, text, (origin[0] + self.padding, origin[1] + text_size[1] + curr_height),
                            self.font, self.font_scale, (255, 255, 255), self.thickness, cv2.LINE_AA)
            
            curr_height += text_size[1] + self.padding
            width = max(width, text_size[0])

        if calculate_only:
            return width, curr_height
        else:
            return frame
    
    def to_teamtext(self, label: dict):
        team_text = ""

        total_actions = {'p': 0, 'p_c': 0}
        for player, actions in label.items():
            if player in ['team1', 'team2']:
                total_actions = actions
            else:
                team_text += f"#{player} - "
                team_text += ", ".join([f"{action}:{count}" for action, count in actions.items()])
                team_text += "\n"

        team_text += "\nTotal stats:\n"
        team_text += ", ".join([f"{action}({count})" for action, count in total_actions.items()])
        
        return team_text

    def visualize_stat(self, label: dict, frame):
        # Calculate rectangle size first
        header_lines = self.header.split('\n')
        team1_text = "Team 1 Actions:\n" + self.to_teamtext(label["team1"])
        team2_text = "Team 2 Actions:\n" + self.to_teamtext(label["team2"])
        team1_lines = team1_text.strip().split('\n')
        team2_lines = team2_text.strip().split('\n')

        # Get sizes for rectangle
        header_width, header_height = self.horizontal(frame.copy(),
                                                      header_lines,
                                                      calculate_only=True)
        team1_width, team1_height = self.horizontal(frame.copy(),
                                                    team1_lines,
                                                    calculate_only=True)
        team2_width, team2_height = self.horizontal(frame.copy(),
                                                    team2_lines,
                                                    calculate_only=True)

        rect_width = max(header_width, team1_width + team2_width + self.team_spacing) + self.padding
        rect_height = header_height + max(team1_height, team2_height) + self.padding

        origin = (0, (frame.shape[0] - rect_height) // 2)

        # Draw rectangle first
        overlay = frame.copy()
        overlay = cv2.rectangle(overlay, origin, (origin[0] + rect_width, origin[1] + rect_height), (0, 0, 0), -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw text on top of rectangle
        frame = self.horizontal(frame, header_lines, origin=origin)

        frame = cv2.line(frame, (0, origin[1] + header_height), (rect_width, origin[1] + header_height), (255, 255, 255), 1)

        frame = self.horizontal(frame, team1_lines, origin=(origin[0], origin[1] + header_height))
        
        frame = cv2.line(frame, (origin[0] + team1_width + self.team_spacing, origin[1] + header_height),
                         (origin[0] + team1_width + self.team_spacing, origin[1] + rect_height), (255, 255, 255), 1)
        
        frame = self.horizontal(frame, team2_lines, origin=(origin[0] + team1_width + self.team_spacing, origin[1] + header_height))

        frame = cv2.rectangle(frame, origin, (origin[0] + rect_width, origin[1] + rect_height), (255, 255, 255), 1)

        return frame