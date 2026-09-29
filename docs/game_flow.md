## End-to-end task flow

Camera frame
  ↓
Task 3 — Person detection, tracking, and persistent player IDs
  Output: { player_id, player bounding box / crop }
  ↓
Task 2 — Pose estimation and action-command output  ← my module
  Output: { player_id, action event/state, confidence }
  ↓
Tasks 4 and 5 — Execute game commands and render visuals
  Output: damage, dodge/block outcome, waves, shield, animations, UI
 &
 Task 1 - receives the head boundary?

## Movement controls

| Movement | Reliable pose evidence | Game command |
|---|---|---|
| Right-arm left → right swing | Right wrist moves rapidly right; elbow/shoulder follow | Horizontal wave right |
| Right-arm right → left swing | Right wrist moves rapidly left; elbow/shoulder follow | Horizontal wave left |
| Right-arm vertical swing | Right wrist moves rapidly up/down | Vertical wave |
| Arms crossed | Both wrists lie near the opposite shoulder/forearm | Block vertical wave |
| Duck / crouch | Hips and shoulders move down; knees bend | Dodge horizontal wave |
| Jump | Hips and ankles move up together | Dodge leg attack |
| Finger rune | A stable, recognised hand gesture | Activate shield |

Use a short motion window, confidence threshold, and cooldown for each action. A low-confidence or missing keypoint must not trigger an action.

## Pipeline


Input from Task 3: tracked player crop + player_id
  → Body-pose estimation inside the supplied crop
  → Hand/finger landmark estimation near each wrist
  → Confidence checks and temporal smoothing
  → Motion features: wrist velocity, torso position, knee angle, arm crossing
  → Action rules and cooldown state machine
  → Output to Tasks 4–5: action command / stable action state + same player_id


## Keypoints

- Body: shoulders, elbows, wrists, hips, knees, ankles, head.
- Hands: 21 finger landmarks per visible hand for the shield rune.
