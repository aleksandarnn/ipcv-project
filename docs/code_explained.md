# The code, explained from scratch

Written for someone who is still learning Python. Every file in `game/` is
covered, line by line where it matters.

Read it in this order: the Python refresher, then the story of one frame,
then the four files. The last section is the questions you should be able to
answer without notes.

---

## Part 0: the Python you need first

**Indentation is not decoration.** The spaces at the start of a line are how
Python knows what belongs to what. Lines indented under an `if` run only when
that `if` is true.

**A variable** is a name for a value. `x = 5` means put 5 into the name `x`.
A single `=` is "put into", not the equals of mathematics.

**A list** is several things in order, written `[a, b, c]`. You add to the end
with `.append(something)`.

**A tuple** is the same but cannot be changed afterwards, written `(a, b, c)`.

**A dictionary** is a lookup table from names to values, written
`{"face": 0, "pose": 0}`. You read one entry with `failures["face"]`.

**A class** is a blank form describing what something has. **An object** is
one filled-in copy of that form. A **method** is an action belonging to an
object, called with a dot: `thing.action()`.

**`self`**, inside a method, means "this particular object". If two objects
exist, `self` is how each refers to its own data instead of the other's.

**`try` / `except`** means attempt something, and if it explodes, do this
instead of crashing.

**`import`** borrows code from another file.

---

## Part 1: the story of one frame

Picture a small office. A photo lands on the table with a blank form next to
it. Four workers each look at the photo, find the one thing they care about,
and write it on the form. A painter then paints everything on the form onto
the photo. Someone with a stopwatch notes how long it took. The form goes in
the bin and a fresh photo arrives.

That happens about thirty times a second, and it is the whole program.

```
webcam  ->  shrink + mirror  ->  blank form (GameState)
              |
              v
   face -> pose -> identity -> control     (each writes on the form)
              |
              v
      renderer paints  ->  screen  ->  stopwatch records
```

---

## Part 2: `game_types.py`, the shapes of things

This file contains almost no behaviour. It only says what our data looks
like, so that all the other files and all five people agree.

### The time budgets

```python
BUDGET_MS = {
    "face": 8.0,
    "pose": 15.0,
    "identity": 3.0,
    "control": 3.0,
    "render": 6.0,
}
```

A dictionary from a task name to how many milliseconds it may take. They add
up to 35, which at 40 milliseconds per frame leaves a little room and gives
about 25 frames per second.

### A keypoint

```python
@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float = 1.0

    @property
    def ok(self) -> bool:
        return self.confidence >= 0.4
```

`@dataclass` is a shortcut that saves writing boring setup code. Without it
you would need about ten more lines to achieve the same thing.

`float` means a decimal number. The `= 1.0` gives confidence a default, so
you can create a keypoint without stating it.

`@property` makes `ok` look like a stored value even though it is calculated
each time. You write `kp.ok`, with no brackets, and it works out whether the
confidence is at least 0.4.

That threshold matters. It is written once, here, so nobody invents their own
idea of when a keypoint is trustworthy.

### A face

```python
@dataclass
class Face:
    cx: float
    cy: float
    w: float
    h: float
    angle: float = 0.0
    confidence: float = 1.0
```

Centre x, centre y, width, height, head tilt in degrees, and confidence. All
in pixels of the frame that was handed to the module.

### A player

```python
@dataclass
class Player:
    player_id: int
    face: Face | None = None
    keypoints: dict[str, Keypoint] = field(default_factory=dict)
    colour: tuple[int, int, int] = (255, 255, 255)
    health: int = 100
    score: int = 0
    frames_missing: int = 0
```

`Face | None` means this holds either a Face or nothing. `None` is Python's
word for "no value at all", and it is how a module says honestly that it
could not find a face, rather than guessing a position.

`field(default_factory=dict)` looks strange and exists for a real reason:
every player needs their **own** empty dictionary of keypoints. Writing
`= {}` would give every player the same shared one, and all of them would
end up with the same keypoints. This is a classic Python trap.

`frames_missing` counts how many frames in a row we have not seen this
person. Zero means they are here right now. It is how the game knows someone
walked out of shot without deleting them and losing their health and score.

Two helpers come with it:

```python
@property
def visible(self) -> bool:
    return self.frames_missing == 0

def point(self, name: str) -> Keypoint | None:
    kp = self.keypoints.get(name)
    return kp if kp is not None and kp.ok else None
```

`point` is the important one. Ask for `player.point("left_wrist")` and you
get either a trustworthy position or nothing. You can never accidentally use
a garbage position that happened to have low confidence. That one method
removes a whole category of bug from everybody's code.

`.get(name)` on a dictionary returns `None` when the name is absent, instead
of crashing the way `dictionary[name]` would.

### The agreed keypoint names

```python
KEYPOINTS = (
    "nose", "left_wrist", "right_wrist", "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
)
```

A tuple, so it cannot be changed by accident. Everyone spells them this way,
so nobody writes `lwrist` while someone else writes `left_wrist`.

### A draw command

```python
@dataclass
class DrawCommand:
    kind: str
    x: float
    y: float
    ...
    layer: int = 1
```

A request to draw something: a circle, a rectangle, a line or some text.
Modules never draw. They write down what they want and the renderer does it.

`layer` decides what ends up on top. 0 is background, 1 is the game, 2 is the
interface. Without it, a health bar could be buried under a skeleton.

### An event

```python
EVENTS = ("attack", "block", "dodge", "damage", "blocked", "missed",
          "ko", "round_start", "player_lost", "player_returned")

@dataclass
class GameEvent:
    kind: str
    player_id: int = -1
    strength: float = 1.0
```

The vocabulary of our duel. Task 4 raises these, task 5 makes them visible.
Spelling matters, because an event nobody recognises is ignored in silence.

`player_id` is who the event is **about**. For `damage` that is the player who
got hit, not the one who swung. Getting this backwards drains the wrong
person's health, and it is the sort of bug that only shows up in front of a
jury.

### The form itself

```python
@dataclass
class GameState:
    frame_index: int = 0
    timestamp: float = 0.0
    players: list[Player] = field(default_factory=list)
    events: list[GameEvent] = field(default_factory=list)
    draw: list[DrawCommand] = field(default_factory=list)
    phase: str = "playing"
    message: str = ""
```

Everything known about one single frame. A fresh one is made every frame, so
nothing carries over by accident. If a module wants to remember something
between frames it must keep it inside itself deliberately.

### The shape of a module

```python
class Module:
    name = "unnamed"

    def process(self, frame, state: GameState) -> None:
        raise NotImplementedError
```

What every task module must look like. `raise NotImplementedError` is a
deliberate landmine: if somebody subclasses this and forgets to write
`process`, they find out immediately instead of wondering why nothing
happens.

---

## Part 3: `main.py`, the loop

### Finding the folder

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
```

`__file__` is this script's own location. `.parent` is the folder holding it.
This line puts that folder at the front of the list of places Python searches
for imports, so `from game_types import ...` works no matter which directory
you were standing in when you ran the program.

### Loading the task modules

```python
for name in TASKS:
    try:
        imported = __import__(f"modules.{name}", fromlist=["Module"])
        instance = imported.Module()
        loaded.append(instance)
    except ModuleNotFoundError:
        missing.append(name)
```

`TASKS` is `("face", "pose", "identity", "control")`, in the order they must
run. The loop tries to import each one by name.

`__import__` is the unusual part. Normally you write `import something` at
the top of a file, with the name fixed. Here the name is only known while the
program runs, so this version takes the name as text.

`imported.Module()` creates one object from the class called `Module` inside
that file. The brackets are what actually builds it.

`ModuleNotFoundError` means the file does not exist yet, so the task is
recorded as missing and skipped. A second `except` catches anything else,
such as a teammate's file that exists but is broken.

The result is that the game runs today with no modules at all, and each task
appears the moment its owner commits a file.

### Opening the camera

```python
cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
```

`source` arrives as text. If it is all digits it is a camera number, so it is
converted with `int()`. Otherwise it is treated as a filename. That one line
is why the same program works with a webcam and with a video file.

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
```

Ask the camera for a small picture rather than shrinking a big one
afterwards. The camera does it before the data ever reaches us, which is
cheaper.

### The loop starts

```python
while True:
    profiler.start_frame()
    grabbed_at = time.perf_counter()
    ok, frame = cap.read()
```

`while True` repeats forever until something says stop.

`time.perf_counter()` is a high-precision clock, meant for measuring
durations. `time.time()` gives the date and time, which is not accurate
enough for milliseconds.

`cap.read()` returns **two** things at once: whether it worked, and the
picture. The line catches both.

```python
if not ok:
    if args.source.isdigit():
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    continue
```

If reading failed and the source was a webcam, the camera is gone, so `break`
leaves the loop. If it was a video file, we have reached the end, so jump
back to frame 0 and `continue`, which means skip the rest of this round and
start the next one.

### Preparing the frame

```python
frame = cv2.resize(frame, (args.width, ...), interpolation=cv2.INTER_AREA)
frame = cv2.flip(frame, 1)
```

The resize is the frame-rate lever. Work per frame is proportional to the
number of pixels, so halving the width quarters the cost. `INTER_AREA` is the
correct method when shrinking, because it averages the pixels being merged
instead of throwing most of them away.

`cv2.flip(frame, 1)` mirrors left to right. Without it, raising your right
hand moves the left side of the screen and the game feels wrong.

### The fresh form

```python
state = GameState(frame_index=frame_index, timestamp=time.time())
```

A brand new empty one, every frame.

### Running the modules

```python
for module in modules:
    if not enabled[module.name]:
        continue
    started = time.perf_counter()
    try:
        module.process(frame, state)
    except Exception as exc:
        failures[module.name] += 1
        if failures[module.name] >= MAX_FAILURES:
            enabled[module.name] = False
    spent = time.perf_counter() - started
    profiler.record(module.name, spent)
```

Four separate ideas in one block.

**The order is fixed.** Pose and identity both need faces to exist first, and
control needs everything before it finished, because it is the one deciding
things.

**Every call is timed**, which is where the report numbers come from.

**Every call is wrapped in `try`**, so a teammate's crash does not kill the
game. `except Exception as exc` catches the error and gives it the name `exc`
so we can print what went wrong.

**Repeated failures switch a module off.** `+= 1` adds one to the count. At
five, `enabled` becomes false and the module is skipped from then on, and the
panel shows it as OFFLINE so the failure is visible rather than silent.

### Drawing and measuring

```python
renderer.react(state)
output = renderer.draw(frame, state)
profiler.record_latency(time.perf_counter() - grabbed_at)
profiler.end_frame()
```

`react` and `draw` are two steps on purpose. Reacting updates how shaken and
flashed the screen should be. Drawing paints. Keeping them apart lets a
reaction fade over several frames instead of existing only on the frame the
hit happened.

The latency line subtracts the time the frame was grabbed from the time it is
finished. That is the delay a player feels.

### The keyboard

```python
key = cv2.waitKey(1) & 0xFF
if key == ord("q"):
    break
```

`waitKey(1)` waits one millisecond for a key and is also what lets the window
redraw. `ord("q")` turns the letter q into its number code, because that is
what `waitKey` returns. `& 0xFF` keeps only the last eight bits, which is a
standard OpenCV habit to make the comparison behave the same on every
platform.

`==` with two equals signs means "is equal to", as opposed to the single `=`
which means "put into".

---

## Part 4: `renderer.py`, the scene

### Reacting to events

```python
if event.kind == "damage":
    self.shake = min(1.0, self.shake + 0.8 * event.strength)
    self.flash = 0.85
    self.flash_colour = (40, 40, 235)
```

`self.shake` and `self.flash` are numbers between 0 and 1 kept **between**
frames, which is why they live on `self` rather than being local variables.

`min(1.0, ...)` stops the shake going above 1 no matter how many hits land at
once.

The colour `(40, 40, 235)` is blue, green, red in that order, because that is
how OpenCV stores colour. This one is red.

The three duel outcomes are deliberately different: a landed hit gives a hard
shake and a red wash, a block gives a white spark with almost no shake, and a
miss gives barely anything.

### Fading

```python
self.shake *= 0.82
self.flash *= 0.85
```

`*=` means multiply what is there and keep the result. Each frame the shake
becomes 82% of what it was, so after ten frames it is about 14%. That is what
makes an impact fade smoothly over roughly half a second.

### The shake

```python
out = np.roll(out, (dy, dx), axis=(0, 1))
```

`np.roll` slides the whole image by a few pixels and wraps the edge around.
It costs almost nothing and sells an impact far better than any overlay.

### The flash

```python
layer = np.full_like(out, self.flash_colour, dtype=np.uint8)
out = cv2.addWeighted(out, 1.0 - 0.35 * self.flash, layer, 0.35 * self.flash, 0)
```

`np.full_like` builds a new image the same size as `out`, filled with one
colour. `addWeighted` blends two images by a fraction each. Blending makes it
read as light on top of the scene, where painting it solid would hide the
picture.

### Drawing in order

```python
for command in sorted(state.draw, key=lambda c: c.layer):
    self._apply(out, command)
```

`sorted` puts the commands in order. `key=lambda c: c.layer` says what to
sort by: `lambda` is a small nameless function, and this one takes a command
`c` and returns its layer number.

This single line is why "nobody draws on the frame" is worth enforcing. One
place decides the order, so nothing can be buried.

### The text outline

```python
for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0),
               (1, 0), (-1, 1), (0, 1), (1, 1)):
    cv2.putText(img, text, (org[0] + dx, org[1] + dy), font, scale,
                (0, 0, 0), thickness, cv2.LINE_AA)
cv2.putText(img, text, org, font, scale, colour, thickness, cv2.LINE_AA)
```

The text is drawn nine times: eight black copies nudged one pixel in each
direction, then the real one on top. The black peeking out around the edges
is the outline, which is what keeps white text readable over a bright
background.

The outline uses the **same** thickness as the main text on purpose. Using a
thicker pen widens every letter slightly, and over a long line that error
adds up until the black copy prints a ghost character past the end. That
happened, it was measured at 18 pixels of drift, and this is the fix.

### The panel

```python
panel = img[0:height, 0:225]
panel[:] = (0.35 * panel).astype(panel.dtype)
```

The first line selects a rectangle of the image. The second multiplies it by
0.35, making that area 35% as bright, which is the dark band behind the
statistics. `panel[:] =` writes back into the original image rather than
making a copy.

---

## Part 5: `profiler.py`, the stopwatch

### Forgetting old frames

```python
self.times[name] = deque(maxlen=window)
```

A `deque` with a maximum length is a list that forgets. Add a 61st item to a
60-long one and the oldest falls off the front automatically.

That gives a rolling average over the last 60 frames, about two seconds. An
average over the whole session would hide a slowdown that started a minute
ago; this one reacts to what is happening now.

### Frames per second

```python
mean = sum(self.frame_times) / len(self.frame_times)
return 1.0 / mean
```

`sum` adds a list up, `len` counts it, so this is the average seconds per
frame. One divided by that is frames per second. A frame taking 0.04 seconds
gives 25 frames per second.

### Two different measurements

**Frame rate** is how many frames you produce per second, which is
throughput. **Latency** is how long one particular frame took from being
grabbed to being displayed, which is delay.

They are not the same, and the player feels the second one.

### The report

```python
for name in sorted(self.times, key=lambda n: -self.ms(n)):
```

Sorts the module names by their time, with a minus sign to get the slowest
first. That is the bottleneck, which the brief asks you to identify.

---

## Part 6: the questions you should be able to answer

**Why do the modules run in that order?** Pose and identity both need faces
first, and control needs everything else up to date before it decides
anything.

**Why do modules return draw commands instead of drawing?** So one place
controls the order and layering, and two modules cannot fight over the same
pixels.

**What happens when a module throws?** The exception is caught, the frame
still renders, the failure is counted, and after five the module is disabled
and shown as offline.

**Why is latency measured separately from frame rate?** Throughput versus
delay. The player feels the delay.

**Why is the frame mirrored?** So moving your right hand moves the right side
of the screen.

**Why is the frame resized?** Work is proportional to pixel count, so halving
the width quarters the cost.

**What does confidence 0.4 mean?** Below it, a keypoint counts as missing
rather than as a position, and `player.point()` returns nothing.

**Why is a fresh GameState made every frame?** So nothing carries over by
accident. Anything a module wants to remember, it must keep deliberately.
