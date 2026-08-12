Generate {n} different ways to break this robot arm task into ordered subtasks:
"{instruction}"

Each decomposition should vary in phrasing, vocabulary, and/or action boundaries, but accomplish the same task.

Rules:
- Each subtask: one continuous action, short natural language (e.g. "pick up the cube", "place the cube in the bowl").
- No motor primitives ("move arm left", "open gripper").
- Each subtask must have a visually obvious end state checkable from a camera image.
- Use as few subtasks as necessary — even just 1.

Respond with JSON only, no markdown:
{{"decompositions": [["subtask 1a", "subtask 2a"], ["subtask 1b", "subtask 2b", "subtask 3b"], ...]}}