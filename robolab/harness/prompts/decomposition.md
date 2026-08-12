You are given an image of the current scene and a high-level task. Use the image to identify the specific objects involved, then break the task into ordered subtasks.

Task: "{instruction}"

Rules:
- Use the image to refer to objects by their distinguishing attributes (e.g. color, size, label) so each subtask is unambiguous. For example, prefer "pick up the red cube" over "pick up the cube" when multiple cubes are visible.
- Each subtask should be a short command for one continuous action (e.g. "pick up the red cube", "put the blue cube in the bowl").
- Do NOT use motor primitives like "move arm left" or "open gripper".
- Each subtask must have a visually verifiable end state (e.g. an object is in a new location).
- Use as few subtasks as strictly necessary, even just 1 if the task is already simple enough.

Respond with JSON only, no markdown:
{{"subtasks": ["subtask 1", "subtask 2", ...]}}