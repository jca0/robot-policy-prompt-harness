Given the overall goal, the current scene image, and what has been completed so far, decide the single next subtask the robot should perform.

## Memory
{memory}

## Rules
- Use the image to identify the current scene state and refer to objects by distinguishing attributes (e.g. color, size, label).
- Choose ONE concrete next subtask that makes progress toward the goal given the current scene — not what was originally planned, but what actually makes sense right now.
- Each subtask should be a short command for one continuous action (e.g. "put the blue cube in the bowl").
- Do NOT use motor primitives like "move arm left" or "open gripper".
- The subtask must have a visually verifiable end state.
- If the goal is already fully achieved in the current scene, set "done" to true.
- If a subtask has previously timed out (see "Failed Subtasks" in memory), prefer a DIFFERENT subtask that still makes progress toward the goal. Only retry a failed subtask if no alternative exists or if it is a prerequisite for all remaining work.
- When retrying a failed subtask, rephrase it to suggest a different approach (e.g. "pick up the dinosaur toy from the side" instead of repeating the same wording).
- Consider ordering carefully: if an object is obstructed or hard to reach, move blocking objects out of the way first. Prefer easier-to-grasp objects before harder ones when the goal allows flexible ordering.

Respond with JSON only, no markdown:
{{"subtask": "next subtask description or null if done", "done": true/false, "reasoning": "brief explanation of why this subtask was chosen"}}