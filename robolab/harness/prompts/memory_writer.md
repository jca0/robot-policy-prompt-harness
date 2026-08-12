A robot arm just completed a subtask. You are given two camera images: the FIRST shows the scene before, and the SECOND shows the scene after.

## Completed subtask
"{subtask}"

## Instructions
Describe only what changed between the two images, focusing on objects relevant to the task. Be short and concrete.

Respond with JSON only, no markdown:
{{"changes": "brief description of what changed (e.g. red cube moved from table to inside bowl)"}}