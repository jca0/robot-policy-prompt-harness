# Edit this file to define which tasks to run.
# Run with: python harness_scripts/run_eval.py
#
# Each entry needs:
#   env         - registered task name (determines the sim scene)
#   instruction - what to tell the policy (overrides the task's built-in instruction)
#
# See docs/scene_previews.md for a full list of available env names and their scenes.

TASKS = [
    {
        "env": "BBQSauceInBinTask",
        "instruction": "Put all the sauce bottles in the gray bin",
    },
    {
        "env": "BlocksInBinTask",
        "instruction": "Clean up all the smaller toys and leave the birdhouse on the table",
    },
    {
        "env": "MarkerInMugTask",
        "instruction": "Clear the space in front of the keyboard",
    },
]
