# notes on this repo

## tasks

unstack rubiks cubes
Clean up all the smaller toys and leave the birdhouse on the table
Put the recyclable cartons in the grey bin
Stack the blocks in the order from bottom to top: red, blue, green, yellow
Put three (3) fruits on the plate
Put all white objects and yellow objects in the grey bin

## commands

```bash
uv run python examples/policy/run_dynamic_prompting.py --task UnstackRubiksCubeTask

# run all tasks
uv run python examples/policy/run_eval.py --policy pi05 --headless

# run specific task(s)
uv run python examples/policy/run_eval.py --policy pi05 --headless --task BananaInBowlTask
```

## scenes and tasks

| Scene | Task | Instruction |
| ------- | ------ | ------------- |
| `bagel_plate_banana_bowl.usda` | BagelsOnPlateTask | Put the bagels on the plate |
| `bagel_plate_banana_bowl.usda` | BananaOnPlateTask | Pick up the banana and put it on the plate |
| `banana_bowl.usda` | BananaInBowlTask | Pick up the banana and place it in the bowl |
| `bananas_5_grey_bin.usda` | BananasInBinOneMoreTask | Put one (1) more bananas in the grey bin. |
| `bananas_5_grey_bin.usda` | BananasInBinThreeTotalTask | Make sure there are 3 (three) bananas in the grey bin. |
| `bananas_5_grey_bin.usda` | BananasOutOfBinTask | Take the bananas out |
| `bananas_5_in_crate.usda` | BananasInCrateTask | Put 2 bananas in the crate |
| `bin_condiments.usda` | BBQSauceInBinTask | Put the red BBQ sauce bottles in the grey bin |
| `bin_condiments.usda` | CannedFoodInBinTask | Put the canned food in the grey bin |
| `bin_condiments.usda` | CoffeePotInBinTask | Put the coffee pot in the grey bin |
| `bin_condiments.usda` | CondimentsInBinTask | Sort the sauce condiments into the grey bin |
| `bin_mug_mustard_marker_bowl.usda` | BowlInBinTask | put the bowl in the grey bin |
| `blue.usda` | PickUpBluePitcherTask | Pick up the large blue pitcher |
| `bottles_crate.usda` | SauceBottlesCrateTask | Put the red bbq sauce bottle in the crate |
| `bowls_2_table.usda` | BowlStackingLeftOnRightTask | Stack the left bowl on the right bowl |
| `bowls_2_table.usda` | BowlStackingRightOnLeftTask | Stack the right bowl on the left bowl |
| `breakfast_table.usda` | GrabABagelTask | Grab a bagel |
| `breakfast_table.usda` | GrabAFruitTask | Pick up a fruit |
| `breakfast_table.usda` | MoveBananaToBagelPlateTask | Move the bananas to the bagel plate |
| `breakfast_table.usda` | UtensilsInMugTask | Put the fork and spoon in the ceramicmug |
| `breakfast_table.usda` | YogurtInBowlTask | Put the small red yogurt in the red bowl |
| `butter_raisin_box.usda` | ButterAboveRaisinTask | Pick up the butter box and place it on top of the raisin box |
| `butter_raisin_box_grey_bin.usda` | LargerObjectRaisinBoxInBinTask | Place the larger object in the grey bin. |
| `butter_raisin_box_grey_bin.usda` | SmallerObjectButterInBinTask | Place the smaller object in the grey bin. |
| `cartons_in_crate.usda` | RecycleCartonTask | Put the recyclable cartons in the grey bin |
| `cartons_in_vertical_crate.usda` | RecycleCartonsVerticalCrateTask | Put the cartons that can be recycled in the vertical crate |
| `cartons_on_box.usda` | RecycleCartonsOnBoxTask | Put the cartons that can be recycled on the box |
| `clutter_fruit_bottle_bluebin.usda` | BigPumpkinInBinTask | Put the bigger pumpkin in the bin |
| `clutter_fruit_bottle_bluebin.usda` | ClearOrganicObjectsTask | Clear away the organic objects |
| `clutter_fruit_bottle_bluebin.usda` | ClutterPlasticTask | Put all plastic bottles away in the bin |
| `clutter_fruit_bottle_bluebin.usda` | ClutterPumpkinTask | Put all the pumpkins away in the bin |
| `clutter_fruit_bottle_bluebin.usda` | SmallPumpkinInBinTask | Put the small pumpkin in the bin |
| `colored_blocks.usda` | BlockStackingOrderAgnosticTask | Stack the blocks into a tower |
| `colored_blocks.usda` | BlockStackingSpecifiedOrderTask | Stack the blocks in the order from bottom to top: red, blue, green, yellow |
| `cooking_table.usda` | CookingClearPlateTask | Put the two measuring cups outside of the plate |
| `cooking_table.usda` | CookingPickPastaToolTask | Move the pink tool from this utensil container to the other utensil holder |
| `cooking_table.usda` | PickOrangeObjectTask | Pick up the orange measuring cup |
| `food_packing.usda` | FoodPackingByColorTask | Pack yellow objects in right container and blue object in the left container |
| `foodpacking_1bin_1box_1can.usda` | FoodPacking1BoxesTask | Pack boxed foods into the bin |
| `foodpacking_1bin_1box_1can.usda` | FoodPacking1CansTask | Pack canned foods into the bin |
| `foodpacking_1bin_2box_2can.usda` | FoodPacking2BoxesTask | Pack boxed foods into the bin |
| `foodpacking_1bin_2box_2can.usda` | FoodPacking2CansTask | Pack canned foods into the bin |
| `foodpacking_1bin_3box_3can.usda` | FoodPacking3BoxesTask | Pack boxed foods into the bin |
| `foodpacking_1bin_3box_3can.usda` | FoodPacking3CansTask | Pack canned foods into the bin |
| `fruits_in_basket.usda` | FruitsMovingOrangeOrLimeTask | Move an orange or a lime to the wood bowl |
| `fruits_in_basket.usda` | FruitsMovingTask | Move an orange to the white bowl |
| `fruits_in_basket.usda` | FruitsOnionTask | Put the onion in the wood bowl |
| `fruits_in_basket.usda` | FruitsOnionToPlateTask | Put the onion on the plate |
| `fruits_in_basket.usda` | WoodSpatulaToBowlTask | Put the wooden spatula in the bowl |
| `fruits_out_of_basket.usda` | FruitsGreenLimesOnPlateTask | Put all the green fruit on the plate |
| `fruits_out_of_basket.usda` | FruitsOnPlate3Task | Put three (3) fruits on the plate |
| `fruits_out_of_basket.usda` | FruitsOnPlateTask | Put all the fruits on the plate |
| `fruits_out_of_basket.usda` | FruitsOrangesOnPlateTask | Put all the oranges on the plate |
| `green.usda` | PickUpGreenObjectTask | Pick up the green vegetable block |
| `ladle_pot.usda` | GreenSpoonsInPotTask | Put the green spoons in the pot |
| `ladle_pot.usda` | PinkSpoonInPotTask | Put the pink spaghetti spoon in the pot |
| `ladle_pot.usda` | SpoonsInPotTask | Put all of the serving spoons with no holes in the pot |
| `mug_banana_ketchup_bowl_rubiks3_bin.usda` | UnstackRubiksCubeTask | Unstack the rubiks cube tower |
| `mug_banana_ketchup_bowl_rubiks3_bin.usda` | YellowAndWhiteObjectsInBinTask | Put all white objects and yellow objects in the grey bin |
| `mugs2_bananas2_ketchup_rubiks3_bin.usda` | DishesInBinTask | Put the dishware in the grey bin |
| `mugs2_bananas2_ketchup_rubiks3_bin.usda` | WhiteMugsInBinTask | Clean up the white mugs |
| `mugs4_measuringcup_drill_bowl.usda` | PickDrillTask | Pick up the cordless drill. |
| `mugs4_measuringcup_drill_bowl.usda` | ReorientAllMugsTask | Reorient all the mugs upright so that the opening is facing upwards. |
| `mugs4_measuringcup_drill_bowl.usda` | ReorientRedMugTask | Put the red mug upright so that the opening is facing upwards. |
| `mugs4_measuringcup_drill_bowl.usda` | StackWhiteMugsTask | Stack the white mugs on top of each other. |
| `mugs4_measuringcup_drill_bowl_v2.usda` | ReorientWhiteMugsTask | Make sure all the white mugs are upright so that the opening is facing upwards. |
| `mugs4_measuringcup_drill_bowl_v2.usda` | TakeMeasuringSpoonOutTask | Take the white colored measuring spoon out of the red bowl and put it on the table. |
| `mugs_on_shelf.usda` | TakeMugsOffOfShelfTask | Take the mugs off the shelf |
| `mustard_raisin_box.usda` | MustardAboveRaisinTask | Place the mustard on the raisin box. |
| `objects_around_table.usda` | WhiteMugInCenterOfTableTask | Put the white mug in the center of the table. |
| `rubiks_cube_3.usda` | Stack3RubiksCubeTask | Stack the rubiks cubes in a tower |
| `rubiks_cube_banana_bowl.usda` | BananaThenRubiksCubeTask | put the banana then the cube in the bowl |
| `rubiks_cube_banana_bowl.usda` | RubiksCubeAndBananaTask | Put the cube and the banana in the bowl |
| `rubiks_cube_banana_bowl.usda` | RubiksCubeBehindBowlTask | Put the rubiks cube behind the bowl |
| `rubiks_cube_banana_bowl.usda` | RubiksCubeInFrontOfBowlTask | Put the rubiks cube in front of the bowl |
| `rubiks_cube_banana_bowl.usda` | RubiksCubeLeftOfBowlTask | Put the rubiks cube to the left of the bowl |
| `rubiks_cube_banana_bowl.usda` | RubiksCubeOrBananaTask | Put the cube or the banana in the bowl |
| `rubiks_cube_banana_bowl.usda` | RubiksCubeThenBananaTask | Put the cube then the banana in the bowl |
| `rubiks_cube_banana_bowl_mug_bin.usda` | RedDishesInBinTask | Put the red dishware in the grey bin |
| `rubiks_cube_banana_bowl_mug_bin.usda` | RedItemsInBinTask | Put all the red things in the grey bin |
| `rubiks_cube_banana_bowl_mug_bin.usda` | RubiksCubeRightOfBowlTask | Put the rubiks cube to the right of the bowl |
| `rubiks_cube_bowl.usda` | RubiksCubeTask | Put the cube in the bowl |
| `shelf_mugs_jug_bowl.usda` | PutBowlOnShelfTopTask | Put the serving bowl anywhere on the shelf in front of you |
| `shelf_mugs_jug_bowl.usda` | PutMugsOnShelfTask | Put the two mugs on the shelf |
| `shelf_with_cleaning_products.usda` | JugsOnShelfTask | Put all the jugs on the shelf |
| `shelf_with_cleaning_products.usda` | OneBottleInSquarePailTask | Put any white plastic bottle in the square pail |
| `shelf_with_cleaning_products.usda` | OneBottleOnShelfTask | Put any white plastic bottle on the shelf |
| `shelf_with_cleaning_products.usda` | PlasticBottlesInSquarePailTask | Put all the small plastic bottles in the square pail |
| `shelf_with_cleaning_products.usda` | ReorientJugTask | Stand the jug upright |
| `tools_container.usda` | ClampInRightBinTask | Put the spring clamp in the right bin |
| `tools_container.usda` | HammersInLeftBinTask | Put the red hammer and black hammer in the left bin |
| `tools_container.usda` | NonHammerToolsInRightBinTask | Put the non-hammer tools in the right bin |
| `tools_container.usda` | ToolOrganizationBothTask | Put hammers in the right bin and do not touch anything else |
| `tools_container.usda` | ToolOrganizationTask | Put the red hammer and black hammer in the left bin |
| `tools_picking.usda` | ToolsPickingAllHammersTask | Take out all the hammers and put them on the table |
| `tools_picking.usda` | ToolsPickingDrillTask | Select the cordless drill and put it on the table |
| `tools_picking.usda` | ToolsPickingHammerTask | Select the blue hammer and put it on the table |
| `toys_cleanup.usda` | AnimalsInBinTask | Put the lizards in the bin |
| `toys_cleanup.usda` | BlocksInBinTask | Sort all colored blocks into the bin |
| `toys_cleanup.usda` | CleanUpToysTask | Clean up all the smaller toys and leave the birdhouse on the table |
| `toys_cleanup.usda` | CubesAndBlocksInBinTask | Put all the cubes and blocks in the bin |
| `toys_cleanup.usda` | RubiksCubesInBinTask | Sort all rubiks cubes into the bin |
| `toys_cleanup.usda` | StackYellowOnRedTask | Stack the yellow block on the red block |
| `two_bin.usda` | MustardInLeftBinTask | Put the mustard in the left bin |
| `two_bin.usda` | MustardInRightBinTask | Put the mustard in the right bin |
| `wire_shelf_mugs_plate_spatula.usda` | PutTwoMugsOnShelfTask | Put two (2) mugs on the wire shelf |
| `wire_shelf_mugs_plate_spatula.usda` | TakeSpatulaOffShelfTask | Take the spatula off the shelf and put it on the table |
| `workdesk.usda` | MarkerInMugTask | Put the whiteboard marker in the mug |
| `workdesk.usda` | MouseOnKeyboardTask | Put the computer mouse on the keyboard |
| `workdesk.usda` | PickGlassesTask | Pick up the eye glasses |
| `workdesk_bin.usda` | BlackItemsInBinTask | Put the black items in the grey bin |
| `workdesk_bin.usda` | ElectronicsInBinTask | Put the electronic devices in the grey bin |
| `workdesk_bin.usda` | KeyboardOutOfBinTask | Take the keyboard out of the bin and put it on the table |
| `workdesk_bin.usda` | PhoneOrRemoteInBinTask | Put the phone or the remote in the grey bin |
| `workdesk_bin.usda` | SmartphoneInBinTask | Put the smartphone in the grey bin |
| `workdesk_bin.usda` | SpoonInMugTask | Put the metal spoon that's in the wooden bowl in the mug |
| `workdesk_bin.usda` | ToyInBinTask | Put the lizard away in the bin |
| `workdesk_snacks.usda` | AppleAndYogurtInBowlTask | Put the apple and yogurt in the bowl |
| `workdesk_snacks.usda` | ThrowAwayAppleTask | Throw away the apple |
| `workdesk_snacks.usda` | ThrowAwaySnacksTask | Put away the snacks in the bin |
