# Per-Category Grid Search Workflow

Each category (e.g. uchen_high, umeh_poor) gets two phases:

## Phase 1: Full Grid Test (1 image)

1. In the grid search notebook, Cell 2 (Configuration):
   - Uncomment the TARGET for the first file in this category
   - Set MODELS_TO_TEST to match the script type:
     - Uchen: ["Woodblock", "Woodblock-Stacks", "Modern"]
     - Umeh: ["Ume_Druma", "Ume_Petsuk", "Modern"]
   - Make sure PARAM_VALUES is the full grid (1,728 combos):
     line_mode: ["line", "layout"]
     class_threshold: [0.7, 0.8, 0.9]
     k_factor: [2.0, 2.5, 3.0]
     bbox_tolerance: [2.5, 3.5, 4.0, 5.0]
     merge_lines: [True, False]
     tps_threshold: [0.1, 0.25, 0.5, 0.9]

2. Reset checkpoint (Cell 10 - uncomment, run, re-comment)

3. Run quick test cell (Cell 7) - processes 1 image, all 1,728 combos (~5 hrs)

4. Run analysis notebook - look at:
   - Average quality by each parameter
   - Which parameters have no effect (identical scores)
   - Which parameters show clear winners

## Phase 2: Trimmed Run (remaining images)

5. Back in grid search notebook, update PARAM_VALUES:
   - Cut parameters that showed no effect
   - Cut clear losers (e.g. if merge_lines=False always loses by 10+ pts)
   - Keep parameters that might matter on other pages in this category

6. Keep same TARGET (same file, it has multiple pages)

7. Run full grid search cell (Cell 8) - checkpoint skips the page already done

8. Run analysis notebook again on the combined results

9. Git commit with category name: "Grid search: uchen_high complete - [key findings]"

## Move to Next Category

10. Change TARGET to first file in next category
11. Change MODELS_TO_TEST if switching between uchen/umeh
12. Restore PARAM_VALUES to full grid
13. Repeat from Phase 1
