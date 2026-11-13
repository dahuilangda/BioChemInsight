# Review grid follow-up

The result toolbar now uses one 32 px control system, 16 px icons, a consistent type scale, shared hover/focus treatment and explicit open/active states. The view switch is one segmented control. Export has a label. Menus are anchored within the result pane, and close outside or with Escape. Opening a completed document no longer produces the redundant extraction-completed toast. Fit-width PDF resizing preserves the same source point at the viewport centre, preventing a jump to another page on narrow screens.

Column headers support drag reordering with before/after insertion feedback. Both headers and cells render from the same column identities, including hidden preview and SMILES fields. Header edges resize columns; keyboard arrows resize a focused separator. Alt + left/right moves a focused header. The Columns menu includes field search, visibility controls and an Arrange columns section with tap-accessible left/right buttons.

Row handles support pointer/touch movement, before/after insertion, viewport edge scrolling, cancellation outside the grid and Alt + up/down. Manual row movement is disabled while automatic sorting is active. Sorting and drag ordering change the displayed view; edits continue to address the original record. Stable opaque IDs are supplied for legacy structure records, rather than using the visible row position or chemical ID as record identity.

Layout is personal to the current browser and task: column order, widths, hidden measurement fields and manual row order survive refresh. Hiding and showing a field retains its prior position. Renaming a measurement retains its position and width. Reset table layout restores defaults. Cards follow the selected measurement visibility and order.

The behavior follows the distinction between manual ordering and active automatic sorting described in Airtable's official guidance:
- [Grid view and reordering fields](https://support.airtable.com/articles/7905594155-airtable-grid-view)
- [Sorting records](https://support.airtable.com/articles/8716174546-sorting-records-in-airtable-views)

## Validation

The Python suite passes 82 checks, including stable record IDs through reload, ID correction and reordering. The production build passes.

`frontend/tests/workbench.grid.browser.cjs` uses actual Chromium pointer interactions and PDF.js with fixture-backed records to verify:
- header/cell alignment after column drag, including hidden columns;
- row movement to the first and last positions, cancellation and keyboard moves;
- editing after reordering writes to the correct record;
- actual column width changes and persistence after reload;
- hidden field persistence and restoration at its previous position;
- automatic sorting blocks manual movement;
- all toolbar control groups measure 32 px high;
- menus remain within the pane and mobile viewport;
- touch-accessible column ordering and Escape dismissal.

Existing navigation and persistence browser regressions also pass. The retained SR document was separately inspected in the production browser; its screenshots are in `output/platform_acceptance_20261113/ui_grid_toolbar.png` and `ui_grid_mobile.png`.

This is not a claim to implement every Airtable feature. Shared named views, multi-condition filter groups, multi-column sorting and bulk record operations remain separate follow-up capabilities. View layout is not currently synchronized across browsers or users.
