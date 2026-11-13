# SAR review workbench interaction contract

The product should support a continuous workflow: choose a document, inspect recommended pages, extract, inspect the evidence behind each result, correct or supplement it, and export reviewed records. A completed background operation is not equivalent to a correct or visible user result.

## Data table

Page cells show numbers and navigate to the PDF. Compound IDs remain the anchor when inspecting many properties. Source structures, assembled structures and source measurements are independently accessible. Experimental data precede predicted/calculated properties; different targets and conditions must not be collapsed merely because an endpoint has the same name.

Common actions have visible labels. Columns can be added, renamed, hidden and removed. Hiding is a view operation; removal is an undoable data edit. Filtering supports filled/empty, contains, equals and numeric comparisons; sorting uses natural ID order and keeps empty cells last. Changing a view never changes record identity or which original record an edit targets. A rename changes the displayed field; it does not convert units or rewrite original source evidence.

Current implementation also supports column drag reordering, pointer/touch row reordering, header-edge resizing and keyboard alternatives. Column order, widths, hidden measurement fields and manual row order are stored per task in the current browser. An opaque record identity protects edits when view positions change. Automatic sorting disables manual row movement. Hidden fields retain their positions when shown again. Menus fit within the results pane and dismiss outside or with Escape. Multi-condition filter groups, named shared views, multi-column sorting, bulk selection and bulk correction remain further product work; they are not claimed complete.

The reference is Airtable's separation of views from records and its discoverable field controls, rather than a visual clone:
- https://support.airtable.com/articles/7905594155-airtable-grid-view
- https://support.airtable.com/articles/6540330000-airtable-field-actions
- https://support.airtable.com/articles/9290731839-filtering-records-using-conditions
- https://support.airtable.com/articles/8716174546-sorting-records-in-airtable-views

## PDF and correction loop

The PDF scrolls continuously. Clicking a result locates its source page and box. Draw box exposes structure, result-table and scaffold/fragment purposes. A green draft box is distinct from an extracted source highlight. A visible action submits the selection; progress and retry errors remain local to that selection.

A structure becomes a new row by default. Users may explicitly replace the selected row's structure. Replacement preserves the previous structure evidence, existing results and user corrections; contradictory detected IDs block replacement. Table extraction links only an unambiguous verified ID; otherwise it creates an unlinked result row. New structure recognition opens the molecular editor for inspection. Pending proposals must not look like accepted structures.

Completed tasks restore selected pages from saved task parameters, with source evidence pages as a fallback. Automatic discovery does not silently erase an existing selection. Selecting Auto retains current pages; clearing them requests fresh discovery on the next run.

## Harness and evidence contract

The runtime plans from document evidence and explicit gaps, chooses bounded atomic operations, validates outputs and updates coverage. Operations include page inspection, host-owned box selection, specialized image recognition, name resolution, fragment assembly and result extraction. PubChem is a cached and globally throttled external fallback after a documented local miss. Independent names can challenge a scaffold hypothesis; previous visual verdicts and generated graphs are excluded from the final blinded review prompt.

IDs, structure graphs, measurements, uncertainty, conditions and source provenance remain separate evidence objects. A graph/ID conflict prevents automatic result linking. Missing or ambiguous attachments stay pending; molecular formula or row order is not identity evidence. Recovery preserves manual edits and returns explicit unresolved items when evidence or providers are insufficient.

Custom peptide/cyclic-peptide recognition remains a first-class path. A bare sequence is not evidence for the full modified or cyclic molecular graph. Acceptance must evaluate stereochemistry, crosslinks, modifications and closure against independent source gold.
