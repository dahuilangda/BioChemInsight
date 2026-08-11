Task
Plan extraction context and entity anchors for candidate bioactivity pages.
Plan only. Do not extract assay values, and do not rewrite assay names.

Target assay list
{{ASSAY_NAMES_JSON}}

Page context JSON
{{PAGE_CONTEXTS_JSON}}

Planning rules
1) Output exactly one decision for each input page.
2) Page roles:
   - `new_record`: new assay record block
   - `continuation`: continuation of a prior table/record block
   - `standalone`: independent record block with complete context
   - `non_assay`: non-assay data page
   - `unknown`: cannot determine
3) Inherit only when prior pages explain missing current-page headers, units, methods, endpoints, compound anchors, or captions.
4) Inheritance may cross multiple pages, but not across a new table, new heading, new assay context, new structure/record block, or obvious reset.
5) Do not inherit old context when the current page has complete context.
6) `entity_anchor_strategy`：
   - `text_compound_id`: anchor by visible text ID
   - `visual_structure_anchor`: anchor mainly by same-page structure drawing
   - `mixed`: both are needed
   - `unknown`: cannot determine
7) Visual anchors are only planning hints; they cannot generate IDs or assay values.

Output contract
Return only JSON:
```json
{
  "pages": [
    {
      "page": 1,
      "role": "new_record|continuation|standalone|non_assay|unknown",
      "use_prior_context": false,
      "context_source_page": null,
      "entity_anchor_strategy": "text_compound_id|visual_structure_anchor|mixed|unknown",
      "confidence": "high|medium|low",
      "reason": "short reason"
    }
  ]
}
```
