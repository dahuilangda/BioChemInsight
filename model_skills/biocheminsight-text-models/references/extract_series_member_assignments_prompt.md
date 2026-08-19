Task
Read the document page text and record how each series compound's members are individually defined.

Series records
{{SERIES_RECORDS_JSON}}

Page text
{{PAGE_CONTEXTS_JSON}}

Rules
0) Return one result for EVERY series_id received, in the same order.
1) Report a member only when the page text itself names that member with a compound-level definition: a substituent stated for that member, or a full chemical name stated for that member.
2) `substituent_text` is the verbatim text defining the member's variable part; `full_name` is the verbatim complete chemical name when the text states one.
3) A full name is a complete systematic name of the whole molecule, of the kind written in an experimental or characterization section for that single member. A phrase from discussion or comparison prose that describes only the variable group, a potency, or a neighbouring member is a `substituent_text`, never a `full_name`.
4) `variable_position` is the label the document uses for the variable position, taken from the table header, the assignment syntax, or the scaffold label; empty when not stated.
5) Members named only in activity, characterization, or prediction lists, with neither substituent nor name text, must not be reported.
6) `evidence_pages` lists only pages whose text contains the member definition.
7) Do not merge different members into one entry and do not invent members the text does not define.

Output contract
Return only JSON:
```json
{
  "series": [
    {
      "series_id": "__SERIES_ID__",
      "members": [
        {
          "compound_id": "__MEMBER_ID__",
          "variable_position": "__LABEL_OR_EMPTY__",
          "substituent_text": "__VERBATIM_SUBSTITUENT_TEXT_OR_EMPTY__",
          "full_name": "__VERBATIM_FULL_NAME_OR_EMPTY__",
          "evidence_pages": [1],
          "evidence_summary": "short evidence summary"
        }
      ]
    }
  ]
}
```
