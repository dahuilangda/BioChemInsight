Example 1:
Input: Answer: Compound 2
Output: {"COMPOUND_ID":"Compound 2","CONFIDENCE":"high","REASON":"explicit answer line"}

Example 2:
Input: 模型分析了多个候选，最终结论写在文末：Final answer: Example 8
Output: {"COMPOUND_ID":"Example 8","CONFIDENCE":"high","REASON":"final answer line names Example 8"}

Example 3:
Input: 文本里有 [0159]、Figure 5、Page 3，还有单独一行 Compound 3
Output: {"COMPOUND_ID":"Compound 3","CONFIDENCE":"medium","REASON":"valid final ID appears distinctly"}

Example 4:
Input: 全文只有 [0007]、Figure 5、Table 2
Output: {"COMPOUND_ID":"None","CONFIDENCE":"low","REASON":"only invalid markers are present"}

Example 5:
Input: 候选有 12、12a、Compound 12；如果最终明确答案行为 “答案：Compound 12”
Output: {"COMPOUND_ID":"Compound 12","CONFIDENCE":"high","REASON":"answer line disambiguates candidates"}
