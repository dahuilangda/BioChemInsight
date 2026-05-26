示例1：
输入：Answer: Compound 2
输出：{"COMPOUND_ID":"Compound 2","CONFIDENCE":"high","REASON":"explicit answer line"}

示例2：
输入：模型分析了多个候选，最终结论写在文末：Final answer: Example 8
输出：{"COMPOUND_ID":"Example 8","CONFIDENCE":"high","REASON":"final answer line names Example 8"}

示例3：
输入：文本里有 [0159]、Figure 5、Page 3，还有单独一行 Compound 3
输出：{"COMPOUND_ID":"Compound 3","CONFIDENCE":"medium","REASON":"valid final ID appears distinctly"}

示例4：
输入：全文只有 [0007]、Figure 5、Table 2
输出：{"COMPOUND_ID":"None","CONFIDENCE":"low","REASON":"only invalid markers are present"}

示例5：
输入：候选有 12、12a、Compound 12；如果最终明确答案行为 “答案：Compound 12”
输出：{"COMPOUND_ID":"Compound 12","CONFIDENCE":"high","REASON":"answer line disambiguates candidates"}
