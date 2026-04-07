import re

with open('/data2/ranran/chenghaoyue/ISGFAN/Baogao/毕业论文模板_复刻排版版.md', 'r', encoding='utf-8') as f:
    text = f.read()

pattern = re.compile(r'# 第四章 实验设计与结果分析.*?\\newpage\n\n# 第五章 全文总结与展望', re.DOTALL)

with open('new_ch4.txt', 'r', encoding='utf-8') as f:
    new_ch4 = f.read()

text_new = pattern.sub(lambda m: new_ch4, text)

with open('/data2/ranran/chenghaoyue/ISGFAN/Baogao/毕业论文模板_复刻排版版.md', 'w', encoding='utf-8') as f:
    f.write(text_new)

print("Chapter 4 updated successfully.")
