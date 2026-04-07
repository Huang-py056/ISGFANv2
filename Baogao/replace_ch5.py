import re

file_path = "毕业论文模板_复刻排版版.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

with open("new_ch5.txt", "r", encoding="utf-8") as f:
    new_ch5_text = f.read()

# Try to find the start of Chapter 5 and the start of the next section
pattern = re.compile(r"# 第五章 全文总结与展望.*?(?=\\newpage\n\n# 致谢)", re.DOTALL)

if pattern.search(content):
    new_content = pattern.sub(new_ch5_text + "\n", content)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Chapter 5 updated successfully.")
else:
    print("Could not find the target section to replace.")
