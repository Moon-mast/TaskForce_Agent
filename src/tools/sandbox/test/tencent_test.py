import os
from dotenv import load_dotenv

# 加载当前目录下的 .env 文件，将其中的变量注入到环境变量中
load_dotenv()

from e2b_code_interpreter import Sandbox

# template 替换为控制台创建的沙箱工具名称
# timeout 指定运行时长，单位为秒；示例为 3600 秒（1 小时）
sandbox = Sandbox.create(template="Sandbox", timeout=3600)

# 执行 python 代码，流式获取输出并打印，代码执行超时时间 600 秒
python_code = """
import time
print("hello python")
time.sleep(2)
print("hello python after 2 sec")
"""

print(sandbox.run_code(
    python_code,
    on_stdout=lambda data: print(data),
    timeout=600,
))

sandbox.kill()
