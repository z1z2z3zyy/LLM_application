import ast
import platform
from typing import List, Callable, Tuple
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
from resources.System_Prompt import system_prompt
import inspect
from string import Template


class Agent:
    def __init__(self, tools:List[Callable],model:str,project_directory:str) -> None:
        self.tools = {func.__name__:func for func in tools}
        self.model = model
        self.project_directory = project_directory
        self.client=OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def run_agent(self,user_input:str)->None:
        messages=[{"role":"system","content":f"{self.generate_prompt()}"},
                 {"role":"user","content":f"<question>{user_input}</question>"}
                 ]
        while True:
            content=self.call_model(messages)
            # print(content)
            thought=re.match(r"<thought>(.*?)</thought>",content,re.DOTALL)
            if thought:
                print(f"\n\n Thought:{thought.group(1)}")
            if "<final_answer>" in content:
                final_answer=re.search(r"<final_answer>(.*?)</final_answer>",content,re.DOTALL).group(1)
                print(f"\n\n Final answer:{final_answer}")
                return
            action=re.search(r"<action>(.*?)</action>",content,re.DOTALL)
            if not action:
                print(f"\n\n {action}")
                raise RuntimeError("模型未输出")
            action=action.group(1)
            tool_name,args=self.parse_action(action)
            print(f"\n\n Action: {tool_name}({', '.join(args)})")
            should_continue = input(f"\n\n是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
            if should_continue != "y":
                print(f"\n\n操作已取消")
                return
            try:
                observation=self.tools[tool_name](*args)
            except Exception as e:
                observation=f"工具执行错误{str(e)}"
            print(f"\n\nObservation:{observation}")
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})

    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 手动解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        i = 0
        paren_depth = 0

        while i < len(args_str):
            char = args_str[i]

            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层逗号，结束当前参数
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i - 1] != '\\'):
                    in_string = False
                    string_char = None

            i += 1

        # 添加最后一个参数
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))

        return func_name, args

    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()

        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
                (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str

        # 尝试使用 ast.literal_eval 解析其他类型
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            return arg_str

    def generate_prompt(self)->str:
        tool_list=self.get_tool_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        return Template(system_prompt).substitute(
            operating_system=platform.system(),
            tool_list=tool_list,
            file_list=file_list,
        )

    def get_tool_list(self):
        tool_description=[]
        for func in self.tools.values():
            func_name = func.__name__
            func_signature = inspect.signature(func)
            func_doc = inspect.getdoc(func)
            tool_description.append(f"-{func_name}{func_signature}: {func_doc}")
        return "\n".join(tool_description)



    def call_model(self, messages):
        print("\n\n正在请求模型，请稍等...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return content

def read_file(file_path):
    """用于读取文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def write_to_file(file_path, content):
    """将指定内容写入指定文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "写入成功"

def run_terminal_command(command):
    """用于执行终端命令"""
    import subprocess
    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功" if run_result.returncode == 0 else run_result.stderr



if __name__=="__main__":
    load_dotenv()
    project_directory="snake"
    project_dir = os.path.abspath(project_directory)
    print(project_dir)
    tools = [read_file, write_to_file, run_terminal_command]
    agent = Agent(tools=tools, model="qwen3-coder-480b-a35b-instruct", project_directory=project_dir)
    task = input("请输入任务：")
    agent.run_agent(task)