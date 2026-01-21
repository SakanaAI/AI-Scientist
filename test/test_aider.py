from aider.io import InputOutput
from aider.coders import Coder
from aider.models import Model
import subprocess
from subprocess import TimeoutExpired

# This is a list of files to add to the chat
fnames = ["greeting.py"]

io = InputOutput(
    yes=True, chat_history_file=f"test/test_aider.txt"
)

# model = Model("OpenAI/gpt-4o")
# model = Model("deepseek/deepseek-chat")
model = Model("deepseek/deepseek-chat")
# model = Model("openai/gpt-4o")

# Create a coder object
coder = Coder.create(
    main_model=model,
    fnames=fnames,
    io=io,
    use_git=False,
)

# This will execute one instruction on those files and then return
coder_out = coder.run("make a script that prints hello world")
print(coder_out)


# LAUNCH COMMAND
command = [
    "python",
    "greeting.py"
]

result = subprocess.run(
    command, stderr=subprocess.PIPE, text=True
)

print('result done!\n')
print('='*20)
print(result)
if result.stderr:
    print(result.stderr, file=sys.stderr)
print('='*20)

