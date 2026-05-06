# 450 行代码，看懂 AI 编程助手的内核

你每天用 Cursor、Claude Code、Copilot 写代码——敲一行注释，它帮你读文件、搜代码、跑命令、写改动，整个流程行云流水。

但你有没有想过：**这背后到底是怎么跑起来的？**

翻 Claude Code 源码？10 万行 TypeScript，50+ 文件，架构深得像迷宫。别说看懂了，找到入口函数都得半天。

**[Minimal Agent Harness](https://github.com/yo666666yo/minimal-agent-harness)** 把这个内核蒸馏成了一个 **450 行的 Python 单文件**——保留了 Claude Code 生产级架构的全部核心模式，但砍掉了所有工程噪音。读完它，你就真的懂了 AI 编程助手是怎么工作的。

---

## 内核长什么样？

```
User Input
  |
  v
AgentHarness.run()           <-- async generator
  |
  +--> 检查上下文是否超限？    <-- 超了就自动压缩
  |
  +--> 调模型（流式输出）      <-- yield text/tool_use 事件
  |      |
  |      +--> 模型要调工具？   --> 立刻开始执行（不等模型说完）
  |      |                          |
  |      |                          +--> 只读工具：并行跑
  |      |                          +--> 写工具：串行排他
  |      |                          +--> bash 报错：取消所有兄弟任务
  |      |
  |      +--> 拿到工具结果 --> 喂回消息列表
  |
  +--> 等剩余工具跑完
  |
  +--> 把 assistant(tool_use) + user(tool_result) 追加进历史
  |
  +--> 检查 turn 上限 / abort 信号
  |
  +--> 回到第一步，继续循环
```

一句话：**调模型 → 拿 tool_use → 跑工具 → 喂结果 → 再调模型**，一个 `while True` 循环撑起整个 agent。

---

## 三大核心机制

### 1. Async Generator 循环——agent 的心跳

```python
async def run(self, user_message: str) -> AsyncGenerator[str, None]:
    messages = [{"role": "user", "content": user_message}]
    turn = 0

    while True:
        turn += 1
        # 调模型，流式拿事件
        async for event in self.api.stream_message(...):
            if event["type"] == "text_delta":
                yield f"[Model] {event['text']}"     # 字还没打完就推给 UI
            elif event["type"] == "tool_use":
                executor.add_tool(block)              # 工具立刻开始执行
                for result in executor.get_completed_results():
                    yield f"[Tool] {result}"           # 结果有了就推
            elif event["type"] == "message_stop":
                break

        if not tool_use_blocks:
            return   # 模型没说要调工具 → 任务结束

        # 等剩余工具跑完，追加到消息历史
        messages.append(assistant_msg)
        messages.append(tool_result_msg)
        # 继续下一轮循环
```

为什么用 `async generator`？因为 **UI 不能卡**。模型吐一个字，你就得立刻显示一个字；工具跑出一个结果，你就得立刻汇报。传统的 "调完模型再统一返回" 在真实产品里完全不可行——用户等不了。

这就是 Claude Code `src/query.ts:241` 里的 `queryLoop()`，一模一样的设计。

### 2. 流式工具执行——模型还没说完，工具已经开跑了

这是最容易被忽略、但体验差距最大的设计。

```
传统做法：                        流式执行：
模型说完所有话                     模型："我需"
  |                                  |    "要读"
  |                                  |    "foo.py"
  v                                  v
解析 tool_use 块                  识别到 tool_use → 立刻启动 read_file("foo.py")
  |                                  |
  v                                  v
执行工具                           模型还在继续说话的同时，文件内容已经返回
  |
  v
返回结果
```

实际效果：**延迟从"模型耗时 + 工具耗时"变成了 max(模型耗时, 工具耗时)**。当模型同时调 3 个只读工具时，三个工具并行跑，几乎同时返回。

更精细的是工具分区：

| 类型 | 例子 | 行为 |
|------|------|------|
| **并发安全** | read_file, grep | 跟其他安全工具并行跑 |
| **排他** | bash, write_file | 独占执行，阻塞所有其他工具 |

bash 还有一个特殊的"兄弟取消"机制——如果一个 bash 命令报错，所有正在并行的 bash 全部取消。因为你的 `mkdir` 失败了，后面的 `cd` 和 `cp` 就算跑完也没意义。

### 3. 上下文自动压缩——token 超限怎么办？

对话长了，上下文窗口装不下。Minimal Agent Harness 的做法和 Claude Code 一样：**把旧消息丢给一个单独的模型调用做摘要，用摘要替换原始历史**。

```python
if estimated_tokens > threshold:
    summary = await summarize_conversation(messages, api_client)
    # [msg1, msg2, ..., msg98, msg99, msg100]  ← 100条消息，~8000 tokens
    #                          |
    #                          v  压缩 msg1..msg96
    # [summary, msg97, msg98, msg99, msg100]    ← 5条消息，~1500 tokens
```

这个过程对用户完全透明。你感觉 agent "一直在干活"，是因为它在 token 快炸的时候悄悄续了命。

---

## 生产级的细节，教学级的可读性

代码里藏了很多"不写出来就意识不到"的生产级设计：

- **Turn limit**：安全阀，防止无限循环把你的 API 额度打光
- **Abort signal**：`asyncio.Event`，Ctrl+C 优雅退出，不丢数据
- **独立 API Client 抽象**：MockAPIClient 和 AnthropicClient 共享同一个接口，你可以先 Mock 模式跑通逻辑再切真 API
- **代码注释精确标注了参考源**：每个关键函数都标注了对应 Claude Code 源码的文件和行号，比如 `# cf. src/query.ts:241`

---

## 一分钟跑起来

```bash
git clone https://github.com/yo666666yo/minimal-agent-harness.git
cd minimal-agent-harness

pip install anthropic

# Mock 模式——不需要 API key，模拟工具调用
python agent_harness.py --mock

# 真实 API 模式
export ANTHROPIC_API_KEY=sk-ant-...
python agent_harness.py
```

Mock 模式下的交互：

```
> read the file agent_harness.py
[Agent] Starting with 4 tools, max 10 turns
[Turn 1/10]
[Agent] Calling model...
[Tool call] read_file({"file_path": "agent_harness.py"})
[Tool result/OK] Minimal Agent Harness — a single-agent tool-use loop...

> search for 'Tool' in the codebase
[Turn 1/10]
[Tool call] grep({"pattern": "Tool", "path": "."})
[Tool result/OK] ./agent_harness.py:47: Tool System...
```

---

## 加一个自定义工具，15 行代码

```python
from agent_harness import Tool, AgentConfig, DEFAULT_TOOLS

class WebSearchTool(Tool):
    name = "web_search"
    description = "搜索网页"
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    is_concurrency_safe = True  # 只读，可以并行

    async def call(self, input, abort_signal):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.search/?q={input['query']}") as resp:
                return await resp.text()

# 注册进去
config = AgentConfig(tools=DEFAULT_TOOLS + [WebSearchTool()])
```

---

## 这个项目适合谁？

- **每天用 Copilot/Cursor/Claude Code 的开发者**——看懂你手里的工具，写出更好的 prompt
- **想入门 LLM Agent 的工程师**——比看论文直观，比啃源码轻松
- **在搭自己的 agent 系统的人**——直接拿这套 query loop + tool executor 做脚手架
- **做 Agentic RL 研究的人**——这个 harness 本身就是强化学习环境的交互层

这也是我正在做的多智能体 LLM 强化学习训练研究的一部分——要让 agent 学会"怎么用工具"，首先得有一个干净的 agent 循环。

---

**[GitHub: yo666666yo/minimal-agent-harness](https://github.com/yo666666yo/minimal-agent-harness)**

如果对你有帮助，欢迎 Star ⭐。有问题直接提 Issue，我回复很快。
