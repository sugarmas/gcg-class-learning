# 敏捷与 Scrum

**敏捷**（Agile）的价值基础是 *Manifesto for Agile Software Development* 与其背后的 principles；它的核心不是"不计划"，而是**以反馈（feedback）驱动**：承认需求无法被完整理解或定义，于是把计划缩短成可检验的假设（short-horizon, testable hypotheses），用短迭代尽快把错误假设的代价压小。讲义一句关键判断：**shorter feedback limits the cost of wrong assumptions**——这正是敏捷方法存在的理由。

**Scrum** 是最广泛使用的 agile 过程，由图定会议（daily and periodic）驱动。按 Scrum Guide，它有三类 accountability：

- **Product Owner**：最大化价值，负责给 Product Backlog 排序。
- **Developers**：产出可用的 Increment，并负责质量。
- **Scrum Master**：团队的教练，帮助团队达到高绩效；**不**做日常指挥、**不**给个人派任务（很多项目里由 PM 兼任）。

三个 artifact 各自配一个 commitment：Product Backlog↔Product Goal，Sprint Backlog↔Sprint Goal，Increment↔Definition of Done。Product Backlog 不只是功能清单，还包括待修 bug、搭建与维护开发环境的技术任务，以及需要获取的知识（例如学习新框架）。

五个 event 构成反馈节奏：**Sprint** 内含 Sprint Planning、Daily Scrum、Sprint Review、Sprint Retrospective。**Daily Scrum** 约 15 分钟，每人说三件事：(1) 昨天做了什么、(2) 今天将做什么、(3) 遇到什么 blocker（队友可当场协助）。**Sprint Review** 向用户/product owner 演示本周期交付物并验收；**Retrospective** 反思哪些做对了、哪些要改进以移除 blocker。Sprint Planning 上 product owner 带着已排序的 backlog 描述头部条目，团队决定本 sprint 能做哪些、把它们展开成 sprint backlog 任务；sprint backlog 每天更新，大小由团队历史交付能力（story points）决定。Task board 通常有 to-do / in-progress / to verify / done 列。

完成与否由两个边界共同定义：**Acceptance Criteria**（用户必须观察到什么）与 **Definition of Done**（团队共享的质量底线），二者都满足才算 complete。

项目管理要素：user story 格式为 *As a \<role\>, I want an \<action\>, so that I can achieve a \<goal\>*——role 来自 RE，action 要 tangible，goal 要可度量。估算用 **story point**（相对估算，如 Fibonacci 或 T-shirt size），因为"说 A 比 B 大"比"说 A 多长"容易；**project velocity** 是若干周期交付 story points 的斜率，**burndown chart** 若出现平台或反弹，说明新增了 story 或原估算偏低。速度单独不等于成功，需与质量、流程、可预测性一起看。

Week 2 的 Scrum feedback loop 实验给了方法论：跑 baseline 与 Sprint 2，**只改一个过程变量**（WIP limit、DoD、pair review、自动化测试、scope negotiation 之一），保持 commitment 与 event stream 不变，这样因果才可解释。度量要 balanced：outcome（完成的有用工作）、flow（carryover 与 WIP）、quality（rework 与 escaped risk）、predictability（forecast error）。claim 必须写清 observed delta + mechanism + limitation；"Sprint 2 完成更多点"不足以证明干预有效。

对照 **XP**（Extreme Programming）：五阶段 planning/design/coding/testing/feedback，实践包括 testing 先于代码、refactoring、pair programming、collective ownership、continuous integration、on-site customer、40-hour week。XP 不设 PM，Scrum 设 PM 但流程比 XP 轻；两者都 lightweight 并含项目管理活动。

## 相关

- [[软件过程模型]]
- [[现代代码评审]]
- [[CS5351 Index]]
