import asyncio
import warnings
import copy 
import time

class BaseNode:
    """
    基础节点类 (BaseNode)，表示工作流中的一个最小可执行单元。
    """
    def __init__(self):
        # 节点的参数和后续节点的映射字典
        self.params = {}
        self.successors = {}

    def set_params(self, params):
        """设置该节点的运行时参数"""
        self.params = params

    def next(self, node, action="default"):
        """
        将目标节点配置为此节点的后继节点。
        :param node: 下一个要执行的目标节点
        :param action: 触发此流转动作的名称（即返回的动作字符串），默认是 "default"
        """
        if action in self.successors:
            warnings.warn(f"Overwriting successor for action '{action}'")
        self.successors[action] = node
        return node

    def prep(self, shared):
        """
        准备阶段 (Preparation)：在节点实际逻辑执行之前运行，它通常用于从共享状态 (shared) 中提取并准备数据。
        :param shared: 全局共享的数据字典（状态库）
        """
        pass

    def exec(self, prep_res):
        """
        执行阶段 (Execution)：节点的核心业务逻辑，处理从 prep 阶段提取的数据。
        :param prep_res: 从 prep 方法返回的准备结果
        """
        pass

    def post(self, shared, prep_res, exec_res):
        """
        后处理阶段 (Post-processing)：执行阶段完成后，将产生的结果更新写入共享状态库。
        并可以返回一个动作字符串 (action) 以决定状态机走向哪一个目标节点。
        :param shared: 全局共享数据字典
        :param prep_res: prep 阶段结果
        :param exec_res: exec 阶段结果
        """
        pass

    def _exec(self, prep_res):
        """内部调用的执行入口。可在子类中用以封装 exec 以支持额外机制（如重试）。"""
        return self.exec(prep_res)

    def _run(self, shared):
        """
        节点的完整单次运行周期：准备 (prep) -> 执行 (_exec) -> 后处理 (post)
        并返回指引下一步方向的 action 字符串。
        """
        p = self.prep(shared)
        e = self._exec(p)
        return self.post(shared, p, e)

    def run(self, shared):
        """
        直接独立运行一个节点时的入口。
        如果你给一个已经挂载了后继节点的 Node 直接调用 .run() 而非包裹在 Flow 里运行，
        它不会向下自动执行子节点，只会发出警告并执行一次自己。
        """
        if self.successors:
            warnings.warn("Node won't run successors. Use Flow.")  
        return self._run(shared)

    def __rshift__(self, other):
        """
        语法糖支持：Node A >> Node B (重载右移运算符)相当于连接默认 default 分支。
        """
        return self.next(other)

    def __sub__(self, action):
        """
        语法糖支持：Node A - "action" >> Node B。
        重载减号运算符（结合右移操作符），返回一个过渡态对象，进而实现带条件的分支注册。
        """
        if isinstance(action, str):
            return _ConditionalTransition(self, action)
        raise TypeError("Action must be a string")


class _ConditionalTransition:
    """
    条件转换辅助类。
    用于处理如 `node - "action" >> next_node` 这样的声明式图结构连线。
    """
    def __init__(self, src, action):
        self.src = src
        self.action = action

    def __rshift__(self, tgt):
        # 让 src 节点注册 tgt 作为在其指定 action 触发时的后继节点
        return self.src.next(tgt, self.action)


class Node(BaseNode):
    """
    标准节点 (Node)，继承自 BaseNode，加入了【重试机制】和【间隔等待】能力。
    """
    def __init__(self, max_retries=1, wait=0):
        super().__init__()
        self.max_retries = max_retries
        self.wait = wait

    def exec_fallback(self, prep_res, exc):
        """重试耗尽时触发的兜底回滚逻辑配置，这里默认抛出捕获的所有异常"""
        raise exc

    def _exec(self, prep_res):
        """包裹重试和延迟等待能力的执行逻辑"""
        for self.cur_retry in range(self.max_retries):
            try:
                # 尝试正常执行业务逻辑
                return self.exec(prep_res)
            except Exception as e:
                # 若抛出异常并且达到最大重试次数，则进行 fallback 兜底动作
                if self.cur_retry == self.max_retries - 1:
                    return self.exec_fallback(prep_res, e)
                # 若没有达到最大重试次数，并且 wait 大于 0，将挂起等待一会儿
                if self.wait > 0:
                    time.sleep(self.wait)


class BatchNode(Node):
    """
    批处理节点。
    它的 prep 阶段必须返回一个列表 (Items)；这之后会对该列表中每个 Item 分别同步调用父类的执行逻辑，输出结果列表。
    """
    def _exec(self, items):
        return [super(BatchNode, self)._exec(i) for i in (items or [])]


class Flow(BaseNode):
    """
    工作流类 (Flow)，用于编排多个节点，本身也是一个节点 (可以作为子 Flow 嵌套调用)。
    基于有向图及状态机的思维自动驱动节点的流转运行。
    """
    def __init__(self, start=None):
        super().__init__()
        self.start_node = start

    def start(self, start):
        """设定工作流起始第一个执行的节点"""
        self.start_node = start
        return start

    def get_next_node(self, curr, action):
        """
        查询路由表：通过当前节点以及其运行结果给出的 action，查询并返回对应的目标节点。
        如果指定动作不匹配且该节点确实注册了连线路由，则给出丢失动作的警告。
        """
        nxt = curr.successors.get(action or "default")
        if not nxt and curr.successors:
            warnings.warn(f"Flow ends: '{action}' not found in {list(curr.successors)}")
        return nxt

    def _orch(self, shared, params=None):
        """
        核心编排引擎调度器。
        以起始节点为起点，根据每个节点返回的 action 不断寻找下个节点执行，直到抵达了没有后续连线的出口终点。
        """
        # copy 一个初始节点，以免并行执行时内部状态被混淆
        curr = copy.copy(self.start_node)
        p = params or {**self.params}
        last_action = None
        
        while curr:
            curr.set_params(p)
            last_action = curr._run(shared) # 运行当前节点完整生命周期以获取 action
            # 流向下一个状态
            curr = copy.copy(self.get_next_node(curr, last_action))
            
        return last_action

    def _run(self, shared):
        """工作流作为一个上层节点的执行流：自己 prep -> 去遍历子集节点(调 _orch) -> 结束时执行自己的 post"""
        p = self.prep(shared)
        o = self._orch(shared)
        return self.post(shared, p, o)

    def post(self, shared, prep_res, exec_res):
        """覆写返回逻辑，Flow的默认逻辑是将内部的节点最终执行动作进行原样透传。"""
        return exec_res


class BatchFlow(Flow):
    """
    批处理工作流。
    工作方式：它的 prep 需要返回由多组 params 参数构成的列表；它会按次序、用每组对应参数，跑完整的工作流编排路径。
    """
    def _run(self, shared):
        pr = self.prep(shared) or []
        for bp in pr:
            # 针对列表里每一个返回的参数元素分别走一个独立的编排周期
            self._orch(shared, {**self.params, **bp})
        return self.post(shared, pr, None)


class AsyncNode(Node):
    """
    异步非阻塞式的标准节点（使用 await / asyncio）。
    对 IO 密集以及外部请求密集的操作推荐使用。
    含有与普通节点相同的各个生命周期阶段，但变成了异步实现方法。
    """
    async def prep_async(self, shared):
        pass

    async def exec_async(self, prep_res):
        pass

    async def exec_fallback_async(self, prep_res, exc):
        raise exc

    async def post_async(self, shared, prep_res, exec_res):
        pass

    async def _exec(self, prep_res):
        """具有重试能力的异步执行层"""
        for self.cur_retry in range(self.max_retries):
            try:
                return await self.exec_async(prep_res)
            except Exception as e:
                if self.cur_retry == self.max_retries - 1:
                    return await self.exec_fallback_async(prep_res, e)
                if self.wait > 0:
                    await asyncio.sleep(self.wait)

    async def run_async(self, shared):
        if self.successors:
            warnings.warn("Node won't run successors. Use AsyncFlow.")  
        return await self._run_async(shared)

    async def _run_async(self, shared):
        p = await self.prep_async(shared)
        e = await self._exec(p)
        return await self.post_async(shared, p, e)

    def _run(self, shared):
        # 覆写普通入口，由于该节点被声明为异步组件，在同步场景调用属于设计错误所以抛出异常。
        raise RuntimeError("Use run_async.")


class AsyncBatchNode(AsyncNode, BatchNode):
    """串行异步批处理节点：输入为一个列表，并以内部循环的方式，await 等待其每一个执行完成后输出列表。"""
    async def _exec(self, items):
        return [await super(AsyncBatchNode, self)._exec(i) for i in items]


class AsyncParallelBatchNode(AsyncNode, BatchNode):
    """并发异步批处理节点：通过 asyncio.gather 在同一时刻并行调度多个执行请求，加速列表批出执行。"""
    async def _exec(self, items):
        return await asyncio.gather(*(super(AsyncParallelBatchNode, self)._exec(i) for i in items))


class AsyncFlow(Flow, AsyncNode):
    """
    支持运行异步节点或同步混合节点的异步工作流类（同样实现了基于动作 action 的有向图）。
    """
    async def _orch_async(self, shared, params=None):
        curr = copy.copy(self.start_node)
        p = params or {**self.params}
        last_action = None
        
        while curr:
            curr.set_params(p)
            # 兼容混合节点，通过类型检查后进入不同周期的运行函数
            if isinstance(curr, AsyncNode):
                last_action = await curr._run_async(shared)
            else:
                last_action = curr._run(shared)
            curr = copy.copy(self.get_next_node(curr, last_action))
            
        return last_action

    async def _run_async(self, shared):
        p = await self.prep_async(shared)
        o = await self._orch_async(shared)
        return await self.post_async(shared, p, o)

    async def post_async(self, shared, prep_res, exec_res):
        return exec_res


class AsyncBatchFlow(AsyncFlow, BatchFlow):
    """
    以异步为底座构建的按列表先后执行的批处理工作流机制
    """
    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        for bp in pr:
            await self._orch_async(shared, {**self.params, **bp})
        return await self.post_async(shared, pr, None)


class AsyncParallelBatchFlow(AsyncFlow, BatchFlow):
    """
    彻底的并发版批处理工作流：接收准备阶段给出的多组配置列表。并利用 asyncio.gather 把这每一种状态执行都
    异步且独立并发的甩入事件循环，最大化编排执行效率。
    """
    async def _run_async(self, shared): 
        pr = await self.prep_async(shared) or []
        await asyncio.gather(*(self._orch_async(shared, {**self.params, **bp}) for bp in pr))
        return await self.post_async(shared, pr, None)
