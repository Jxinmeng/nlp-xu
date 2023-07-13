# 将模型 m 的状态字典保存到文件中，并在另一个模型对象 new_m 中加载了该状态字典，从而实现了模型状态的传递和复用。
import self
import torch


class MyModule(torch.nn.Module):
    def __init__(self):
        # 调用了父类的构造函数super(MyModule,self).__init__()来初始化父类torch.nn.Module
        super(MyModule, self).__init__()
        # 定义了一个线性层，输入维度为4，输出维度为2。该线性层包含了4个输入特征和2个输出特征。
        self.l0 = torch.nn.Linear(4, 2)
        # 定义了另一个线性层，输入维度为2，输出维度为1。该线性层包含了2个输入特征和1个输出特征。
        self.l1 = torch.nn.Linear(2, 1)


m = MyModule()
m.state_dict()

torch.save(m.state_dict(), 'mymodule.pt')
m_state_dict = torch.load('mymodule.pt')
new_m = MyModule()
# 调用 load_state_dict() 方法将 m_state_dict 加载到 new_m 中，从而将 new_m 的状态设置为与 m 相同。
new_m.load_state_dict(m_state_dict)
# 使用torch.jit.script函数将MyModule模型转换为脚本模块（scripted module），然后使用torch.jit.save将脚本模块保存到名为'mymodule.pt'的文件中。
scripted_module = torch.jit.script(MyModule())
torch.jit.save(scripted_module, 'mymodule.pt')
torch.jit.load('mymodule.pt')

class ControlFlowModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(4, 2)
        self.l1 = torch.nn.Linear(2, 1)

    def forward(self, input):
        if input.dim() > 1:
            return torch.tensor(0)

        out0 = self.l0(input)
        out0_relu = torch.nn.functional.relu(out0)
        return self.l1(out0_relu)

# 使用torch.jit.trace函数对ControlFlowModule类进行跟踪，传入一个4维的随机张量作为输入，生成一个跟踪模块
traced_module = torch.jit.trace(ControlFlowModule(), torch.randn(4))
# 将跟踪模块保存到名为'controlflowmodule_traced.pt'的文件中。
torch.jit.save(traced_module, 'controlflowmodule_traced.pt')
# 加载名为'controlflowmodule_traced.pt'的跟踪模块文件，返回一个可以直接调用的模块对象。
loaded = torch.jit.load('controlflowmodule_traced.pt')
# 调用加载的跟踪模块，传入一个2x4的随机张量作为输入，对模型进行推理。
loaded(torch.randn(2, 4))

# 这行代码使用torch.jit.script函数对ControlFlowModule类进行脚本化，传入一个4维的随机张量作为输入，生成一个脚本模块。
scripted_module = torch.jit.script(ControlFlowModule(), torch.randn(4))
# 将脚本模块保存到名为'controlflowmodule_scripted.pt'的文件中。
torch.jit.save(scripted_module, 'controlflowmodule_scripted.pt')
# 加载名为'controlflowmodule_scripted.pt'的脚本模块文件，返回一个可以直接调用的模块对象。
loaded = torch.jit.load('controlflowmodule_scripted.pt')

class ControlFlowModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(4, 2)
        self.l1 = torch.nn.Linear(2, 1)

    def forward(self, input):
        if input.dim() > 1:
            return torch.tensor(0)

        out0 = self.l0(input)
        out0_relu = torch.nn.functional.relu(out0)
        return self.l1(out0_relu)

# 使用torch.jit.trace函数对ControlFlowModule类进行跟踪，传入一个4维的随机张量作为输入，生成一个跟踪模块
traced_module = torch.jit.trace(ControlFlowModule(), torch.randn(4))
# 将跟踪模块保存到名为'controlflowmodule_traced.pt'的文件中。
torch.jit.save(traced_module, 'controlflowmodule_traced.pt')
# 加载名为'controlflowmodule_traced.pt'的跟踪模块文件，返回一个可以直接调用的模块对象。
loaded = torch.jit.load('controlflowmodule_traced.pt')
# 调用加载的跟踪模块，传入一个2x4的随机张量作为输入，对模型进行推理。
loaded(torch.randn(2, 4))

# 这行代码使用torch.jit.script函数对ControlFlowModule类进行脚本化，传入一个4维的随机张量作为输入，生成一个脚本模块。
scripted_module = torch.jit.script(ControlFlowModule(), torch.randn(4))
# 将脚本模块保存到名为'controlflowmodule_scripted.pt'的文件中。
torch.jit.save(scripted_module, 'controlflowmodule_scripted.pt')
# 加载名为'controlflowmodule_scripted.pt'的脚本模块文件，返回一个可以直接调用的模块对象。
loaded = torch.jit.load('controlflowmodule_scripted.pt')
