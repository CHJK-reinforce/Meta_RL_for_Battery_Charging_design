import pybamm
import numpy as np

def cal_soc(c):
    return (c - 873.0) / (30171.3 - 873.0)


class SPM:
    def __init__(self, init_v=3.2, init_t=298, param="Chen2020"):
        # 传递参数
        self.reward = None
        self.param = param
        self.sett = {'sample_time': 30*3,
                     'constraints temperature max': 273 + 25 + 11,
                     'constraints voltage max': 4.2}
        # 设置一下日志信息
        # pybamm.set_logging_level("DEBUG")
        # 模型初始化
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options=options)
        param = pybamm.ParameterValues(self.param)
        param["Upper voltage cut-off [V]"] = 4.4

        # 根据所给的电压初始化参数
        param.set_initial_stoichiometries("{} V".format(init_v))
        # param.set_initial_stoichiometries(init_soc)
        # 改变电流函数为输入模型
        param["Current function [A]"] = "[input]"
        self.model = model
        self.param = param
        self.temp = init_t
        self.voltage = init_v
        self.soc = cal_soc(param["Initial concentration in negative electrode [mol.m-3]"])
        self.soc_d = None
        self.temp_d = None
        self.voltage_d = None
        self.sol = None
        self.info = None
        self.done = False
        self.reward = 0

    def step(self, action, st=None):

        # 连续求解状态替换
        if self.sol is not None:
            self.model.set_initial_conditions_from(self.sol)
        # 仿真设置
        simulation = pybamm.Simulation(self.model, parameter_values=self.param)
        # 时间间隔设置
        if st is not None:
            t_eval = np.linspace(0, st, 2)
        else:
            t_eval = np.linspace(0, self.sett['sample_time'], 2)
        sol = simulation.solve(t_eval, inputs={"Current function [A]": -action})
        self.voltage = sol["Voltage [V]"].entries[-1]
        self.temp = sol["X-averaged cell temperature [K]"].entries[-1]
        c = sol["R-averaged negative particle concentration [mol.m-3]"].entries[-1][-1]
        self.soc = cal_soc(c)
        self.info = sol.termination
        # 数据的更新
        self.sol = sol

        time_reward = -0.1
        if self.temp > self.sett['constraints temperature max']:
            temp_reward = -1 * (self.temp - self.sett['constraints temperature max'])
        else:
            temp_reward = 0
        
        if self.voltage > self.sett['constraints voltage max']:
            volt_reward = -10 * (self.voltage - self.sett['constraints voltage max'])
        else:
            volt_reward = 0

        self.reward = time_reward + temp_reward + volt_reward

        # 检查充电是否完成
        if self.soc > 0.8:
            self.done = True

        return [self.voltage, self.temp, self.soc], self.reward, self.done, self.info

    def reset(self, init_v=3.2, init_t=298):
        self.__init__(init_v, init_t)

        return

