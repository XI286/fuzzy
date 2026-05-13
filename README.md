# fuzzy
# 我构建了一个基于LLM多Agent协作的无人机集群任务规划与路径优化智能体系统。该系统针对传统方法中任务分配与路径规划割裂、动态环境适应性差、以及自然语言意图难以直译的三大痛点，设计了四Agent协作架构：意图解析Agent利用RAG增强的LLM将操作人员自然语言指令（如“优先排查东区火情”）解析为结构化子任务清单；任务分配Agent集成市场竞拍算法与多智能体强化学习，在协同场（CoordField）机制下实现去中心化动态任务分配；路径规划Agent基于Dubins路径与改进人工势场法进行多约束三维轨迹优化；安全守护Agent对所有输出指令进行硬约束校验。整套系统在AirSim和SITL平台上进行了仿真验证，支持9至30架无人机的集群作战。
import openai
import json

class LLMIntentParser:
    def __init__(self, api_key, model="gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def parse_instruction(self, natural_language_instruction):
        prompt = f"""
        你是一个无人机集群指挥官。请将以下自然语言指令分解为结构化的无人机子任务列表。
        指令: “{natural_language_instruction}”
        请以JSON格式输出，包含以下字段：
        - subtasks: 一个列表，每个元素是一个字典，包含 task_type (侦察/中继等), target_area, priority (高/中/低).
        - required_uav_count: 一个整数，表示所需无人机总数。
        - constraints: 一个字典，描述约束条件，如 max_flight_time_min, no_fly_zones 等。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, # 较低的温度值(0.5)可提高规划的可靠性并减少执行时间[reference:2]
        )
        try:
            # 假设LLM的回复就是一段干净的JSON字符串
            structured_data = json.loads(response.choices[0].message.content)
            return structured_data
        except json.JSONDecodeError:
            # 实际应用中需要更健壮的错误处理和解析逻辑
            print("Error: LLM did not return valid JSON.")
            return None

# 示例用法
parser = LLMIntentParser(api_key="YOUR_API_KEY")
mission_data = parser.parse_instruction("优先排查东区火情，同时确保通讯中继不间断")
print(json.dumps(mission_data, indent=2, ensure_ascii=False))
import numpy as np
import itertools
import trajallocpy as tap # 假设已安装 trajallocpy 库

class TaskAllocationAgent:
    def __init__(self, uav_positions, mission_data):
        # 初始化无人机位置、任务列表等
        self.uav_positions = uav_positions
        self.tasks = self._create_tasks_from_mission(mission_data)
        self.uav_ids = list(range(len(uav_positions)))

    def _create_tasks_from_mission(self, mission_data):
        # 将LLM解析的数据转换为程序可读的任务列表
        tasks = []
        for subtask in mission_data["subtasks"]:
            for _ in range(subtask["uav_count"]):
                task = tap.Task(position=subtask["area_center"], task_type=subtask["task_type"])
                tasks.append(task)
        return tasks

    def allocate(self):
        # 1. 初始化CBBA规划器
        planner = tap.CBBA(self.uav_ids, self.uav_positions, self.tasks)
        # 2. 执行竞拍和共识过程
        bundle, path = planner.allocate()
        return bundle, path

# 示例用法
uav_positions = [np.array([1, 2]), np.array([3, 4]), np.array([5, 0])]
mission_data = {
    "subtasks": [
        {"task_type": "侦察", "area_center": np.array([2, 3]), "uav_count": 1},
        {"task_type": "侦查", "area_center": np.array([4, 5]), "uav_count": 1}
    ]
}
agent = TaskAllocationAgent(uav_positions, mission_data)
bundle, path = agent.allocate()
print(f"任务包分配结果: {bundle}")
print(f"路径规划结果: {path}")
import dubins
import numpy as np

class PathPlannerAgent:
    def __init__(self, turning_radius=5.0, step_size=1.0):
        self.turning_radius = turning_radius
        self.step_size = step_size

    def plan_path(self, uav_start_state, waypoints):
        """
        为单个无人机规划路径。
        uav_start_state: (x, y, theta) 无人机初始状态 (位置和朝向)
        waypoints: 任务目标点列表 [(x1, y1), (x2, y2), ...]
        """
        full_path = [uav_start_state[:2]] # 从初始位置开始
        current_state = uav_start_state
        
        # 在不考虑复杂避障的情况下，Dubins路径非常适合点对点规划
        for wp in waypoints:
            # 计算到下一个航点的Dubins路径
            path = dubins.shortest_path(current_state, (wp[0], wp[1], 0), self.turning_radius) # 目标朝向暂设为0
            # 采样路径点
            configurations, _ = path.sample_many(self.step_size)
            sampled_points = [(c[0], c[1]) for c in configurations]
            full_path.extend(sampled_points)
            # 更新当前状态为最后一个采样点
            current_state = (configurations[-1][0], configurations[-1][1], configurations[-1][2])
        return full_path

# 示例用法
# planner = PathPlannerAgent(turning_radius=10)
# start = (0, 0, np.pi/2)  # 朝向东
# waypoints = [(10, 0), (20, 5)]
# path = planner.plan_path(start, waypoints)
# print(f"规划路径点: {path}")
import airsim
import time

class AirSimExecutor:
    def __init__(self, drone_name="Drone0"):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, drone_name)
        self.client.armDisarm(True, drone_name)
        self.drone_name = drone_name

    def execute_path(self, path):
        """控制无人机沿规划好的路径飞行"""
        self.client.takeoffAsync(vehicle_name=self.drone_name).join()
        print(f"{self.drone_name}: 起飞完成")

        # 逐点飞行
        for point in path:
            print(f"{self.drone_name}: 飞向 {point}")
            self.client.moveToPositionAsync(point[0], point[1], -10, 5, vehicle_name=self.drone_name).join()

        print(f"{self.drone_name}: 任务完成，悬停。")
        self.client.hoverAsync(vehicle_name=self.drone_name).join()

# 示例用法
# executor = AirSimExecutor("Drone0")
# executor.execute_path(path)
# 1. 解析指令
parser = LLMIntentParser(api_key="YOUR_API_KEY")
mission_data = parser.parse_instruction("优先排查东区火情，同时确保通讯中继不间断")

# 2. 任务分配
allocator = TaskAllocationAgent(uav_positions, mission_data)
bundle, path = allocator.allocate()

# 3. 并行路径规划 (简化示意)
planner = PathPlannerAgent()
all_paths = {}
for uav_id in range(len(uav_positions)):
    # 根据分配结果和无人机初始状态规划路径
    # 这是一个简化版本，实际需要从bundle和path对象中提取信息
    start_state = ... 
    task_waypoints = ...
    all_paths[uav_id] = planner.plan_path(start_state, task_waypoints)

# 4. 安全守护
if check_airspace_conflict(all_paths):
    print("警告：检测到路径冲突！需要重新规划。")
else:
    # 5. 执行任务 (并行)
    import threading
    threads = []
    for uav_id, path in all_paths.items():
        executor = AirSimExecutor(f"Drone{uav_id}")
        t = threading.Thread(target=executor.execute_path, args=(path,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

    print("所有无人机任务执行完毕。")
