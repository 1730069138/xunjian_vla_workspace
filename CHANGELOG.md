# 修改记录

## 2026-09-13 23:19:49 CST

- 修改文件：`docs/文件树.txt`、`scripts/collect/collect_tidy_B.py`、`scripts/deploy/deploy_screwdriver_client.py`（删除）、`scripts/deploy/deploy_screwdriver_client8.29备份没用gpt前 copy.py`（新增）、`AGENTS.md`（新增）、`CHANGELOG.md`（新增）。
- 修改点：将当前工作区的全部已有变动纳入版本控制，包括采集脚本与文件树更新、原部署脚本删除、历史备份脚本加入，以及项目协作规则和修改记录文件加入。
- 功能说明：按用户确认的完整范围，将当前巡检 VLA 项目状态提交并同步到 GitHub 的 `origin/main`。
- 验证结果：两个现存 Python 文件均通过 AST 语法解析检查；`git diff --check` 检出 `docs/文件树.txt:145` 文件末尾新增空行，因此差异格式检查未完全通过。按“全部原样上传”的确认要求保留该空行，未擅自修正。GitHub 推送结果将在提交后核验。

## 2026-08-29 14:50:27 CST

- 修改文件：`scripts/deploy/deploy_screwdriver_client.py`、`CHANGELOG.md`
- 修改点：将推理场景从 `common.robot_spec` 默认的 `scenes/xunjian_arm_scene.xml` 切换为收集脚本使用的 `scenes/tidy_B_record_preview.xml`；适配 `screwdriver`、`fj_screwdriver`、`delivery_box` 实体名；同步 tidy_B 的螺丝刀生成范围、绕世界 z 轴偏航扰动、夹爪张开初态和任务夹爪伺服增益；成功判定改为读取快递盒实际位置与内腔尺寸；进度签名加入场景和新生成分布。
- 功能说明：当前优先保证无障碍、无 APF、无接触保护的纯 VLA（Case 1）在与 `collect_tidy_B.py` 相同的 XML、相机覆盖、初始状态及物体分布下运行，避免推理画面和训练采集场景不一致。障碍物相关工况本次未修改。
- 验证结果：系统 Python 的 `py_compile` 和本次文件范围 `git diff --check` 通过；在 `dummyx` Conda 环境实际加载 `tidy_B_record_preview.xml`，确认双相机覆盖可应用，机械臂 8 维状态地址为 `[0,1,2,3,4,5,6,7]`，`ee_site`、`screwdriver`、`delivery_box` 均能解析，随机生成范围与四元数归一性检查通过。未连接 OpenPI WebSocket 策略服务器运行完整 Case 1，因此端到端推理未验证。

## 2026-08-29 13:02:36 CST

- 修改文件：`scripts/collect/collect_tidy_B.py`、`CHANGELOG.md`
- 修改点：新增 `solve_level_jaw_ik()` 离线 IK 与 `IK_PREALIGN` 阶段，抓取前先解出"夹爪闭合轴水平"的关节位形再关节空间到位；闭合轴目标符号由 IK 解确定并在 `PREALIGN_GRIPPER`/`DESCEND` 全程沿用；新增 `REPOSE_OBJECT` 备用路径（闭爪贴桌把螺丝刀推向可达中心后重走抓取）；`PREALIGN_GRIPPER` 增加位置冻结子步与归一化残差停滞检测；`DESCEND` 增加调平超时分支；调平伺服增益与阻尼改为实测选定值；`metadata.json` 增补 IK/调平/预调整相关字段。
- 功能说明：解决"螺丝刀较远时两夹爪与桌面不平行（有高低差）"。根因不是增益或奇异，而是运动学分支与关节限位——`joint6` 行程为 [0°, 272.7°]，原来"就近选闭合轴符号"总把它推到下限 0° 顶死，残留十几度高低差压不下来；夹爪两指对称，等价解在 `joint6` 约 84~203° 的另一侧，但速度级伺服跨不过分支。现在改为离线 DLS IK（约束为位置 3 + 闭合轴方向 2，绕闭合轴自转留作冗余；两侧符号都试、多起点、按关节限位余量选解、要求接近方向偏离竖直不超过 40°），再用关节空间运动到位，余下几厘米仍由带轴约束的伺服下降。无 IK 解时走预调整备用路径（最多 1 次），仍无解才判失败；关节转移途中若碰动螺丝刀则该次尝试作废。这段关节空间转移会完整录入数据集（已确认），`metadata.json` 中的 `ik_prealign_solved`、`jaw_axis_sign`、`repose_used`、`prealign_unreachable` 可用于事后筛选。
- 验证结果：`python3 -m py_compile`、`python3 -m pyflakes`、`--inspect`、`git diff --check` 均通过。仓库内 `solve_level_jaw_ik` 在 20 个随机 spawn 上 18/20 有解，`joint6` 落在 84~203°、限位余量 8.8~34°，全部为另一侧符号。无图像 MuJoCo 端到端回归（seed 7/42/99/123/2024，各采 1 条成功 episode）：失败尝试合计由旧版 84 次降至 3 次（剩余 2 次为"无夹爪水平的关节解且预调整已用尽"，1 次为既有的 `[MOVE_ARC]` 阶段死锁）；抓取瞬间闭合轴误差 0.559~0.947°、左右指高度差 0.171~0.662mm，全部在 4°/2mm 容差内，对照旧版为 0.415~3.999° / 0.628~1.972mm。过程中还纠正了本次自行引入的两处退步：按最小奇异值自适应阻尼实测完全不收敛（0/6）已删除，符号翻转改为仅在残差真停滞时触发。完整分辨率渲染采集未验证；`REPOSE_OBJECT` 备用路径仅在上述回归中被触发过，未做针对性专项验证。

## 2026-08-29 10:29:24 CST

- 修改文件：`scripts/collect/collect_tidy_B.py`、`CHANGELOG.md`
- 修改点：增强任务内夹爪位置伺服；加入双指接触检测和 `VERIFY_GRASP` 慢速预抬阶段；收紧 DESCEND 到位判据；把抓取相对位姿基准延后到物体确实上升且双指接触稳定之后；统一限制预抬、抬升、搬运和对齐阶段的重抓次数。
- 功能说明：本任务运行时只将 Joint8/Joint9 的 `kp` 从 100 提高到 400，并按比例调整阻尼，不修改共享机器人 XML。闭爪后先试提 5cm，物体相对起点上升至少 12mm、双指同时接触并稳定 0.2 秒后才进入正式搬运；正常的夹内就位不再被当成滑移后主动张爪。单次 attempt 最多纠错重抓一次，避免反复滑移产生数千帧污染轨迹。
- 验证结果：`--inspect`、Python 编译、`pyflakes` 和本次文件范围内的 `git diff --check` 均通过。无图像 MuJoCo 回归首先验证 5 个固定 seed 均能最终成功；加入统一重抓上限后，对较困难的 seed 7/42/99 复测均成功，最终有效 episode 分别为 670/719/689 帧，最大抓取平移滑移分别为 27.05/7.03/6.15mm，均低于 35mm 阈值，且全部通过预抬和双指接触验证。完整分辨率渲染采集未验证。

## 2026-08-29 10:13:11 CST

- 修改文件：`scripts/collect/collect_tidy_B.py`、`CHANGELOG.md`
- 修改点：修正采集脚本的项目根目录推导和 MuJoCo 场景路径。
- 功能说明：直接执行 `python scripts/collect/collect_tidy_B.py ...` 时，脚本现在会把 `/home/jun/xunjian_vla_workspace` 加入模块搜索路径，并从项目根目录下的 `scenes/tidy_B_record_preview.xml` 加载场景。
- 验证结果：在 `dummyx` 环境直接运行 `python scripts/collect/collect_tidy_B.py --inspect`，场景加载、相机覆盖、50 Hz × 20 子步契约、夹爪反号约定及抓取/走廊检查均通过；Python 编译检查通过。

## 2026-08-29 09:52:20 CST

- 修改文件：`scripts/collect/collect_tidy_B.py`、`CHANGELOG.md`
- 修改点：将笛卡尔伺服控制点从 `ee_site` 改为两指真实接触端面中心，并使用偏移点雅可比；将抓取滑移检测改为物体相对夹持中心的平移/旋转位姿误差；为 episode/attempt 派生独立确定性 seed；新增 episode 完整性识别、原子写盘、`metadata.json` 与 `complete.json`。
- 功能说明：抓取和搬运轨迹现在控制真实交互点；滑移检测不再受世界坐标系运动干扰；固定 seed 续采时不同 episode 不会重复使用同一随机流；半成品目录不会被当作新格式的完整 episode，且旧格式完整数据仍可识别；元数据记录初始/最终物体位姿、投放点、seed、阶段耗时、纠错状态及轨迹质量指标。
- 验证结果：`python3 -m py_compile scripts/collect/collect_tidy_B.py`、`python3 -m pyflakes scripts/collect/collect_tidy_B.py` 与 `git diff --check` 通过；相对位姿误差、seed 唯一性/确定性、数据流长度校验、原子发布及完成标记的独立逻辑测试通过。在 `dummyx` 环境通过运行时路径覆盖完成 MuJoCo `--inspect`；无图像诊断采集以 seed `20260829` 成功生成 1 条 605 帧 episode，并验证 `metadata.json`、`complete.json`、两路图像和 NPZ 帧数一致。完整分辨率渲染采集未验证。另只读检查发现脚本现有项目根目录注入多上溯一级、`SCENE_PATH` 指向不存在的 `scripts/scenes/tidy_B_record_preview.xml`；这两个既有路径问题不在本次确认范围内，未修改。

## 2026-08-29 09:31:18 CST

- 修改文件：`AGENTS.md`、`CHANGELOG.md`
- 修改点：建立项目代码修改前的需求确认规则，以及每次修改后的记录规范。
- 功能说明：后续修改必须先确认目标、范围、预期行为和验证方式；得到用户明确授权后方可实施，并在本文件记录修改时间、涉及文件、修改内容、功能与验证结果。
- 验证结果：已检查两份文件内容和 Git 工作区状态；规则文件与修改记录均已创建。
