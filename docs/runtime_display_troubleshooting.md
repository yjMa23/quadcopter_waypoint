# Isaac Sim 默认 Display 诊断与 GUI 启动说明

## 1. 本次实际诊断结果

在 2026-08-04 当前执行会话中检查到：

```text
DISPLAY=
WAYLAND_DISPLAY=
XDG_SESSION_TYPE=
/tmp/.X11-unix/X0 不存在
~/.Xauthority 不存在
```

因此 Isaac Sim 启动日志中的

```text
failed to open the default display. Can't verify X Server version.
```

不是 GPU、驱动或 Vulkan 初始化失败，而是当前进程运行在一个没有图形桌面会话的非交互 shell 中。进程没有 X11/Wayland display 地址，也没有 X11 socket 和授权文件，自然无法连接默认显示服务器。

当前 GPU 和 Vulkan 仍可被 Isaac Sim 识别，因此 `--headless` 训练、数据采集和闭环数值评估能够正常运行。

## 2. `DISPLAY` 到底是什么

Linux 图形程序通常不直接操作显示器，而是连接图形服务器：

- X11/XWayland 使用 `DISPLAY`，常见值为 `:0`、`:1` 或 SSH 转发生成的 `localhost:10.0`；
- 原生 Wayland 使用 `WAYLAND_DISPLAY`，常见值为 `wayland-0`；
- X11 本地连接通常还依赖 `/tmp/.X11-unix/X*` socket；
- 某些环境还需要 `XAUTHORITY` 或 `~/.Xauthority` 完成授权。

仅执行

```bash
export DISPLAY=:0
```

不会创建显示服务器。如果 `/tmp/.X11-unix/X0` 不存在，或者当前用户无权限连接，强行设置 `DISPLAY=:0` 仍然会失败。

## 3. 为什么当前会话没有 Display

最符合当前证据的原因是：项目命令由 DevSpace/远程任务执行器启动，该执行器提供的是独立的非交互 shell，而不是从 Ubuntu 桌面终端继承的图形会话。

常见的同类场景还有：

1. 通过普通 SSH 登录，但没有启用 X11 forwarding；
2. 在 systemd、cron、后台任务或 CI 中运行；
3. tmux/screen 会话在登录桌面前创建，未继承后来的 `DISPLAY`；
4. 使用 `sudo` 后图形环境变量或授权被清除；
5. Docker 容器没有挂载 X11 socket 和授权文件；
6. 机器本身没有登录图形桌面，只运行纯 TTY 或远程 shell。

## 4. 本地桌面终端启动 GUI

应在已经登录 Ubuntu 图形桌面的终端中执行，而不是在当前无图形任务 shell 中执行。

先检查：

```bash
echo "$DISPLAY"
echo "$WAYLAND_DISPLAY"
echo "$XDG_SESSION_TYPE"
ls -l /tmp/.X11-unix/
```

正常情况下至少应满足一种：

```text
DISPLAY=:0 或 :1，并存在对应 /tmp/.X11-unix/X0 或 X1
```

或者：

```text
WAYLAND_DISPLAY=wayland-0
```

### 4.1 推荐：查看 frozen PPO teacher

下面这条命令可直接复制执行：

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint && \
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --num_envs=1 \
  --checkpoint=logs/rl_games/quadcopter_ship_landing_physical_deck_attitude/expanded_from_physical_deck_ep990_16to22.pth
```

### 4.2 查看 BC-only

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint && \
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --num_envs=1 \
  --checkpoint=logs/imitation/behavior_cloning/bc_init_rlgames.pth
```

### 4.3 查看 BC+PPO 主实验 seed 42

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint && \
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --num_envs=1 \
  --checkpoint=logs/rl_games/bc_ppo/seed42/nn/bc_ppo.pth
```

### 4.4 查看 PPO-from-scratch seed 42

```bash
cd /home/j/Isaac_RL_Projects/quadcopter_waypoint && \
PYTHONPATH=source/quadcopter_waypoint \
/home/j/anaconda3/envs/env_isaaclab/bin/python scripts/rl_games/play.py \
  --task=Isaac-Quadcopter-ShipLanding-PhysicalDeckAttitude-Direct-v0 \
  --num_envs=1 \
  --checkpoint=logs/rl_games/ppo_scratch/seed42/nn/ppo_scratch.pth
```

四条命令都不要添加 `--headless`。窗口打开后可观察无人机接近、对准、下降和接触甲板的过程；关闭 Isaac Sim 窗口或在终端按 `Ctrl+C` 可结束运行。

### 4.5 只检查显示环境，不启动 Isaac Sim

```bash
printf 'DISPLAY=%s\nWAYLAND_DISPLAY=%s\nXDG_SESSION_TYPE=%s\n' \
  "$DISPLAY" "$WAYLAND_DISPLAY" "$XDG_SESSION_TYPE"
ls -l /tmp/.X11-unix/
```

如果桌面终端中 `DISPLAY` 正常，但仍提示无权限，可检查：

```bash
xhost
```

只为当前本地用户授权可使用：

```bash
xhost +SI:localuser:$USER
```

不建议使用宽泛的 `xhost +`。

## 5. tmux 中恢复图形变量

若 tmux 是从无图形 shell 中创建的，即使后来登录桌面，旧会话也可能保持空变量。可以在桌面终端中获取变量：

```bash
echo "$DISPLAY"
echo "$XAUTHORITY"
echo "$WAYLAND_DISPLAY"
```

再在 tmux 内设置对应值，或者更稳妥地从桌面终端重新创建 tmux 会话：

```bash
tmux new -s isaac_gui
```

应同时确认对应的 X11 socket 实际存在，不能只复制变量字符串。

## 6. SSH 场景

### 6.1 X11 转发

客户端可尝试：

```bash
ssh -Y user@host
```

登录后检查：

```bash
echo "$DISPLAY"
```

通常会得到类似：

```text
localhost:10.0
```

不过 Isaac Sim GUI 图形负载较高，传统 X11 forwarding 通常性能较差，也可能遇到 OpenGL/Vulkan 限制。更实际的方案通常是远程桌面、VNC、NoMachine 或直接在主机桌面会话中运行。

### 6.2 普通 SSH

普通

```bash
ssh user@host
```

默认不会提供图形 display。此时 `DISPLAY` 为空属于正常现象，适合执行 `--headless` 训练和评估。

## 7. Docker 场景

容器需要显式传递显示变量并挂载 X11 socket，例如：

```bash
docker run --rm -it \
  --gpus all \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY="$XAUTHORITY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$XAUTHORITY:$XAUTHORITY:ro" \
  <IMAGE>
```

前提是宿主机本身已有有效图形会话。宿主机 `DISPLAY` 为空时，传进容器也没有意义。

## 8. GUI、Headless 与 MP4 的区别

需要区分三个概念：

1. **交互 GUI**：需要可用的 X11/Wayland display；当前会话无法完成。
2. **Headless 数值仿真**：不需要 display；当前训练和正式评估均属于此类，结果有效。
3. **Headless 离屏渲染/视频录制**：理论上不一定需要桌面 display，但需要启用渲染经验文件、相机或 viewport、视频 wrapper/编码器，并编写对应录制流程。

当前 `scripts/imitation/record_rollout_case.py` 只保存数值状态—动作轨迹，不包含离屏视频录制实现。因此“没有默认 display”只说明不能打开交互窗口；它不等价于 Isaac Sim 永远无法在 headless 模式生成 MP4。若后续需要视频，应单独实现和验证 render-enabled recorder。

## 9. 当前可复核结论

- GPU 驱动和 Vulkan 已被 Isaac Sim 正常识别；
- 当前 shell 没有继承任何桌面 display；
- `--headless` 实验不受该问题影响；
- 当前未完成的是人工 GUI 目视验收，不是数值评估；
- 设置一个虚假的 `DISPLAY=:0` 不能解决问题；
- 最直接的 GUI 运行方式是在已登录 Ubuntu 桌面的终端中执行不带 `--headless` 的播放命令。
