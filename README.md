# 能绕过限制多开，但需要一些技巧，请耐心看完。
# Inference（原 Kuzco）Epoch 2 最新一键自动重启 + 多开脚本

**作者：J1N，KuzCommunityCN Founder**  
**推特：[ @J1N226 ](https://twitter.com/J1N226)**（关注推特私信拉交流群）

---

## 📌 项目简介

该脚本为 Inference Epoch 2 提供了 **一键部署、自动重启和多开** 的完整方案。  
基于 [Singosol 原项目](https://github.com/singosol/kuzco-docker) 进行 Fork，并根据实测对官方 Epoch 2 的更新进行了适配与优化。

主要优化内容包括：
- 自动重启
- 根据显存自动多开运行
- 支持单卡多开、多卡多开
- 脚本内含详细注释，方便根据不同设备灵活调整参数

---

## ✅ 环境要求

- **操作系统**：Linux  
- **显卡驱动**：NVIDIA 550  
- **显存要求**：显存大于 6GB（详见[官方文档](https://docs.inference.supply/hardware)）  
- **Python**：3.8 及以上版本
- **Docker**：最新版

---

## 🚀 脚本功能

- 🔁 **自动重启**：无需人工干预，持续稳定运行  
- 🧩 **多开支持**：根据显存自动分配线程数量  
  - 例如：6GB 显存 = 单开，12GB 显存 = 双开，以此类推

---

## 🛠️ 使用方法

在终端输入以下指令运行脚本： 

```bash
python3 kzco.py -c "--worker XXX --code XXX"
```

> 参数 `"--worker XXX --code XXX"` 来自官方 Inference 平台创建 Worker 时选择 Docker 方式所自动生成的命令。

---

## 📖 注意事项

- ✅ **支持单卡多开和多卡多开**
- 当你**创建一个 Worker 并运行一次脚本时**，你机器上的**每张显卡将单开**
- 如果你**再创建一个新的 Worker 并再次运行脚本**，每张显卡将会**双开**
- 通过重复创建 Worker 和运行脚本，可以逐步实现每张卡的多开

### 📊 举例说明：

| 显卡显存 | 每张卡最大进程数 | 所需 Worker 数 |
|----------|------------------|----------------|
| 12GB     | 2                | 2              |
| 18GB     | 3                | 3              |
| 24GB     | 4                | 4              |

> 💡 每创建一个新的 Worker 并再次运行脚本，系统就会为每张显卡会多开一个进程


# Inference (formerly Kuzco) Epoch 2 One-Click Auto-Restart & Multi-Instance Script

**Author: J1N, Founder of KuzCommunityCN**  
**Twitter: [@J1N226](https://twitter.com/J1N226)** 

---

## 📌 Overview

This script provides a **complete solution for deploying Inference Epoch 2** with one-click setup, **automatic restarts**, and **multi-instance execution**.  
It is a **fork of the [original Singosol project](https://github.com/singosol/kuzco-docker)** and has been optimized based on real-world testing to support the latest Epoch 2 updates.

### Key Improvements:
- Auto-Restart
- Multi-instance execution based on available memory  
- Fully commented for easy customization based on your hardware setup
- Supports multi-instance on a single GPU and across multiple GPUs

---

## ✅ Requirements

- **Operating System**: Linux  
- **GPU Driver**: NVIDIA version 550  
- **GPU Memory**: More than 6GB (see [official hardware documentation](https://docs.inference.supply/hardware))  
- **Python**: Version 3.8 or higher  
- **Docker**: Latest version

---

## 🚀 Features

- 🔁 **Auto-Restart** – Runs continuously without manual intervention  
- 🧩 **Multi-Instance Support** – Automatically allocates threads based on available VRAM  
  - Example: 6GB = 1 thread, 12GB = 2 threads, and so on

---

## 🛠️ How to Use

Run the following command in your terminal:

```bash
python3 kzco.py -c "--worker XXX --code XXX"
```

> Replace `"--worker XXX --code XXX"` with the command generated by the **official Inference platform** when creating a new worker using the **Docker** option.

---

## 📖 Notes

- ✅ **Supports both single-GPU multi-instance and multi-GPU setups**
- When you **create one worker** and run the script once, **each GPU on your machine will launch one process**.
- When you **create a second worker** and run the script again, **each GPU will then run two processes**.
- Repeat this process to scale up the number of instances per GPU.

### 📊 Example:

| GPU VRAM | Max Instances | Required Workers |
|----------|---------------|------------------|
| 12GB     | 2             | 2                |
| 18GB     | 3             | 3                |
| 24GB     | 4             | 4                |

> 💡 Each time you create a new worker and run the script again, the system adds another process per GPU.

