# Inference（原 Kuzco）Epoch 2 最新一键自动重启 + 多开脚本

**作者：J1N，KuzCommunityCN Founder**  
**推特：[ @J1N226 ](https://twitter.com/J1N226)**（关注推特私信拉交流群）

---

## 📌 项目简介

该脚本为 Inference Epoch 2 提供了 **一键部署、自动重启和多实例运行** 的完整方案。  
基于 [Singosol 原项目](https://github.com/singosol/kuzco-docker) 进行 Fork，并根据实测对官方 Epoch 2 的更新进行了适配与优化。

主要优化内容包括：
- 自动识别 GPU 显存
- 根据显存自动多开运行
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
  - 例如：6GB 显存 = 1 线程，12GB 显存 = 2 线程，以此类推

---

## 🛠️ 使用方法

在终端输入以下指令运行脚本：

```bash
python3 kzco.py -c "--worker XXX --code XXX"
```

> 参数 `"--worker XXX --code XXX"` 来自官方 Inference 平台创建 Worker 时选择 Docker 方式所自动生成的命令。

当然可以，以下是你修改后的说明的英文翻译版，格式和风格保持专业清晰，适合直接用于 GitHub README：

---

# Inference (formerly Kuzco) Epoch 2 One-Click Auto-Restart & Multi-Instance Script

**Author: J1N, Founder of KuzCommunityCN**  
**Twitter: [@J1N226](https://twitter.com/J1N226)** 

---

## 📌 Overview

This script provides a **complete solution for deploying Inference Epoch 2** with one-click setup, **automatic restarts**, and **multi-instance execution**.  
It is a **fork of the [original Singosol project](https://github.com/singosol/kuzco-docker)** and has been optimized based on real-world testing to support the latest Epoch 2 updates.

### Key Improvements:
- Automatic detection of GPU VRAM  
- Multi-instance execution based on available memory  
- Fully commented for easy customization based on your hardware setup

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
