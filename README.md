

# IGN_ANO

paper code for IGN

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/2679373161/IGN_ANO">
    <img src="figs/logo.jpg"Logo" width="80" height="80">
  </a>

  <h3 align="center">基于深度特征重建的缺陷检测算法方法研究</h3>
  <p align="center">
    基于IGN、AE、SAM等模型思想，对样本特征空间进行修复！
    <br />
    <a href="https://github.com/2679373161/IGN_ANO"><strong>研究详情 »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/2679373161/IGN_ANO">查看Demo</a>
    · -->
    <a href="https://github.com/2679373161/IGN_ANO/issues">报告Bug</a>
    ·
    <a href="https://github.com/2679373161/IGN_ANO/issues">提出Bug</a>
  </p>

</p>
 
## 目录

- [IGN\_ANO](#ign_ano)
  - [目录](#目录)
    - [核心思路](#核心思路)
          - [开发前的配置要求](#开发前的配置要求)
          - [**安装步骤**](#安装步骤)
    - [部署](#部署)
    - [使用到的框架](#使用到的框架)
    - [版本控制](#版本控制)
    - [作者](#作者)

### 核心思路

- 基于IGN模型的双阶段重建模型
- 基于SAM模型的特征提取重建模型
- 基于大语言模型的zero-shot模型研究



###### 开发前的配置要求

1. 运行环境 win/linux
2. cuda11.8
3. pytorch2.0

###### **安装步骤**

1. Clone the repo
```sh
git clone https://github.com/2679373161/IGN_ANO
```
2. pip install -r requirements.txt
3. [Ano_Create.ipynb](Ano_Create.ipynb)使用IGN网络生成局部缺陷；
   [demo_SAM_only.ipynb](demo_SAM_only.ipynb)使用SAM作为特征提取器重建图像；
   [SAM_IGN.ipynb](SAM_IGN.ipynb)使用SAM作为特征提取器结合IGN多次重建图像




### 部署

暂无

### 使用到的框架

- [pytorch](https://pytorch.org/)




### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

### 作者

2679373161@qq.com

CSDN:[爱神的箭呵呵](https://blog.csdn.net/qq_44646352?spm=1000.2115.3001.5343)  &ensp; qq:2679373161    



<!-- links -->
[your-project-path]:2679373161/IGN_ANO
[contributors-shield]: https://img.shields.io/github/contributors/2679373161/IGN_ANO.svg?style=flat-square
[contributors-url]: https://github.com/2679373161/IGN_ANO/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/2679373161/IGN_ANO.svg?style=flat-square
[forks-url]: https://github.com/2679373161/IGN_ANO/network/members
[stars-shield]: https://img.shields.io/github/stars/2679373161/IGN_ANO.svg?style=flat-square
[stars-url]: https://github.com/2679373161/IGN_ANO/stargazers
[issues-shield]: https://img.shields.io/github/issues/2679373161/IGN_ANO.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/2679373161/IGN_ANO.svg
[license-shield]: https://img.shields.io/github/license/2679373161/IGN_ANO.svg?style=flat-square
[license-url]: https://github.com/2679373161/IGN_ANO/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian




