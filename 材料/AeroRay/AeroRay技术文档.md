# AeroRay 技术文档

> **AeroRay · 大模型驱动的低空电磁智能孪生平台**
> 用语言驱动仿真,以对话洞察信道,以多基站协同低空

---

## 目录

- [一、项目概述](#一项目概述)
- [二、系统架构总览](#二系统架构总览)
- [三、数据获取层（C++ / Python）](#三数据获取层c--python)
  - [3.1 地理数据获取（Python）](#31-地理数据获取python)
  - [3.2 SHP → ODB 格式转换（C++）](#32-shp--odb-格式转换c)
  - [3.3 射线追踪仿真引擎（C++）](#33-射线追踪仿真引擎c)
- [四、Agent 架构层（Python）](#四agent-架构层python)
  - [4.1 LLM 集成与多模型支持](#41-llm-集成与多模型支持)
  - [4.2 ReAct Agent 与工具调用](#42-react-agent-与工具调用)
  - [4.3 Skill 编排系统](#43-skill-编排系统)
  - [4.4 WebSocket 实时通信协议](#44-websocket-实时通信协议)
  - [4.5 会话管理与状态持久化](#45-会话管理与状态持久化)
  - [4.6 System Prompt 工程](#46-system-prompt-工程)
- [五、前端展示层（React + Three.js）](#五前端展示层react--threejs)
  - [5.1 前端架构与组件树](#51-前端架构与组件树)
  - [5.2 Three.js 3D 场景渲染](#52-threejs-3d-场景渲染)
  - [5.3 AI 对话交互界面](#53-ai-对话交互界面)
  - [5.4 覆盖分析与 MIMO 分析面板](#54-覆盖分析与-mimo-分析面板)
  - [5.5 射线可视化与交互检查](#55-射线可视化与交互检查)
  - [5.6 设计语言与样式系统](#56-设计语言与样式系统)
- [六、数据流与协议详解](#六数据流与协议详解)
  - [6.1 端到端数据流水线](#61-端到端数据流水线)
  - [6.2 坐标系统与变换](#62-坐标系统与变换)
  - [6.3 进程间通信（Python ↔ C++）](#63-进程间通信python--c)
  - [6.4 文件格式与中间产物](#64-文件格式与中间产物)
- [七、核心算法详解](#七核心算法详解)
  - [7.1 WinProp DPM 主导路径模型](#71-winprop-dpm-主导路径模型)
  - [7.2 WinProp IRT 智能射线追踪](#72-winprop-irt-智能射线追踪)
  - [7.3 MIMO 信道矩阵分析（SVD）](#73-mimo-信道矩阵分析svd)
  - [7.4 覆盖统计分析](#74-覆盖统计分析)
- [八、低空电磁智能孪生子系统(平台核心)](#八低空电磁智能孪生子系统平台核心)
  - [8.1 子系统目标与定位](#81-子系统目标与定位)
  - [8.2 端到端工作流](#82-端到端工作流)
  - [8.3 多高度 DPM 仿真与场聚合](#83-多高度-dpm-仿真与场聚合)
  - [8.4 三维 A* 路径规划算法](#84-三维-a-路径规划算法)
  - [8.5 前端体素渲染与无人机动画](#85-前端体素渲染与无人机动画)
  - [8.6 WebSocket 协议扩展](#86-websocket-协议扩展)
- [九、完整技术栈清单](#九完整技术栈清单)
- [十、项目目录结构](#十项目目录结构)
- [十一、部署与运行](#十一部署与运行)

---

## 一、项目概述

AeroRay 是一个**基于大语言模型(LLM)驱动的低空电磁智能孪生平台**。
用户通过自然语言对话,即可完成从地理数据提取、建筑场景建模、多基站三维射线仿真、
干扰态势分析,到无人机/蜂群航线规划的全链路操作,结果以实时 3D 体素与动画渲染在浏览器中。

**核心能力(🚁 低空电磁智能孪生 — 平台主战场):**
- **🚁 低空多基站三维仿真**:N 个基站 × M 个高度层循环执行 DPM,聚合为完整的三维信号场,
  per-BS 路径损耗张量供实时干扰分析使用
- **📡 干扰态势 4 模式**:RSRP / 主导基站 / SINR / 重叠覆盖,前端从 per-BS 张量实时计算与切换
- **✈️ 多基站感知航线规划**:把 A* 状态空间从 `(x,y,z)` 扩展到 `(x,y,z,k)`,
  算法自动决定何时切换基站,产出"分段着色"的航线
- **🐝 蜂群协同规划**:1~10 架无人机共享同一起终点,通过强制每架初始连接不同基站来产生多样化路径,
  飞行动画中每架无人机用一条白线连到当前服务基站
- **自然语言 → 结构化参数**:LLM Agent 理解用户意图,自动编排 10 个工具完成多步骤任务
- **实时 3D 可视化**:Three.js 渲染建筑、多基站塔、3D 信号场体素、分段着色航线、蜂群无人机动画

**扩展能力(🔬 地面单基站精细分析 — 辅助工具):**
- **专业 RF 仿真**:集成 Altair WinProp 引擎,地面单基站支持 DPM / IRT 两种传播模型
- **射线交互检查**:点击 RadioMap 任意点查看主导射线及反射/衍射点
- **MIMO 信道分析**:4×4 天线阵列仿真 + SVD 分解 + 信道容量 / 秩 / 条件数可视化
- **覆盖统计**:Impulse Response / PDF / CDF / Threshold 四种视图

**技术定位:** 把传统需要 GUI 操作的无线网络规划仿真工具(如 WinProp)的能力,
通过 AI Agent 包装为自然语言交互界面,**以"低空多基站协同 + 蜂群航线"作为核心应用场景**,
对接低空经济需求。地面单基站精细仿真作为研究工具的扩展能力保留。

---

## 二、系统架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           用户浏览器 (Frontend)                          │
│   React 18 + Three.js + WebSocket                                        │
│   ┌─────────────┐ ┌────────────────────┐ ┌────────────────────────────┐  │
│   │ FeaturePanel│ │   ThreeScene (3D)  │ │       ChatPanel            │  │
│   │ (核心 / 扩展│ │  · 建筑 + 多基站塔 │ │  · 流式对话                │  │
│   │  功能 + 区域│ │  · 干扰热力图(4种)  │ │  · 工具调用追踪           │  │
│   │  坐标输入)  │ │  · per-BS 3D 信号场│ │  · 多会话管理              │  │
│   │             │ │  · 蜂群无人机动画  │ │  · Markdown 渲染           │  │
│   └─────────────┘ │  · 动态 BS 连接线  │ └────────────────────────────┘  │
│                   │  + Overlay 浮窗:    │                                 │
│                   │  · MultiBSPanel 🌟 │                                 │
│                   │  · InterferenceOverlay 🌟                            │
│                   │  · UAVPlanPanel 🌟 (单/多机统一)                     │
│                   │  · ParamsPanel(地面单基站)                            │
│                   │  · CoverageOverlay / MimoOverlay / RayInfoPanel       │
│                   │  · TopDownView                                        │
│                   └────────────────────┘                                  │
├────────────────────────── WebSocket / REST ───────────────────────────────┤
│                         FastAPI 后端 (server.py)                          │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │              LangGraph ReAct Agent                                │    │
│   │   ┌─────────┐    ┌───────────┐    ┌──────────┐                   │    │
│   │   │ LLM 推理 │←──│ 工具执行  │←──│ 状态记忆  │                   │    │
│   │   │(Claude/  │──→│(11 Tools) │──→│(Checkpoint│                   │    │
│   │   │ MiniMax) │    └───────────┘    │  Memory) │                   │    │
│   │   └─────────┘                      └──────────┘                   │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│   ┌─────────────────────── Tool 层(LLM 可见 11 个) ────────────────┐    │
│   │ 🌟 prepare_multi_bs (低空主入口,默认 mode='low_alt')            │    │
│   │ 地理:    resolve_area / extract_buildings / export_odb         │    │
│   │           visualize_scene                                        │    │
│   │ 地面扩展: prepare_ray_tracing / execute_ray_tracing             │    │
│   │           visualize_radiomap / analyze_coverage / analyze_mimo  │    │
│   │ 内部(WS 直接驱动,不暴露给 LLM):                                 │    │
│   │   execute_multi_bs / plan_multi_bs_route / plan_swarm_routes    │    │
│   └──────────────────────────────────────────────────────────────────┘    │
├─────────────── subprocess 调用 ───────────────────────────────────────────┤
│                         C++ 仿真引擎 (WinProp SDK)                        │
│   ┌────────────────────────┐    ┌───────────────────────────────────┐    │
│   │  SHP2ODB_debug.exe     │    │ PropagationRunMSMIMOSingle.exe    │    │
│   │  (Shapefile → ODB/OIB) │    │ (DPM/IRT 传播预测 + MIMO RunMS)   │    │
│   └────────────────────────┘    └───────────────────────────────────┘    │
├───────────────────────────────────────────────────────────────────────────┤
│                         外部数据源 & API                                  │
│   · OpenStreetMap Nominatim  (地名 → 坐标)                                │
│   · OpenStreetMap Overpass   (建筑几何数据)                               │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 三、数据获取层（C++ / Python）

### 3.1 地理数据获取（Python）

地理数据获取是整个仿真流水线的第一步，由 Python Skill 层负责，分为**区域解析**和**建筑提取**两个阶段。

#### 3.1.1 区域解析（resolve_area）

**文件：** `skill/find_area/skill_find_area.py`

当用户通过自然语言给出目标区域（如"西安电子科技大学南校区"），系统首先需要将其转化为精确的地理坐标边界。

**解析流程：**

1. **自然语言 → 坐标**：调用 OpenStreetMap Nominatim API 进行地理编码
   ```
   GET https://nominatim.openstreetmap.org/search
   ?q=西安电子科技大学南校区
   &format=json
   &polygon_geojson=1
   ```

2. **手动坐标输入**：如果用户直接提供了坐标（支持多种格式），通过 `parse_coordinate_text()` 进行容错解析：
   - `[[108.91, 34.23], [108.92, 34.24], ...]` — 标准 JSON 格式
   - `(108.91, 34.23), (108.92, 34.24)` — 元组格式
   - `108.91 34.23 108.92 34.24` — 纯数字空格分隔
   - `polygon_coords = [[...]]` — 带赋值的格式

3. **坐标规范化**：
   - `normalize_coordinate_order()`：自动判断经纬度顺序（通过范围检测 -180~180 vs -90~90）
   - `normalize_polygon_vertices()`：按质心角度排序，统一为逆时针方向
   - `ensure_closed_polygon()`：确保多边形首尾闭合

4. **输出**：创建时间戳目录 `resu/YYYYMMDD_HHMMSS/`，保存 `polygon.json`

#### 3.1.2 建筑数据提取（extract_buildings）

**文件：** `skill/__init__.py` 中的 `extract_buildings()` 编排函数（第268–431行）

这是一个复合编排函数，内部串联 5 个子 Skill 完成完整的建筑提取流水线：

```
extract_buildings()
  ├─ [数据源选择]
  │   ├─ force_online=True  → Overpass API 在线获取
  │   └─ force_online=False → 本地 Shapefile 匹配
  │       ├─ infer_shp_dir() → 根据区域名/多边形自动匹配本地 SHP 目录
  │       ├─ find_target_shp() → 在 SHP 目录中定位目标文件
  │       └─ clip_geometries() → 按用户多边形精确裁剪
  │
  ├─ [步骤1] 获取原始建筑数据 → buildings[] (id, polygon, height, tags)
  │
  ├─ [步骤2] GeoDataFrame 构建 + 二次裁剪
  │   ├─ gpd.GeoDataFrame(geometries, heights, crs="EPSG:4326")
  │   └─ clip_geometries(gdf, target_polygon) → 精确裁剪
  │
  ├─ [步骤3] 多边形提取 + 简化 + 去重叠
  │   ├─ extract_polygons_and_heights(tolerance=1e-5) → 简化几何
  │   └─ remove_overlaps(threshold=1e-10) → IoU 去重
  │
  ├─ [步骤4] 像素坐标映射 (shp_to_json)
  │   ├─ calc_pixel_params(bounds) → 计算投影参数
  │   ├─ save_json_outputs() → 生成 buildings_with_height.json
  │   └─ 保存 area_meta.json (场景尺寸元信息)
  │
  └─ [步骤5] 3D Shapefile 生成 (json_to_shp)
      └─ json_to_shp() → 生成含 Z 值的 result_final.shp
```

**Overpass API 查询（在线模式）：**

```python
# skill/common.py: fetch_buildings_from_overpass()
query = f"""
[out:json][timeout:180];
(
  way["building"]({south},{west},{north},{east});
);
out geom tags;
"""
```

查询通过 3 个 Overpass 镜像端点进行故障转移：
- `https://overpass-api.de/api/interpreter`（主）
- `https://lz4.overpass-api.de/api/interpreter`（备）
- `https://overpass.kumi.systems/api/interpreter`（备）

**高度推断逻辑：**

```python
# skill/common.py: parse_height()
def parse_height(tags):
    # 优先级: height 标签 > building:levels × 3m > 默认 20m
    if tags.get("height"):         return max(3.0, float(height))
    if tags.get("building:levels"): return max(3.0, float(levels) * 3.0)
    return 20.0
```

**像素坐标投影：**

系统使用各向同性投影，将 WGS84 经纬度映射到以米为单位的局部像素坐标：

```python
# skill/common.py: calc_pixel_params()
scale = 111320.0 * cos(radians(mid_lat))  # 东西向: 1像素 = 1米
width  = round(lon_range * scale)          # 像素宽度
height = round(lat_range * scale)          # 像素高度
```

- 东西方向精确 1px = 1m
- 南北方向 1px ≈ 1/cos(lat) m（纬度 34° 处约 1.21m）
- 这种设计保证仿真网格与真实距离一致

**本地 Shapefile 支持：**

项目内置了陕西省和郑州市的行政区划 Shapefile 数据（`skill/find_shp/shanxisheng/`、`skill/find_shp/zhengzhoushi/`），支持离线环境下的建筑数据提取。`infer_shp_dir()` 通过区域名模糊匹配和空间包含关系自动定位合适的 SHP 目录。

---

### 3.2 SHP → ODB 格式转换（C++）

**源文件：** `skill/create_odb/convert_database.cpp`（433行）
**编译产物：** `SHP2ODB_debug.exe`（648 KB）
**依赖 SDK：** Altair WinProp（头文件 `convert_database.h`、`IRT_preprocess_urban.h`）

ODB（Outdoor Database）是 WinProp 射线追踪引擎的专有建筑数据库格式。本程序负责将标准 Shapefile 转换为 ODB，并在 IRT 模式下执行额外的预处理生成 OIB 文件。

#### 3.2.1 转换核心流程

```
Shapefile (.shp)
    │
    ▼
[SHP2ODB_debug.exe]
    │
    ├── DPM 模式 (--mode dpm)
    │   └── WinProp_Convert() → .odb 文件（二进制建筑数据库）
    │
    └── IRT 模式 (--mode irt)
        ├── WinProp_Convert() → .odb 文件
        └── OutdoorPlugIn_ComputePrePro() → .oib 文件（IRT 预处理数据库）
```

#### 3.2.2 SHP → ODB 转换（DPM / IRT 共用）

```cpp
// convert_database.cpp 第258-285行
WinProp_Converter WinPropConverter;
WinProp_Structure_Init_Converter(&WinPropConverter);

WinPropConverter.measurementUnit = "Meter";       // 使用米制单位
WinPropConverter.ConverterID     = 205;            // SHP→ODB 转换器 ID
WinPropConverter.SaveAsciiFormat = 0;              // 输出二进制格式（更小、更快）
WinPropConverter.BuildingHeightMode = 1;           // 从 z-value 属性读取建筑高度

WinPropConverter.databaseNameSource = source_path.c_str();
WinPropConverter.databaseNameDest   = dest_path.c_str();

WinProp_Convert(&WinPropConverter, &Callback);     // 执行转换
```

**关键回调函数：**

- `CallbackAutoCADLayerSelection()`：当 SHP 包含多图层时，自动选择所有图层参与转换
- `HeightTest()`：在 SHP 属性表中查找 `z-value` 字段作为建筑高度来源

```cpp
// convert_database.cpp 第49-80行
int HeightTest(const char* const* const properties,
    const int nrProperties,
    int* const heightPropertyIndex, ...)
{
    for (int i = 0; i < nrProperties; i++) {
        if (strcmp(properties[i], "z-value") == 0) {
            *heightPropertyIndex = i;  // 找到高度属性列
        }
    }
    return 0;
}
```

#### 3.2.3 IRT 预处理（仅 IRT 模式）

IRT（Intelligent Ray Tracing）模式需要对建筑场景进行预处理，将 3D 建筑几何离散化为适合射线追踪的数据结构。**预处理阶段的分辨率一旦确定就不可更改**，这是 IRT 与 DPM 的关键区别之一。

```cpp
// convert_database.cpp 第290-390行
WinProp_PreProUrban PreproPara;
WinProp_Structure_Init_PreProUrban(&PreproPara);

PreproPara.Model               = PREDMODEL_IRT;
PreproPara.Mode                = PP_MODE_IRT_3D;
PreproPara.Resolution          = irt_resolution;     // 用户指定, 默认 10m
PreproPara.Heights[0]          = irt_height;          // 预测高度, 默认 1.5m
PreproPara.SegmentHorizontal   = irt_segment_h;       // 空间离散化, 默认 100m
PreproPara.SegmentVertical     = irt_segment_v;
PreproPara.TileHorizontal      = irt_tile_h;          // 瓦片大小, 默认 100m
PreproPara.TileVertical         = irt_tile_v;
PreproPara.MultiThreading      = irt_multi_thread;     // 多线程, 默认 2
PreproPara.ConsiderTopography  = irt_consider_topo;    // 可选地形
PreproPara.MultipleInteractions = 1;                   // 启用多次交互

OutdoorPlugIn_ComputePrePro(&PreproPara, irt_output.c_str(), ...);
```

**Python 调用入口：**

```python
# skill/create_odb/skill_create_odb.py
subprocess.run([
    exe_path,
    "--shp_file", input_shp,
    "--output", output_base,
    "--mode", "irt",              # 或 "dpm"
    "--irt_resolution", "10.0",   # IRT 分辨率
], capture_output=True, text=True)
```

---

### 3.3 射线追踪仿真引擎（C++）

**源文件：** `skill/single_dpm_mimo/outdoor_propagation_runms_mimo_single.cpp`（629行）
**编译产物：** `PropagationRunMSMIMOSingle_debug.exe`
**依赖 SDK：** Altair WinProp（头文件 `outdoor_propagation_runms_mimo_single.h`）

这是整个系统的计算核心，分为两个阶段执行：

#### 3.3.1 第一阶段：室外传播预测

根据传播模型选择，加载不同的场景数据库并执行电磁波传播计算。

**通用配置：**

```cpp
// outdoor_propagation_runms_mimo_single.cpp 第306-360行
WinProp_ParaMain GeneralParameters;
GeneralParameters.ScenarioMode     = SCENARIOMODE_URBAN;        // 城市场景
GeneralParameters.Resolution       = resolution;                 // 默认 0.5m
GeneralParameters.NrLayers         = 1;                          // 单层预测
GeneralParameters.PredictionHeights = &prediction_height;        // 默认 1.5m

// 仿真区域边界
GeneralParameters.UrbanLowerLeftX  = lower_left_x;
GeneralParameters.UrbanLowerLeftY  = lower_left_y;
GeneralParameters.UrbanUpperRightX = upper_right_x;
GeneralParameters.UrbanUpperRightY = upper_right_y;
```

**天线配置：**

```cpp
// 第365-378行
WinProp_Antenna Antenna;
Antenna.Height      = antenna_height;      // 默认 15m
Antenna.Power       = antenna_power;       // 默认 23 dBm
Antenna.Frequency   = antenna_freq;        // 默认 3500 MHz (5G NR n78)
Antenna.Longitude_X = antenna_x;           // 天线 X 坐标（米）
Antenna.Latitude_Y  = antenna_y;           // 天线 Y 坐标（米）
Antenna.Azimuth     = antenna_azimuth;     // 方位角（度），默认 0°
Antenna.Downtilt    = antenna_downtilt;    // 下倾角（度），默认 80°
// 0°=垂直向下, 90°=水平, 180°=垂直向上

// 加载天线方向图文件
WinProp_Pattern antennaPattern;
antennaPattern.Mode = PATTERN_MODE_FILE;
sprintf(antennaPattern.Filename, "%s", pattern_file.c_str());
Antenna.Pattern = &antennaPattern;
```

**DPM 模式（Dominant Path Model）：**

```cpp
// 第344-348行
GeneralParameters.PredictionModelUrban = PREDMODEL_UDP;        // DPM 模型
GeneralParameters.BuildingsMode        = BUILDINGSMODE_BINARY;  // 加载 .odb

// 第435-438行
OutdoorPlugIn_ComputePrediction(
    &Antenna, &GeneralParameters,
    NULL, 0, NULL, NULL, NULL, NULL,   // 无 IRT 额外参数
    &Callback, &Resultmatrix, &RayMatrix, NULL, NULL);
```

DPM 是一种简化的传播模型，基于主导传播路径（直射、反射或衍射中损耗最小的路径）来估算信号强度。计算速度快，适合大范围覆盖预测。

**IRT 模式（Intelligent Ray Tracing）：**

```cpp
// 第325-341行
GeneralParameters.PredictionModelUrban = PREDMODEL_IRT;
GeneralParameters.BuildingsMode        = BUILDINGSMODE_IRT;     // 加载 .oib

// IRT 路径参数
GeneralParameters.MaxPathLoss       = irt_max_path_loss;  // 200 dB
GeneralParameters.MaxNumberPaths    = irt_max_num_paths;   // 20

// 第406-432行
Model_UrbanIRT ParameterIRT;
ParameterIRT.MaxReflections          = irt_max_reflections;     // 默认 2
ParameterIRT.MaxDiffractions         = irt_max_diffractions;    // 默认 1
ParameterIRT.MaxScatterings          = irt_max_scatterings;     // 默认 0
ParameterIRT.MaxSumReflDiff          = irt_max_sum_refl_diff;   // 默认 2
ParameterIRT.BreakpointExponentBeforeLOS = 2.3;   // LOS 断点前衰减指数
ParameterIRT.BreakpointExponentAfterLOS  = 3.3;   // LOS 断点后衰减指数
ParameterIRT.DiffractionModel        = 'e';         // 衍射模型

OutdoorPlugIn_ComputePrediction(
    &Antenna, &GeneralParameters,
    NULL, 0, &ParameterIRT, NULL, NULL, NULL,   // 传入 IRT 参数
    &Callback, &Resultmatrix, &RayMatrix, NULL, NULL);
```

IRT 是基于物理的射线追踪，模拟电磁波在建筑间的反射、衍射、散射等多径效应。计算精度高但速度慢，适合小范围精细分析。

**输出控制：**

```cpp
// 第382-391行
WinProp_Propagation_Results OutputResults;
OutputResults.FieldStrength          = out_field_strength;     // 场强
OutputResults.PathLoss               = out_path_loss;          // 路径损耗
OutputResults.StatusLOS              = out_status_los;         // LOS/NLOS 状态
OutputResults.RayFilePropPaths       = out_ray_file;           // 射线路径（.ray）
OutputResults.StrFilePropPaths       = out_str_file;           // 射线路径（.str）
OutputResults.AdditionalResultsASCII = out_additional_ascii;   // ASCII 结果文件
```

#### 3.3.2 第二阶段：MIMO RunMS 信道矩阵计算

当用户启用 MIMO 模式时，在传播预测完成后自动执行 MIMO 信道矩阵计算。

**天线阵列配置：**

```cpp
// outdoor_propagation_runms_mimo_single.cpp 第452-500行
WinProp_SuperposeMS superposeMS;

// 发射阵列 (TX)
superposeMS.setArray(false, txArrayAPI,
    num_tx_antennas,    // 默认 4 天线
    tx_azimuth,         // 默认 60°
    antenna_spacing,    // 默认 0.5λ
    0.f, coupling_flag);

for (int i = 0; i < num_tx_antennas; i++) {
    superposeMS.setArrayElement(
        false, i, &WinPropPattern, nullptr,
        tx_azimuth,     // 60° 方位角
        tx_tilt,        // 80° 俯仰角
        txPol,          // 交叉极化
        {0., 0., 0.});
}

// 接收阵列 (RX) — 类似配置
superposeMS.setArray(true, rxArrayAPI,
    num_rx_antennas, rx_azimuth, antenna_spacing, 0.f, coupling_flag);
```

**信道计算参数：**

```cpp
// 第507-523行
WinProp_MS_Para msPara;
msPara.coherentSuperposition  = 1;                           // 相干叠加
msPara.channelType            = WINPROP_MS_CHANNEL_PROPAGATION;
msPara.channelBandwidth       = channel_bandwidth;            // 默认 50 MHz
msPara.channelNormalization   = WINPROP_MS_NORMALIZE_TIME_FROBENIUS;
msPara.snirMode               = WINPROP_MS_SNIR_CALC;

// 输出配置
WinProp_MS_AdditionalResults msAdditionalResult;
msAdditionalResult.channelMatricesPerPoint = 1;    // 每点输出信道矩阵
msAdditionalResult.channelMatricesPerRay   = 0;    // 不按射线输出
msAdditionalResult.channelMatrixResultMode = WINPROP_MS_CHANNEL_MATRIX_REAL_IMAG;

superposeMS.compute(msPara, &msAdditionalResult, &Callback);
```

**输出文件：**
- `Antenna_1 Path Loss.txt` — 路径损耗网格数据
- `Antenna_1 Power.fpp` — 功率网格数据
- `runMS/Antenna_1 ChannelMatricesPerPoint.txt` — MIMO 信道矩阵
- `Antenna_1 Propagation Paths.str` — 射线传播路径

**内存管理：**

```cpp
// 第567-570行
WinProp_FreeResult(&Resultmatrix);
WinProp_FreeRayMatrix(&RayMatrix);
```

#### 3.3.3 C++ 编译环境

- **平台：** Windows（使用 `_WIN32`、`windows.h`、`_mkdir`）
- **编译器：** MSVC (Visual Studio)
- **C++ 标准：** C++17（使用 `<filesystem>`）
- **SDK：** Altair WinProp（商业 RF 仿真库，通过 DLL 链接）
- **调用约定：** `__stdcall`（Windows 标准调用约定）

---

## 四、Agent 架构层（Python）

### 4.1 LLM 集成与多模型支持

**文件：** `server.py` 第62-77行、`skill/common.py` 第83-102行、`config.yaml`

系统通过 `config.yaml` 的 profile 机制支持多个 LLM 提供商的快速切换：

```yaml
llm:
  active: guizhou            # 当前激活的 profile 名称
  profiles:
    api123:                  # Anthropic Claude
      base_url: https://api123.icu
      model: claude-sonnet-4-6
      api_type: anthropic
      temperature: 0.2
    guizhou:                 # MiniMax (通过第三方代理)
      base_url: https://gpt-agent.cc
      model: MiniMax-M2.7-highspeed
      api_type: anthropic
      temperature: 0.2
    localqwen:               # 本地部署的 Qwen (通过 ngrok 穿透)
      base_url: https://xxx.ngrok-free.dev
      model: qwen3.5-9b
      api_type: anthropic
```

**模型实例化：**

```python
# server.py 第62-77行
def _build_agent():
    llm_config = get_active_llm_config(APP_CONFIG)
    api_type = str(llm_config.get("api_type", "openai")).strip().lower()

    if api_type == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(api_key=api_key, base_url=base_url,
                            model=model_name, temperature=0)
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(api_key=api_key, base_url=base_url,
                         model=model_name, temperature=0)
```

**设计特点：**
- `temperature=0`：确保推理结果的确定性和一致性
- 统一使用 `api_type: anthropic` 协议：MiniMax 等第三方模型也兼容 Anthropic 的 Messages API 格式
- 支持代理 URL：通过 `base_url` 指向 API 代理服务（如 `api123.icu`、`gpt-agent.cc`），绕过直连限制

### 4.2 ReAct Agent 与工具调用

**核心框架：** LangChain + LangGraph

系统采用 **ReAct（Reasoning + Acting）** 模式构建 Agent。ReAct 的核心思想是让 LLM 交替进行"思考"和"行动"：

```
[用户输入] → LLM 思考要调用哪个工具
           → 调用工具 → 获取结果
           → LLM 再思考是否需要继续调用
           → ... (循环直到任务完成)
           → LLM 生成最终回复
```

**Agent 创建：**

```python
# server.py 第148-153行
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()                              # 内存级会话记忆
tools = get_all_tools()                              # 10 个注册工具
agent = create_react_agent(llm, tools,
                           prompt=system_prompt,     # 147 行系统提示词
                           checkpointer=memory)      # 状态检查点
```

**流式执行：**

```python
# server.py 第404-526行
for msg, metadata in agent_executor.stream(
    {"messages": [("user", user_input)]},
    config=config,                    # 包含 thread_id 用于状态隔离
    stream_mode="messages",           # 逐消息流式输出
):
    if isinstance(msg, AIMessageChunk):
        # 流式文本 token → 前端实时显示
        # 工具调用片段 → 累积直到完整
    elif isinstance(msg, ToolMessage):
        # 工具执行完毕 → 推送结果
        # 拦截 prepare_ray_tracing → 推送参数面板
```

### 4.3 Skill 编排系统

**文件：** `skill/__init__.py`（523行）

Skill 系统是 Agent 和底层功能之间的桥梁，由三部分组成：

#### 4.3.1 Pydantic 输入模型

每个 Skill 都有对应的 Pydantic 模型定义输入参数、类型约束和自然语言描述：

```python
class PrepareRayTracingInput(BaseModel):
    odb_file: str = Field(..., description="已生成的场景数据库文件的绝对路径")
    run_dir: Optional[str] = Field(default=None, description="输出目录")
    pred_model: Optional[str] = Field(default="dpm", description="传播预测模型: dpm 或 irt")
    enable_mimo: Optional[bool] = Field(default=False, description="是否启用 MIMO")
    overrides: Optional[Dict[str, Any]] = Field(default=None, description="参数覆盖字典")

    @field_validator("overrides", mode="before")
    @classmethod
    def _parse_overrides_str(cls, v):
        if isinstance(v, str): return json.loads(v)  # LLM 可能输出字符串格式
        return v
```

LLM 的工具调用（Function Calling）输出会被 Pydantic 模型自动验证和类型转换。`@field_validator` 处理 LLM 有时将 JSON 对象输出为字符串的情况。

#### 4.3.2 工具注册表

当前 `skill/__init__.py` 中注册给 LLM 的工具已更新为 **11 个**。其中多基站低空链路已成为默认主入口；多基站执行、单 UAV 航线规划、蜂群规划等能力仍由 server 在 WebSocket 层直接驱动，**不暴露给 LLM**。

| # | 工具名 | 类型 | 功能 | 关键参数 |
|---|--------|------|------|----------|
| 1 | `resolve_area` | LLM | 地名/坐标 → 标准化范围 + 创建 run_dir | user_request, mode, manual_coords |
| 2 | `extract_buildings` | LLM | 从 OSM/本地 SHP 提取建筑数据 | run_dir, area_name, force_online |
| 3 | `export_odb` | LLM | SHP → ODB/OIB 格式转换 | final_shp_path, pred_model, irt_resolution |
| 4 | `visualize_scene` | LLM | 建筑 → 2D 预览图 + 3D scene_data | run_dir, buildings_json_path |
| 5 | `prepare_multi_bs` | LLM | 平台主入口，多基站低空/平面参数准备 | odb_file, n_bs, mode, res_xy, z_min, z_max, res_z |
| 6 | `visualize_multi_bs` | LLM | 直接加载已有多基站结果并恢复可视化 | run_dir |
| 7 | `prepare_ray_tracing` | LLM | 地面单基站参数面板 | odb_file, pred_model, enable_mimo, overrides |
| 8 | `execute_ray_tracing` | LLM | 执行 WinProp 单基站仿真 | params |
| 9 | `visualize_radiomap` | LLM | Path Loss → 3D 热力图渲染 | run_dir, path_loss_file |
| 10 | `analyze_coverage` | LLM | 激活覆盖分析面板 | run_dir |
| 11 | `analyze_mimo` | LLM | MIMO 信道矩阵 SVD 分析 | run_dir, channel_matrix_file, snr_db |
| — | `execute_multi_bs` | 内部 | 执行多基站循环仿真 | params |
| — | `plan_multi_bs_route` | 内部 | 单 UAV 多基站感知航线规划 | start, goal, threshold_db |
| — | `plan_swarm_routes` | 内部 | 1~10 架蜂群规划 | start, goal, n_drones, threshold_db |

需要特别说明的是：`prepare_low_altitude`、`execute_low_altitude`、`plan_uav_route` 仍保留在代码中，但当前默认低空主路径已统一收口到 `prepare_multi_bs(mode="low_alt")`。

#### 4.3.3 编排函数（extract_buildings）

`extract_buildings` 是一个典型的编排函数示例，它不直接执行算法，而是按照固定流程串联多个子 Skill：

```python
# 简化的执行流程
def extract_buildings(run_dir, area_name, polygon_coords, ...):
    # 1. 加载坐标
    polygon_coords = json.load(run_dir / "polygon.json")
    target_polygon = Polygon(ensure_closed_polygon(polygon_coords))

    # 2. 数据获取（在线/离线）
    if force_online:
        buildings = fetch_buildings_from_overpass(bbox)
    else:
        shp_path, gdf = find_target_shp(source_dir, target_polygon)
        clipped_gdf = clip_geometries(gdf, target_polygon)

    # 3. 后处理
    gdf_clipped = clip_geometries(gdf_raw, target_polygon)
    geometries, heights = extract_polygons_and_heights(gdf_clipped)
    geometries, heights = remove_overlaps(geometries, heights)

    # 4. 格式转换
    save_json_outputs(geometries, heights, ...)  # → buildings_with_height.json
    json_to_shp(buildings, final_shp_path)       # → result_final.shp

    return {"building_count": len(geometries), "final_shp_path": str(final_shp_path)}
```

### 4.4 WebSocket 实时通信协议

**文件：** `server.py` 第341-1145行

系统使用双通道 WebSocket 实现前后端实时通信：一个负责业务交互(`/ws/chat`)，一个负责 Agent 执行追踪(`/ws/trace`)。

#### 4.4.1 主通信通道 `/ws/chat`

**客户端 → 服务端消息类型：**

| type | 描述 | 触发场景 |
|------|------|---------|
| `message` | 用户聊天消息 | 用户输入文本并发送 |
| `cancel` | 取消当前任务 | 用户点击停止按钮 |
| `reset` | 重置会话 | 新建/切换会话后同步 `session_id` |
| `params_confirmed` | (扩展)确认地面单基站仿真参数 | 用户在 ParamsPanel 点击确认 |
| `reopen_params` | (扩展)重新打开参数面板 | 用户要求重新修改参数 |
| 🌟 `multi_bs_confirmed` | 确认多基站仿真参数 | 用户在 MultiBSPanel 点"开始多基站仿真" |
| 🌟 `plan_multi_bs_route` | 触发单 UAV 多基站航线规划 | 旧单机入口或兼容链路 |
| 🌟 `plan_swarm` | 统一低空航线规划请求 | 当前 `UAVPlanPanelV2` 统一承载单机/蜂群规划，`n_drones=1` 即单机 |

**服务端 → 客户端消息类型：**

| type | 描述 | 数据结构 |
|------|------|---------|
| `system` | 系统通知 | `{content: "..."}` |
| `ai_token` | 流式文本 token | `{content: "token片段"}` |
| `ai_token_end` | 文本流结束 | `{}` |
| `ai_message` | 完整消息（非流式回退） | `{content: "完整消息"}` |
| `tool_start` | 工具开始执行 | `{tool: "resolve_area", args: {...}}` |
| `tool_end` | 工具执行完毕 | `{tool: "resolve_area", result_preview: "..."}` |
| `params_confirm` | (扩展)地面单基站参数面板数据 | `{params, descriptions, run_dir}` |
| 🌟 `multi_bs_config` | 多基站参数面板数据 | `{params, descriptions, run_dir}` |
| 🌟 `multi_bs_progress` | 多基站仿真进度 | `{phase, current, total, bs_id, height, error?}` |
| 🌟 `low_alt_progress` | 历史单基站低空进度 | `{phase, current, total, height, error?}` |
| 🌟 `uav_route_done` | 多基站航线规划结果 | `{success, metrics, handover_count, message}` |
| 🌟 `swarm_progress` | 蜂群逐机规划进度 | `{phase, current, total, drone_id, success, handover_count, total_handover, capacity_violations}` |
| 🌟 `swarm_done` | 蜂群规划完成 | `{n_drones, success_count, failed_count, capacity_violations, total_handover}` |
| `viz_update` | 3D 场景数据更新 | `{session_id: "..."}` |
| `error` | 错误信息 | `{content: "错误描述"}` |

#### 4.4.2 参数确认快速通道

当用户在前端参数面板点击"确认执行"时，系统**绕过 LLM**，直接执行仿真：

```python
# server.py
if data.get("type") == "params_confirmed":
    confirmed_params = data.get("params", {})
    run_dir = data.get("run_dir", "")

    result = execute_ray_tracing(params=confirmed_params, run_dir=run_dir)
    if result.get("success"):
        visualize_radiomap(run_dir=run_dir)

    await websocket.send_json({"type": "viz_update", "session_id": session_id})
```

同理，多基站参数确认后，`server.py` 会直接执行 `execute_multi_bs(progress_cb=...)`；统一低空航线规划面板提交后，会直接执行 `plan_swarm_routes(...)`，其中 `n_drones=1` 时自然退化为单机规划。

#### 4.4.3 监控通道 `/ws/trace`

独立的只读 WebSocket 通道，用于实时监控 Agent 执行事件：

```python
# server.py
@app.websocket("/ws/trace")
async def websocket_trace(websocket: WebSocket):
    # 广播模式: 所有订阅者收到相同事件流
    # 事件格式: {"phase": "input|llm|tool|output|error", ...}
```

配套的 `trace.html` 会把事件渲染为时间线卡片，按 `input / llm / tool / output / error` 分层展示，并统计 Rounds / Tools / Total。
可通过 `http://localhost:7860/trace` 访问可视化监控页面。

### 4.5 会话管理与状态持久化

**Agent 状态：** LangGraph 的 `MemorySaver` 为每个 `thread_id` 维护独立的对话历史和工具调用记录。

**会话隔离：** 每个前端会话对应一个 `web_session_{session_id}`，Agent 与场景缓存都按会话隔离。

**前端会话持久化：**

```javascript
const SESSIONS_KEY = 'agentray_sessions'
// localStorage 中存 { id, name, messages }
```

当前实现会在前端初始化时创建一条默认会话，并在新建会话或切换会话时向后端发送：

```javascript
ws.send(JSON.stringify({ type: 'reset', session_id: currentSessionId }))
```

**场景数据缓存：**

```python
_SESSION_SCENES: Dict[str, Optional[dict]] = {}  # session_id → scene_data
```

每次工具执行后，后端对比最新 `scene_data.json` 的路径和修改时间；如果有更新，就把对应场景写入当前 `session_id` 的缓存，并通过 `viz_update` 通知前端重新拉取。

**仿真参数缓存：**

- `last_params`：WebSocket 会话内缓存最近一次参数面板载荷，支持 `reopen_params`
- `ray_tracing_params.json` / `multi_bs_params.json`：将用户确认后的参数持久化到 `run_dir`

### 4.6 System Prompt 工程

**文件：** `server.py` 第80-147行

系统提示词约 68 行（第80行到第147行），精心设计了工具调用的编排规则和约束条件：

**关键规则示例：**

1. **严格的前置条件链**：`resolve_area` → `extract_buildings` → `visualize_scene`
   > "必须首先调用 resolve_area——即使用户已直接给出坐标也必须调用，因为它负责创建本次任务的输出目录"

2. **模式互斥**：RadioMap / MIMO / 覆盖分析三者互斥
   > "Path Loss.txt → visualize_radiomap；ChannelMatricesPerPoint.txt → analyze_mimo，绝不可混用"

3. **参数面板交互**：LLM 不直接展示参数表格
   > "你不需要在聊天中用 Markdown 表格展示参数，只需告诉用户：'仿真参数已展示在左侧 3D 界面的参数面板中'"

4. **方位角联动**：
   > "当用户指定'方位角 X 度'时，你需要同时设置 antenna_azimuth=X 和 tx_azimuth=X"

5. **IRT 分辨率默认值**：
   > "如果用户没有提到分辨率，则默认使用 10m，不要反复询问"

---

## 五、前端展示层（React + Three.js）

### 5.1 前端架构与组件树

**构建工具：** Vite 6.0.3（ES Modules + HMR）
**框架：** React 18.3.1（Hooks）
**3D 引擎：** Three.js 0.169.0
**开发服务器：** localhost:5173 → 代理到后端 :7860

**布局结构：**

补充说明：当前前端低空航线交互已经统一到 `UAVPlanPanelV2`。旧的 `MultiBSUavPanel`、`SwarmPanel`、`LowAltitudePanel`、`UavRoutePanel` 文件仍保留，但主流程已不再分别使用它们。

```
┌──────────────────────────────────────────────────────────────┐
│  AeroRay · 智能无线仿真平台                    [状态灯]   │
├───────────┬────────────────────────────────┬──┬──────────────┤
│           │                                │▎│              │
│ Feature   │     ThreeScene (WebGL 3D)      │拖│  ChatPanel   │
│ Panel     │                                │拽│              │
│           │  ┌─────────────┐               │条│ [+] 新会话   │
│ · 功能介绍│  │ CoverageOvl │               │  │ ─────────── │
│ · 坐标输入│  └─────────────┘               │  │ 用户消息    │
│           │         ┌──────────┐           │  │ AI 流式回复  │
│           │         │MimoOverly│           │  │ 工具调用卡片  │
│           │         └──────────┘           │  │             │
│           │  ┌──────┐ ┌─────────────┐      │  │ [输入框]    │
│           │  │TopDwn│ │ RayInfoPanel│      │  │ [发送/停止] │
│ (260px)   │  └──────┘ └─────────────┘      │  │ (280-720px) │
│           │  [ParamsPanel - 模态弹窗]       │  │             │
├───────────┴────────────────────────────────┴──┴──────────────┤
│                CSS Grid: 260px 1fr 10px var(--chat-width)    │
└──────────────────────────────────────────────────────────────┘
```

**组件树：**

```
App (根组件 - 状态编排)
├── FeaturePanel          功能卡片 + 手动坐标输入
├── section.scene-panel
│   ├── ThreeScene        WebGL 3D 场景渲染（含低空体素 + 无人机动画）
│   ├── CoverageOverlay   覆盖分析统计图表 (Canvas 2D)
│   ├── MimoOverlay       MIMO 指标切换面板
│   ├── ParamsPanel       仿真参数确认弹窗
│   ├── MultiBSPanel       🌟 多基站参数配置弹窗(模式切换 + 基站表格)
│   ├── InterferenceOverlay 🌟 4 种干扰分析模式选择浮窗
│   ├── InterferenceLegend  🌟 干扰分析图例
│   ├── UAVPlanPanelV2    🌟 当前统一低空航线规划面板(单机 / 蜂群 + 清除路线)
│   ├── TopDownView       2D 俯视图坐标选择（支持航线起终点拾取）
│   └── RayInfoPanel      射线详情浮动表格
├── div.chat-resizer      可拖拽宽度调节器
└── ChatPanel             AI 对话聊天面板
```

**状态管理：** 纯 React Hooks（useState/useEffect/useCallback/useRef），无 Redux/Context

```jsx
// App.jsx 核心状态
const [sceneData, setSceneData] = useState(null)       // 3D 场景数据
const [currentSessionId, setCurrentSessionId] = useState(null) // 当前会话 ID
const [ws, setWs] = useState(null)                     // WebSocket 连接
const [wsReady, setWsReady] = useState(false)          // WebSocket 在线状态
const [manualCoords, setManualCoords] = useState([])   // 左侧手动输入坐标
const [thresholdRange, setThresholdRange] = useState(null)  // 覆盖分析阈值
const [mimoMetric, setMimoMetric] = useState(null)     // MIMO 指标选择
const [pendingParams, setPendingParams] = useState(null) // 待确认仿真参数
const [rayModeEnabled, setRayModeEnabled] = useState(false) // 射线模式
const [rayData, setRayData] = useState(null)           // 射线数据
// 多基站低空主流程状态
const [pendingMultiBs, setPendingMultiBs] = useState(null)      // 待确认多基站参数
const [multiBsProgress, setMultiBsProgress] = useState(null)    // 多基站仿真进度
const [lowAltProgress, setLowAltProgress] = useState(null)      // 历史单基站低空进度
const [swarmProgress, setSwarmProgress] = useState(null)        // 蜂群规划进度
const [interfMode, setInterfMode] = useState('dominance')       // 干扰模式
const [interfVisibleBs, setInterfVisibleBs] = useState(null)    // 可见基站集合
const [mbsUavPanelVisible, setMbsUavPanelVisible] = useState(true) // 统一航线面板显隐
const [uavPickMode, setUavPickMode] = useState(null)            // 'start'|'goal'|null
const [uavPickedStart, setUavPickedStart] = useState(null)      // 俯视图拾取的起点 XY
const [uavPickedGoal, setUavPickedGoal] = useState(null)        // 俯视图拾取的终点 XY
```

### 5.2 Three.js 3D 场景渲染

**文件：** `frontend/src/components/ThreeScene.jsx`（659行）

这是整个前端最复杂的组件，负责所有 3D 渲染逻辑。

#### 5.2.1 坐标系统

采用 **Z-up 工程坐标系**（与 WinProp 物理坐标一致）：

```
X 轴 = 东 (右)       三维坐标
Y 轴 = 北 (前)       1 单位 = 1 米
Z 轴 = 上 (高度)
```

**像素坐标 → Three.js 坐标变换：**

```javascript
// ThreeScene.jsx 注释
three_x = pixel_x - cx        // cx = scene_width / 2
three_y = cy - pixel_y         // cy = scene_height / 2 (Y 轴翻转)
three_z = height               // 直接使用高度值（米）
```

#### 5.2.2 场景初始化

```javascript
// ThreeScene.jsx 第27-99行
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
renderer.toneMapping = THREE.ACESFilmicToneMapping  // ACES 电影色调映射
renderer.toneMappingExposure = 1.6                   // 曝光补偿

const camera = new THREE.PerspectiveCamera(55, aspect, 1, 20000)
camera.up.set(0, 0, 1)           // Z-up
camera.position.set(0, -800, 600) // 初始视角

const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true      // 惯性阻尼
controls.maxPolarAngle = Math.PI * 0.48  // 不能低于地面

// 光照
new THREE.AmbientLight(0x667788, 5)               // 环境光
new THREE.DirectionalLight(0xffffff, 2.2)          // 主方向光
new THREE.DirectionalLight(0xaabbcc, 1.0)          // 补光
```

#### 5.2.3 建筑渲染

```javascript
// 简化逻辑
buildings.forEach(building => {
    const shape = new THREE.Shape()
    building.coordinates.forEach(([x, y]) => {
        shape.lineTo(x - cx, cy - y)  // 像素 → Three.js 坐标
    })

    const geometry = new THREE.ExtrudeGeometry(shape, {
        depth: building.height,        // Z 轴挤出高度
        bevelEnabled: false
    })

    // 砖块纹理材质
    const material = new THREE.MeshStandardMaterial({
        map: brickTexture,             // 砖块.png 纹理
        roughness: 0.85
    })

    // 半透明线框叠加
    const wireframe = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({ color: 0x00d4ff, opacity: 0.12 })
    )
})
```

#### 5.2.4 RadioMap 热力图渲染

使用 **InstancedMesh** 高效渲染数千个信号强度点：

```javascript
// 简化逻辑
const sphereGeo = new THREE.SphereGeometry(resolution / 2, 8, 6)
const instancedMesh = new THREE.InstancedMesh(sphereGeo, material, pointCount)

points.forEach(([px, py, value], i) => {
    // 位置
    const x = px - cx
    const y = cy - py
    const z = predictionHeight
    matrix.setPosition(x, y, z)
    instancedMesh.setMatrixAt(i, matrix)

    // HSL 颜色映射: 红(强信号) → 绿 → 青 → 蓝 → 紫(弱信号)
    const t = (value - vMin) / (vMax - vMin)  // 归一化
    const hue = (1 - t) * 0.75                 // H: 0.75(紫) → 0(红)
    color.setHSL(hue, 1.0, 0.5)
    instancedMesh.setColorAt(i, color)
})
```

#### 5.2.5 天线模型

```javascript
// 基站杆 + 天线板
const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.3, 0.3, antennaZ),
    new THREE.MeshStandardMaterial({ color: 0x888888 })
)

// 3 个天线扇区板（120° 间隔）
for (let i = 0; i < 3; i++) {
    const panel = new THREE.Mesh(
        new THREE.BoxGeometry(2, 0.3, 4),
        new THREE.MeshStandardMaterial({ color: 0x00d4ff, emissive: 0x00d4ff })
    )
    panel.rotation.z = (i * 120) * Math.PI / 180
}

// MIMO 模式：添加波束锥体显示方位角/俯仰角
const cone = new THREE.Mesh(
    new THREE.ConeGeometry(beamRadius, beamLength, 32),
    new THREE.MeshBasicMaterial({ color: 0x00d4ff, opacity: 0.15 })
)
```

#### 5.2.6 交互系统

**OrbitControls：** 默认模式 — 旋转、平移、缩放

**Ray Mode：** 射线检查模式
1. 用户点击热力图上的球体
2. Raycaster 命中检测 → 获取球体位置
3. 反查像素坐标 → REST API `/api/rays?x=...&y=...`
4. 后端从 `.str` 文件解析射线数据
5. 渲染射线路径（颜色编码 + 交互点标记）

```javascript
// 点击处理
raycaster.setFromCamera(mouse, camera)
const intersects = raycaster.intersectObject(instancedMesh)
if (intersects.length > 0) {
    const instanceId = intersects[0].instanceId
    const point = filteredPoints[instanceId]
    onPointClick({ pixelX: point[0], pixelY: point[1] })
}
```

### 5.3 AI 对话交互界面

**文件：** `frontend/src/components/ChatPanel.jsx`（427行）

#### 5.3.1 多会话管理

```javascript
// 会话持久化到 localStorage
const SESSIONS_KEY = 'agentray_sessions'
// 每个会话: { id, name, messages }
```

- 前端初始化时创建默认会话 `当前对话`
- 当前版本以**单槽多次重置**为主：新建会话时会直接覆盖 localStorage 为仅包含新会话的数组
- 切换/新建会话时通过 `reset + session_id` 同步后端 LangGraph thread 与 `_SESSION_SCENES`

#### 5.3.2 流式渲染

```javascript
// WebSocket 消息处理
const handler = (event) => {
    const data = JSON.parse(event.data)
    switch(data.type) {
        case 'ai_token':
            isStreamingRef.current = true
            setStreamingText(prev => prev + (data.content || ''))
            break
        case 'ai_token_end':
            // 结束后再固化为 assistant 消息
            flushStreamingMessage()
            break
        case 'tool_start':
            // 若正在流式输出，先收束上一条 assistant 消息，再插入工具卡片
            finalizeStreamingBeforeTool()
            break
        case 'viz_update':
            onVizUpdate(data.session_id)
            break
    }
}
```

当前实现把流式文本先缓存在 `streamingText`，等 `ai_token_end` 再固化进消息列表；这样可以避免 token 级频繁重排，也保证工具调用卡片能正确插入到流式回复之间。

#### 5.3.3 消息类型

| 类型 | 视觉表现 | 触发条件 |
|------|---------|---------|
| 用户消息 | 绿色气泡，右对齐 | 用户发送文本 |
| AI 回复 | 蓝色气泡，左对齐，Markdown 渲染 | LLM 流式输出 |
| 工具调用 | 灰色可折叠卡片 | `tool_start` 事件 |
| 工具结果 | 绿色可折叠卡片 | `tool_end` 事件 |
| 系统通知 | 橙色通知条 | 任务中断、会话切换 |
| 错误消息 | 红色气泡 | `error` 事件 |
| 思考中 | 动画指示器 | LLM 推理进行中 |

#### 5.3.4 坐标联动

当用户在 FeaturePanel 输入手动坐标后，ChatPanel 会在发送消息时自动附加坐标信息，但只在**用户文本里没有显式给出区域坐标**时才触发：

```javascript
const normalizedManualCoords = normalizeManualCoords(manualCoords)
const shouldUseManualCoords = normalizedManualCoords.length >= 4 && !hasExplicitArea(text)
const finalText = shouldUseManualCoords
  ? `${text}\n\n请优先使用左侧输入框中的四个顶点坐标作为本次区域：polygon_coords = ${JSON.stringify(normalizedManualCoords)}`
  : text
```

这里有两个关键细节：

1. `normalizeManualCoords()` 会去重、计算质心、按极角排序，统一成逆时针顶点序列
2. `hasExplicitArea()` 会检测用户消息里是否已经带 `[...]` / `(x,y)` 形式的坐标，避免重复注入

### 5.4 覆盖分析与 MIMO 分析面板

#### 5.4.1 CoverageOverlay（覆盖分析）

**文件：** `frontend/src/components/CoverageOverlay.jsx`（315行）

使用 **Canvas 2D API** 手绘统计图表（无第三方图表库）：

| 模式 | 可视化 | 数据源 |
|------|--------|--------|
| IR (Impulse Response) | 散点图 — 所有信号强度值 | radiomap.points 的值域 |
| PDF (概率密度) | 直方图 — 信号强度分布 | 值域分桶统计频率 |
| CDF (累积分布) | S 曲线 — 累积百分比 + P50 中位线 | 排序后累积 |
| Threshold (门限过滤) | 滑块 — 设置 [min, max] 范围 | 过滤后更新 3D 球体显示 |

```javascript
// CDF 绘制简化逻辑
const sorted = allValues.sort((a, b) => a - b)
sorted.forEach((val, i) => {
    const x = mapToCanvas(val, vMin, vMax, 0, canvasWidth)
    const y = canvasHeight * (1 - (i + 1) / sorted.length)
    ctx.lineTo(x, y)
})
// P50 中位线
const p50 = sorted[Math.floor(sorted.length / 2)]
ctx.setLineDash([5, 3])
ctx.moveTo(mapToCanvas(p50, ...), 0)
ctx.lineTo(mapToCanvas(p50, ...), canvasHeight)
```

#### 5.4.2 MimoOverlay（MIMO 分析）

**文件：** `frontend/src/components/MimoOverlay.jsx`（94行）

展示 4 个 MIMO 指标的切换按钮和色标：

| 指标 | 单位 | 物理含义 |
|------|------|---------|
| Capacity (C) | bps/Hz | 信道容量 — 最大可靠传输速率 |
| Rank (R) | — | MIMO 秩 — 空间复用能力 |
| Condition Number (K) | dB | 条件数 — 信道矩阵"健康度" |
| Max Singular Value (S) | dB | 最大奇异值 — 主信道增益 |

点击切换指标时，通过 `onMetricChange` 回调通知 ThreeScene 重新渲染热力图颜色。

### 5.5 射线可视化与交互检查

#### 5.5.1 射线路径渲染（ThreeScene）

当用户在 Ray Mode 下点击热力图上某点时：

1. 前端发送 `GET /api/rays?run_dir=...&x=...&y=...&max_rays=5`
2. 后端 `show_ray` Skill 解析 `.str` 文件，找到最近接收点的射线
3. 返回射线数据（路径点、延迟、场强、到达角等）
4. ThreeScene 渲染射线路径

```javascript
// 射线渲染
rays.forEach((ray, idx) => {
    const colors = [0xff4444, 0x44ff44, 0xff8800, 0xff44ff, 0x44ffff]
    const points = [antenna, ...interactions, receiver]

    const geometry = new THREE.BufferGeometry().setFromPoints(points)
    const line = new THREE.Line(geometry,
        new THREE.LineBasicMaterial({
            color: colors[idx],
            opacity: highlightedIdx === idx ? 1.0 : 0.3
        }))

    // 交互点标记（小球）
    interactions.forEach(pt => {
        const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(1),
            new THREE.MeshBasicMaterial({ color: colors[idx] })
        )
        sphere.position.copy(pt)
    })
})
```

#### 5.5.2 RayInfoPanel（射线详情）

**文件：** `frontend/src/components/RayInfoPanel.jsx`（118行）

可拖拽的浮动面板，显示射线参数表格：

| 列 | 含义 | 来源 |
|----|------|------|
| Delay (ns) | 传播延迟 | 路径长度 / 光速 |
| Field (dBuV/m) | 场强 | WinProp 计算 |
| Phase (rad) | 相位 | 电磁波相位 |
| DoD Azi/Ele | 出发方向 | 天线端角度 |
| DoA Azi/Ele | 到达方向 | 接收端角度 |
| Int. | 交互类型 | D=衍射, R=反射 |

鼠标悬停表格行时，3D 场景中对应射线高亮显示。

### 5.6 设计语言与样式系统

**文件：** `frontend/src/index.css`、`frontend/src/App.css` 及各组件 CSS

**设计主题：** 科幻 / 数据仪表盘风格

**CSS 变量系统：**

```css
:root {
  --bg-deep:    #020810;    /* 深空背景 */
  --bg-panel:   #040d1a;    /* 面板背景 */
  --bg-card:    #071426;    /* 卡片背景 */
  --border:     #0d2644;    /* 边框 */
  --accent:     #00d4ff;    /* 主强调色（青色） */
  --accent2:    #f0c040;    /* 次强调色（金色） */
  --accent3:    #ff6b35;    /* 第三强调色（橙色） */
  --text:       #c8e0f4;    /* 主文本 */
  --text-dim:   #4a7a9b;    /* 弱化文本 */
  --glow:       0 0 12px #00d4ff66, 0 0 24px #00d4ff22;  /* 发光效果 */
}
```

**字体：**
- **Orbitron**：科技感标题字体（Google Fonts）
- **Noto Sans SC**：中文正文字体

**动画效果：**
- `sweepMove`：标题栏扫描光线
- `pulse`：连接状态指示灯脉冲
- `fadeSlideIn`：组件入场动画
- `cursorBlink`：AI 打字光标闪烁

---

## 六、数据流与协议详解

### 6.1 端到端数据流水线

一次完整的"建筑提取 → 仿真 → 分析"流程：

```
用户: "仿真西安电子科技大学南校区的信号覆盖，开启 MIMO"
                    │
                    ▼
[1] LLM 推理 → 调用 resolve_area(user_request="...", mode="auto")
                    │
                    ▼
    Nominatim API → polygon.json (4个顶点坐标)
    创建 resu/20260322_143000/
                    │
                    ▼
[2] LLM 推理 → 调用 extract_buildings(run_dir="...", area_name="西安电子科技大学")
                    │
                    ├─ Overpass API → 建筑几何数据
                    ├─ GeoDataFrame 裁剪 + 去重
                    ├─ → json/buildings_with_height.json (像素坐标 + 高度)
                    ├─ → json/area_meta.json (场景尺寸)
                    └─ → shp/result_final.shp (3D Shapefile)
                    │
                    ▼
[3] LLM 推理 → 调用 export_odb(final_shp_path="...", pred_model="dpm")
                    │
                    ├─ subprocess → SHP2ODB_debug.exe
                    └─ → shp/result_final.odb
                    │
                    ▼
[4] LLM 推理 → 调用 visualize_scene(run_dir="...")
                    │
                    └─ → scene_data.json (建筑 3D 数据)
                    → WebSocket viz_update → 前端 Three.js 渲染建筑
                    │
                    ▼
[5] LLM 推理 → 调用 prepare_ray_tracing(odb_file="...", enable_mimo=true)
                    │
                    ├─ 读取 area_meta.json → 推算仿真区域边界
                    ├─ 天线默认位置 = 区域中心
                    └─ → WebSocket params_confirm → 前端弹出参数面板
                    │
                    ▼
[6] 用户在面板中确认参数 → WebSocket params_confirmed
                    │
                    ▼
[7] 绕过 LLM → 直接执行:
    execute_ray_tracing(params={...})
                    │
                    ├─ subprocess → PropagationRunMSMIMOSingle_debug.exe
                    │   ├─ Stage 1: DPM/IRT 传播预测 → Path Loss.txt, Power.fpp
                    │   └─ Stage 2: MIMO RunMS → ChannelMatricesPerPoint.txt
                    │
                    └─ Python 后处理 → .npz + 预览图
                    │
                    ▼
[8] 自动执行: visualize_radiomap(run_dir="...")
                    │
                    ├─ 解析 Path Loss.txt → 像素坐标点云
                    └─ → scene_data.json (追加 radiomap 字段)
                    → WebSocket viz_update → 前端渲染 3D 热力图
                    │
                    ▼
[9] 用户: "MIMO 分析" → LLM → analyze_mimo(run_dir="...")
                    │
                    ├─ 解析 ChannelMatricesPerPoint.txt
                    ├─ SVD 分解 → 容量/秩/条件数/最大奇异值
                    └─ → scene_data.json (替换为 mimo_analysis 字段)
                    → 前端渲染 MIMO 热力图 + 指标面板
```

### 6.2 坐标系统与变换

系统涉及 4 个坐标系统之间的转换：

#### 6.2.1 WGS84 地理坐标

```
(经度 lon, 纬度 lat)
范围: lon ∈ [-180, 180], lat ∈ [-90, 90]
来源: OSM API, Nominatim, 用户输入
```

#### 6.2.2 像素坐标（各向同性投影）

```
(pixel_x, pixel_y)
单位: 像素 (≈1 米/像素)
原点: 左上角
X轴: 向东递增
Y轴: 向南递增
转换: pixel_x = (lon - min_lon) × scale
      pixel_y = (max_lat - lat) × scale
      scale = 111320 × cos(mid_lat)
```

#### 6.2.3 ODB/WinProp 坐标

```
(odb_x, odb_y)
单位: 米
原点: 左下角
X轴: 向东递增
Y轴: 向北递增 (与像素坐标 Y 方向相反!)
转换: odb_x = pixel_x
      odb_y = pixel_height - pixel_y  (Y 轴翻转)
```

#### 6.2.4 Three.js 场景坐标

```
(three_x, three_y, three_z)
单位: 米
原点: 场景中心
X轴: 东 (右)
Y轴: 北 (前)
Z轴: 上
转换: three_x = pixel_x - cx       (cx = scene_width / 2)
      three_y = cy - pixel_y        (cy = scene_height / 2)
      three_z = height
```

**完整转换链：**

```
WGS84 (lon, lat)
    ↓ calc_pixel_params() / latlon_to_pixel()
像素坐标 (px, py)
    ↓ json_to_shp() (Y翻转: odb_y = max_y - py)
ODB 坐标 (ox, oy) → WinProp 仿真
    ↓ visualize_radiomap() (反转: py = upper_right_y - oy)
像素坐标 (px, py)
    ↓ ThreeScene.jsx (居中: tx = px - cx, ty = cy - py)
Three.js 坐标 (tx, ty, tz)
```

### 6.3 进程间通信（Python ↔ C++）

Python 通过 `subprocess.run()` 调用 C++ 可执行文件，采用**文件级 IPC**：

```python
# 调用示例
cmd = [
    str(exe_path),
    "--odb_file",     str(odb_path),
    "--antenna_x",    "500.0",
    "--antenna_y",    "350.0",
    "--antenna_freq",  "3500",
    "--enable_mimo",   "1",
    "--pred_model",    "dpm",
    # ... 更多参数
]

result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
```

**通信特征：**
- **同步阻塞**：Python 等待 C++ 进程完成
- **参数传递**：命令行 `--key value` 格式
- **数据交换**：全部通过文件系统（输入: .odb/.oib; 输出: .txt/.fpp/.str）
- **错误处理**：检查 `returncode` 和文件是否生成
- **无网络/共享内存/管道**：完全依赖文件 I/O

### 6.4 文件格式与中间产物

每次任务创建一个时间戳目录，完整的文件结构如下：

```
resu/20260322_143000/
├── polygon.json                    # 用户指定的区域坐标
│   # [[108.91, 34.23], [108.92, 34.23], [108.92, 34.24], [108.91, 34.24]]
│
├── json/
│   ├── buildings_with_height.json  # 建筑像素坐标 + 高度
│   │   # [{"coordinates": [[x1,y1],[x2,y2],...], "height": 25.0}, ...]
│   ├── area_meta.json              # 场景元信息
│   │   # {"pixel_width": 1024, "pixel_height": 768, "scale": 92345.6, "geo_bounds": {...}}
│   └── irt_meta.json               # (仅 IRT) 预处理参数
│       # {"irt_resolution": 10.0, "irt_height": 1.5}
│
├── shp/
│   ├── result_final.shp            # 3D Shapefile (含 Z=height)
│   ├── result_final.dbf            # 属性表
│   ├── result_final.shx            # 索引
│   ├── result_final.cpg            # 编码
│   ├── result_final.prj            # 投影定义
│   ├── result_final.odb            # WinProp 二进制建筑数据库 (DPM)
│   └── result_final_IRT.oib        # (仅 IRT) 预处理数据库
│
├── ray_tracing/
│   ├── Antenna_1 Path Loss.txt     # 路径损耗结果 (TSV 格式)
│   ├── Antenna_1 Power.fpp         # 功率结果 (浮点网格)
│   ├── Antenna_1 Propagation Paths.str  # 射线路径数据
│   └── runMS/                      # (仅 MIMO) 信道矩阵输出
│       ├── Antenna_1 ChannelMatricesPerPoint.txt
│       ├── Antenna_1 Power.fpp
│       └── Antenna_1_power.npz     # Python 后处理压缩
│
├── scene_data.json                 # 统一 3D 场景数据 (供前端渲染)
│   # {buildings: [...], stations: [...], radiomap: {...}, mimo_analysis: {...}}
│
├── ray_tracing_params.json         # 最近一次仿真参数快照
├── sim_params.json                 # 用户确认的参数缓存
└── final_visualization.png         # 2D 合成预览图 (Matplotlib)
```

**scene_data.json 关键结构：**

```json
{
  "buildings": [
    {"coordinates": [[x1,y1],[x2,y2],...], "height": 25.0}
  ],
  "stations": [
    {"x": 500, "y": 350, "name": "基站-001", "operator": "China Mobile"}
  ],
  "radiomap": {
    "points": [[px, py, -85.3], ...],
    "resolution": 2.0,
    "value_range": [-149.0, -36.0],
    "antenna": {"x": 500, "y": 350, "z": 35, "frequency": 3500}
  },
  "mimo_analysis": {
    "enabled": true,
    "metrics": {
      "capacity": {"points": [[x,y,val],...], "value_range": [0, 25]},
      "rank":     {"points": [[x,y,val],...], "value_range": [1, 4]},
      "condition_number": {"points": [...], ...},
      "max_singular":     {"points": [...], ...}
    }
  },
  "coverage_analysis": {"enabled": true},
  "scene_width": 1024,
  "scene_height": 768
}
```

---

## 七、核心算法详解

### 7.1 WinProp DPM 主导路径模型

DPM（Dominant Path Model）是 WinProp 提供的快速城市传播预测模型。

**原理：** 对每个预测点，只计算对信号贡献最大的"主导路径"（通常是直射路径、一次反射路径或绕射路径中最强的那条），而不像 IRT 那样追踪所有可能的传播路径。

**特点：**
- 计算速度快（适合大范围覆盖预测）
- 精度中等（适合网规初期估算）
- 输入：`.odb` 二进制建筑数据库
- 分辨率可在每次仿真时灵活调整

**默认参数：**
```
分辨率: 0.5m
频率: 3500 MHz (5G NR n78)
功率: 23 dBm
天线高度: 15m
预测高度: 1.5m
```

### 7.2 WinProp IRT 智能射线追踪

IRT（Intelligent Ray Tracing）是基于物理的精确射线追踪模型。

**原理：** 从发射天线出发，模拟电磁波在建筑场景中的传播，考虑：
- **反射**（最多 2 次）：电磁波遇到建筑墙面发生镜面反射
- **衍射**（最多 1 次）：电磁波在建筑棱边发生衍射弯曲
- **散射**（可选）：粗糙表面引起的散射

**断点路径损耗模型：**

IRT 使用分段幂律模型计算路径损耗：

```
LOS（视距）:
  d < d_breakpoint: PL = PL_0 + 10 × 2.3 × log10(d)
  d > d_breakpoint: PL = PL_0 + 10 × 3.3 × log10(d)

NLOS（非视距）:
  d < d_breakpoint: PL = PL_0 + 10 × 2.5 × log10(d)
  d > d_breakpoint: PL = PL_0 + 10 × 3.3 × log10(d)
```

**关键约束：**
- 预处理分辨率一旦生成 `.oib` 后**不可更改**
- 仿真分辨率必须与预处理分辨率一致
- 计算量远大于 DPM，适合小范围精细分析

### 7.3 MIMO 信道矩阵分析（SVD）

**文件：** `skill/mimo_analysis/skill_mimo_analysis.py`

#### 7.3.1 信道矩阵解析

WinProp 输出的 `ChannelMatricesPerPoint.txt` 格式：

```
Transmitter settings
  Nbr of antenna elements: 4
Receiver settings
  Nbr of antenna elements: 4

Point: 500.00000 350.00000 1.50000 [m]

-6.080e-19+1.431e-19j   3.215e-19-2.108e-19j   ...
 1.002e-19+4.553e-20j  -5.877e-19+1.224e-19j   ...
 ...
```

每个空间点有一个 Nr×Nt 的复数信道矩阵 H。

#### 7.3.2 全局归一化

WinProp 输出的 H 矩阵元素通常极小（~10⁻¹⁹），需要归一化：

```python
# skill_mimo_analysis.py 第340-354行
frob_norms = [np.linalg.norm(pt["H"], 'fro') for pt in points]
median_frob = np.median(frob_norms[frob_norms > 0])

if median_frob < 1e-6:
    target_norm = np.sqrt(nr_tx * nr_rx)    # 目标 Frobenius 范数
    scale_factor = target_norm / median_frob
    for pt in points:
        pt["H"] = pt["H"] * scale_factor     # 标量缩放不影响秩和条件数
```

#### 7.3.3 四大指标计算

```python
# skill_mimo_analysis.py 第218-289行

# 对每个点执行 SVD
singular_values = np.linalg.svd(H, compute_uv=False)
sv = np.sort(singular_values)[::-1]  # 降序排列

# 1. 信道容量 (等功率分配, Shannon 公式)
# C = Σ log₂(1 + (SNR / Nt) × σᵢ²)
capacity = sum(np.log2(1.0 + (snr_linear / nr_tx) * s**2) for s in sv)

# 2. 有效 MIMO 秩 (奇异值 > 阈值 × 最大奇异值)
# 阈值 = 10^(-20/20) = 0.1 (-20dB)
rank = np.sum(sv > 0.1 * sv[0])

# 3. 条件数 (dB)
# κ = σ_max / σ_min → 越大信道越"病态"
cond_db = 20 × log10(sv[0] / sv_min_nonzero[-1])

# 4. 最大奇异值 (dB)
sv_max_db = 20 × log10(sv[0])
```

**物理意义：**
- **信道容量**：给定 SNR 下的理论最大传输速率（bps/Hz）
- **MIMO 秩**：可以并行传输的独立数据流数量（最大 = min(Nt, Nr)）
- **条件数**：信道矩阵的"健康度"，越小越好（低条件数 = 多流性能均衡）
- **最大奇异值**：主信道的增益强度

### 7.4 覆盖统计分析

前端 CoverageOverlay 对 RadioMap 数据进行四种统计分析：

**IR（脉冲响应）：** 将所有点的信号强度值绘制为散点图，直观展示值域分布

**PDF（概率密度函数）：**
```
将 [vMin, vMax] 分为 N 个桶
对每个桶计算 count / total
绘制直方图
```

**CDF（累积分布函数）：**
```
将所有值排序
CDF(x) = P(信号强度 ≤ x) = rank(x) / total
P50 = 中位数信号强度
```

**Threshold（门限过滤）：**
```
用户设置 [min, max] 范围
3D 场景中仅显示 value ∈ [min, max] 的球体
用于直观查看"哪些区域信号弱于 -100dBm"
```

---

## 八、低空电磁智能孪生子系统(平台核心)

### 8.0 子系统总览

低空电磁智能孪生子系统是平台的**核心**,面向低空经济场景的三大需求:

1. **低空通信走廊评估** — 现网基站天线为对地优化(下倾 6~10°),100 m 以上存在大量"覆盖空洞",
   需要在多个高度层快速评估**多基站联合三维信号场**,识别低空盲区
2. **多基站协同与干扰分析** — 低空空间中通常被多个基站同时覆盖,
   需要 RSRP / 主导基站 / SINR / 重叠覆盖 4 种态势分析,评估协同与干扰
3. **无人机/蜂群航线规划** — 无人机物流/巡检/eVTOL 场景下,航线需要在物理可行的同时
   满足通信链路质量约束,**且能感知基站切换**(避免在小区边界出现链路中断)

子系统包含两层:

- **`skill/multi_bs/`(主层)**:多基站三维仿真 + 干扰分析 + 多基站感知 A* + 蜂群规划。
  唯一暴露给 LLM 的入口是 `prepare_multi_bs(mode="low_alt")`。
- **`skill/low_altitude/`(基础层)**:单基站多高度 DPM 与 26 邻接 A* 实现,作为 multi_bs 上层的基础参考。
  (注:LLM 不再直接调用此层,所有低空场景统一走 multi_bs。)

### 8.1 单基站低空场景实现(基础层)

`skill/low_altitude/` 仍保留单基站多高度 DPM、3D 场聚合与基础 A* 规划能力,主要价值有两点:

1. 作为 `skill/multi_bs/` 的实现参考与算法基线
2. 兼容历史 `run_dir` 与旧实验链路

需要强调的是:当前生产主路径已经不再由 LLM 直接调用 `prepare_low_altitude`,而是统一收口到 `prepare_multi_bs(mode="low_alt")`。

### 8.2 当前端到端主工作流

当前低空主路径如下:

```
用户自然语言
   ↓
LLM 调 resolve_area → extract_buildings → export_odb(pred_model="dpm")
   ↓
LLM 调 prepare_multi_bs(mode="low_alt")
   ↓
server 拦截 suggested_params → ws: multi_bs_config
   ↓
前端弹 MultiBSPanel,用户确认参数 → ws: multi_bs_confirmed
   ↓
server 直接调 execute_multi_bs(progress_cb=...)  (绕过 LLM)
   ↓
ws: tool_start / multi_bs_progress / tool_end / viz_update
   ↓
ThreeScene 加载 `multi_bs_field` + `/api/multibs/field`
   ↓
InterferenceOverlay 实时切换 RSRP / Dominance / SINR / Overlap
   ↓
UAVPlanPanel 发起 plan_multi_bs_route 或 plan_swarm
   ↓
server 调 `plan_multi_bs_route` / `plan_swarm_routes`
   ↓
scene_data.json 注入 `uav_route` 或 `swarm_routes` → viz_update
```

### 8.3 多高度 DPM 仿真与场聚合

当前真正用于低空生产链路的是 `execute_multi_bs()`:

- `mode="plane"` 时输出 `(N_bs, nx, ny)` 单平面 per-BS 张量
- `mode="low_alt"` 时输出 `(N_bs, nx, ny, nz)` 低空 3D per-BS 张量
- 对每个 `(基站, 高度)` 组合调用一次底层 `execute_ray_tracing(pred_model="dpm")`
- 将每层解析后的 path loss 切片写入 `multi_bs/multi_bs_field.npz`
- 将元数据写入 `multi_bs/multi_bs_field_meta.json` 并注入 `scene_data.json["multi_bs_field"]`

单基站基础层的 `field_aggregator.py` 仍然保留,但当前主要作为历史实现和算法参考。

### 8.4 路径规划算法演进

基础层 `astar_3d.py` 解决的是单基站三维路径问题;当前主流程已经升级为 `handover_astar.py` 中的多基站切换感知规划:

- 基础层状态: `(x, y, z)`
- 主流程状态: `(x, y, z, k)`

也就是说,当前生产路径不只考虑空间位置,还把"当前连接哪个基站"纳入搜索状态,从而自然表达切换代价与分段服务关系。

### 8.5 前端体素渲染与无人机动画

前端保留了对 `low_alt_field` 的兼容加载逻辑,但当前主显示链路优先使用 `multi_bs_field`:

- `ThreeScene.jsx` 通过 `/api/multibs/field?run_dir=...` 拉取原始 float32 张量
- `multiBsField.js` 在浏览器侧即时计算 dominance / SINR / overlap
- `UAVPlanPanelV2.jsx` 统一显示单机和蜂群规划结果，支持清除路线，并渲染连接时间轴(Gantt)
- 无人机动画、服务基站连接线、分段着色轨迹都以 `scene_data.json` 中的 `uav_route` / `swarm_routes` 为准

### 8.6 WebSocket 协议扩展

当前低空主流程主要依赖以下消息:

| 消息 | 方向 | 用途 |
|---|---|---|
| `multi_bs_config` | server → client | 下发多基站参数面板 |
| `multi_bs_confirmed` | client → server | 用户确认多基站参数 |
| `multi_bs_progress` | server → client | 多基站仿真分层进度 |
| `plan_multi_bs_route` | client → server | 旧单 UAV 切换感知规划入口 |
| `plan_swarm` | client → server | 当前统一航线规划入口,`n_drones=1~10` |
| `swarm_progress` | server → client | 蜂群逐机规划进度与累计统计 |
| `uav_route_done` | server → client | 单 UAV 规划完成(兼容链路) |
| `swarm_done` | server → client | 蜂群规划完成 |
| `viz_update` | server → client | scene_data 已刷新,前端重新拉取 |

旧的 `low_alt_config` / `low_alt_confirmed` / `low_alt_progress` 仍保留在代码中,但不再是默认主流程。

---

### 8.7 multi_bs 主层 — 多基站三维仿真 + 切换 A* + 蜂群

`skill/multi_bs/` 是当前生产路径,在基础层之上做了根本性的扩展:从"单基站三维场"
升级为"**N 基站 per-BS 张量**",并由此衍生出干扰分析、切换感知规划、蜂群规划。

**模块文件**:

| 文件 | 职责 |
|---|---|
| `skill/multi_bs/skill_multi_bs.py` | 编排层:`prepare_multi_bs` / `execute_multi_bs` / `plan_multi_bs_route` / `plan_swarm_routes` |
| `skill/multi_bs/field_store.py` | per-BS 张量存取(npz)、`build_obstacle_mask_from_buildings`、默认基站布局 |
| `skill/multi_bs/handover_astar.py` | **状态空间 (x,y,z,k) 的分层图 A***,自动切换基站 |
| `skill/multi_bs/swarm_planner.py` | 优先级解耦 + 容量约束 + **强制初始 BS 多样化** 的蜂群规划 |

#### 8.7.1 per-BS 张量

把每个基站的 path loss 单独保存为张量的一个切片:

- **mode="plane"**:`(N_bs, nx, ny)` — 地面单平面态势
- **mode="low_alt"**:`(N_bs, nx, ny, nz)` — 低空 3D 多高度

执行 `execute_multi_bs` 时,对每个 (基站, 高度) 组合调用一次 `execute_ray_tracing(prediction_height=h, antenna_x=...)`,
解析 `Prediction Path Loss.txt` 后写入张量对应的切片。最终聚合保存为
`multi_bs/multi_bs_field.npz`(单 key `field`,float32),前端通过
`/api/multibs/field?run_dir=...` 拿到原始字节流并 `new Float32Array(buf)` 解析。

#### 8.7.2 干扰分析(前端实时计算)

干扰分析的 4 种模式 **完全在前端从 per-BS 张量计算**(`frontend/src/components/multiBsField.js`):

| 模式 | 公式 | 颜色映射 |
|---|---|---|
| **RSRP** | 单 BS 的 path loss | 红强 → 蓝弱 |
| **主导基站(Dominance)** | `argmax_k(PL_k)`,选 PL 最大的 BS | 每个 BS 一种身份色 |
| **SINR** | `S = max_k(P_k_lin)`,`I = sum - S`,`SINR = 10 log10(S/(I+N))` | 红 → 黄绿(高 SINR) |
| **重叠覆盖(Overlap)** | `count(PL_k >= threshold)` | 蓝/青/黄/红(1~4+) |

**为什么放在前端**:per-BS 张量已经在浏览器内存里,切换模式毫秒级响应,不需要再发 WS 请求。
也支持用户用复选框临时屏蔽某几个 BS(在计算前把对应行设为 NaN),交互极顺滑。

#### 8.7.3 切换感知 A*(handover_astar.py)

把 A* 的状态从 `(ix, iy, iz)` 扩展到 `(ix, iy, iz, k)` —— 即"在该格且当前连接基站 k"。
两类边:

```
移动边:  (cell, k) → (相邻 cell, 同 k)
         代价 = α · 距离 + β_eff · 信号惩罚(对 BS k 在 nb 处的 PL)

切换边:  (cell, k1) → (同 cell, k2)
         代价 = handover_penalty * res_xy * 0.4
         前提:k2 在 cell 的 PL >= threshold - 25(避免无意义切换)
```

跑 Dijkstra/A*(欧氏距离作启发函数,可采纳),算法**自动决定何时切换基站**。
回溯路径时记录每个 waypoint 当前的 BS 索引,再做"同 BS 段内 LOS 简化"得到平滑分段路径。

**建筑避障**:从 `buildings_with_height.json` 通过 `build_obstacle_mask_from_buildings` 构造 (nx, ny) bool 掩膜,
A* 跳过 mask 中为 True 的格子;此外**所有 BS 在该格都没数据** (`np.all(np.isnan(col))`) 也视为不可达。

**关键参数**:

| 参数 | 默认 | 含义 |
|---|---|---|
| `handover_penalty` | 8.0 | 切换基站的等效距离惩罚,**滑块可调** |
| `signal_priority` | 0.5 | 0~1,指数缩放放大 β,值越大越偏好高 SNR 路径 |
| `forced_initial_bs` | None | **蜂群多样性的关键** — 强制起点连接特定 BS |

#### 8.7.4 蜂群规划(swarm_planner.py)

蜂群中所有无人机 **共享同一起点和同一终点**(数量 1~10 任意调)。多样性通过两种机制实现:

1. **强制初始 BS 轮询**:第 i 架被强制 `forced_initial_bs = i % N_bs`,这样 5 架 4 BS 蜂群会有
   `[BS0, BS1, BS2, BS3, BS0]` 的初始基站序列,因起始连接的 BS 不同,A* 会走出明显不同的路径
2. **优先级解耦的容量约束**:按 priority 顺序逐架规划,维护 `bs_load[t][bs]` 时刻表,
   后规划的能感知前面的基站占用,如果某基站在某时段已满则统计为 `capacity_violations`(简化版不强制重规划)

#### 8.7.5 scene_data 注入与互斥

`_inject_scene_data` 把 `multi_bs_field`(只含元数据,实际张量走二进制 API)写入 `scene_data.json`,
**同时主动 pop 掉**:`radiomap` / `mimo_analysis` / `coverage_analysis` / `low_alt_field` / `uav_route` / `swarm_routes`,
确保任意时刻只激活一种场景模式。

`plan_multi_bs_route` 注入 `uav_route`(标记 `multi_bs: True` 字段),`plan_swarm_routes` 注入 `swarm_routes`,
两者互斥(写入新的会 pop 旧的)。

#### 8.7.6 WebSocket 协议(multi_bs)

| 消息 | 方向 | 用途 |
|---|---|---|
| `multi_bs_config` | server → client | LangGraph 拦截 `prepare_multi_bs` 后推送的面板配置 |
| `multi_bs_confirmed` | client → server | 用户确认 MultiBSPanel 后回传 |
| `multi_bs_progress` | server → client | 多基站仿真进度,字段 `phase / current / total / bs_id / height` |
| `plan_multi_bs_route` | client → server | 单 UAV 规划请求 |
| `plan_swarm` | client → server | 蜂群规划请求 |
| `swarm_done` | server → client | 蜂群规划完成,带 `n_drones / capacity_violations / total_handover` |

---

## 九、完整技术栈清单

### 后端

| 技术 | 版本/说明 | 用途 |
|------|-----------|------|
| **Python** | 3.10+ | 主后端语言 |
| **FastAPI** | latest | 异步 Web 框架, REST + WebSocket |
| **Uvicorn** | latest | ASGI 服务器 |
| **LangChain** | latest | LLM 工具调用框架 |
| **LangGraph** | latest | ReAct Agent 编排 + 状态检查点 |
| **langchain-anthropic** | latest | Anthropic Claude 集成 |
| **langchain-openai** | latest | OpenAI / 兼容接口集成 |
| **GeoPandas** | latest | 地理空间矢量数据处理 |
| **Shapely** | latest | 几何运算 (裁剪、包含判断) |
| **Fiona** | latest | Shapefile 读写 |
| **Matplotlib** | latest | 2D 预览图生成 (Agg backend) |
| **NumPy** | latest | MIMO 矩阵运算, SVD |
| **Requests** | latest | HTTP 客户端 (Overpass/Nominatim) |
| **PyYAML** | latest | config.yaml 解析 |
| **Pydantic** | v2 | 输入参数验证 |

### 前端

| 技术 | 版本 | 用途 |
|------|------|------|
| **React** | 18.3.1 | UI 框架 (Hooks) |
| **Three.js** | 0.169.0 | WebGL 3D 渲染引擎 |
| **Vite** | 6.0.3 | 前端构建工具 (ESM + HMR) |
| **react-markdown** | 9.0.1 | AI 回复 Markdown 渲染 |
| **OrbitControls** | Three.js addon | 3D 相机交互控制 |
| **Canvas 2D API** | 浏览器原生 | 统计图表绘制 (无第三方库) |
| **WebSocket API** | 浏览器原生 | 实时双向通信 |
| **localStorage** | 浏览器原生 | 会话持久化 |
| **CSS Custom Properties** | 原生 CSS | 设计系统变量 |
| **Google Fonts** | Orbitron, Noto Sans SC | 界面字体 |

### C++ / 仿真

| 技术 | 说明 | 用途 |
|------|------|------|
| **C++17** | MSVC 编译 | 仿真引擎语言 |
| **Altair WinProp SDK** | 商业库 | RF 传播预测 + MIMO 信道计算 |
| **Windows API** | `windows.h` | 进程控制、控制台着色 |
| **STL** | `<filesystem>`, `<map>`, `<string>` | 标准库 |

### 外部服务

| 服务 | 用途 |
|------|------|
| **OpenStreetMap Nominatim** | 地名 → 经纬度地理编码 |
| **OpenStreetMap Overpass** | 建筑几何数据查询 (3 镜像故障转移) |
| **LangSmith** (可选) | Agent 执行链路追踪与调试 |

### 数据资源

| 资源 | 规模 | 用途 |
|------|------|------|
| 陕西省 Shapefile | 42 个 SHP 文件 | 本地建筑数据 |
| 郑州市 Shapefile | 多个 SHP 文件 | 本地建筑数据 |
| WinProp 天线方向图 | Sector.apb, dipole.apa | 5G/偶极子天线 |

---

## 十、项目目录结构

```
AeroRay/
├── config.yaml                          # 全局配置 (LLM profile, 网络, 路径)
├── server.py                            # FastAPI 主服务 (662 行)
├── trace.html                           # Agent 执行监控页面
│
├── skill/                               # Python Skill 层
│   ├── __init__.py                      # Skill 编排 + Pydantic 模型 + 工具注册 (523 行)
│   ├── common.py                        # 公共配置/常量/工具函数 (451 行)
│   │
│   ├── find_area/                       # [Skill 1] 区域解析
│   │   └── skill_find_area.py           # Nominatim 地理编码 + 手动坐标解析
│   │
│   ├── find_shp/                        # [子Skill] 本地 SHP 匹配
│   │   ├── skill_find_shp.py
│   │   ├── shanxisheng/                 # 陕西省行政区划 SHP (42 文件)
│   │   └── zhengzhoushi/                # 郑州市 SHP
│   │
│   ├── split_shp/                       # [子Skill] SHP 裁剪/简化
│   │   └── skill_split_shp.py
│   │
│   ├── shp_to_json/                     # [子Skill] 几何 → JSON 像素坐标
│   │   └── skill_shp_to_json.py
│   │
│   ├── json_to_shp/                     # [子Skill] JSON → 3D SHP
│   │   └── skill_json_to_shp.py
│   │
│   ├── create_odb/                      # [Skill 3] SHP → ODB/OIB
│   │   ├── skill_create_odb.py          # Python 调用入口
│   │   ├── convert_database.cpp         # C++ 源码 (433 行)
│   │   └── SHP2ODB_debug.exe            # 编译产物
│   │
│   ├── visualize/                       # [Skill 4] 2D/3D 场景可视化
│   │   └── skill_visualize.py
│   │
│   ├── single_dpm_mimo/                 # [Skill 5-6] 射线追踪仿真
│   │   ├── skill_single_dpm_mimo.py     # prepare + execute (578 行)
│   │   ├── outdoor_propagation_runms_mimo_single.cpp  # C++ 源码 (629 行)
│   │   └── source_tool/                 # 仿真工具包
│   │       ├── PropagationRunMSMIMOSingle_debug.exe
│   │       ├── Sector.apb               # 5G 扇区天线方向图
│   │       ├── dipole.apa               # 偶极子天线方向图
│   │       └── map0.odb                 # 示例建筑数据库
│   │
│   ├── show_ray/                        # [API] 射线数据解析
│   │   └── skill_show_ray.py            # .str 文件解析
│   │
│   ├── visualize_radiomap/              # [Skill 7] RadioMap 热力图
│   │   └── skill_visualize_radiomap.py  # Path Loss → 3D 点云 (391 行)
│   │
│   ├── coverage_analysis/               # [Skill 8] 覆盖分析
│   │   └── skill_coverage_analysis.py
│   │
│   ├── mimo_analysis/                   # [Skill 9] MIMO 分析
│   │   └── skill_mimo_analysis.py       # SVD 分解 + 四指标计算 (525 行)
│   │
│   ├── low_altitude/                    # 低空场景基础层(单基站,见 §8.1~§8.6)
│   │   ├── skill_low_altitude.py        # prepare/execute/plan 三个入口
│   │   ├── field_aggregator.py          # N 层 Path Loss → 3D npz 体素场
│   │   └── astar_3d.py                  # 26 邻接 A* + 三轮平滑 + 信号优先度
│   │
│   └── multi_bs/                        # 🌟 平台核心 — 多基站低空电磁孪生(见 §8.7)
│       ├── skill_multi_bs.py            # 编排:prepare/execute_multi_bs +
│       │                                #       plan_multi_bs_route + plan_swarm_routes
│       ├── field_store.py               # per-BS 张量存取 + 默认基站布局 + 建筑障碍 mask
│       ├── handover_astar.py            # 状态空间 (x,y,z,k) 的分层图 A*,自动切换基站
│       └── swarm_planner.py             # 优先级解耦 + 容量约束 + 强制初始 BS 多样化的蜂群规划
│
├── frontend/                            # React + Three.js 前端
│   ├── package.json                     # 依赖声明
│   ├── vite.config.js                   # Vite 配置 (代理规则)
│   ├── index.html                       # HTML 入口
│   ├── dist/                            # 生产构建输出
│   └── src/
│       ├── main.jsx                     # React 入口
│       ├── App.jsx                      # 根组件,所有面板状态编排
│       ├── App.css                      # 全局布局样式
│       ├── index.css                    # CSS 变量定义
│       ├── 砖块.png                     # 建筑砖块纹理
│       └── components/
│           ├── ThreeScene.jsx           # 3D 场景渲染(建筑/热力图/多基站塔/3D 信号场/蜂群动画)
│           ├── ChatPanel.jsx            # AI 对话面板(含 multi_bs ws 消息分发)
│           ├── FeaturePanel.jsx         # 功能面板(核心 / 扩展 两个分组)
│           │
│           ├── MultiBSPanel.jsx         # 🌟 多基站参数配置(模式切换 + 基站表格)
│           ├── InterferenceOverlay.jsx  # 🌟 4 种干扰分析模式选择浮窗
│           ├── multiBsField.js          # 🌟 per-BS 张量解码 + SINR/dominance/overlap 计算
│           ├── UAVPlanPanelV2.jsx       # 🌟 当前统一低空航线规划面板(单机 / 蜂群 / 清除路线)
│           ├── UAVPlanPanel.jsx         # 早期统一版面板(已由 V2 接管主流程)
│           ├── MultiBSUavPanel.jsx      # 旧版单 UAV 面板(保留兼容)
│           ├── SwarmPanel.jsx           # 旧版蜂群面板(保留兼容)
│           │
│           ├── ParamsPanel.jsx          # 地面单基站仿真参数面板
│           ├── CoverageOverlay.jsx      # 覆盖分析浮窗
│           ├── MimoOverlay.jsx          # MIMO 分析面板
│           ├── RayInfoPanel.jsx         # 射线详情
│           ├── TopDownView.jsx          # 2D 俯视图(航线起终点拾取)
│           └── *.css                    # 各组件样式
│
├── resu/                                # 仿真结果存储
│   └── YYYYMMDD_HHMMSS/                 # 时间戳目录 (每次任务一个)
│       ├── polygon.json                 # 区域多边形坐标
│       ├── json/buildings_with_height.json
│       ├── json/area_meta.json          # 像素尺寸/缩放/经纬度边界
│       ├── shp/result_final.{shp,odb}
│       ├── ray_tracing/                 # 单基站 RT 结果(Path Loss.txt + .fpl 等)
│       ├── scene_data.json              # 前端 3D 场景统一数据
│       ├── ray_tracing_params.json      # 用户确认后的单基站仿真参数
│       ├── multi_bs_params.json         # 🌟 多基站仿真用户确认参数
│       ├── multi_bs/                    # 🌟 多基站子系统输出
│       │   ├── BS1/PathLoss_h100.txt    # 每个 (基站, 高度) 独立子目录
│       │   ├── BS1/PathLoss_h120.txt
│       │   ├── BS2/...
│       │   ├── multi_bs_field.npz       # per-BS path loss 张量(单 key: field)
│       │   └── multi_bs_field_meta.json # mode/shape/stations/heights/...
│       └── low_altitude/                # 单基站低空子系统输出(基础层)
│           ├── h100/PathLoss.txt
│           ├── ...
│           ├── low_alt_field.npz
│           └── low_alt_field_meta.json
│
└── ref/                                 # 参考脚本
    ├── main.py                          # CLI Agent (终端交互版)
    └── learn_langchain.py               # LangChain 学习笔记
```

---

## 十一、部署与运行

### 环境要求

- **操作系统：** Windows 10/11（C++ EXE 为 Windows 编译）
- **Python：** 3.10+
- **Node.js：** 18+（前端构建）

### 启动步骤

```bash
# 1. 安装 Python 依赖
pip install fastapi uvicorn[standard] langchain langgraph langchain-anthropic langchain-openai
pip install geopandas shapely fiona matplotlib numpy pandas requests pyyaml

# 2. 构建前端
cd frontend
npm install
npm run build
cd ..

# 3. 配置 LLM (编辑 config.yaml 中的 api_key 和 active profile)

# 4. 启动服务
python server.py
# 或
uvicorn server:app --host 0.0.0.0 --port 7860
```

### 访问地址

- **主界面：** `http://localhost:7860`
- **监控面板：** `http://localhost:7860/trace`
- **场景 API：** `GET http://localhost:7860/api/scene?session_id=xxx`
- **射线 API：** `GET http://localhost:7860/api/rays?run_dir=...&x=...&y=...`
- **多基站 per-BS 张量 API:** `GET http://localhost:7860/api/multibs/field?run_dir=...`
- **单基站低空信号场 API:** `GET http://localhost:7860/api/lowalt/field?run_dir=...`

### REST API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/scene` | 获取当前会话 3D 场景数据 |
| GET | `/api/rays` | 根据像素坐标获取射线数据 |
| GET | `/api/multibs/field` | 获取多基站 per-BS path loss 张量二进制(float32 C 顺序,Header: `X-Field-Shape`) |
| GET | `/api/lowalt/field` | 获取单基站低空 3D 信号场二进制(float32 C 顺序,Header: `X-Field-Shape`) |
| GET | `/trace` | 监控页面 |
| WS | `/ws/chat` | 主通信通道 (双向) |
| WS | `/ws/trace` | 监控通道 (只读广播) |
| GET | `/{path}` | 前端 SPA 路由 |

---

> **AeroRay** — 自然语言驱动 · 多基站低空电磁孪生 · 干扰态势分析 · 蜂群航线规划 · 实时三维可视化
