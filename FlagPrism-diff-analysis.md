# FlagPrism 相对 FlagTree 3.5.x 改动分析

## 1. 基线与统计边界

- FlagTree 基线：`flagos-ai/triton_v3.5.x@317f15a426466633c4f37f164b2c58ae9c31bd03`。
- 当前父仓 HEAD：`e7a173733872b2c6c9a9da81492d4a0f9fd74dc3`，本轮修改尚未提交。
- 当前 FlagPrism HEAD：`a52b38cee72d92ac78cdffcc810a2868e59283b3`，本轮修改尚未提交。
- 主仓统计排除基线 `third_party/proton/**` 的迁移删除、FlagPrism gitlink、三个审计文件
  及构建生成物。
- Debugger 与 Profiler 作为 FlagPrism submodule 随 FlagTree 共同打包到一个 wheel；只
  支持默认联合构建和 `TRITON_BUILD_FLAGPRISM=OFF` 的 core-only 构建。

当前命名边界如下：

| 层次 | 名称 |
| --- | --- |
| 公开 Python API | `flagtree.debugger`、`flagtree.profiler` |
| 工具源码 | `third_party/FlagPrism/Debugger`、`third_party/FlagPrism/Profiler` |
| Profiler native | `flagtree/profiler/_native*.so`，target 为 `flagtree_profiler_native` |
| CLI | `flagtree-profiler`、`flagtree-profiler-viewer` |
| Profiler 环境变量 | FlagPrism runtime 使用 `FLAGTREE_PROFILER_*`；父仓原有 kernel metadata 选项保留旧名 |
| 内部编译协议 | `proton`/`proton_gpu` MLIR dialect、`TritonProton`、`__PROTON__` |
| Host 接口 | `triton._flagprism`，core 不直接导入组件实现 |

不提供 `triton.profiler`、旧 Proton CLI 或 `libproton` 兼容入口。Proton 作为既有
compiler dialect 名以及父仓未改动的局部 helper/别名保留，避免无功能收益的
机械重命名。

## 2. 当前改动规模

### 2.1 FlagTree 主仓

相对上述 3.5.x 基线，主仓接口和公开命名改动为 **36 个文件，`+1010/-148`**，另有
一个不计代码行的 `third_party/FlagPrism` gitlink。

| 审计区域 | 文件数 | 新增 | 删除 | 主要所有权 |
| --- | ---: | ---: | ---: | --- |
| 构建、发布与仓库边界 | 4 | 95 | 30 | 父仓构建/发行生命周期 |
| Python Host 网关与 Statement/DSL | 13 | 742 | 4 | Host 协议与 frontend/JIT 接点 |
| Profiler 公开命名传播 | 5 | 13 | 8 | 父仓示例和测试的实际 import/CLI |
| Profiler 内部 Proton 注册与 C++ test 迁移 | 5 | 21 | 73 | 静态 registry 与条件链接 |
| Ascend backend/launcher 接点 | 4 | 97 | 29 | 后端序列化和 launch ABI |
| Enflame/TsingMicro 构建兼容 | 5 | 42 | 4 | 后端私有构建/registry |
| **合计** | **36** | **1010** | **148** | 另有 FlagPrism gitlink |

新增行主要集中在 `python/triton/_flagprism.py`（345 行）和
`python/test/unit/test_flagprism.py`（297 行）。两者共占新增行的 62.0%，分别是 Host
协议实现和边界回归测试，不是 Debugger/Profiler 业务实现。

### 2.2 与上一份审计快照比较

上一份快照为 40 个文件、`+1035/-177`。本轮删除不影响执行的公开命名
传播后为 36 个文件、`+1010/-148`，即 **减少 4 个文件、25 行新增和 29 行删除**。

恢复到 3.5.x 的 4 个文件是 `.github/PULL_REQUEST_TEMPLATE.md`、
`python/triton/knobs.py`、`python/triton_kernels/triton_kernels/matmul_ogs_details/_common.py`
和 `python/triton_kernels/triton_kernels/proton_opts.py`。benchmark、test 和 tutorial 中的局部
`proton` 别名也恢复，仅保留实际执行所需的 `flagtree.profiler` import 和新 CLI。

### 2.3 FlagPrism 子模块

子模块相对当前 HEAD 为 272 个文件、`+1619/-1510`；Git 检测为 232 个 rename、21 个
modify、8 个 add 和 11 个 delete。文件数高主要来自 `proton/ -> Profiler/` 的目录迁移，
不是 272 份独立功能修改。

Profiler 的外部边界已完成改名：源码目录、Python package、native module、runtime CMake
target、CLI、环境变量、文档、脚本和测试均使用 Profiler。保留的 Proton 名称只位于
compiler dialect/target、内部编译 binding 和既有 C++ implementation namespace。

## 3. 必须留在 FlagTree 的接口

### 3.1 Host 协议

`python/triton/_flagprism.py` 定义 Host API 2.x、capability 协商和结构化事件。兼容性按
接口 major/minor 与 capability 判断，不按 FlagTree 3.5/3.6 版本字符串判断：

| Capability | Host 提供的接点 |
| --- | --- |
| `compiler.dialects.v1` | 新 MLIR context 的 dialect 注册 |
| `compiler.options.v1` | cache key 固化前的编译选项注入 |
| `compiler.events.v1` | override 后的 compiler lifecycle event |
| `frontend.statement_events.v1` | 规范化 Triton statement event |
| `language.debug_collect.v1` | `tl.debug_collect_start/end` builtin 转发 |
| `runtime.launch_context.v1` | 带 backend 名称的通用 launch context |

因此同一 FlagPrism 可兼容任何提供 API 2.x 和上述 capability 的 FlagTree 版本。新版本
FlagTree 仍需移植这层薄 Host 接口，但 FlagPrism 不需要为 3.5/3.6 复制组件实现。

### 3.2 生命周期硬接点

下列接点由 FlagTree 自身拥有，不能从 submodule 单方面注入：

- 两套 compiler 在 context 创建、IR override 完成和 Ascend 序列化前发布事件。
- 两套 CodeGenerator 在 AST 与新生成 IR value 同时可用时发布 statement event。
- 两套 JIT 在 specialization/cache key 固化前注入 instrumentation option。
- 两套 language 静态导出 `debug_collect_start/end` builtin。
- Ascend compiler/driver 提供 pre-serialize 和 hidden argument launch ABI 接点。
- 原生工具 registry 调用 Profiler 提供的内部 Proton 集中注册函数。
- 根 CMake/setup 把 submodule target、Python package、native extension 和 CLI 放进主 wheel。

实现、buffer 管理、报告导出、statement annotation 和后端 adapter 均留在 FlagPrism。

## 4. Profiler 改名后的主仓影响

改名本身只要求以下父仓变化：

- `setup.py` 从 FlagPrism policy 获取 `flagtree.profiler` package mapping、新 native 输出和
  新 CLI；父仓不包含 Profiler 构建实现。
- benchmark/tutorial/test 从 `triton.profiler` 切换到 `flagtree.profiler`。
- 父仓原有 `knobs.proton`、`proton_opts.py` 和局部别名不改；它们不决定
  FlagPrism Profiler 的公开 package 或 CLI。
- C++ registry include 路径从旧 `proton/...` 指向 `FlagPrism/Profiler/Dialect/...`，注释
  明确这些 Proton symbol 是 Profiler 的内部 compiler dialect。
- Enflame nested build 的源码和 object helper 路径切换到 `FlagPrism/Profiler`。

以下内部名称不改，因它们已经进入 MLIR 文本、pass 名、生成 header、C++ namespace 和
测试路径，改名会制造大范围无语义 churn：

- `proton`、`proton_gpu` dialect 文本名；
- `mlir::triton::proton` namespace；
- `ProtonIR`、`ProtonGPUIR`、`TritonProton`、`TritonTestProton` target；
- `triton._C.libtriton.proton` compiler binding；
- `__PROTON__` 编译宏和父仓历史 `test/Proton` lit suite。
- 父仓原有 `knobs.proton`、`proton_opts.py` 及
  `PROTON_LAUNCH_METADATA_NOSYNC`。

## 5. 构建与运行边界

- FlagPrism CMake policy：`third_party/FlagPrism/cmake/FlagPrism.cmake`。
- wheel policy：`third_party/FlagPrism/python/flagprism_build.py`。
- 唯一开关：`TRITON_BUILD_FLAGPRISM=ON/OFF`。
- 联合构建生成 `flagtree.debugger`、`flagtree.profiler`、两个 runtime `_native`，并把
  Debugger 和内部 Proton compiler plugin 链入 `libtriton`。
- core-only 构建不发布两个 package/CLI，并清除复用 build tree 中的旧 native artifact。
- Profiler runtime `_native` 不作为 `libproton` 发布；wheel 中没有 `triton.profiler`。

## 6. 验证状态

| 检查 | 当前结果 |
| --- | --- |
| v3.5.x Ascend 联合 wheel 增量编译 | 通过，已生成 `flagtree/profiler/_native*.so` |
| v3.5.x Ascend core-only wheel | 通过；无 Debugger/Profiler package、native 或 CLI 残留 |
| wheel package/entry point 审计 | 仅 `flagtree.profiler` 和两个 `flagtree-profiler*` CLI |
| 隔离 wheel import | `flagtree.profiler`、`_native`、内部 compiler binding 均通过 |
| `triton.profiler` 负向检查 | `find_spec(...) is None` |
| Host 边界测试 | 18 passed |
| Profiler API/CANN 非实机回归 | API 10 passed/2 CANN-contract skips；CANN 20 passed |
| Profiler CLI | 4 passed，三种执行模式均在 device 1 生成报告 |
| Profiler CANN 实机 direct-export | 3 passed，物理 device 1/逻辑 device 0 |
| Profiler viewer | 19 passed（隔离环境安装 `llnl-hatchet` 及其依赖后） |
| 四个 Debugger example | device 1 全部 `output_allclose=True`、`exported_runs=1` |
| level 2 artifacts | abs 11 个 `.npy`；softmax 16 个 `.npy` |
| Python AST | 父仓 23 个、子模块 57 个改动文件通过 |
| 父仓与子模块 `git diff --check` | 通过 |

上表 wheel 和实机结果来自上一轮完整验证。本轮只恢复不影响执行的局部别名和
helper 名称，按约定未重复构建 wheel；已在 quan 的 Python 3.11 容器中对 7 个相关
Python 文件执行语法检查。

最终联合 wheel：

`/tmp/flagprism-profiler-final-wheel/flagtree-0.6.0+ascend.gite7a17373-cp311-cp311-linux_aarch64.whl`

SHA-256：`0bee17ce6529d6bdf50412500b1abcadcaff85b8398e213187b0481f7e047b2d`。

core-only wheel：

`/tmp/flagprism-profiler-core-only-wheel/flagtree-0.6.0+ascend.gite7a17373-cp311-cp311-linux_aarch64.whl`

SHA-256：`5ae410c40ec2bf28c3282b31d6e6925c935a7c51f878da9b6fab584f85550707`。

## 7. Diff 与提交状态

`FlagPrism-core.diff` 保存当前父仓接线相对 3.5.x 的 zero-context raw diff；同一内容内嵌
在 `FlagTree-DevTools-file-by-file-analysis.md` 的 36 个对应文件小节中，不再集中放在
报告末尾。两者排除旧 `third_party/proton/**` 的
迁移删除、FlagPrism gitlink/内部 diff 和审计文件自身。

父仓和子模块当前都有未提交修改。可复现提交必须先提交并推送 FlagPrism，再更新父仓
gitlink；本轮不执行 commit 或 push。
