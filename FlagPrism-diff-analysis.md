# FlagPrism 相对 3.5.x 改动分析

## 1. 基线与统计边界

- 官方基线：`flagos-ai/triton_v3.5.x@317f15a426466633c4f37f164b2c58ae9c31bd03`
- 当前主仓 HEAD：`890daf9326c8806432e75642135d49ffe4ec8cb5`
- 当前 FlagPrism 子模块 HEAD：`a2e07e574307ddae48aae9a181bcfaceb1a5787f`
- 统计包含当前工作区中属于 FlagPrism 接入的 tracked/untracked 源文件。
- 统计排除官方基线旧 `third_party/proton/**` 的文件级删除、审计产物、缓存、
  build、egg-info 和其他生成物。
- 主仓与子模块修改尚未提交；当前 gitlink 仍指向修改前的子模块 HEAD。

本轮采用“公开入口改名，内部协议不改名”的边界：

| 层次 | 当前约定 |
| --- | --- |
| 公开 Python API | `flagtree.debugger`、`flagtree.profiler` |
| 源码目录 | `third_party/FlagPrism/Debugger`、`third_party/FlagPrism/proton` |
| Profiler 内部名称 | Proton dialect、`libproton`、`PROTON_*`、Proton CLI 均保留 |
| Debugger 编译 binding | `triton._C.libtriton.debugger`，作为标准 libtriton plugin 构建 |
| Debugger runtime binding | `flagtree.debugger._native`，不链接 `libtriton` 或 MLIR |
| 构建开关 | `TRITON_BUILD_FLAGPRISM=ON/OFF` 统一控制 Debugger 与 Profiler |
| 发布方式 | 两个组件随 FlagTree 主 wheel 共同构建和发布 |

FlagTree 原有 `-fvisibility=hidden` 和链接器 `--exclude-libs,ALL` 保持不变，没有为了
Debugger 导出 core C++ ABI。

## 2. 改动规模

### 2.1 FlagTree 主仓

相对上述 3.5.x 基线，主仓接入代码为 **34 个文件，新增 718 行，删除 145 行**。
其中 31 个是 tracked diff（`+344/-145`），3 个是当前新增文件（共 374 行）：

- `python/flagtree/__init__.py`
- `python/triton/_flagprism.py`
- `python/test/unit/test_flagprism.py`

另有一个 `third_party/FlagPrism` gitlink，不计代码行。34 个文件中有 5 个只把
示例或 benchmark 的公开 import 改为 `flagtree.profiler`。没有 Proton dialect syntax、
native 库名或环境变量的机械重命名。

### 2.2 FlagPrism 子模块

相对子模块当前 HEAD，工作区改动为：

| 分类 | 文件数 | 新增 | 删除 | 说明 |
| --- | ---: | ---: | ---: | --- |
| Debugger | 60 | 513 | 829 | 联合构建、compiler/runtime native 拆分、FlagPrism 品牌与测试 |
| Proton | 53 | 394 | 449 | 联合构建、公开入口、集中 registration、独立 C++ test target 与 Ascend runtime |
| 共享文件 | 4 | 319 | 7 | `.gitignore`、顶层 README、单一模式 CMake 与 packaging policy |
| 合计 | 117 | 1226 | 1285 | 包含 9 个尚未 tracked 的新增源文件 |

删除量主要来自移除两套独立 wheel 的 `setup.py`、`pyproject.toml` 和 `MANIFEST.in`。

## 3. 主仓逐文件说明

### 3.1 构建、打包与 submodule 边界（4 个文件）

| 文件 | 改动 | 目的 |
| --- | ---: | --- |
| `.gitmodules` | +4 | 声明唯一 `FlagPrism` submodule 并标注源码边界 |
| `CMakeLists.txt` | +33 | 块注释保留旧 Proton 接线，可选加载 submodule policy，并通过统一入口在正确生命周期加入组件 |
| `Makefile` | +5/-2 | 注释保留旧 Proton 测试命令并更新实际物理路径 |
| `setup.py` | +54/-23 | 用 `runpy` 加载 submodule packaging policy，并在父 wheel 生命周期中调用；各接点带 FlagPrism 所有权注释 |

`third_party/FlagPrism/python/flagprism_build.py` 是唯一打包策略实现。主 `setup.py`
仅负责动态加载，并在 package discovery、CMake 参数、CLI 和 build tree 生命周期调用。
policy 只接受联合构建和 core-only 两种模式；submodule 缺失时只有 core-only 可继续。

### 3.2 通用组件网关与 DSL 接点（13 个文件）

| 文件 | 改动 | 目的 |
| --- | ---: | --- |
| `python/flagtree/__init__.py` | +5 | 建立公开 `flagtree` namespace |
| `python/src/main.cc` | +5/-2 | 将既有 plugin 宏容量从 6 扩到 8，并标明由 FlagPrism plugin 使用 |
| `python/test/unit/test_flagprism.py` | +202 | 锁定组件注册、兼容性、统一构建提示、hook 契约和两种模式的构建目录清理 |
| `python/triton/_flagprism.py` | +167 | core 调用两个组件的唯一 Python 网关 |
| `python/triton/compiler/code_generator.py` | +7/-2 | 在 assignment/expression 生成点调用 statement hook |
| `python/triton/compiler/compiler.py` | +8/-1 | 注册组件 dialect，并在 IR override 后处理最终 module |
| `python/triton/language/__init__.py` | +6 | 导出 collect DSL marker |
| `python/triton/language/core.py` | +18 | 定义 `debug_collect_start/end` 薄入口 |
| `python/triton/runtime/jit.py` | +4/-1 | 在 specialization/cache key 前合并组件 compile options |
| `python/triton/spec/ascend/compiler/code_generator.py` | +7/-2 | Ascend 镜像 statement 接点 |
| `python/triton/spec/ascend/compiler/compiler.py` | +8/-1 | Ascend 镜像 compiler 接点 |
| `python/triton/spec/ascend/language/core.py` | +18 | Ascend 镜像 DSL marker |
| `python/triton/spec/ascend/runtime/jit.py` | +4/-1 | Ascend 镜像 JIT options 接点 |

这些文件不包含组件实现，只依赖 `triton._flagprism`。`create_store` 仍返回 `None`，
Debugger compiler plugin 的 `annotate_statement_operation` 直接从 insertion point 获取
刚创建的 StoreOp。普通和 Ascend `ir.cc` 相对 3.5.x 均已恢复为零 diff，因此没有改变
StoreOp、Python builder ABI 或其他后端的 launcher ABI。

### 3.3 公开 Profiler import 更新（5 个文件）

| 文件 | 改动 | 目的 |
| --- | ---: | --- |
| `python/triton_kernels/bench/bench_mlp.py` | +2/-1 | 使用 `flagtree.profiler as proton` 并标注 FlagPrism API |
| `python/triton_kernels/bench/roofline.py` | +2/-1 | 使用新的公开入口并标注所有权 |
| `python/triton_kernels/tests/test_routing.py` | +2/-1 | 测试使用新的公开入口并标注所有权 |
| `python/tutorials/09-persistent-matmul.py` | +4/-2 | 更新 import/提示，保留局部别名 `proton` 并标注所有权 |
| `python/tutorials/10-block-scaled-matmul.py` | +2/-2 | 仅更新必要 import；按要求不增加注释 |

### 3.4 Proton 工具注册与测试（4 个修改文件，1 个父仓文件恢复基线）

| 文件 | 改动 | 目的 |
| --- | ---: | --- |
| `bin/CMakeLists.txt` | 0 | 恢复 3.5.x 链接列表；FlagPrism 提供真实或空的 `TritonTestProton` target |
| `bin/RegisterTritonDialects.h` | +12/-21 | 只调用 Proton 自有的 production/test 集中 registration API，并标注被移除 dialect 的 FlagPrism 所有权 |
| `test/lib/CMakeLists.txt` | +3 | 块注释保留并禁用父仓 Proton C++ test 入口 |
| `test/lib/Proton/CMakeLists.txt` | +3 | target 定义迁入 FlagPrism，父仓旧定义仅保留为块注释 |
| `test/lib/Proton/TestScopeIdAllocation.cpp` | +1/-51 | test pass 源码和注册迁入 FlagPrism，父仓保留一行 tombstone |

### 3.5 Ascend 后端接点（4 个修改文件，2 个父仓文件恢复基线）

| 文件 | 改动 | 目的 |
| --- | ---: | --- |
| `third_party/ascend/backend/compiler.py` | +8 | 最后结构化 IR hook 与 hashable instrumentation mode |
| `third_party/ascend/backend/driver.py` | +67/-6 | 以 `_DebuggerHiddenArgABI` 集中 hidden argument 和 launch context，并标注 FlagPrism 接入边界 |
| `third_party/ascend/bin/CMakeLists.txt` | 0 | 恢复 3.5.x 链接列表；复用真实或空的 `TritonTestProton` target |
| `third_party/ascend/bin/RegisterTritonDialects.h` | +12/-21 | 使用 Proton production/test 集中 registration API，并标注 dialect 迁移 |
| `third_party/ascend/python/src/ir.cc` | 0 | operation 注解 helper 已迁入 Debugger compiler plugin |
| `third_party/ascend/python/src/main.cc` | +5/-2 | 与 normal binding 相同地扩展 plugin 宏容量并标明用途 |

hidden argument 只有 metadata 明确包含 `debug_launch_hidden_arg` 时才进入 Ascend launcher；
CUDA/HIP 等 launcher 不会因全局 Debugger 状态改变 ABI。

### 3.6 其他后端构建兼容（4 个文件）

| 文件 | 改动 | 目的 |
| --- | ---: | --- |
| `third_party/enflame/cmake/triton_gcu.cmake` | +21 | 从 submodule helper 获取 Proton object 列表并传递统一开关；块注释保留旧 glob |
| `third_party/enflame/cmake/triton_gcu300.cmake` | +4 | 块注释保留旧硬编码 Proton object 清单 |
| `third_party/enflame/cmake/triton_gcu400.cmake` | +4 | 块注释保留另一份旧清单 |
| `third_party/tsingmicro/bin/RegisterTritonDialects.h` | +11/-2 | Proton ON 时调用集中 registration API，并标注 FlagPrism 所有权 |

## 4. 集中化与剩余重复

本轮已经完成的集中化：

- CMake policy 集中在 `third_party/FlagPrism/cmake/FlagPrism.cmake`；主
  `CMakeLists.txt` 只保留可选 include、缺失检查和两个调用同一
  `flagprism_add_components()` 的生命周期点。
- Python 生命周期集中在 `triton._flagprism`；compiler、JIT、DSL 和 launcher 不直接
  import Debugger/Profiler 实现。
- wheel policy 集中在 `third_party/FlagPrism/python/flagprism_build.py`。
- Ascend hidden argument 的 parse、struct、call fragments 集中在一个 dataclass。
- Proton dialect/pass 注册和 Enflame object list 由 Proton submodule 自己维护。
- Debugger runtime 不再依赖 libtriton，因此无需放宽 FlagTree 全局符号可见性。

当前不继续合并的重复：

- `python/triton/*` 与 `python/triton/spec/ascend/*` 是 FlagTree 3.5.x 已存在的两套前端
  实现；四个薄 hook 必须成对接入，否则不同 Ascend 路径行为不一致。
- `python/src/main.cc` 与 `third_party/ascend/python/src/main.cc` 是两个独立 binding
  target，每侧仍需保留 4 行 plugin 宏容量扩展；两份 `ir.cc` 已恢复到 3.5.x。
- void statement operation 注解已集中到一份 Debugger `CompilerBindings.cpp`，该源码
  会按当前构建选择的普通或 Ascend `ir.h` 编译，不再在父仓复制 helper。
- `bin`、Ascend `bin` 和 TsingMicro 各自拥有 dialect registry header；现在每处只调用
  一个 Proton registration 函数。进一步统一需要先重构 FlagTree 原有工具架构，超出
  FlagPrism 接入范围。

## 5. 子模块内部边界

- `cmake/FlagPrism.cmake` 维护单一工具套件开关、源码检查、target 和 dialect/plugin
  接入策略；FlagTree 主仓不再保存该实现。
- `python/flagprism_build.py` 维护 package 映射、CLI、CMake 参数和复用 build tree 清理；
  FlagTree `setup.py` 只保留加载与调用接点。

### 5.1 Debugger

- `CompilerPlugin.cpp` 与 compiler IR/pass 源码编入 `libtriton`，沿用标准 plugin loader。
- `annotate_statement_operation` 在插件内完成无结果 operation 定位和属性写入，父仓
  normal/Ascend IR binding 无需增加 Debugger API。
- `_native` 只包含 buffer、transfer、decode、report 和 host runtime；ELF `NEEDED` 中无
  `libtriton`/MLIR。
- compiler/runtime 两侧通过版本化 metadata/record 协议通信。
- `statement.py`、`language.py` 和全部实现模块直接进入主 wheel，不依赖未 tracked shim。
- 移除独立 Debugger wheel 文件。

### 5.2 Proton

- 保留 `proton/`、Proton dialect、C++ namespace、`libproton`、`PROTON_*` 与 CLI。
- 新增 `Integration/Registration` 和 `ProtonDialectObjects.cmake`，让 FlagTree 只依赖稳定
  接口；原 `test/lib/Proton` pass 迁入独立 `TritonTestProton` target，并通过单独的 test
  registration API 注册，不混入生产 `ProtonRegistration`。
- CANN、instrumentation、trace、Hatchet 和 legacy CUDA/HIP 语义均保留。
- 移除独立 Proton wheel 文件，仅把公开 Python 安装路径映射为 `flagtree.profiler`。

## 6. FlagPrism 改名与验证结果

联合与 core-only 的全量编译、wheel 和实机结果来自本轮功能修改完成后的独立构建目录。
随后新增的审计注释和统一 CMake 入口不改变生成代码；当前工作区又分别使用全新目录完成
联合、core-only 和非 Python 工具三种 CMake configure，避免复用旧 target graph。

环境：`flagtree-cann9-quan`，Ascend 物理 device 1。

| 检查 | 结果 |
| --- | --- |
| 联合 wheel 全量编译 | 通过；851 个 target，Debugger、Profiler、compiler plugin 与 native runtime 均进入同一 wheel |
| 联合 wheel 内容检查 | 仅保留当前 ABI 的 `libproton.cpython-311-aarch64-linux-gnu.so`，无旧 package、旧网关或残留 `.pyc` |
| 联合 wheel 隔离测试 | 32 passed，3 deselected；`triton._flagprism`、Debugger native、Profiler CANN smoke 均从安装目录加载 |
| Debugger level 2 实机示例 | device 1，`output_allclose=True`，导出 1 个 report 与 `.npy` artifacts |
| Profiler CANN 实机示例 | device 1，生成 Hatchet、metadata、timeline 和 vendor 四类输出，180 个 association |
| core-only wheel 全量编译 | 通过；762 个 target，不构建 Debugger/Profiler target |
| core-only wheel 内容检查 | 无 `flagtree.debugger`、`flagtree.profiler`、`libproton`、console script 或旧组件残留 |
| core-only 实机示例 | device 1，vector-add 最大误差 `0.0`，缺失组件提示符合预期 |
| 当前 CMake 接线 | 联合、core-only、非 Python 工具三种全新 configure 均通过；target graph 分别包含完整套件、无组件、Proton dialect/pass |
| FlagPrism 网关/清理单测 | 10 passed |
| root/submodule `git diff --check` | 当前工作区通过 |

当前联合 wheel：

`/tmp/flagtree-verify-20260804-joint-v2/wheel-clean/flagtree-0.6.0+ascend.git890daf93-cp311-cp311-linux_aarch64.whl`

SHA-256：

`f46d48e16e919340332fb52c48f2145b56fc937ba9a2314d8181898da3331516`

当前 core-only wheel：

`/tmp/flagtree-verify-20260804-core/wheel/flagtree-0.6.0+ascend.git890daf93-cp311-cp311-linux_aarch64.whl`

SHA-256：

`f974779ac3fca7bfae90691e1d6161d422724edbd49c93ff3afb8188a40843cc`

本轮实机报告为：

`/tmp/flagtree_debugger_level2_example/abs_level2_debug_abs_kernel_20260804_023215_916_run1.txt`

旧 `TRITON_BUILD_DEVTOOLS`、`TRITON_BUILD_PROTON` 等变量仅作为兼容输入保留；新构建
统一使用 `TRITON_BUILD_FLAGPRISM`，只支持联合构建和 core-only。稳定公开 API
`flagtree.debugger`、`flagtree.profiler`，以及 Proton/Debugger 的内部 IR 与 native ABI
名称没有机械改名。

## 7. 当前提交状态

修改仍在工作区，尚未提交。必须先在 `FlagPrism` 仓库提交当前修改，再在主仓
更新 gitlink；否则 clean checkout 只能取得 `a2e07e5`，不能复现上述构建。

`FlagPrism-core.diff` 只导出 FlagTree 主仓接线，不包含旧
`third_party/proton/**` 文件删除或 submodule 内部实现。由于 3 个新增主仓文件当前仍
untracked，diff 快照需要在每轮实现后重新生成。当前 `.gitmodules` 已使用
`third_party/FlagPrism` 作为 section/path；按仓库约束，远程 URL 不参与品牌改名，
固定保留 `https://github.com/ZQ-Struggle/FlagTree_DevTools.git`。
