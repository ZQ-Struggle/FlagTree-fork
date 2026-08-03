# FlagPrism 接入相对 FlagTree v3.5.x 的逐文件审计

## 1. 审计范围

- 对比基线：`317f15a426466633c4f37f164b2c58ae9c31bd03`。
- 审计对象：当前工作区中由 FlagTree 主仓承担的 FlagPrism 接入改动。
- 主仓接入规模：31 个 tracked 文件，`+344/-145`；3 个新增未跟踪文件，
  共 374 行；合计 34 个文件，`+718/-145`。
- `third_party/FlagPrism` gitlink 单独计入仓库结构，不计代码行。
- 基线中的 `third_party/proton/**` 共 197 个文件、16,573 行删除，不纳入下面
  34 个文件的接线统计。它们不是“无法迁移”的主仓改动，而是已经迁入
  `FlagPrism/proton` 的旧实现。
- 子模块内部 Debugger/Proton 实现、构建产物、审计文档和本地临时文件不属于本次
  FlagTree 主体审计。

本文中的“不能迁移”特指：在保持 FlagTree 3.5.x 现有编译、JIT 和 launcher 架构的
前提下，不能把该修改完全移入可选 submodule，并让主仓恢复为零改动。它不表示永远
不能重构，而是表示若要消除该接点，必须先给 FlagTree 增加通用扩展接口，改动通常
会比当前薄接点更大。

## 2. 判定标签

| 标签 | 含义 |
| --- | --- |
| H | 主仓硬接点。生命周期或 ABI 由 FlagTree 持有，不能完全移入 submodule。 |
| P | 主仓策略文件。实现不在主仓，但 Git、发布或构建策略必须由主仓声明。 |
| R | 可继续迁移或消除。当前改动有效，但并非架构上不可迁移。 |
| X | 与 FlagPrism 无关，应拆分提交或从本次改动移除。 |

## 3. 总体结论

当前边界的核心原则是：Debugger 和 Profiler 的功能实现留在
`third_party/FlagPrism`，FlagTree 只保留它自身才能提供的生命周期调用点。
3.5.x 没有覆盖以下需求的统一插件协议，因此主仓不可能保持零修改：

1. AST statement 创建后回调；void statement 的 operation 访问已由 Debugger plugin
   自身处理，不再扩展 core IR pybind。
2. 新 MLIR context 创建、IR override 完成和后端序列化前的回调。
3. JIT specialization/cache key 固化前的组件编译选项注入。
4. `tl.debug_collect_start/end` 这类 DSL builtin 的静态导出。
5. Ascend launcher 的隐藏参数解析、参数结构布局和 launch context。
6. `triton-opt` 等原生工具在编译期构造的 dialect/pass registry。
7. 主 wheel 对 submodule 源码、Python package、native library 和 CLI 的打包声明。

不过，34 个文件并非全部都是硬接点。Proton C++ 测试和两个 IR pybind helper 已完成
迁移；五个示例 import 和若干工程配置仍有继续迁移或消除的空间。下面逐文件说明。

## 4. 构建、发布与仓库边界

### 4.1 当前改动矩阵

本节相对 3.5.x 共有 4 个父仓代码/配置文件，合计 `+96/-25`；另有一个不计代码行的
gitlink。构建和发布实现已经移入 submodule，父仓只保留所有权无法外移的调用点。

| 文件或对象 | 当前 diff | 分类 | 是否必须保留 | 审计结论 |
| --- | ---: | --- | --- | --- |
| `.gitmodules` | `+4` | P | 是 | 声明 submodule 路径和远端，并标注 FlagPrism 所有权 |
| `third_party/FlagPrism` gitlink | mode `160000` | P | 是 | 固定 clean checkout 使用的工具版本 |
| `CMakeLists.txt` | `+33` | H/R | 是 | 块注释保留旧 Proton 接线、include、缺失 guard 和统一生命周期入口 |
| `Makefile` | `+5/-2` | R | 否 | 注释保留旧测试命令，并保持父仓 `make test-proton` 新入口 |
| `setup.py` | `+54/-23` | P/H | 是 | 让两个组件进入同一个 FlagTree wheel |
| `.pre-commit-config.yaml` | `0` | 已恢复 | 否 | submodule lint 由 FlagPrism 自己维护 |
| `.gitignore` | `0` | 已恢复 | 否 | 不处理父仓 egg-info 或本地生成物 |
| `MANIFEST.in` | `0` | 已恢复 | 否 | 复用 3.5.x 已有的 `graft third_party` |

### 4.2 Submodule 元数据与可复现性 [P]

- **`.gitmodules` 的职责**：section/path 为 `third_party/FlagPrism`，远端按约束继续使用
  `https://github.com/ZQ-Struggle/FlagTree_DevTools.git`。工具品牌改名不改变远端 URL。
- **gitlink 的职责**：父仓 index 当前固定到
  `a2e07e574307ddae48aae9a181bcfaceb1a5787f`。只有 gitlink 能规定 clean checkout 获取
  哪个 FlagPrism 版本，submodule 无法在内部声明自己属于哪个父仓版本。
- **初始化方式**：clean checkout 必须执行
  `git submodule update --init --recursive`。默认 wheel 构建启用组件，未初始化时会在
  `setup.py` 加载 policy 阶段直接报出上述命令。
- **当前审计风险**：FlagPrism 工作树仍有未提交修改，而 gitlink 仍指向修改前的 commit。
  因此当前本地代码可以审计，但父仓 commit 尚不能复现它；正式提交顺序必须是先提交并
  推送 FlagPrism，再更新父仓 gitlink。

### 4.3 根 `CMakeLists.txt` (`+33`) [H/R]

父仓 CMake 仅保留三个接入区域：

| 当前位置 | 父仓动作 | 为什么必须位于父仓 |
| --- | --- | --- |
| `CMakeLists.txt:114` | 可选 include `FlagPrism.cmake`；组件开启但源码缺失时失败；core-only 时提供空 `TritonTestProton` | 根工程拥有源码存在性和全局 target graph |
| `CMakeLists.txt:355` | Python 构建调用统一 `flagprism_add_components()` | 必须在 backend 子目录读取 `TRITON_LIBS`、`TRITON_PLUGINS` 前创建 target |
| `CMakeLists.txt:596` | 非 Python 工具调用同一 `flagprism_add_components()` | `triton-opt` 等工具需要 Proton dialect/pass，但不需要 Python runtime |

3.5.x 原有的两条 `TRITON_BUILD_PROTON` option、Python standalone Proton target 块和
非 Python dialect 入口均在原位置用 CMake 块注释保留；前置注释明确说明独立 Proton
配置已移除，Profiler 与 Debugger 由 FlagPrism 统一构建。这些注释不再创建 target 或
cache option，只用于保留基线语义和方便审计。

父仓的两个生命周期点使用同一个 `flagprism_add_components()`。Python 与非 Python
所需 target 不同，两个物理调用点不能在不重排主 CMake 生命周期的情况下合成一个；
具体分支判断已收回 FlagPrism helper，父仓不再了解内部函数名。

以下策略均已迁入 `third_party/FlagPrism/cmake/FlagPrism.cmake`，不再由父仓实现：单一
`TRITON_BUILD_FLAGPRISM` 开关、旧变量兼容、源码完整性检查、`__PROTON__` 定义、Debugger
plugin/native target、Proton runtime/dialect target，以及真实或空的 `TritonTestProton`
选择。兼容变量最终同步为同一个值，不能再表达单组件模式。

`TritonTestProton` 的空 INTERFACE target 只用于兼容 3.5.x 工具已有的固定链接列表。它在
FlagPrism OFF 时不包含源码、符号或运行时代码；联合构建时由 FlagPrism 提供真实测试
target。采用该兼容 target 后，普通和 Ascend 的两个 `bin/CMakeLists.txt` 都已恢复为零
diff。

直接 CMake 与 setuptools 都只产生联合构建或 core-only；setuptools 总是把统一开关
显式传给 CMake。

| 入口 | `TRITON_BUILD_FLAGPRISM` 默认值 | 结果 |
| --- | --- | --- |
| `setup.py`/wheel | ON | Debugger + Profiler 联合构建 |
| 直接 CMake，无 `FLAGTREE_BACKEND` | ON | Debugger + Profiler 联合构建 |
| 直接 CMake，有 `FLAGTREE_BACKEND` | OFF | core-only |

非 Python 的 `triton-opt` 类工具构建即使套件为 ON，也只创建该工具实际需要的 Proton
dialect/pass target，不构建 Python package 或 Debugger runtime；这属于目标类型裁剪，
不是第三种用户可选模式。

需要单独审计一个退化行为：submodule 缺失时无法加载上述直接 CMake 默认值。如果调用者
也未显式传入任何 FlagPrism 开关，根 CMake 会按 core-only 继续并创建空测试 target；
如果调用者显式要求 FlagPrism，则立即失败。setuptools 默认联合构建总会显式要求组件，
因此 wheel 不会因漏拉 submodule 而静默降级。

### 4.4 `setup.py` (`+54/-23`) [P/H]

`setup.py` 仍是统一 wheel 的唯一所有者，submodule policy 位于
`third_party/FlagPrism/python/flagprism_build.py`。父仓当前只在以下既有生命周期调用它：

| 当前位置 | 接点 | 结果 |
| --- | --- | --- |
| `setup.py:141-163` | 用 `runpy` 加载 policy、处理缺失 submodule | 默认构建失败并给出初始化命令；统一开关 OFF 时允许 core-only |
| `setup.py:428-438` | `build_py` 前后清理复用 build tree | CMake 前删除旧组件；`build_py` 复制源码后再次删除旧 package、`.pyc` 和错误 ABI 的 `libproton`，防止联合/core-only wheel 被源码树残留污染 |
| `setup.py:517-526` | 追加组件 CMake 参数 | 显式传递统一开关、Debugger Python 输出目录和 extension suffix |
| `setup.py:564-577` | 开关转换与 Proton 依赖参数 | legacy 开关由 policy 统一转换，仅在联合构建时解析 Proton native 依赖 |
| `setup.py:683-715` | 合并 `package_dir` 和 package list | 把源码映射为 `flagtree.debugger`、`flagtree.profiler`，不创建软链接 |
| `setup.py:778` | 保留 backend link 流程 | 删除旧 `add_link_to_proton()`，避免在父仓生成 `python/triton/profiler` |
| `setup.py:830-835` | 合并 console scripts | 联合构建时发布 `proton` 和 `proton-viewer` |

bootstrap 不能完全删除：`setup.py` 执行时 FlagPrism 尚未安装为 Python package，且 clean
checkout 可能尚未初始化 submodule。父仓必须先判断是否允许 core-only，再从确定路径加载
policy。当前使用 `runpy` 执行受控 helper，已经删除手工 `ModuleSpec`、loader 和
`sys.modules` 注册；具体 package/CMake/CLI 决策仍全部留在 submodule。每个保留接点均以
`FlagPrism` 注释标明其所有权。

wheel 只支持两种组合。`TRITON_BUILD_FLAGPRISM` 是唯一推荐开关；旧
`TRITON_BUILD_DEVTOOLS` 和 `TRITON_BUILD_PROTON` 仅作为兼容输入，同时出现时必须一致，
并统一转换为整套工具的 ON/OFF。

| wheel 组合 | 环境变量 | 进入发行包的组件 |
| --- | --- | --- |
| 联合构建（默认） | 新旧相关开关均不设置 | Debugger + Profiler |
| core-only | `TRITON_BUILD_FLAGPRISM=OFF` | 仅 FlagTree core 和 no-op 网关 |

统一 wheel 中的物理归属如下。Debugger compiler plugin 作为 object 进入 `libtriton`，
不是第二个可独立 import 的 MLIR extension；Debugger runtime 则保持独立 `_native`，避免
要求父仓导出 MLIR C++ ABI。

| wheel 内容 | 构建/拷贝来源 | 关闭组件后的行为 |
| --- | --- | --- |
| `flagtree/debugger/*.py` | `FlagPrism/Debugger/python/flagtree_debugger` package mapping | 整个 package 不发布 |
| `flagtree/debugger/_native*.so` | Debugger runtime CMake target | 从复用 build tree 删除 |
| `triton/_C/libtriton*.so` 中的 `debugger` 子模块 | `TritonDebugger` compiler plugin object | 不注册 Debugger plugin |
| `flagtree/profiler/*.py` 和 `hooks/` | `FlagPrism/proton/proton` package mapping | 整个 package 不发布 |
| `triton/_C/libproton*.so` | Proton native CMake target | 从复用 build tree 删除 |
| `proton`、`proton-viewer` | FlagPrism console-script policy | core-only 时不注册 |

不能把上述调用点全部迁入 submodule：最终 `packages`、`package_dir`、`ext_modules`、CMake
命令和 entry points 都由父仓的 `setuptools.setup()` 创建。可以迁移决策实现，但
submodule 无法自行修改一个已经由父仓执行的 setup 配置。

### 4.5 发布清单与开发入口 [R]

- **`MANIFEST.in` (`0`)**：无需新增 FlagPrism 规则。3.5.x 原文件已经包含
  `graft third_party`，从已初始化 submodule 的工作树生成 sdist 时会带入 FlagPrism
  源码。未初始化 submodule 时，默认组件策略会在 sdist 开始前失败，不会静默生成缺少
  工具源码的发行包。
- **`.gitignore` (`0`)**：此前针对 `python/flagtree.egg-info/` 的修改已回撤。该文件不是
  功能或发布接点，等待主线统一处理本地生成物。
- **`Makefile` (`+5/-2`)**：把 `test-proton` 的两条 pytest 路径改到 submodule，并以
  shell 注释保留旧命令供审计。
  删除此改动不会影响 wheel 或运行时，只会失去父仓原有测试命令的兼容性。
- **`.pre-commit-config.yaml` (`0`)**：父仓改动已回撤。FlagPrism 自己维护 submodule
  内部 lint/CI，避免为一个父仓 hook 增加接入文件。

### 4.6 本节审计结论

必须保留的父仓边界是 `.gitmodules`、gitlink、根 CMake 的三个区域，以及 `setup.py` 的
薄调用点。`Makefile` 只影响开发测试入口，可单独决定是否保留。
`.pre-commit-config.yaml`、`.gitignore`、`MANIFEST.in`、普通/Ascend `bin/CMakeLists.txt`
均已恢复 3.5.x，不应在后续 FlagPrism 功能提交中重新引入修改。

## 5. Python 网关与 Statement/DSL 接点

### 5.1 `python/flagtree/__init__.py` (`+5`) [P]

- **改动**：建立统一公开 namespace，并复用 core `__version__`。
- **必要性**：共同 wheel 对外暴露 `flagtree.debugger` 和 `flagtree.profiler` 时需要稳定
  的父 package；两个组件都关闭时 namespace 的所有权也应保持明确。
- **为何不能迁移**：若把父 package 交给可选 submodule，core-only wheel 中它会消失，
  或需要依赖 PEP 420 namespace 并放弃当前版本入口。五行主仓声明比隐式 namespace
  更容易审计。

### 5.2 `python/triton/_flagprism.py` (`+167`) [H]

- **改动**：定义版本化组件注册、惰性加载、缺失组件错误、兼容性检查，以及 compiler、
  JIT、statement、DSL 和 Ascend launch 的统一回调门面。
- **必要性**：core 文件只依赖这个稳定网关，不直接 import Debugger 或 Profiler；组件
  关闭时 compiler/JIT hook 必须安全地退化为 no-op。
- **为何不能迁移**：它必须在没有初始化 submodule、两个组件都未打包的 core-only wheel
  中仍可导入。若整个网关放入可选包，core 的无条件生命周期调用会失败。可以压缩 API，
  但至少一个主仓 stub/registry 必须保留。

### 5.3 `python/test/unit/test_flagprism.py` (`+202`) [P/R]

- **改动**：测试公开 namespace、统一构建提示、一次加载、API 版本校验、no-op hook、
  Ascend context 转发、未知组件拒绝，以及联合/core-only 两种模式下构建前后残留清理。
- **必要性**：它锁定的是 FlagTree core 与可选 submodule 的边界契约，尤其覆盖 core-only
  行为，因此属于主仓回归测试。
- **迁移判断**：测试可以物理放进 FlagPrism，但那会使主仓在组件关闭或 submodule 未拉取
  时不再验证自己的网关。建议保留；它不是运行时代码，若只追求主仓文件数则可以迁移。

### 5.4 `python/src/ir.cc` (`0`) [已消除]

- **当前状态**：相对 FlagTree 3.5.x 已无 diff。原 `operation.set_attr`、
  `builder.get_last_op` 和 `FLAGPRISM_BUILD_DEBUGGER` 宏均已删除。
- **替代实现**：Debugger compiler plugin 提供 `annotate_statement_operation(builder,
  source, result_name, statement_id)`，直接从 builder insertion point 取得无结果 StoreOp
  并设置属性；有结果 tensor 继续使用 3.5.x 原有 `Value.set_attr`。
- **审计结论**：operation 注解实现已完全迁入 FlagPrism，不再属于父仓修改。普通与
  Ascend 构建共用同一 `CompilerBindings.cpp`，动态验证将在本轮统一构建时执行。

### 5.5 `python/src/main.cc` (`+5/-2`) [H]

- **改动**：把静态 `FOR_EACH` plugin 初始化宏的容量从 6 扩到 8，并用注释说明新增
  容量由 FlagPrism 的 Proton/Debugger plugin 占用。
- **必要性**：加入 Proton 和 Debugger 后，当前 plugin 名单超过 3.5.x 宏支持的最大数量，
  否则 `libtriton` 无法编译或无法初始化新增 plugin。
- **为何不能迁移**：该宏在 core extension 编译期间展开，submodule 无法从外部改变它。
  只有把 FlagTree plugin loader 重构成不限制数量的通用注册机制，才能消除此修改；那是
  更大的 core 重构。

### 5.6 `python/triton/compiler/code_generator.py` (`+7/-2`) [H]

- **改动**：assignment 在符号绑定前调用 statement hook；expression 改为显式访问子
  expression 并把返回值传给 hook。
- **必要性**：Debugger 需要同时拿到 Python AST 源位置、左值名称和刚生成的 IR 值。
  对 `tl.store` 这类 void statement，原 `generic_visit` 会丢弃上下文。
- **为何不能迁移**：submodule 无法在不 monkey-patch `CodeGenerator` 的情况下拦截这个
  精确时点。monkey-patch 依赖导入顺序且容易随 3.5.x 实现变化。若未来 core 提供
  `on_statement_emitted` 通用扩展点，本文件可恢复；当前薄回调是可审计的最小接点。

### 5.7 `python/triton/compiler/compiler.py` (`+8/-1`) [H]

- **改动**：每次创建 MLIR context 后加载已注册组件 dialect；每个 stage 在 IR override
  完成后对最终 module 调用 compiler hook。
- **必要性**：未向新 context 注册 dialect 时不能 parse 含 Proton/Debugger op 的 IR；
  hook 若放在 override 前会插桩一个随后被丢弃的 module，并导致 metadata 与最终 IR
  不一致。
- **为何不能迁移**：context 和 stage replacement 的控制流属于 core compiler。外部包
  无法可靠观察“最终 override 已确定但尚未缓存”的位置。要删除这些行，FlagTree 需要
  原生的 context/stage listener API。

### 5.8 `python/triton/language/__init__.py` (`+6`) [H]

- **改动**：在 `triton.language` 导出 `debug_collect_start/end` 并加入 `__all__`。
- **必要性**：用户 kernel 使用 `tl.debug_collect_start/end`，名称必须在 DSL 公共 namespace
  中稳定存在，才能被 JIT frontend 解析。
- **为何不能迁移**：可选包只能在 import 后动态 monkey-patch `tl`，这会使“先定义 kernel
  还是先 import debugger”影响结果。主仓静态导出两个薄入口可避免导入顺序协议。

### 5.9 `python/triton/language/core.py` (`+18`) [H]

- **改动**：定义带 `@builtin` 的两个 collect marker，实际实现转发到 `_flagprism`。
- **必要性**：`@builtin` 是 Triton frontend 识别语义函数的机制；普通 submodule Python
  函数不能自动获得相同 AST/semantic 注入行为。
- **为何不能迁移**：可以把业务逻辑移走，当前也已经移走，但 builtin 声明必须位于 JIT
  可识别的 DSL 中。除非 core 增加“外部 DSL builtin 注册”机制，否则这 18 行不能完全
  消失。

### 5.10 `python/triton/runtime/jit.py` (`+4/-1`) [H]

- **改动**：在 backend option parse、specialization 和 cache key 固化前调用组件选项
  hook。
- **必要性**：Debugger/Profiler 的 instrumentation mode 必须参与编译选项和缓存键；
  否则可能复用未插桩 binary，或把插桩 binary 用于普通运行。
- **为何不能迁移**：submodule 在 compiler hook 阶段再修改 metadata 已经太晚，cache
  lookup 可能已完成。3.5.x 没有 option-provider 注册接口，因此这一时点只能由 JIT
  主流程暴露。

### 5.11 `python/triton/spec/ascend/compiler/code_generator.py` (`+7/-2`) [H]

- **改动**：与通用 CodeGenerator 相同，增加 assignment/expression statement hook。
- **必要性**：FlagTree 3.5.x 的 Ascend spec 维护独立 frontend 实现；只修改通用文件会
  导致不同构建路径下 statement report 缺失。
- **为何不能迁移**：原因与 5.6 相同。两份修改来自基线已有的代码镜像，不是 FlagPrism
  主动拆分；消除重复需要先让 Ascend spec 复用通用 CodeGenerator。

### 5.12 `python/triton/spec/ascend/compiler/compiler.py` (`+8/-1`) [H]

- **改动**：Ascend spec 的 context/dialect 和最终 stage hook。
- **必要性**：该 compiler 自己创建 context、执行 override 和缓存，通用 compiler 的
  hook 不会覆盖它。
- **为何不能迁移**：原因与 5.7 相同。要删掉本文件改动，必须先统一两套 compiler
  pipeline 或在 core 提供共同的 stage callback abstraction。

### 5.13 `python/triton/spec/ascend/language/core.py` (`+18`) [H]

- **改动**：Ascend spec DSL 中定义相同 collect builtin。
- **必要性**：该路径使用自己的 `language/core.py`；没有镜像定义时 Ascend kernel 无法
  识别 collect marker。
- **为何不能迁移**：原因与 5.9 相同。真正的去重方向是让两套 DSL 共享定义，而不是让
  可选组件 monkey-patch 两份模块。

### 5.14 `python/triton/spec/ascend/runtime/jit.py` (`+4/-1`) [H]

- **改动**：在 Ascend spec JIT 的 cache key 前注入组件选项。
- **必要性**：该 JIT 独立于通用 `runtime/jit.py`，必须保证两条路径使用相同的插桩缓存
  语义。
- **为何不能迁移**：原因与 5.10 相同。消除重复需要先合并基线中的两套 JIT 实现。

## 6. FlagTree 示例和测试的公开 import

以下五个文件只反映公开 API 从 `triton.profiler` 调整为
`flagtree.profiler`。它们不包含 Profiler 实现，也不是组件接线硬点。

### 6.1 `python/triton_kernels/bench/bench_mlp.py` (`+2/-1`) [R]

- **改动与必要性**：benchmark 改从实际发布 namespace import Profiler。
- **迁移判断**：文件属于 FlagTree benchmark，不能搬到 FlagPrism 来修复其 import；但可
  通过恢复一个 `triton.profiler` 兼容 facade 来撤销此修改。

### 6.2 `python/triton_kernels/bench/roofline.py` (`+2/-1`) [R]

- **改动与必要性**：viewer import 改为 `flagtree.profiler.viewer`。
- **迁移判断**：与 6.1 相同。它是 API 迁移，不是架构上无法迁移的接点。

### 6.3 `python/triton_kernels/tests/test_routing.py` (`+2/-1`) [R]

- **改动与必要性**：benchmark helper 使用新的公开 Profiler namespace。
- **迁移判断**：与 6.1 相同；若保留旧 facade，可恢复基线内容。

### 6.4 `python/tutorials/09-persistent-matmul.py` (`+4/-2`) [R]

- **改动与必要性**：主 import 和 viewer import 改名，局部变量仍保留 `proton` 以减少
  教程正文变化。
- **迁移判断**：教程归父仓所有，组件不能从 submodule 修改它；兼容 facade 可以消除
  这两处 diff。

### 6.5 `python/tutorials/10-block-scaled-matmul.py` (`+2/-2`) [R]

- **改动与必要性**：与 6.4 相同。
- **迁移判断**：与 6.4 相同，不是必须保留的 core 接线。按审计要求，该文件不额外添加
  FlagPrism 注释，只保留防止教程导入失败所必需的两处 namespace 修改。

## 7. Proton 原生工具注册与测试

### 7.1 `bin/CMakeLists.txt` (`0`) [已恢复]

- **当前状态**：相对 FlagTree 3.5.x 已无 diff。FlagPrism 在 Proton 开启时提供真实
  `TritonTestProton`，关闭时提供空 INTERFACE compatibility target，因此父仓四个工具
  保持原链接列表。
- **审计结论**：测试 target 的实现和可选策略均在 submodule，父仓无需条件分支。

### 7.2 `bin/RegisterTritonDialects.h` (`+12/-21`) [H]

- **改动**：删除对 Proton 各具体 pass/dialect header 和 test pass symbol 的逐项依赖；
  在 `__PROTON__` 下分别调用 submodule 提供的 production 和 test 集中 registration API。
- **必要性**：`triton-opt/reduce/lsp` 在 C++ 启动时构造静态 registry。旧源码路径已迁走，
  core-only 构建又不能引用 Proton symbol。
- **为何不能迁移**：submodule 已经持有全部 production registration 实现，但 host 工具
  仍必须在自己的 registry 构造函数中调用一次。完全消除需要动态 dialect plugin loader，
  3.5.x 没有该机制。

### 7.3 `test/lib/CMakeLists.txt` (`+3`) [R]

- **改动**：用 CMake 块注释禁用并原样保留父仓 `add_subdirectory(Proton)`。
- **必要性**：Proton C++ test pass 已由 FlagPrism 内的 `TritonTestProton` 编译，并通过
  submodule registration API 注册；父仓 test library 图不再拥有该实现。
- **迁移判断**：父仓 `test/Proton/*.mlir` 仍保留，旧 lit 用例继续调用父仓
  `triton-opt --test-print-scope-id-allocation`；变化的是 pass 所有权，不是测试执行器。

### 7.4 `test/lib/Proton/CMakeLists.txt` (`+3`) [已迁移]

- **改动**：target 定义已迁到
  `third_party/FlagPrism/proton/Dialect/test/CMakeLists.txt`；父仓原定义仅用 CMake
  块注释保留为审计参考，不再执行。
- **审计结论**：父仓文件不再拥有 target 实现，保留的内容全部不可执行。

### 7.5 `test/lib/Proton/TestScopeIdAllocation.cpp` (`+1/-51`) [已迁移]

- **改动**：源码迁到
  `third_party/FlagPrism/proton/Dialect/test/TestScopeIdAllocation.cpp`，并由集中注册入口调用。
- **审计结论**：父仓只保留一行 FlagPrism 迁移 tombstone 和消费该 pass 的 lit 文件，
  不再维护 Proton analysis 测试实现。

## 8. Ascend 后端硬接点

### 8.1 `third_party/ascend/backend/compiler.py` (`+8`) [H]

- **改动**：在 Ascend adapter 序列化前调用 `ttadapter.pre_serialize` hook，并给
  `NPUOptions` 增加可哈希的 `instrumentation_mode`。
- **必要性**：这是 Ascend 最后一个仍可进行结构化 IR 插桩的位置；mode 必须进入 options
  hash，避免插桩和非插桩 binary 共用缓存。
- **为何不能迁移**：序列化边界和 `NPUOptions` 定义由 Ascend backend 持有。submodule
  无法在序列化后恢复结构化 IR，也不能事后修改已经形成的 cache key。除非 backend
  新增通用 pre-serialize/options 扩展协议，否则这处薄 hook 必须保留。

### 8.2 `third_party/ascend/backend/driver.py` (`+67/-6`) [H]

- **改动**：用 `_DebuggerHiddenArgABI` 集中描述一个隐藏指针在 Python parse format、
  C++ declaration、kernel argument struct 和 call site 中的全部片段；launcher 只在
  metadata 明确要求时进入 Debugger launch context 并追加参数。关键边界均以
  `FlagPrism` 注释标明，普通 launch 路径不读取组件实现。
- **必要性**：Debugger runtime buffer 指针必须按 Ascend kernel ABI 传到设备；Host
  buffer 生命周期还必须覆盖一次 launch。仅靠 compiler 插桩无法让 launcher 自动多传
  一个参数。
- **为何不能迁移**：launcher C++ 源码由本文件动态生成，参数个数、顺序、对齐和
  `PyArg_ParseTuple` format 都是后端私有协议。submodule 无法安全改写已生成的 launcher。
  若要完全移走，需要先为所有 backend 设计 hidden-argument/launch-context adapter API，
  修改面会显著大于当前 dataclass 加一个调用点。

### 8.3 `third_party/ascend/bin/CMakeLists.txt` (`0`) [已恢复]

- **当前状态**：相对 FlagTree 3.5.x 已无 diff；与 7.1 相同，真实或空的兼容 target
  保持 Ascend 原工具链接列表不变。

### 8.4 `third_party/ascend/bin/RegisterTritonDialects.h` (`+12/-21`) [H]

- **改动**：Ascend 工具 registry 改为条件调用 Proton 集中 registration API。
- **必要性**：Ascend 使用独立 registry，且还注册 Ascend/HIVM 等自身 dialect，不能复用
  通用 header。
- **为何不能迁移**：与 7.2 相同。submodule 可拥有注册函数，但 host registry 中的一次
  条件调用不能消失，除非重构 FlagTree 的工具级动态插件机制。

### 8.5 `third_party/ascend/python/src/ir.cc` (`0`) [已消除]

- **当前状态**：相对 FlagTree 3.5.x 已无 diff；与 5.4 相同，operation 注解由同一份
  Debugger compiler plugin 实现。
- **审计结论**：父仓不再维护两套重复 IR helper；Ascend 动态验证已加入 store statement
  metadata 断言，统一构建后执行。

### 8.6 `third_party/ascend/python/src/main.cc` (`+5/-2`) [H]

- **改动**：Ascend 独立 `libtriton` binding 的 plugin 宏容量从 6 扩到 8。
- **必要性**：这份 target 有自己的 plugin 初始化宏，通用 `main.cc` 修改不会生效。
- **为何不能迁移**：与 5.5 相同。只有统一两个 binding target 或重构不限数量的 loader
  才能消除。

## 9. 其他后端的构建兼容

### 9.1 `third_party/enflame/cmake/triton_gcu.cmake` (`+21`) [H/R]

- **改动**：Proton 开启时把 submodule 源码加入 nested build 依赖，调用 submodule
  helper 追加 dialect object，并把统一 `TRITON_BUILD_FLAGPRISM` 开关传入嵌套 CMake。
- **必要性**：Enflame 自己维护一套 nested Triton build 和静态 object 聚合；主 CMake
  的 plugin 调用不会自动传播到该子构建。
- **为何不能完全迁移**：只有该文件知道 `triton_${arch}_objs`、nested binary dir 和
  `triton_cmake_args`。submodule 已持有 object 清单，但 host 至少要传入这些变量并调用
  helper。可继续把 source glob 和参数追加封装成一个函数，主仓保留一次调用。

### 9.2 `third_party/enflame/cmake/triton_gcu300.cmake` (`+4`) [H]

- **改动**：19 个旧硬编码 object 路径用 CMake 块注释原样保留，实际对象由
  `triton_gcu.cmake` 调用 FlagPrism helper 统一追加。
- **必要性**：源码移动后这些路径失效，且 object 清单已由 Proton submodule 自己生成；
  若取消块注释会导致 Profiler-enabled Enflame 链接失败并重复维护清单。
- **为何不能迁移**：旧文本属于 backend 父仓文件；当前仅作为不可执行审计参考保留，
  活跃 object 清单已经完全迁入 submodule。

### 9.3 `third_party/enflame/cmake/triton_gcu400.cmake` (`+4`) [H]

- **改动与必要性**：与 9.2 相同，块注释保留 GCU400 的第二份旧 Proton object 清单。
- **为何不能迁移**：与 9.2 相同；活跃清单已迁移，父仓只保留审计文本。

### 9.4 `third_party/tsingmicro/bin/RegisterTritonDialects.h` (`+11/-2`) [H]

- **改动**：移除无条件旧 Proton include/dialect 插入，在 `__PROTON__` 下调用集中注册
  API。
- **必要性**：即使 TsingMicro 默认关闭 Proton，旧 header 已不存在，无条件 include
  仍会破坏编译；开启时又必须注册完整 dialect/pass。
- **为何不能迁移**：该工具自己构造静态 registry，原因与 7.2 相同。保留 compatibility
  旧路径可以避免 diff，但会重新引入重复源码或软链接，不符合单一 submodule 所有权。
- **已消除接点**：`third_party/tsingmicro/scripts/build_tsingmicro.sh` 已恢复 3.5.x。脚本
  原有 `TRITON_BUILD_PROTON=OFF` 作为兼容输入会关闭整个 FlagPrism，无需新增第二个变量。

## 10. 已迁移的旧 `third_party/proton` 目录

相对基线，旧目录显示为 197 个文件、16,573 行删除。这些文件包括 Proton dialect、
conversion passes、native profiler、Python API、测试和教程。它们的统一理由如下：

- **为什么删除必要**：同一实现已经由 `third_party/FlagPrism/proton` submodule
  所有。父仓继续保留会形成两份源码、两套修复历史和不明确的链接来源。
- **为什么不逐个解释为“无法迁移”**：这些文件事实上已经迁移；删除正是迁移结果。
- **为什么不能用软链接保留旧路径**：软链接会让 sdist、Windows checkout、GitHub
  archive 和构建依赖解析出现不同语义，也会继续暴露两套所有权边界。
- **审计方式**：功能变化应在 FlagPrism 仓库相对其导入基线审计；FlagTree 主仓只审计
  本文列出的接线和 gitlink 更新。

## 11. 收敛状态与下一步

1. **已回撤无关构建/发布改动**：CMake 全局符号可见性、`.pre-commit-config.yaml`、
   `.gitignore`、`MANIFEST.in`、TsingMicro 构建脚本和两份 `bin/CMakeLists.txt` 均已恢复
   3.5.x，不再混入 FlagPrism 接入提交。
2. **已完成 Proton C++ 测试迁移**：测试 pass 源码和注册已进入 FlagPrism；父仓
   `triton-opt` 继续执行原 `test/Proton` lit 用例，两份父仓 bin CMake 已恢复基线。
3. **已完成 IR helper 内聚**：两个 `ir.cc` 共 38 行 FlagPrism diff 已清零；普通与
   Ascend 构建共用插件 helper，联合 wheel 编译、native binding 测试和 level 2 实机
   编译运行均已通过。
4. **已收敛构建模式**：`TRITON_BUILD_FLAGPRISM` 同时控制 Debugger 和 Profiler，只保留
   联合构建与 core-only；旧变量不能再表达单组件模式。两种模式均已完成独立全量编译，
   FlagPrism Python 网关与打包清理测试为 10 passed；统一 CMake 入口及审计注释完成后，
   联合、core-only 和非 Python 工具三种全新 configure 均通过。
5. **决定兼容 API**：若接受保留薄 `triton.profiler` facade，可恢复 5 个 benchmark/
   tutorial 文件；若明确只支持 `flagtree.profiler`，则保留当前 API 迁移。

## 12. 最小不可消除边界

在不大改 FlagTree 3.5.x 扩展架构的前提下，最终仍需由主仓承担以下内容：

- `.gitmodules` 和 gitlink。
- 主 CMake 在 target/plugin 生命周期中的少量调用。
- `setup.py` 对统一 wheel package、native extension 和 CLI 的少量声明。
- 一个始终可导入的 Python 组件 registry/no-op 网关。
- AST、context/stage、JIT option/cache key 和 DSL builtin 的薄回调点。
- Ascend compiler pre-serialize 与 launcher hidden-argument 接点。
- 每套静态原生工具 registry 对 Proton 集中注册函数的一次条件调用。

这部分不是 Debugger/Profiler 业务实现，而是 FlagTree 3.5.x 当前缺少通用插件接口时，
由 host 必须提供的接入能力。若目标是把主仓改动进一步降为零，正确方向不是继续搬文件，
而是先在 FlagTree 中设计并上游化上述通用扩展协议。

## 13. Diff 快照与审计索引

### 13.1 生成边界

以下 raw diff 以 `flagos-ai/triton_v3.5.x@317f15a426466633c4f37f164b2c58ae9c31bd03`
为唯一基线，包含 31 个 tracked 文件以及 3 个尚未 tracked 的新增主仓源文件，共 34 个
文件、`+718/-145`。3 个新增文件通过 `git diff --no-index /dev/null <file>` 追加，因此
不会因尚未进入 index 而从审计快照中消失。

快照明确排除以下内容：

- 旧 `third_party/proton/**` 的 197 文件迁移删除；该目录单独在第 10 节解释。
- `third_party/FlagPrism` 内部实现和 gitlink；它们必须在 FlagPrism 仓库及父仓 raw
  gitlink 中分别审计。
- `FlagPrism-core.diff`、本报告、其他审计文档、构建目录、缓存和 egg-info。
- `.pre-commit-config.yaml`、`.gitignore`、`MANIFEST.in`、TsingMicro 构建脚本、普通/Ascend
  两份 `ir.cc` 和两份 `bin/CMakeLists.txt`；这些文件相对 3.5.x 为零 diff，不会出现在
  raw diff 中。

同一份 raw diff 保存在 `FlagPrism-core.diff`。报告内嵌内容与该文件通过 `cmp` 逐字校验，
避免说明文字和实际补丁来自不同工作区快照。

### 13.2 分类统计

| 审计区域 | 文件数 | 新增 | 删除 | 主要所有权 |
| --- | ---: | ---: | ---: | --- |
| 构建、发布与仓库边界 | 4 | 96 | 25 | 父仓构建/发行生命周期 |
| Python 网关与 Statement/DSL | 13 | 459 | 10 | 父仓 frontend/JIT 扩展点 |
| Profiler 公开 import | 5 | 12 | 7 | 父仓示例和测试 |
| Proton 工具注册与 C++ test 迁移 | 4 | 19 | 72 | 父仓静态 registry、审计注释与迁移删除 |
| Ascend backend/launcher 接点 | 4 | 92 | 29 | Ascend 编译和 launch ABI |
| Enflame/TsingMicro 构建兼容 | 4 | 40 | 2 | 后端私有构建/registry |
| **合计** | **34** | **718** | **145** | 另有不计行数的 FlagPrism gitlink |

该统计把 `python/flagtree/__init__.py`、`python/triton/_flagprism.py` 和
`python/test/unit/test_flagprism.py` 计入 Python 网关区域。零 diff 的审计项只在正文
列出，不人为计入文件数。

### 13.3 审计顺序

建议按以下顺序阅读 raw diff：先审核第 4 节的构建/发布边界，再审核 core Python 薄
hook，随后检查 Proton registry/测试迁移、Ascend ABI，最后检查其他后端兼容。这样可以
先确认组件是否被正确构建和打包，再判断运行时 hook 是否必要。

当前 raw diff 已通过 `git diff --check`；但它只是工作区快照，不替代提交状态检查。
FlagPrism 内部新增文件仍需先在 submodule 仓库提交，父仓 gitlink 随后更新，才能形成可
复现的最终 patch。

### 13.4 Raw Diff

````diff
diff --git a/.gitmodules b/.gitmodules
new file mode 100644
index 000000000..e67d98cea
--- /dev/null
+++ b/.gitmodules
@@ -0,0 +1,4 @@
+# FlagPrism: bundled Debugger and Profiler source boundary.
+[submodule "third_party/FlagPrism"]
+	path = third_party/FlagPrism
+	url = https://github.com/ZQ-Struggle/FlagTree_DevTools.git
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 3b39b5421..42078ba73 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -99,13 +99,31 @@ list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
 
 # Options
 option(TRITON_BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)
+# Standalone Proton configuration was removed; FlagPrism builds Proton and Debugger together.
 if(FLAGTREE_BACKEND)
+  #[[
   option(TRITON_BUILD_PROTON "Build the Triton Proton profiler" OFF)
+  ]]
   option(TRITON_BUILD_UT "Build C++ Triton Unit Tests" OFF)
 else()
+  #[[
   option(TRITON_BUILD_PROTON "Build the Triton Proton profiler" ON)
+  ]]
   option(TRITON_BUILD_UT "Build C++ Triton Unit Tests" ON)
 endif()
+# FlagPrism integration: load the combined Debugger and Profiler build policy.
+set(FLAGPRISM_CMAKE_FILE
+    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/FlagPrism/cmake/FlagPrism.cmake")
+if(EXISTS "${FLAGPRISM_CMAKE_FILE}")
+  include("${FLAGPRISM_CMAKE_FILE}")
+elseif(TRITON_BUILD_FLAGPRISM OR TRITON_BUILD_PROTON OR
+       TRITON_BUILD_DEVTOOLS)
+  message(FATAL_ERROR
+    "FlagPrism source is missing. Run `git submodule update --init --recursive`.")
+elseif(NOT TARGET TritonTestProton)
+  # Keep core-only builds compatible with the baseline tool link list.
+  add_library(TritonTestProton INTERFACE)
+endif()
 option(TRITON_BUILD_WITH_CCACHE "Build with ccache (if available)" ON)
 set(TRITON_CODEGEN_BACKENDS "" CACHE STRING "Enable different codegen backends")
 
@@ -334,6 +352,11 @@ if(TRITON_BUILD_PYTHON_MODULE)
     add_subdirectory(third_party/flir)
   endif()
 
+  # FlagPrism integration: add the bundled tools before TRITON_LIBS is read.
+  if(COMMAND flagprism_add_components)
+    flagprism_add_components()
+  endif()
+
   if (DEFINED TRITON_PLUGIN_DIRS)
     foreach(PLUGIN_DIR ${TRITON_PLUGIN_DIRS})
       # Read the plugin name under dir/backend/name.conf
@@ -355,6 +378,8 @@ if(TRITON_BUILD_PYTHON_MODULE)
     add_subdirectory(third_party/${CODEGEN_BACKEND})
   endforeach()
 
+  # Standalone Proton wiring was removed; FlagPrism registers these targets.
+  #[[
   if(TRITON_BUILD_PROTON)
     add_definitions(-D__PROTON__)
     add_subdirectory(third_party/proton)
@@ -362,6 +387,7 @@ if(TRITON_BUILD_PYTHON_MODULE)
   # We always build proton dialect
   list(APPEND TRITON_PLUGIN_NAMES "proton")
   add_subdirectory(third_party/proton/Dialect)
+  ]]
 
   # The 3.2 DSA CMake tree defines the same TLE dialect/plugin targets as
   # third_party/tle on 3.5, so do not add it as a second CMake subdirectory.
@@ -568,10 +594,17 @@ if(NOT TRITON_BUILD_PYTHON_MODULE)
   if(FLAGTREE_BACKEND STREQUAL "ascend")
     add_subdirectory(third_party/flir)
   endif()
+  # FlagPrism integration: add the targets required by non-Python tools.
+  if(COMMAND flagprism_add_components)
+    flagprism_add_components()
+  endif()
   foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
     add_subdirectory(third_party/${CODEGEN_BACKEND})
   endforeach()
+  # Standalone Proton dialect wiring was removed; FlagPrism owns this target.
+  #[[
   add_subdirectory(third_party/proton/Dialect)
+  ]]
 endif()
 
 find_package(Threads REQUIRED)
diff --git a/Makefile b/Makefile
index 54ec7f3e8..a16607ae0 100644
--- a/Makefile
+++ b/Makefile
@@ -73,8 +73,11 @@ test-interpret: all
 
 .PHONY: test-proton
 test-proton: all
-	$(PYTEST) -s -n 8 third_party/proton/test --ignore=third_party/proton/test/test_override.py
-	$(PYTEST) -s third_party/proton/test/test_override.py
+	# Proton tests moved to the FlagPrism submodule; legacy commands are retained for audit.
+	# $(PYTEST) -s -n 8 third_party/proton/test --ignore=third_party/proton/test/test_override.py
+	# $(PYTEST) -s third_party/proton/test/test_override.py
+	$(PYTEST) -s -n 8 third_party/FlagPrism/proton/test --ignore=third_party/FlagPrism/proton/test/test_override.py
+	$(PYTEST) -s third_party/FlagPrism/proton/test/test_override.py
 
 .PHONY: test-python
 test-python: test-unit test-regression test-interpret test-proton
diff --git a/bin/RegisterTritonDialects.h b/bin/RegisterTritonDialects.h
index 39bd4a3a9..dc4db9b2b 100644
--- a/bin/RegisterTritonDialects.h
+++ b/bin/RegisterTritonDialects.h
@@ -3,13 +3,10 @@
 #include "amd/include/TritonAMDGPUTransforms/Passes.h"
 #include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
 #include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
-#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
-#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
-#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
-#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"
-#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
-#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
-#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h"
+#ifdef __PROTON__
+// FlagPrism: Proton owns its dialect and production-pass registry.
+#include "FlagPrism/proton/Dialect/include/Integration/Registration.h"
+#endif
 #ifdef __TLE__
 #include "third_party/tle/dialect/include/Transforms/Passes.h"
 #include "tle/dialect/include/IR/Dialect.h" // flagtree tle raw
@@ -52,9 +49,7 @@ void registerTestMembarPass();
 void registerTestAMDGPUMembarPass();
 void registerTestTritonAMDGPURangeAnalysis();
 void registerTestLoopPeelingPass();
-namespace proton {
-void registerTestScopeIdAllocationPass();
-} // namespace proton
+// FlagPrism declares the migrated Proton test pass in its integration header.
 } // namespace test
 } // namespace mlir
 
@@ -115,15 +110,11 @@ inline void registerTritonDialects(mlir::DialectRegistry &registry) {
   // NVGPU transform passes
   mlir::registerNVHopperTransformsPasses();
 
-  // Proton passes
-  mlir::test::proton::registerTestScopeIdAllocationPass();
-  mlir::triton::proton::registerConvertProtonToProtonGPU();
-  mlir::triton::proton::gpu::registerConvertProtonNvidiaGPUToLLVM();
-  mlir::triton::proton::gpu::registerConvertProtonAMDGPUToLLVM();
-  mlir::triton::proton::gpu::registerAllocateProtonSharedMemoryPass();
-  mlir::triton::proton::gpu::registerAllocateProtonGlobalScratchBufferPass();
-  mlir::triton::proton::gpu::registerScheduleBufferStorePass();
-  mlir::triton::proton::gpu::registerAddSchedBarriersPass();
+  // FlagPrism owns Proton's production and legacy lit-test registration.
+#ifdef __PROTON__
+  mlir::triton::proton::registerFlagTreeProtonTestPasses();
+  mlir::triton::proton::registerFlagTreeProtonPassesAndDialects(registry);
+#endif
 
   registry.insert<
       mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
@@ -134,8 +125,8 @@ inline void registerTritonDialects(mlir::DialectRegistry &registry) {
       mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
       mlir::triton::nvgpu::NVGPUDialect, mlir::triton::nvws::NVWSDialect,
       mlir::triton::amdgpu::TritonAMDGPUDialect,
-      mlir::triton::proton::ProtonDialect,
-      mlir::triton::proton::gpu::ProtonGPUDialect, mlir::ROCDL::ROCDLDialect,
+      // FlagPrism registers Proton dialects through the centralized call above.
+      mlir::ROCDL::ROCDLDialect,
 #ifdef __TLE__
       mlir::triton::gluon::GluonDialect,
       mlir::triton::tle::TleDialect // flagtree tle raw
diff --git a/python/src/main.cc b/python/src/main.cc
index 9bbd5e636..a21c15a74 100644
--- a/python/src/main.cc
+++ b/python/src/main.cc
@@ -10,11 +10,14 @@ namespace py = pybind11;
 #define FOR_EACH_4(MACRO, X, ...) MACRO(X) FOR_EACH_3(MACRO, __VA_ARGS__)
 #define FOR_EACH_5(MACRO, X, ...) MACRO(X) FOR_EACH_4(MACRO, __VA_ARGS__)
 #define FOR_EACH_6(MACRO, X, ...) MACRO(X) FOR_EACH_5(MACRO, __VA_ARGS__)
+// FlagPrism adds Proton and Debugger plugins; extend the static loader capacity.
+#define FOR_EACH_7(MACRO, X, ...) MACRO(X) FOR_EACH_6(MACRO, __VA_ARGS__)
+#define FOR_EACH_8(MACRO, X, ...) MACRO(X) FOR_EACH_7(MACRO, __VA_ARGS__)
 
 #define FOR_EACH_NARG(...) FOR_EACH_NARG_(__VA_ARGS__, FOR_EACH_RSEQ_N())
 #define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__)
-#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, N, ...) N
-#define FOR_EACH_RSEQ_N() 6, 5, 4, 3, 2, 1, 0
+#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
+#define FOR_EACH_RSEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0
 
 #define CONCATENATE(x, y) CONCATENATE1(x, y)
 #define CONCATENATE1(x, y) x##y
diff --git a/python/triton/compiler/code_generator.py b/python/triton/compiler/code_generator.py
index 66b3ef4e3..dac288fe9 100644
--- a/python/triton/compiler/code_generator.py
+++ b/python/triton/compiler/code_generator.py
@@ -12,7 +12,8 @@ from types import ModuleType
 from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, Iterable, List
 import importlib
 
-from .. import knobs, language
+# FlagPrism: use the core no-op gateway; optional tools remain lazily loaded.
+from .. import _flagprism, knobs, language
 from .._C.libtriton import ir, gluon_ir
 from ..language import constexpr, str_to_ty, tensor, tuple as tl_tuple
 from ..language.core import _unwrap_if_constexpr, base_value, base_type
@@ -745,6 +746,8 @@ class CodeGenerator(ast.NodeVisitor):
                 values = _sanitize_value(self.visit(node.value))
         else:
             values = _sanitize_value(self.visit(node.value))
+        # FlagPrism: expose the source assignment before symbol binding.
+        _flagprism.annotate_statement("assignment", self, node, target, values)
         self.assignTarget(target, values)
 
     def visit_AugAssign(self, node):
@@ -1599,7 +1602,9 @@ class CodeGenerator(ast.NodeVisitor):
 
     def visit_Expr(self, node):
         node.value._is_unused = True
-        ast.NodeVisitor.generic_visit(self, node)
+        value = self.visit(node.value)
+        # FlagPrism: retain the operation created by a void expression.
+        _flagprism.annotate_statement("expression", self, node, None, value)
 
     def visit_NoneType(self, node):
         return None
diff --git a/python/triton/compiler/compiler.py b/python/triton/compiler/compiler.py
index 1f38e9ddf..27d5a4b33 100644
--- a/python/triton/compiler/compiler.py
+++ b/python/triton/compiler/compiler.py
@@ -5,7 +5,8 @@ from .._C.libtriton import get_cache_invalidating_env_vars, ir
 from ..backends import backends
 from ..backends.compiler import Language
 from ..backends.compiler import BaseBackend, GPUTarget
-from .. import __version__, knobs
+# FlagPrism: use the core no-op gateway; optional tools remain lazily loaded.
+from .. import _flagprism, __version__, knobs
 from ..runtime.autotuner import OutOfResources
 from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager, get_cache_key
 from ..runtime.driver import driver
@@ -94,6 +95,8 @@ class IRSource:
         self.src = path.read_text()
         ir.load_dialects(context)
         backend.load_dialects(context)
+        # FlagPrism: register optional dialects in every fresh context.
+        _flagprism.load_dialects(context)
 
         # We don't have a easy-to-use PTX parser that we can use, so keep that regex for now.
         # TODO - replace with a proper parser
@@ -293,6 +296,8 @@ def compile(src, target=None, options=None, _env_vars=None):
         context = ir.context()
         ir.load_dialects(context)
         backend.load_dialects(context)
+        # FlagPrism: register optional dialects in every fresh context.
+        _flagprism.load_dialects(context)
 
     codegen_fns = backend.get_codegen_implementation(options)
     module_map = backend.get_module_map()
@@ -327,6 +332,8 @@ def compile(src, target=None, options=None, _env_vars=None):
         elif full_name := fn_override_manager.get_file(ir_filename):
             print(f"\nOverriding kernel with file {full_name}")
             next_module = parse(full_name, ext, context)
+        # FlagPrism: instrument the final module after any IR override.
+        _flagprism.run_compiler_hook(ext, next_module, metadata)
         # If TRITON_STORE_BINARY_ONLY is 1, only store cubin/hsaco/json
         if (not store_only_binary) or (ext in ("cubin", "hsaco", "json")):
             metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
diff --git a/python/triton/language/__init__.py b/python/triton/language/__init__.py
index de88cee95..ac16d6099 100644
--- a/python/triton/language/__init__.py
+++ b/python/triton/language/__init__.py
@@ -72,6 +72,9 @@ from .core import (
     constexpr,
     constexpr_type,
     debug_barrier,
+    # FlagPrism: explicit source-level capture controls.
+    debug_collect_end,
+    debug_collect_start,
     device_assert,
     device_print,
     dot,
@@ -191,6 +194,9 @@ __all__ = [
     "cumprod",
     "cumsum",
     "debug_barrier",
+    # FlagPrism: public DSL names for optional debugger regions.
+    "debug_collect_end",
+    "debug_collect_start",
     "device_assert",
     "device_print",
     "div_rn",
diff --git a/python/triton/language/core.py b/python/triton/language/core.py
index 5b16c4597..7e12d815d 100644
--- a/python/triton/language/core.py
+++ b/python/triton/language/core.py
@@ -2852,6 +2852,24 @@ def debug_barrier(_semantic=None):
     return _semantic.debug_barrier()
 
 
+# FlagPrism: keep public DSL markers thin; implementation stays in the
+# optional component behind triton._flagprism.
+@builtin
+def debug_collect_start(level=1, addr_level=None, _semantic=None):
+    """Begin a FlagTree debug collect region."""
+    from triton import _flagprism
+
+    return _flagprism.debug_collect_start(_semantic, level, addr_level)
+
+
+@builtin
+def debug_collect_end(_semantic=None):
+    """End a FlagTree debug collect region."""
+    from triton import _flagprism
+
+    return _flagprism.debug_collect_end(_semantic)
+
+
 @builtin
 def multiple_of(input, values, _semantic=None):
     """
diff --git a/python/triton/runtime/jit.py b/python/triton/runtime/jit.py
index e66b7b749..dc3026a6e 100644
--- a/python/triton/runtime/jit.py
+++ b/python/triton/runtime/jit.py
@@ -14,7 +14,8 @@ from typing import Callable, Generic, Iterable, Optional, TypeVar, Union, overlo
 
 from triton.tools.tensor_descriptor import TensorDescriptor
 from types import ModuleType
-from .. import knobs
+# FlagPrism: use the core no-op gateway; optional tools remain lazily loaded.
+from .. import _flagprism, knobs
 from .driver import driver
 from . import _async_compile
 from .._utils import find_paths_if, get_iterable_path, type_canonicalisation_dict, canonicalize_dtype
@@ -717,6 +718,8 @@ class JITFunction(JITCallable, KernelInterface[T]):
 
     def run(self, *args, grid, warmup, **kwargs):
         kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
+        # FlagPrism: options must affect specialization and cache keys.
+        _flagprism.apply_compile_options(kwargs)
 
         # parse options
         device = driver.active.get_current_device()
diff --git a/python/triton/spec/ascend/compiler/code_generator.py b/python/triton/spec/ascend/compiler/code_generator.py
index f6a023476..6835cc22a 100644
--- a/python/triton/spec/ascend/compiler/code_generator.py
+++ b/python/triton/spec/ascend/compiler/code_generator.py
@@ -16,7 +16,8 @@ import importlib
 import triton.language.extra.cann.extension as extension
 from triton.extension.buffer.language.builder import setup_unified_builder_with_buffer_builder
 
-from .. import knobs, language
+# FlagPrism: mirror the core no-op gateway in the Ascend frontend.
+from .. import _flagprism, knobs, language
 from .._C.libtriton import ir, gluon_ir, buffer_ir
 from .._C.libtriton.ascend import ir as ascend_ir
 from ..language import constexpr, str_to_ty, tensor, tuple as tl_tuple
@@ -727,6 +728,8 @@ class CodeGenerator(ast.NodeVisitor):
                 values = _sanitize_value(self.visit(node.value))
         else:
             values = _sanitize_value(self.visit(node.value))
+        # FlagPrism: expose the source assignment before symbol binding.
+        _flagprism.annotate_statement("assignment", self, node, target, values)
         self.assignTarget(target, values)
 
     def visit_AugAssign(self, node):
@@ -1540,7 +1543,9 @@ class CodeGenerator(ast.NodeVisitor):
 
     def visit_Expr(self, node):
         node.value._is_unused = True
-        ast.NodeVisitor.generic_visit(self, node)
+        value = self.visit(node.value)
+        # FlagPrism: retain the operation created by a void expression.
+        _flagprism.annotate_statement("expression", self, node, None, value)
 
     def visit_NoneType(self, node):
         return None
diff --git a/python/triton/spec/ascend/compiler/compiler.py b/python/triton/spec/ascend/compiler/compiler.py
index 505141ac9..22218afde 100644
--- a/python/triton/spec/ascend/compiler/compiler.py
+++ b/python/triton/spec/ascend/compiler/compiler.py
@@ -6,7 +6,8 @@ from .._C.libtriton.ascend import ir as ascend_ir
 from ..backends import backends
 from ..backends.compiler import Language
 from ..backends.compiler import BaseBackend, GPUTarget
-from .. import __version__, knobs
+# FlagPrism: mirror the core no-op gateway in the Ascend compiler.
+from .. import _flagprism, __version__, knobs
 from ..runtime.autotuner import OutOfResources
 from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager, get_cache_key
 from ..runtime.driver import driver
@@ -96,6 +97,8 @@ class IRSource:
         self.src = path.read_text()
         ir.load_dialects(context)
         backend.load_dialects(context)
+        # FlagPrism: register optional dialects in every fresh context.
+        _flagprism.load_dialects(context)
 
         # We don't have a easy-to-use PTX parser that we can use, so keep that regex for now.
         # TODO - replace with a proper parser
@@ -297,6 +300,8 @@ def compile(src, target=None, options=None, _env_vars=None):
         buffer_ir.load_dialects(context)
         ascend_ir.load_dialects(context)
         backend.load_dialects(context)
+        # FlagPrism: register optional dialects in every fresh context.
+        _flagprism.load_dialects(context)
 
     codegen_fns = backend.get_codegen_implementation(options)
     module_map = backend.get_module_map()
@@ -353,6 +358,8 @@ def compile(src, target=None, options=None, _env_vars=None):
         elif full_name := fn_override_manager.get_file(ir_filename):
             print(f"\nOverriding kernel with file {full_name}")
             next_module = parse(full_name, ext, context)
+        # FlagPrism: instrument the final module after any IR override.
+        _flagprism.run_compiler_hook(ext, next_module, metadata)
         # If TRITON_STORE_BINARY_ONLY is 1, only store cubin/hsaco/json
         if (not store_only_binary) or (ext in ("cubin", "hsaco", "json")):
             metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
diff --git a/python/triton/spec/ascend/language/core.py b/python/triton/spec/ascend/language/core.py
index 0a8daeee1..0089a8b40 100644
--- a/python/triton/spec/ascend/language/core.py
+++ b/python/triton/spec/ascend/language/core.py
@@ -2864,6 +2864,24 @@ def debug_barrier(_semantic=None):
     return _semantic.debug_barrier()
 
 
+# FlagPrism: keep public DSL markers thin; implementation stays in the
+# optional component behind triton._flagprism.
+@builtin
+def debug_collect_start(level=1, addr_level=None, _semantic=None):
+    """Begin a FlagTree debug collect region."""
+    from triton import _flagprism
+
+    return _flagprism.debug_collect_start(_semantic, level, addr_level)
+
+
+@builtin
+def debug_collect_end(_semantic=None):
+    """End a FlagTree debug collect region."""
+    from triton import _flagprism
+
+    return _flagprism.debug_collect_end(_semantic)
+
+
 @builtin
 def multiple_of(input, values, _semantic=None):
     """
diff --git a/python/triton/spec/ascend/runtime/jit.py b/python/triton/spec/ascend/runtime/jit.py
index 1cdaf1dfc..7304b407a 100644
--- a/python/triton/spec/ascend/runtime/jit.py
+++ b/python/triton/spec/ascend/runtime/jit.py
@@ -14,7 +14,8 @@ from typing import Callable, Generic, Iterable, Optional, TypeVar, Union, overlo
 
 from triton.tools.tensor_descriptor import TensorDescriptor
 from types import ModuleType
-from .. import knobs
+# FlagPrism: mirror the core no-op gateway in the Ascend JIT path.
+from .. import _flagprism, knobs
 from .driver import driver
 from . import _async_compile
 from .._utils import find_paths_if, get_iterable_path, type_canonicalisation_dict, canonicalize_dtype
@@ -708,6 +709,8 @@ class JITFunction(JITCallable, KernelInterface[T]):
 
     def run(self, *args, grid, warmup, **kwargs):
         kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
+        # FlagPrism: options must affect specialization and cache keys.
+        _flagprism.apply_compile_options(kwargs)
 
         # parse options
         device = driver.active.get_current_device()
diff --git a/python/triton_kernels/bench/bench_mlp.py b/python/triton_kernels/bench/bench_mlp.py
index 8fb72a86d..8e7213894 100644
--- a/python/triton_kernels/bench/bench_mlp.py
+++ b/python/triton_kernels/bench/bench_mlp.py
@@ -1,7 +1,8 @@
 from itertools import chain
 from pathlib import Path
 from copy import deepcopy
-import triton.profiler as proton
+# FlagPrism: use the bundled Profiler's stable public namespace.
+import flagtree.profiler as proton
 import torch
 import argparse
 import triton_kernels
diff --git a/python/triton_kernels/bench/roofline.py b/python/triton_kernels/bench/roofline.py
index 16789fa7d..d059f0432 100644
--- a/python/triton_kernels/bench/roofline.py
+++ b/python/triton_kernels/bench/roofline.py
@@ -19,7 +19,8 @@ def parse_profile(profile_path, useful_op_regex):
     """
     construct a PerfRecord from a (proton) profile path and a regex for useful operations
     """
-    from triton.profiler import viewer
+    # FlagPrism: use the bundled Profiler's stable public namespace.
+    from flagtree.profiler import viewer
     gf, _, _, _ = viewer.read(profile_path)
     # aggregate "useful" flops + bytes
     useful = gf.filter(f"MATCH ('*', c) WHERE c.'name' =~ '{useful_op_regex}' AND c IS LEAF").dataframe
diff --git a/python/triton_kernels/tests/test_routing.py b/python/triton_kernels/tests/test_routing.py
index 60bb35d26..6eca61f38 100644
--- a/python/triton_kernels/tests/test_routing.py
+++ b/python/triton_kernels/tests/test_routing.py
@@ -77,7 +77,8 @@ def test_op(n_tokens_pad, n_tokens_raw, n_expts_tot, n_expts_act, sm_first, use_
 
 
 def bench_routing():
-    import triton.profiler as proton
+    # FlagPrism: use the bundled Profiler's stable public namespace.
+    import flagtree.profiler as proton
     n_tokens = 8192
     n_expts_tot, n_expts_act = 128, 4
     tri_logits = init_data(n_tokens, n_expts_tot)
diff --git a/python/tutorials/09-persistent-matmul.py b/python/tutorials/09-persistent-matmul.py
index 621378332..a632ceecd 100644
--- a/python/tutorials/09-persistent-matmul.py
+++ b/python/tutorials/09-persistent-matmul.py
@@ -26,7 +26,8 @@ import sys
 import torch
 import triton
 import triton.language as tl
-import triton.profiler as proton
+# FlagPrism: use the bundled Profiler's stable public namespace.
+import flagtree.profiler as proton
 from triton.tools.tensor_descriptor import TensorDescriptor
 from contextlib import contextmanager
 
@@ -703,7 +704,8 @@ def validate(M, N, K, dtype):
 
 
 def show_profile(precision, profile_name):
-    import triton.profiler.viewer as proton_viewer
+    # FlagPrism: use the bundled Profiler viewer namespace.
+    import flagtree.profiler.viewer as proton_viewer
     metric_names = ["time/ms"]
     if precision == 'fp8':
         metric_names = ["tflop8/s"] + metric_names
diff --git a/python/tutorials/10-block-scaled-matmul.py b/python/tutorials/10-block-scaled-matmul.py
index 3d18c7020..57a5250bb 100644
--- a/python/tutorials/10-block-scaled-matmul.py
+++ b/python/tutorials/10-block-scaled-matmul.py
@@ -70,7 +70,7 @@ import argparse
 import torch
 import triton
 import triton.language as tl
-import triton.profiler as proton
+import flagtree.profiler as proton
 from triton.tools.tensor_descriptor import TensorDescriptor
 from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
 
@@ -331,7 +331,7 @@ def bench_block_scaled(K, block_scale_type="nvfp4", reps=10):
 
 
 def show_profile(profile_name):
-    import triton.profiler.viewer as proton_viewer
+    import flagtree.profiler.viewer as proton_viewer
 
     metric_names = ["time/ms"]
     metric_names = ["tflop/s"] + metric_names
diff --git a/setup.py b/setup.py
index 6bf11d75d..33936dd8e 100644
--- a/setup.py
+++ b/setup.py
@@ -2,6 +2,7 @@ import os
 import platform
 import re
 import contextlib
+import runpy  # FlagPrism policy is loaded before it can be installed as a package.
 import shlex
 import shutil
 import subprocess
@@ -137,6 +138,31 @@ def check_env_flag(name: str, default: str = "") -> bool:
     return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]
 
 
+# FlagPrism integration: FlagTree owns this bootstrap; wheel policy stays in the submodule.
+def _load_flagprism_build_config():
+    project_root = Path(__file__).resolve().parent
+    helper_path = project_root / "third_party" / "FlagPrism" / "python" / "flagprism_build.py"
+    if not helper_path.is_file():
+        options = (
+            "TRITON_BUILD_FLAGPRISM",
+            "TRITON_BUILD_DEVTOOLS",
+            "TRITON_BUILD_PROTON",
+        )
+        configured = [name for name in options if name in os.environ]
+        if not configured or any(check_env_flag(name) for name in configured):
+            raise RuntimeError(
+                "FlagPrism sources are missing. Run "
+                "`git submodule update --init --recursive`."
+            )
+        return None
+
+    policy = runpy.run_path(str(helper_path), run_name="_flagprism_build")
+    return policy["create_build_config"](project_root)
+
+
+FLAGPRISM = _load_flagprism_build_config()
+
+
 def get_build_type():
     if check_env_flag("DEBUG"):
         return "Debug"
@@ -400,9 +426,16 @@ class CMakeClean(clean):
 class CMakeBuildPy(build_py):
 
     def run(self) -> None:
+        # FlagPrism: sanitize a reused build_lib before native components are written.
+        if FLAGPRISM is not None:
+            FLAGPRISM.prepare_build_tree(self.build_lib)
         self.run_command('build_ext')
         helper.write_flagtree_backend_file()
-        return super().run()
+        result = super().run()
+        # FlagPrism: remove stale source artifacts copied by setuptools afterward.
+        if FLAGPRISM is not None:
+            FLAGPRISM.finalize_build_tree(self.build_lib)
+        return result
 
 
 class CMakeExtension(Extension):
@@ -488,6 +521,9 @@ class CMakeBuild(build_ext):
             "-DTRITON_PLUGIN_DIRS=" + ';'.join([b.src_dir for b in backends if b.is_external]),
             "-DTRITON_WHEEL_DIR=" + wheeldir
         ]
+        # FlagPrism: forward the unified suite switch and wheel output paths.
+        if FLAGPRISM is not None:
+            cmake_args += FLAGPRISM.cmake_args(self.build_lib)
         cmake_args += helper.get_backend_cmake_args(build_ext=self)
         if lit_dir is not None:
             cmake_args.append("-DLLVM_EXTERNAL_LIT=" + lit_dir)
@@ -528,15 +564,16 @@ class CMakeBuild(build_ext):
                 "-DCMAKE_CXX_FLAGS=-fsanitize=address",
             ]
 
-        # environment variables we will pass through to cmake
+        # FlagPrism translates legacy component switches into its unified CMake option.
+        # Only unrelated core build switches still pass through directly.
         passthrough_args = [
-            "TRITON_BUILD_PROTON",
             "TRITON_BUILD_WITH_CCACHE",
             "TRITON_PARALLEL_LINK_JOBS",
         ]
         cmake_args += [f"-D{option}={os.getenv(option)}" for option in passthrough_args if option in os.environ]
 
-        if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
+        # FlagPrism: resolve profiler-native dependencies only for the combined build.
+        if FLAGPRISM is not None and FLAGPRISM.enabled:
             cmake_args += self.get_proton_cmake_args()
 
         if is_offline_build():
@@ -647,6 +684,9 @@ else:
 
 def get_package_dirs():
     yield ("", "python")
+    # FlagPrism: map submodule sources directly into the parent wheel namespace.
+    if FLAGPRISM is not None:
+        yield from FLAGPRISM.package_dirs()
 
     for backend in backends:
         # we use symlinks for external plugins
@@ -667,13 +707,13 @@ def get_package_dirs():
             for x in os.listdir(backend.tools_dir):
                 yield (f"triton.tools.extra.{x}", os.path.join(backend.tools_dir, x))
 
-    if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
-        yield ("triton.profiler", "third_party/proton/proton")
-        yield ("triton.profiler.hooks", "third_party/proton/proton/hooks")
 
 
 def get_packages():
-    yield from find_packages(where="python", include=["triton", "triton.*"])
+    # FlagPrism publishes its stable public API below the parent `flagtree` package.
+    yield from find_packages(where="python", include=["flagtree", "triton", "triton.*"])
+    if FLAGPRISM is not None:
+        yield from FLAGPRISM.packages()
 
     for backend in backends:
         yield f"triton.backends.{backend.name}"
@@ -695,8 +735,7 @@ def get_packages():
     elif helper.flagtree_backend == "mthreads":
         yield f"triton/language/extra/musa"
 
-    if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
-        yield "triton.profiler"
+    # FlagPrism package mappings above replace the legacy `triton.profiler` entry.
 
 
 def add_link_to_backends(external_only):
@@ -736,16 +775,9 @@ if helper.flagtree_backend == "xpu":
 # }
 
 
-def add_link_to_proton():
-    proton_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "third_party", "proton", "proton"))
-    proton_install_dir = os.path.join(os.path.dirname(__file__), "python", "triton", "profiler")
-    update_symlink(proton_install_dir, proton_dir)
-
-
+# FlagPrism uses package_dir mappings, so the parent tree no longer creates a Proton symlink.
 def add_links(external_only):
     add_link_to_backends(external_only=external_only)
-    if not external_only and check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
-        add_link_to_proton()
 
 
 class plugin_bdist_wheel(bdist_wheel):
@@ -799,11 +831,10 @@ class plugin_sdist(sdist):
 
 def get_entry_points():
     entry_points = {}
-    if check_env_flag("TRITON_BUILD_PROTON", "ON"):  # Default ON
-        entry_points["console_scripts"] = [
-            "proton-viewer = triton.profiler.viewer:main",
-            "proton = triton.profiler.proton:main",
-        ]
+    # FlagPrism: publish Profiler CLIs only when the combined suite is enabled.
+    if FLAGPRISM is not None:
+        if console_scripts := FLAGPRISM.console_scripts():
+            entry_points["console_scripts"] = console_scripts
     entry_points["triton.backends"] = [f"{b.name} = triton.backends.{b.name}" for b in backends]
     return entry_points
 
diff --git a/test/lib/CMakeLists.txt b/test/lib/CMakeLists.txt
index ae9229519..58a025b7b 100644
--- a/test/lib/CMakeLists.txt
+++ b/test/lib/CMakeLists.txt
@@ -1,4 +1,7 @@
 add_subdirectory(Analysis)
 add_subdirectory(Dialect)
 add_subdirectory(Instrumentation)
+# Proton tests moved to FlagPrism; the legacy parent target is intentionally disabled.
+#[[
 add_subdirectory(Proton)
+]]
diff --git a/test/lib/Proton/CMakeLists.txt b/test/lib/Proton/CMakeLists.txt
index 0aad8417c..eb4569c3f 100644
--- a/test/lib/Proton/CMakeLists.txt
+++ b/test/lib/Proton/CMakeLists.txt
@@ -1,3 +1,5 @@
+# Proton C++ tests moved to FlagPrism; retain the legacy target for audit only.
+#[[
 add_mlir_library(TritonTestProton
   TestScopeIdAllocation.cpp
 
@@ -5,3 +7,4 @@ add_mlir_library(TritonTestProton
   MLIRPass
   ${triton_libs}
 )
+]]
diff --git a/test/lib/Proton/TestScopeIdAllocation.cpp b/test/lib/Proton/TestScopeIdAllocation.cpp
index 7140c0508..8236e7501 100644
--- a/test/lib/Proton/TestScopeIdAllocation.cpp
+++ b/test/lib/Proton/TestScopeIdAllocation.cpp
@@ -1,51 +1 @@
-#include "mlir/Pass/Pass.h"
-#include "third_party/proton/Dialect/include/Analysis/ScopeIdAllocation.h"
-
-using namespace mlir;
-using namespace triton::proton;
-
-namespace {
-
-struct TestScopeIdAllocationPass
-    : public PassWrapper<TestScopeIdAllocationPass, OperationPass<ModuleOp>> {
-
-  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestScopeIdAllocationPass);
-
-  TestScopeIdAllocationPass() = default;
-  TestScopeIdAllocationPass(const TestScopeIdAllocationPass &other)
-      : PassWrapper<TestScopeIdAllocationPass, OperationPass<ModuleOp>>(other) {
-  }
-
-  StringRef getArgument() const final {
-    return "test-print-scope-id-allocation";
-  }
-  StringRef getDescription() const final {
-    return "print the result of the scope id allocation pass";
-  }
-
-  void runOnOperation() override {
-    ModuleOp moduleOp = getOperation();
-    // Convert to std::string can remove quotes from opName
-    ModuleScopeIdAllocation moduleScopeIdAllocation(moduleOp);
-    moduleOp.walk([&](triton::FuncOp funcOp) {
-      auto opName = SymbolTable::getSymbolName(funcOp).getValue().str();
-      mlir::emitRemark(funcOp.getLoc(), opName);
-      funcOp.walk([&](RecordOp recordOp) {
-        auto scopeId = moduleScopeIdAllocation.getOpScopeId(recordOp);
-        mlir::emitRemark(recordOp.getLoc()) << "scope id = " << scopeId;
-      });
-    });
-  }
-};
-
-} // namespace
-
-namespace mlir {
-namespace test {
-namespace proton {
-void registerTestScopeIdAllocationPass() {
-  PassRegistration<TestScopeIdAllocationPass>();
-}
-} // namespace proton
-} // namespace test
-} // namespace mlir
+// FlagPrism: implementation moved to proton/Dialect/test in the submodule.
diff --git a/third_party/ascend/backend/compiler.py b/third_party/ascend/backend/compiler.py
index 7803fc46e..0a43e58c6 100644
--- a/third_party/ascend/backend/compiler.py
+++ b/third_party/ascend/backend/compiler.py
@@ -32,6 +32,8 @@ from pathlib import Path
 from types import ModuleType
 from typing import Any, Dict, Optional, Tuple, Union
 
+# FlagPrism: use the core no-op gateway at the Ascend serialization boundary.
+from triton import _flagprism
 from triton._C.libtriton import ir, passes, ascend
 from triton.backends.ascend.utils import (
     _check_bishengir_api_change,
@@ -165,6 +167,10 @@ def ttir_to_linalg(mod, metadata, opt, *, named_ops=False):
 
         pm.run(mod)
 
+        # FlagPrism: this is the last structured IR point before the
+        # Ascend adapter serializes the module.
+        _flagprism.run_compiler_hook("ttadapter.pre_serialize", mod, metadata)
+
         if opt.debug:
             dump_manager = get_dump_manager(metadata["hash"])
             dump_manager.put(str(mod), "kernel.ttadapter.mlir", binary=False)
@@ -775,6 +781,8 @@ def get_libdevice():
 @dataclass(frozen=True)
 class NPUOptions:
     debug: bool = False
+    # FlagPrism: instrumentation mode participates in the option hash.
+    instrumentation_mode: str = ""
     sanitize_overflow: bool = True
     llvm_version: int = 15
     kernel_name: str = "triton_"
diff --git a/third_party/ascend/backend/driver.py b/third_party/ascend/backend/driver.py
index 5f48ab3af..8d73a8d1d 100644
--- a/third_party/ascend/backend/driver.py
+++ b/third_party/ascend/backend/driver.py
@@ -18,16 +18,20 @@
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 
-from pathlib import Path
-import tempfile
+from dataclasses import dataclass
 import os
 import os.path
 import re
 import subprocess
 import sysconfig
+import tempfile
+from pathlib import Path
 from typing import Optional
 import functools
 import hashlib
+
+# FlagPrism uses the core no-op gateway so core-only builds keep this backend importable.
+from triton import _flagprism
 from triton.runtime.cache import get_cache_manager, get_dump_manager
 from triton.backends.driver import DriverBase
 from triton.backends.compiler import GPUTarget
@@ -94,6 +98,33 @@ class NPUUtils(object):
         return self.get_device_properties("npu")["num_vectorcore"]
 
 
+@dataclass(frozen=True)
+class _DebuggerHiddenArgABI:
+    """FlagPrism: generated-launcher fragments for one hidden pointer."""
+
+    parse_format: str = ""
+    extracted_declaration: str = ""
+    parse_argument: str = ""
+    launch_declaration: str = ""
+    struct_field: str = ""
+    struct_value: str = ""
+    call_argument: str = ""
+
+    @classmethod
+    def from_metadata(cls, metadata):
+        if not bool(getattr(metadata, "debug_launch_hidden_arg", False)):
+            return cls()
+        return cls(
+            parse_format="K",
+            extracted_declaration="uint64_t _debugHiddenArg = 0;",
+            parse_argument=", &_debugHiddenArg",
+            launch_declaration=", uint64_t debugHiddenArg",
+            struct_field="void* debug_hidden_arg __attribute__((aligned(8)));",
+            struct_value="reinterpret_cast<void*>(debugHiddenArg),",
+            call_argument=", _debugHiddenArg",
+        )
+
+
 class NPULauncher(object):
 
     def __init__(self, src, metadata):
@@ -108,6 +139,8 @@ class NPULauncher(object):
         signature = {cst_key(key): value for key, value in src.signature.items()}
         wrapper_src = make_launcher(constants, signature, metadata)
         so_launcher_path = make_npu_launcher_stub(header_src, wrapper_src, metadata.debug)
+        # FlagPrism reads the compile-time hidden-argument contract at launch.
+        self.metadata = metadata
         # setup for remote run
         # TODO: use a var to pack all vars required to run on a remote machine
         self.mix_mode = metadata.mix_mode
@@ -132,7 +165,22 @@ class NPULauncher(object):
         else:
             if self.compile_only:
                 return
-            profiler_registered = self.launch(*args, **kwargs)
+            debug_enabled = bool(getattr(self.metadata, "debug_enabled", False))
+            if debug_enabled:
+                # FlagPrism: the core gateway owns optional-component
+                # loading; the backend only provides Ascend launch state.
+                with _flagprism.ascend_debugger_launch_context(
+                    self.metadata,
+                    args[:3],
+                    args[3],
+                    args[6],
+                    args[9:],
+                ) as hidden_args:
+                    profiler_registered = self.launch(
+                        *args, *hidden_args, **kwargs
+                    )
+            else:
+                profiler_registered = self.launch(*args, **kwargs)
             import triton
             triton.backends.ascend.utils.TRITON_PROFILER_REGISTERED = True if profiler_registered == 1 else False
 
@@ -517,12 +565,17 @@ def make_launcher(constants, signature, metadata):
         PyObject* launch_enter_hook, *launch_exit_hook;
         *args_expand
     """
+    # FlagPrism: keep every generated hidden-argument ABI fragment in
+    # one object so the normal launcher template remains auditable.
+    debug_abi = _DebuggerHiddenArgABI.from_metadata(metadata)
+
     args_format = ''.join([format_of(ty) for ty in signature.values()])
-    format = "iiiKKOOOO" + args_format
+    format = "iiiKKOOOO" + args_format + debug_abi.parse_format
     signature = ','.join(map(_serialize_signature, signature.values()))
     signature = list(filter(bool, signature.split(',')))
     signature = {i: s for i, s in enumerate(signature)}
     args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
+    args_list += debug_abi.parse_argument
     # Record the end of regular arguments;
     # subsequent arguments are architecture-specific descriptors.
     arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")
@@ -799,7 +852,8 @@ extern "C" {
 
 {cpp_device_pointer}
 
-static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds{', ' + arg_decls if len(signature) > 0 else ''}) {{
+// FlagPrism: append the optional debugger pointer to the generated launch ABI.
+static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds{', ' + arg_decls if len(signature) > 0 else ''}{debug_abi.launch_declaration}) {{
   // only 1D parallelization is supported for NPU
   // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
   // base_ptr offset shape and stride are not used, arbitrarily set for now
@@ -857,6 +911,8 @@ static void _launch(const char* kernelName, const void* func, rtStream_t stream,
       {'void* syncBlockLock __attribute__((aligned(8)));' if not metadata.force_simt_only else ''}
       {'void* workspace_addr __attribute__((aligned(8)));' if not metadata.force_simt_only else ''}
       {' '.join(f'{ty_to_cpp(ty)} arg{i} __attribute__((aligned({4 if ty[0] != "*" and ty[-2:] != "64" else 8})));' for i, ty in signature.items() if i not in constants and ty != "constexpr")}
+      // FlagPrism: optional debugger control pointer; empty in normal launches.
+      {debug_abi.struct_field}
       {' '.join(f'{ty_to_cpp(ty)} grid{mark} __attribute__((aligned(4)));' for mark, ty in grid_info.items())}
       {'void* DTData __attribute__((aligned(8)));' if enable_device_print else ''}
     }} args = {{
@@ -866,6 +922,8 @@ static void _launch(const char* kernelName, const void* func, rtStream_t stream,
       {(lambda _rt: (', '.join(_rt) + ',') if _rt else '')(
         [f'static_cast<{ty_to_cpp(ty)}>(arg{i})' for i, ty in signature.items() if i not in constants and ty != "constexpr"]
       )}
+      // FlagPrism: initialize the optional pointer in its packed ABI slot.
+      {debug_abi.struct_value}
       {', '.join(f'static_cast<{ty_to_cpp(ty)}>(grid{mark})' for mark, ty in grid_info.items())}
       {', static_cast<void*>(DTData)' if enable_device_print else ''}
     }};
@@ -921,6 +979,8 @@ static PyObject* launch(PyObject* self, PyObject* args) {{
   PyObject *launch_exit_hook = NULL;
   std::vector<std::vector<int64_t>> tensorShapes;
 
+  // FlagPrism: decode the optional hidden argument after user arguments.
+  {debug_abi.extracted_declaration}
   {newline.join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])}
   if(!PyArg_ParseTuple(
       args, \"{format}\",
@@ -963,7 +1023,8 @@ static PyObject* launch(PyObject* self, PyObject* args) {{
 
   // raise exception asap
   {newline.join(ptr_decls)}
-  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
+  // FlagPrism: forward the optional pointer after all user arguments.
+  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''}{debug_abi.call_argument});
   if (PyErr_Occurred()) {{
     return NULL;
   }}
diff --git a/third_party/ascend/bin/RegisterTritonDialects.h b/third_party/ascend/bin/RegisterTritonDialects.h
index c76daf291..adc76ffc1 100644
--- a/third_party/ascend/bin/RegisterTritonDialects.h
+++ b/third_party/ascend/bin/RegisterTritonDialects.h
@@ -23,13 +23,10 @@
 #include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
 #include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
 #include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
-#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
-#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
-#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
-#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"
-#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
-#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
-#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h"
+#ifdef __PROTON__
+// FlagPrism: Proton owns its dialect and production-pass registry.
+#include "FlagPrism/proton/Dialect/include/Integration/Registration.h"
+#endif
 #include "triton/Dialect/Gluon/Transforms/Passes.h"
 #include "triton/Dialect/Triton/IR/Dialect.h"
 #include "triton/Dialect/TritonGPU/IR/Dialect.h"
@@ -68,9 +65,7 @@ void registerTestMembarPass();
 void registerTestAMDGPUMembarPass();
 void registerTestTritonAMDGPURangeAnalysis();
 void registerTestLoopPeelingPass();
-namespace proton {
-void registerTestScopeIdAllocationPass();
-} // namespace proton
+// FlagPrism declares the migrated Proton test pass in its integration header.
 } // namespace test
 } // namespace mlir
 
@@ -142,15 +137,11 @@ inline void registerTritonDialects(mlir::DialectRegistry &registry) {
   // NVGPU transform passes
   mlir::registerNVHopperTransformsPasses();
 
-  // Proton passes
-  mlir::test::proton::registerTestScopeIdAllocationPass();
-  mlir::triton::proton::registerConvertProtonToProtonGPU();
-  mlir::triton::proton::gpu::registerConvertProtonNvidiaGPUToLLVM();
-  mlir::triton::proton::gpu::registerConvertProtonAMDGPUToLLVM();
-  mlir::triton::proton::gpu::registerAllocateProtonSharedMemoryPass();
-  mlir::triton::proton::gpu::registerAllocateProtonGlobalScratchBufferPass();
-  mlir::triton::proton::gpu::registerScheduleBufferStorePass();
-  mlir::triton::proton::gpu::registerAddSchedBarriersPass();
+  // FlagPrism owns Proton's production and legacy lit-test registration.
+#ifdef __PROTON__
+  mlir::triton::proton::registerFlagTreeProtonTestPasses();
+  mlir::triton::proton::registerFlagTreeProtonPassesAndDialects(registry);
+#endif
 
   // DynamicCVPipeline passes
   mlir::triton::registerAddControlFlowConditionPasses();
@@ -165,8 +156,8 @@ inline void registerTritonDialects(mlir::DialectRegistry &registry) {
       mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
       mlir::triton::nvgpu::NVGPUDialect, mlir::triton::nvws::NVWSDialect,
       mlir::triton::amdgpu::TritonAMDGPUDialect,
-      mlir::triton::proton::ProtonDialect,
-      mlir::triton::proton::gpu::ProtonGPUDialect, mlir::ROCDL::ROCDLDialect,
+      // FlagPrism registers Proton dialects through the centralized call above.
+      mlir::ROCDL::ROCDLDialect,
       mlir::triton::gluon::GluonDialect,
       mlir::triton::ascend::TritonAscendDialect, mlir::hivm::HIVMDialect,
       mlir::scope::ScopeDialect, mlir::hacc::HACCDialect,
diff --git a/third_party/ascend/python/src/main.cc b/third_party/ascend/python/src/main.cc
index 76f52e08b..c42fcdd4e 100644
--- a/third_party/ascend/python/src/main.cc
+++ b/third_party/ascend/python/src/main.cc
@@ -10,11 +10,14 @@ namespace py = pybind11;
 #define FOR_EACH_4(MACRO, X, ...) MACRO(X) FOR_EACH_3(MACRO, __VA_ARGS__)
 #define FOR_EACH_5(MACRO, X, ...) MACRO(X) FOR_EACH_4(MACRO, __VA_ARGS__)
 #define FOR_EACH_6(MACRO, X, ...) MACRO(X) FOR_EACH_5(MACRO, __VA_ARGS__)
+// FlagPrism adds Proton and Debugger plugins; extend the static loader capacity.
+#define FOR_EACH_7(MACRO, X, ...) MACRO(X) FOR_EACH_6(MACRO, __VA_ARGS__)
+#define FOR_EACH_8(MACRO, X, ...) MACRO(X) FOR_EACH_7(MACRO, __VA_ARGS__)
 
 #define FOR_EACH_NARG(...) FOR_EACH_NARG_(__VA_ARGS__, FOR_EACH_RSEQ_N())
 #define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__)
-#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, N, ...) N
-#define FOR_EACH_RSEQ_N() 6, 5, 4, 3, 2, 1, 0
+#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
+#define FOR_EACH_RSEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0
 
 #define CONCATENATE(x, y) CONCATENATE1(x, y)
 #define CONCATENATE1(x, y) x##y
diff --git a/third_party/enflame/cmake/triton_gcu.cmake b/third_party/enflame/cmake/triton_gcu.cmake
index e16789af0..589a69db2 100644
--- a/third_party/enflame/cmake/triton_gcu.cmake
+++ b/third_party/enflame/cmake/triton_gcu.cmake
@@ -24,15 +24,36 @@ include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
 # 使用本地的 triton 文件，不需要下载（使用根目录的 triton）
 set(third_party_triton_${arch}_fetch_src "${CMAKE_SOURCE_DIR}")
 set(third_party_triton_${arch}_fetch_bin "${CMAKE_CURRENT_BINARY_DIR}/third_party_triton_${arch}_bin")
+# Legacy Proton source glob retained for audit; FlagPrism owns these sources.
+#[[
 file(GLOB_RECURSE third_party_triton_${arch}_src "${CMAKE_SOURCE_DIR}/include/*" "${CMAKE_SOURCE_DIR}/lib/*" "${CMAKE_SOURCE_DIR}/third_party/f2reduce/*" "${CMAKE_SOURCE_DIR}/third_party/proton/*")
+]]
+file(GLOB_RECURSE third_party_triton_${arch}_src
+  "${CMAKE_SOURCE_DIR}/include/*"
+  "${CMAKE_SOURCE_DIR}/lib/*"
+  "${CMAKE_SOURCE_DIR}/third_party/f2reduce/*"
+)
+if(TRITON_BUILD_PROTON)
+  # FlagPrism: rebuild the nested compiler when Proton sources change.
+  file(GLOB_RECURSE _flagtree_proton_src
+    "${CMAKE_SOURCE_DIR}/third_party/FlagPrism/proton/*")
+  list(APPEND third_party_triton_${arch}_src ${_flagtree_proton_src})
+endif()
 
 include(${CMAKE_CURRENT_LIST_DIR}/triton_${arch}.cmake)
+if(TRITON_BUILD_PROTON)
+  include(
+    "${CMAKE_SOURCE_DIR}/third_party/FlagPrism/proton/cmake/ProtonDialectObjects.cmake")
+  flagtree_append_proton_dialect_objects(
+    triton_${arch}_objs "${third_party_triton_${arch}_fetch_bin}")
+endif()
 
 file(MAKE_DIRECTORY ${third_party_triton_${arch}_fetch_bin})
 
 list(APPEND triton_cmake_args -DMLIR_DIR=${MLIR_DIR})
 list(APPEND triton_cmake_args -DLLVM_LIBRARY_DIR=${LLVM_LIBRARY_DIR})
 list(APPEND triton_cmake_args -DTRITON_BUILD_UT=OFF)
+list(APPEND triton_cmake_args -DTRITON_BUILD_FLAGPRISM=${TRITON_BUILD_FLAGPRISM})
 list(APPEND triton_cmake_args -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
 list(APPEND triton_cmake_args -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
 list(APPEND triton_cmake_args -DCMAKE_BUILD_TYPE=Release)
diff --git a/third_party/enflame/cmake/triton_gcu300.cmake b/third_party/enflame/cmake/triton_gcu300.cmake
index 3c74eecb1..df336d4ca 100644
--- a/third_party/enflame/cmake/triton_gcu300.cmake
+++ b/third_party/enflame/cmake/triton_gcu300.cmake
@@ -214,6 +214,9 @@ ${third_party_triton_${arch}_fetch_bin}/third_party/nvidia/lib/TritonNVIDIAGPUTo
 ${third_party_triton_${arch}_fetch_bin}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeFiles/TritonNVIDIAGPUToLLVM.dir/TMAToLLVM.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeFiles/TritonNVIDIAGPUToLLVM.dir/TritonGPUToLLVM.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeFiles/TritonNVIDIAGPUToLLVM.dir/Utility.cpp.o
+# FlagPrism appends the active Proton objects centrally. Keep the legacy list
+# below as a non-executable audit reference.
+#[[
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/Analysis/CMakeFiles/ProtonAnalysis.dir/ScopeIdAllocation.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/Dialect/ProtonGPU/IR/CMakeFiles/ProtonGPUIR.dir/Dialect.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/Dialect/ProtonGPU/IR/CMakeFiles/ProtonGPUIR.dir/Ops.cpp.o
@@ -233,4 +236,5 @@ ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/ProtonGPU
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/CMakeFiles/ProtonNVIDIAGPUToLLVM.dir/NvidiaPatternProtonGPUOpToLLVM.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/CMakeFiles/ProtonNVIDIAGPUToLLVM.dir/TargetInfo.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/ProtonToProtonGPU/CMakeFiles/ProtonToProtonGPU.dir/ProtonToProtonGPUPass.cpp.o
+]]
 )
diff --git a/third_party/enflame/cmake/triton_gcu400.cmake b/third_party/enflame/cmake/triton_gcu400.cmake
index 3c74eecb1..df336d4ca 100644
--- a/third_party/enflame/cmake/triton_gcu400.cmake
+++ b/third_party/enflame/cmake/triton_gcu400.cmake
@@ -214,6 +214,9 @@ ${third_party_triton_${arch}_fetch_bin}/third_party/nvidia/lib/TritonNVIDIAGPUTo
 ${third_party_triton_${arch}_fetch_bin}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeFiles/TritonNVIDIAGPUToLLVM.dir/TMAToLLVM.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeFiles/TritonNVIDIAGPUToLLVM.dir/TritonGPUToLLVM.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/CMakeFiles/TritonNVIDIAGPUToLLVM.dir/Utility.cpp.o
+# FlagPrism appends the active Proton objects centrally. Keep the legacy list
+# below as a non-executable audit reference.
+#[[
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/Analysis/CMakeFiles/ProtonAnalysis.dir/ScopeIdAllocation.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/Dialect/ProtonGPU/IR/CMakeFiles/ProtonGPUIR.dir/Dialect.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/Dialect/ProtonGPU/IR/CMakeFiles/ProtonGPUIR.dir/Ops.cpp.o
@@ -233,4 +236,5 @@ ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/ProtonGPU
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/CMakeFiles/ProtonNVIDIAGPUToLLVM.dir/NvidiaPatternProtonGPUOpToLLVM.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/CMakeFiles/ProtonNVIDIAGPUToLLVM.dir/TargetInfo.cpp.o
 ${third_party_triton_${arch}_fetch_bin}/third_party/proton/Dialect/lib/ProtonToProtonGPU/CMakeFiles/ProtonToProtonGPU.dir/ProtonToProtonGPUPass.cpp.o
+]]
 )
diff --git a/third_party/tsingmicro/bin/RegisterTritonDialects.h b/third_party/tsingmicro/bin/RegisterTritonDialects.h
index fa140a076..d63746c78 100644
--- a/third_party/tsingmicro/bin/RegisterTritonDialects.h
+++ b/third_party/tsingmicro/bin/RegisterTritonDialects.h
@@ -4,7 +4,10 @@
 #include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
 #include "amd/include/TritonAMDGPUTransforms/Passes.h"
 #include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
-#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
+#ifdef __PROTON__
+// FlagPrism: Proton is optional for this backend too.
+#include "FlagPrism/proton/Dialect/include/Integration/Registration.h"
+#endif
 #include "triton/Dialect/Triton/IR/Dialect.h"
 #include "triton/Dialect/TritonGPU/IR/Dialect.h"
 #include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
@@ -139,6 +142,11 @@ inline void registerTritonDialects(mlir::DialectRegistry &registry) {
   // Math dialect passes
   mlir::test::registerTestMathPolynomialApproximationPass();
 
+#ifdef __PROTON__
+  // FlagPrism centralizes Proton dialect and pass registration.
+  mlir::triton::proton::registerFlagTreeProtonPassesAndDialects(registry);
+#endif
+
   // FIXME: May not need all of these
   // mlir::registerAllDialects(registry);
   // Register all external models.
@@ -195,7 +203,8 @@ inline void registerTritonDialects(mlir::DialectRegistry &registry) {
       mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
       mlir::triton::nvgpu::NVGPUDialect,
       mlir::triton::amdgpu::TritonAMDGPUDialect,
-      mlir::triton::proton::ProtonDialect, mlir::ROCDL::ROCDLDialect,
+      // FlagPrism registers Proton dialects through the centralized call above.
+      mlir::ROCDL::ROCDLDialect,
       mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect,
       mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
       mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
diff --git a/python/flagtree/__init__.py b/python/flagtree/__init__.py
new file mode 100644
index 000000000..226622f39
--- /dev/null
+++ b/python/flagtree/__init__.py
@@ -0,0 +1,5 @@
+"""FlagPrism debugger and profiler components bundled with FlagTree."""
+
+from triton import __version__
+
+__all__ = ["__version__"]
diff --git a/python/triton/_flagprism.py b/python/triton/_flagprism.py
new file mode 100644
index 000000000..1a178ed43
--- /dev/null
+++ b/python/triton/_flagprism.py
@@ -0,0 +1,167 @@
+"""Integration boundary between FlagTree core and bundled FlagPrism.
+
+Core compiler and runtime code must use this module instead of importing the
+Debugger or Profiler implementations directly. Optional callbacks are no-ops
+until their package is imported and registers a compatible component.
+"""
+
+from __future__ import annotations
+
+from importlib import import_module
+from threading import RLock
+from typing import Any
+
+
+COMPONENT_API_VERSION = 1
+_COMPONENT_MODULES = {
+    "debugger": "flagtree.debugger",
+    "profiler": "flagtree.profiler",
+}
+_COMPONENT_BUILD_OPTION = "TRITON_BUILD_FLAGPRISM"
+
+
+class ComponentNotInstalledError(ModuleNotFoundError):
+    pass
+
+
+class ComponentCompatibilityError(ImportError):
+    pass
+
+
+_lock = RLock()
+_components: dict[str, Any] = {}
+
+
+def _module_name(name: str) -> str:
+    try:
+        return _COMPONENT_MODULES[name]
+    except KeyError as error:
+        raise ComponentCompatibilityError(
+            f"unsupported FlagPrism component {name!r}"
+        ) from error
+
+
+def _validate_component(name: str, component: Any) -> Any:
+    actual_name = str(getattr(component, "name", ""))
+    if actual_name != name:
+        raise ComponentCompatibilityError(
+            f"FlagPrism component {name!r} returned {actual_name!r}"
+        )
+    api_version = int(getattr(component, "api_version", -1))
+    if api_version != COMPONENT_API_VERSION:
+        raise ComponentCompatibilityError(
+            f"FlagTree {name} API mismatch: core={COMPONENT_API_VERSION}, "
+            f"component={api_version}. Use a matching FlagPrism submodule revision."
+        )
+    core_series = str(getattr(component, "core_version_series", ""))
+    if core_series:
+        from triton import __version__
+
+        installed_series = ".".join(str(__version__).split(".")[:2])
+        if installed_series != core_series:
+            raise ComponentCompatibilityError(
+                f"FlagTree {name} targets core {core_series}, but the installed "
+                f"FlagTree core is {__version__}."
+            )
+    return component
+
+
+def register_component(name: str, component: Any) -> Any:
+    _module_name(name)
+    component = _validate_component(name, component)
+    with _lock:
+        current = _components.get(name)
+        if current is not None and current is not component:
+            raise ComponentCompatibilityError(
+                f"FlagPrism component {name!r} is already loaded"
+            )
+        _components[name] = component
+    return component
+
+
+def load_component(name: str, *, required: bool = True) -> Any | None:
+    module_name = _module_name(name)
+    with _lock:
+        if name in _components:
+            return _components[name]
+        try:
+            module = import_module(module_name)
+        except ModuleNotFoundError as error:
+            if error.name != module_name:
+                raise
+            if not required:
+                return None
+            raise ComponentNotInstalledError(
+                f"FlagTree {name} is not included in this build. Rebuild FlagTree "
+                "with its FlagPrism submodule initialized and "
+                f"`{_COMPONENT_BUILD_OPTION}=ON`."
+            ) from None
+        return register_component(name, getattr(module, "component", module))
+
+
+def _registered_components() -> tuple[Any, ...]:
+    with _lock:
+        return tuple(_components.values())
+
+
+def _call_registered(method: str, *args: Any) -> None:
+    for component in _registered_components():
+        callback = getattr(component, method, None)
+        if callable(callback):
+            callback(*args)
+
+
+def _call_required(name: str, method: str, *args: Any):
+    component = load_component(name)
+    callback = getattr(component, method, None)
+    if not callable(callback):
+        raise ComponentCompatibilityError(
+            f"FlagTree {name} does not implement required callback {method!r}"
+        )
+    return callback(*args)
+
+
+def load_dialects(context: Any) -> None:
+    _call_registered("load_dialects", context)
+
+
+def apply_compile_options(options: dict[str, Any]) -> None:
+    _call_registered("apply_compile_options", options)
+
+
+def run_compiler_hook(stage: str, module: Any, metadata: dict[str, Any]) -> None:
+    _call_registered("run_compiler_hook", stage, module, metadata)
+
+
+def annotate_statement(
+    kind: str, generator: Any, node: Any, target: Any, value: Any
+) -> None:
+    _call_registered("annotate_statement", kind, generator, node, target, value)
+
+
+def debug_collect_start(semantic: Any, level: Any, addr_level: Any):
+    return _call_required(
+        "debugger", "debug_collect_start", semantic, level, addr_level
+    )
+
+
+def debug_collect_end(semantic: Any):
+    return _call_required("debugger", "debug_collect_end", semantic)
+
+
+def ascend_debugger_launch_context(
+    metadata: Any,
+    grid: Any,
+    stream: Any,
+    launch_metadata: Any,
+    kernel_args: Any,
+):
+    return _call_required(
+        "debugger",
+        "ascend_launch_context",
+        metadata,
+        grid,
+        stream,
+        launch_metadata,
+        kernel_args,
+    )
diff --git a/python/test/unit/test_flagprism.py b/python/test/unit/test_flagprism.py
new file mode 100644
index 000000000..1b538da24
--- /dev/null
+++ b/python/test/unit/test_flagprism.py
@@ -0,0 +1,202 @@
+from contextlib import nullcontext
+import importlib.util
+from pathlib import Path
+import sys
+import sysconfig
+from types import SimpleNamespace
+
+import pytest
+
+from triton import _flagprism
+
+
+def _load_build_helper():
+    path = (
+        Path(__file__).resolve().parents[3]
+        / "third_party"
+        / "FlagPrism"
+        / "python"
+        / "flagprism_build.py"
+    )
+    spec = importlib.util.spec_from_file_location("_test_flagprism_build", path)
+    module = importlib.util.module_from_spec(spec)
+    sys.modules[spec.name] = module
+    spec.loader.exec_module(module)
+    return module
+
+
+_build_helper = _load_build_helper()
+
+
+@pytest.fixture(autouse=True)
+def isolated_components():
+    previous = dict(_flagprism._components)
+    _flagprism._components.clear()
+    try:
+        yield
+    finally:
+        _flagprism._components.clear()
+        _flagprism._components.update(previous)
+
+
+def _component(name, **attributes):
+    values = {
+        "name": name,
+        "api_version": _flagprism.COMPONENT_API_VERSION,
+    }
+    values.update(attributes)
+    return SimpleNamespace(**values)
+
+
+def test_public_component_modules_use_flagtree_namespace():
+    assert _flagprism._COMPONENT_MODULES == {
+        "debugger": "flagtree.debugger",
+        "profiler": "flagtree.profiler",
+    }
+
+
+@pytest.mark.parametrize(
+    "component",
+    ("debugger", "profiler"),
+)
+def test_missing_component_has_build_instruction(
+    monkeypatch, component
+):
+    def missing(name):
+        raise ModuleNotFoundError(name=name)
+
+    monkeypatch.setattr(_flagprism, "import_module", missing)
+    with pytest.raises(_flagprism.ComponentNotInstalledError) as error:
+        _flagprism.load_component(component)
+    assert "TRITON_BUILD_FLAGPRISM=ON" in str(error.value)
+
+
+def test_known_component_is_loaded_once(monkeypatch):
+    component = _component("debugger")
+    module = SimpleNamespace(component=component)
+    loads = []
+
+    def import_component(name):
+        loads.append(name)
+        return module
+
+    monkeypatch.setattr(_flagprism, "import_module", import_component)
+    assert _flagprism.load_component("debugger") is component
+    assert _flagprism.load_component("debugger") is component
+    assert loads == ["flagtree.debugger"]
+
+
+def test_component_api_mismatch_is_rejected():
+    component = _component("debugger", api_version=999)
+    with pytest.raises(_flagprism.ComponentCompatibilityError, match="API mismatch"):
+        _flagprism.register_component("debugger", component)
+
+
+def test_optional_hooks_are_noops_until_a_component_registers():
+    events = []
+    component = _component(
+        "debugger",
+        apply_compile_options=lambda options: options.update(
+            instrumentation_mode="debug"
+        ),
+        run_compiler_hook=lambda stage, module, metadata: events.append(
+            (stage, module, metadata)
+        ),
+        annotate_statement=lambda *args: events.append(args),
+    )
+    options = {}
+    _flagprism.apply_compile_options(options)
+    assert options == {}
+
+    _flagprism.register_component("debugger", component)
+    _flagprism.apply_compile_options(options)
+    _flagprism.run_compiler_hook("ttir", "module", {"key": "value"})
+    _flagprism.annotate_statement("expression", "generator", "node", None, "value")
+
+    assert options == {"instrumentation_mode": "debug"}
+    assert events == [
+        ("ttir", "module", {"key": "value"}),
+        ("expression", "generator", "node", None, "value"),
+    ]
+
+
+def test_required_ascend_launch_context_is_forwarded():
+    context = nullcontext((123,))
+    component = _component(
+        "debugger",
+        ascend_launch_context=lambda *args: context,
+    )
+    _flagprism.register_component("debugger", component)
+
+    result = _flagprism.ascend_debugger_launch_context(
+        "metadata", (1, 2, 3), "stream", "launch_metadata", ("arg",)
+    )
+    assert result is context
+
+
+def test_unknown_components_are_rejected():
+    with pytest.raises(_flagprism.ComponentCompatibilityError, match="unsupported"):
+        _flagprism.load_component("custom")
+
+
+@pytest.mark.parametrize("enabled", (True, False))
+def test_build_tree_cleanup_prevents_split_wheel_artifacts(tmp_path, enabled):
+    build_lib = tmp_path / "build-lib"
+    triton_root = build_lib / "triton"
+    flagtree_root = build_lib / "flagtree"
+    native_root = triton_root / "_C"
+    cache_root = triton_root / "__pycache__"
+    config = _build_helper.FlagPrismBuildConfig(
+        enabled=enabled,
+        relative_root=Path("third_party/FlagPrism"),
+        root=tmp_path / "FlagPrism",
+    )
+
+    for path in (
+        triton_root / "debugger" / "old.py",
+        build_lib / "flagtree_debugger" / "old.py",
+        flagtree_root / "debugger" / "old.py",
+        native_root / "libproton.so",
+        cache_root / "_components.cpython-311.pyc",
+    ):
+        path.parent.mkdir(parents=True, exist_ok=True)
+        path.write_bytes(b"stale")
+
+    config.prepare_build_tree(str(build_lib))
+    assert not (triton_root / "debugger").exists()
+    assert not (flagtree_root / "debugger").exists()
+    assert not list(native_root.glob("libproton*"))
+
+    expected_native = native_root / (
+        "libproton" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so")
+    )
+    if enabled:
+        for path in (
+            flagtree_root / "debugger" / "__init__.py",
+            flagtree_root / "profiler" / "__init__.py",
+            expected_native,
+        ):
+            path.parent.mkdir(parents=True, exist_ok=True)
+            path.write_bytes(b"current")
+
+    # build_py can copy these stale source-tree files after CMake completes.
+    for path in (
+        native_root / "libproton.so",
+        cache_root / "_components.cpython-311.pyc",
+    ):
+        path.parent.mkdir(parents=True, exist_ok=True)
+        path.write_bytes(b"stale")
+
+    config.finalize_build_tree(str(build_lib))
+    assert not (cache_root / "_components.cpython-311.pyc").exists()
+    if enabled:
+        assert expected_native.is_file()
+        assert sorted(path.name for path in native_root.glob("libproton*")) == [
+            expected_native.name
+        ]
+        assert (flagtree_root / "debugger").is_dir()
+        assert (flagtree_root / "profiler").is_dir()
+    else:
+        assert not list(native_root.glob("libproton*"))
+        assert not (flagtree_root / "debugger").exists()
+        assert not (flagtree_root / "profiler").exists()
````
