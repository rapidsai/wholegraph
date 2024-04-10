# wholegraph 24.04.00 (10 Apr 2024)

## 🐛 Bug Fixes

- Update pre-commit-hooks to v0.0.3 ([#152](https://github.com/rapidsai/wholegraph/pull/152)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fixed README links to point to cuGraph API ([#145](https://github.com/rapidsai/wholegraph/pull/145)) [@acostadon](https://github.com/acostadon)
- [Bugfix] Fix to compile when NVSHMEM is ON ([#142](https://github.com/rapidsai/wholegraph/pull/142)) [@chang-l](https://github.com/chang-l)
- handle more RAPIDS version formats in update-version.sh ([#122](https://github.com/rapidsai/wholegraph/pull/122)) [@jameslamb](https://github.com/jameslamb)

## 🚀 New Features

- Support CUDA 12.2 ([#116](https://github.com/rapidsai/wholegraph/pull/116)) [@jameslamb](https://github.com/jameslamb)

## 🛠️ Improvements

- Use `conda env create --yes` instead of `--force` ([#155](https://github.com/rapidsai/wholegraph/pull/155)) [@bdice](https://github.com/bdice)
- add round-robin shard strategy ([#154](https://github.com/rapidsai/wholegraph/pull/154)) [@linhu-nv](https://github.com/linhu-nv)
- Switch to scikit-build-core ([#150](https://github.com/rapidsai/wholegraph/pull/150)) [@vyasr](https://github.com/vyasr)
- Update script input name ([#147](https://github.com/rapidsai/wholegraph/pull/147)) [@AyodeAwe](https://github.com/AyodeAwe)
- Add upper bound to prevent usage of NumPy 2 ([#146](https://github.com/rapidsai/wholegraph/pull/146)) [@bdice](https://github.com/bdice)
- Replace local copyright check with pre-commit-hooks verify-copyright ([#144](https://github.com/rapidsai/wholegraph/pull/144)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- remove an unnecessary sync in exchange_embeddings_nccl_func ([#143](https://github.com/rapidsai/wholegraph/pull/143)) [@linhu-nv](https://github.com/linhu-nv)
- Use default `rapids-cmake` CUDA_ARCHITECTURES ([#140](https://github.com/rapidsai/wholegraph/pull/140)) [@trxcllnt](https://github.com/trxcllnt)
- Add support for Python 3.11, require NumPy 1.23+ ([#139](https://github.com/rapidsai/wholegraph/pull/139)) [@jameslamb](https://github.com/jameslamb)
- [Bugfix] Host full-neighbor sampling returns wrong results in unit test ([#138](https://github.com/rapidsai/wholegraph/pull/138)) [@chang-l](https://github.com/chang-l)
- use enum to implement log_level in wholememory ([#136](https://github.com/rapidsai/wholegraph/pull/136)) [@linhu-nv](https://github.com/linhu-nv)
- target branch-24.04 for GitHub Actions workflows ([#135](https://github.com/rapidsai/wholegraph/pull/135)) [@jameslamb](https://github.com/jameslamb)
- Add environment-agnostic scripts for running ctests and pytests ([#128](https://github.com/rapidsai/wholegraph/pull/128)) [@trxcllnt](https://github.com/trxcllnt)

# wholegraph 24.02.00 (12 Feb 2024)

## 🐛 Bug Fixes

- Revert &quot;Exclude tests from builds ([#127)&quot; (#130](https://github.com/rapidsai/wholegraph/pull/127)&quot; (#130)) [@raydouglass](https://github.com/raydouglass)
- Exclude tests from builds ([#127](https://github.com/rapidsai/wholegraph/pull/127)) [@vyasr](https://github.com/vyasr)
- fix a bug for embedding optimizer, which leads to undefined behavior ([#108](https://github.com/rapidsai/wholegraph/pull/108)) [@linhu-nv](https://github.com/linhu-nv)
- fix inferencesample option ([#107](https://github.com/rapidsai/wholegraph/pull/107)) [@chuangz0](https://github.com/chuangz0)

## 🚀 New Features

- allow users to control gather/scatter sms ([#124](https://github.com/rapidsai/wholegraph/pull/124)) [@linhu-nv](https://github.com/linhu-nv)

## 🛠️ Improvements

- Logging level ([#123](https://github.com/rapidsai/wholegraph/pull/123)) [@linhu-nv](https://github.com/linhu-nv)
- Fix pip dependencies ([#118](https://github.com/rapidsai/wholegraph/pull/118)) [@trxcllnt](https://github.com/trxcllnt)
- Remove usages of rapids-env-update ([#117](https://github.com/rapidsai/wholegraph/pull/117)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- refactor CUDA versions in dependencies.yaml ([#115](https://github.com/rapidsai/wholegraph/pull/115)) [@jameslamb](https://github.com/jameslamb)
- Don&#39;t overwrite wholegraph_ROOT if provided ([#114](https://github.com/rapidsai/wholegraph/pull/114)) [@vyasr](https://github.com/vyasr)
- added Direct IO support for WholeMemory loading ([#113](https://github.com/rapidsai/wholegraph/pull/113)) [@dongxuy04](https://github.com/dongxuy04)
- Align versions for cudnn, clang-tools, cython, and doxygen with the rest of RAPIDS. ([#112](https://github.com/rapidsai/wholegraph/pull/112)) [@bdice](https://github.com/bdice)
- Reset WholeGraph communicators during the finalize call ([#111](https://github.com/rapidsai/wholegraph/pull/111)) [@chang-l](https://github.com/chang-l)
- Forward-merge branch-23.12 to branch-24.02 ([#102](https://github.com/rapidsai/wholegraph/pull/102)) [@bdice](https://github.com/bdice)

# wholegraph 23.12.00 (6 Dec 2023)

## 🐛 Bug Fixes

- move_vector_clear_outside_loop ([#103](https://github.com/rapidsai/wholegraph/pull/103)) [@chuangz0](https://github.com/chuangz0)
- change pytorch cu121 to stable to fix ci ([#97](https://github.com/rapidsai/wholegraph/pull/97)) [@dongxuy04](https://github.com/dongxuy04)

## 🚀 New Features

- Integrate NVSHMEM into WholeGraph ([#91](https://github.com/rapidsai/wholegraph/pull/91)) [@chuangz0](https://github.com/chuangz0)
- Grace Hopper support and add benchmark ([#87](https://github.com/rapidsai/wholegraph/pull/87)) [@dongxuy04](https://github.com/dongxuy04)

## 🛠️ Improvements

- Fix dependencies on librmm and libraft. ([#96](https://github.com/rapidsai/wholegraph/pull/96)) [@bdice](https://github.com/bdice)
- gather/scatter optimizations ([#90](https://github.com/rapidsai/wholegraph/pull/90)) [@linhu-nv](https://github.com/linhu-nv)
- Use branch-23.12 workflows. ([#84](https://github.com/rapidsai/wholegraph/pull/84)) [@AyodeAwe](https://github.com/AyodeAwe)
- Setup Consistent Nightly Versions for Pip and Conda ([#82](https://github.com/rapidsai/wholegraph/pull/82)) [@divyegala](https://github.com/divyegala)
- Add separate init, expose gather/scatter for WholeMemoryTensor and update example ([#81](https://github.com/rapidsai/wholegraph/pull/81)) [@dongxuy04](https://github.com/dongxuy04)
- Use RNG (random number generator) provided by RAFT ([#79](https://github.com/rapidsai/wholegraph/pull/79)) [@linhu-nv](https://github.com/linhu-nv)
- Build CUDA 12.0 ARM conda packages. ([#74](https://github.com/rapidsai/wholegraph/pull/74)) [@bdice](https://github.com/bdice)
- upload xml docs ([#73](https://github.com/rapidsai/wholegraph/pull/73)) [@AyodeAwe](https://github.com/AyodeAwe)
- replace optparse  with argparser ([#61](https://github.com/rapidsai/wholegraph/pull/61)) [@chuangz0](https://github.com/chuangz0)

# wholegraph 23.10.00 (11 Oct 2023)

## 🐛 Bug Fixes

- Update all versions to 23.10 ([#71](https://github.com/rapidsai/wholegraph/pull/71)) [@raydouglass](https://github.com/raydouglass)
- Use `conda mambabuild` not `mamba mambabuild` ([#67](https://github.com/rapidsai/wholegraph/pull/67)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Update image names ([#70](https://github.com/rapidsai/wholegraph/pull/70)) [@AyodeAwe](https://github.com/AyodeAwe)
- Update to clang 16.0.6. ([#68](https://github.com/rapidsai/wholegraph/pull/68)) [@bdice](https://github.com/bdice)
- Simplify wheel build scripts and allow alphas of RAPIDS dependencies ([#66](https://github.com/rapidsai/wholegraph/pull/66)) [@divyegala](https://github.com/divyegala)
- Fix docs build and slightly optimize ([#63](https://github.com/rapidsai/wholegraph/pull/63)) [@dongxuy04](https://github.com/dongxuy04)
- Use `copy-pr-bot` ([#60](https://github.com/rapidsai/wholegraph/pull/60)) [@ajschmidt8](https://github.com/ajschmidt8)
- PR: Use top-k from RAFT ([#53](https://github.com/rapidsai/wholegraph/pull/53)) [@chuangz0](https://github.com/chuangz0)

# wholegraph 23.08.00 (9 Aug 2023)

## 🚨 Breaking Changes

- Refactoring into 23.08 ([#24](https://github.com/rapidsai/wholegraph/pull/24)) [@BradReesWork](https://github.com/BradReesWork)

## 🛠️ Improvements

- Correct syntax in GHA workflow ([#46](https://github.com/rapidsai/wholegraph/pull/46)) [@tingyu66](https://github.com/tingyu66)
- Refactoring into 23.08 ([#24](https://github.com/rapidsai/wholegraph/pull/24)) [@BradReesWork](https://github.com/BradReesWork)
