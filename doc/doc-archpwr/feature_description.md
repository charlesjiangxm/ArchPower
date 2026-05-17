# ArchPower Feature Description

All event parameters (except [14] ipc, [15] cpi, [16] numCycles which are raw values) are **normalized by numCycles**.

## Global 101-D Feature Vector (feature.npy)

### Hardware Parameters (Index 0–13, 14 total)

| Index | Name | Description |
|-------|------|-------------|
| 0 | FetchWidth | Fetch width |
| 1 | DecodeWidth | Decode width |
| 2 | FetchBufferEntry | Fetch buffer entries |
| 3 | RobEntry | Reorder buffer entries |
| 4 | IntPhyRegister | Integer physical registers |
| 5 | FpPhyRegister | Floating-point physical registers |
| 6 | LDQ/STQEntry | Load/store queue entries |
| 7 | BranchCount | Branch predictor entries |
| 8 | Mem/FpIssueWidth | Memory/FP issue width |
| 9 | IntIssueWidth | Integer issue width |
| 10 | DCache/ICacheWay | D-Cache/I-Cache ways |
| 11 | DCacheTLBEntry | D-TLB entries |
| 12 | DCacheMSHR | D-Cache MSHR entries |
| 13 | ICacheFetchBytes | I-Cache fetch bytes |

### Event Parameters (Index 14–100, 87 total)

| Index | Name | gem5 Stat |
|-------|------|-----------|
| 14 | totalIpc | system.cpu.ipc (raw) |
| 15 | totalCpi | system.cpu.cpi (raw) |
| 16 | numCycles | system.cpu.numCycles (raw) |
| 17 | idleCycles | system.cpu.idleCycles |
| 18 | BTBLookups | branchPred.BTBLookups |
| 19 | condPredicted | branchPred.condPredicted |
| 20 | condIncorrect | branchPred.condIncorrect |
| 21 | intAluAccesses | system.cpu.intAluAccesses |
| 22 | fpAluAccesses | system.cpu.fpAluAccesses |
| 23 | numLoadInsts | system.cpu.numLoadInsts |
| 24 | functionCalls | commit.functionCalls |
| 25 | numSquashedInsts | system.cpu.numSquashedInsts |
| 26 | committedInsts | system.cpu.committedInsts |
| 27 | commit.numDist::mean | commit.numCommittedDist::mean |
| 28 | commit.memRefs | commit.memRefs |
| 29 | numBranches | system.cpu.numBranches |
| 30 | decode.runCycles | decode.runCycles |
| 31 | decode.blockedCycles | decode.blockedCycles |
| 32 | decode.decodedInsts | decode.decodedInsts |
| 33 | InstPrefetch | commit.committedInstType_0::InstPrefetch (all zeros) |
| 34 | (unused) | (all zeros, not in any component mask) |
| 35 | fetch.insts | fetch.insts |
| 36 | fetch.branches | fetch.branches / branchPred.lookups |
| 37 | fetch.cycles | fetch.cycles |
| 38 | numRefs | system.cpu.numRefs |
| 39 | numStoreInsts | system.cpu.numStoreInsts |
| 40 | numInsts | system.cpu.numInsts |
| 41 | intInstQueueWakeupAccesses | intInstQueueWakeupAccesses |
| 42 | numBranches (dup) | same as [29], not in Others mask |
| 43 | numIssuedDist::mean | numIssuedDist::mean |
| 44 | intInstQueueReads | intInstQueueReads |
| 45 | intInstQueueWrites | intInstQueueWrites |
| 46 | intInstQueueWakeupAccesses (dup) | same as [41] |
| 47 | fpInstQueueReads | fpInstQueueReads |
| 48 | fpInstQueueWrites | fpInstQueueWrites |
| 49 | fpInstQueueWakeupAccesses | fpInstQueueWakeupAccesses |
| 50 | intAluAccesses (dup) | same as [21], not in Others mask |
| 51 | fpAluAccesses (dup) | same as [22], not in Others mask |
| 52 | statIssuedInstType_0::total | total issued instructions |
| 53 | IssuedMemRead | statIssuedInstType_0::MemRead |
| 54 | IssuedMemWrite | statIssuedInstType_0::MemWrite |
| 55 | IssuedFloatMemRead | statIssuedInstType_0::FloatMemRead |
| 56 | IssuedFloatMemWrite | statIssuedInstType_0::FloatMemWrite |
| 57 | IssuedIntAlu | statIssuedInstType_0::IntAlu |
| 58 | IssuedIntMult | statIssuedInstType_0::IntMult |
| 59 | IssuedIntDiv | statIssuedInstType_0::IntDiv |
| 60 | IssuedFloatMult | statIssuedInstType_0::FloatMult (all zeros) |
| 61 | IssuedFloatDiv | statIssuedInstType_0::FloatDiv (all zeros) |
| 62 | fuBusy | system.cpu.fuBusy |
| 63 | (unused) | (all zeros, not in Others mask) |
| 64 | (unused) | (all zeros, not in Others mask) |
| 65 | conflictingLoads | MemDepUnit.conflictingLoads |
| 66 | conflictingStores | MemDepUnit.conflictingStores |
| 67 | insertedLoads | MemDepUnit.insertedLoads |
| 68 | insertedStores | MemDepUnit.insertedStores |
| 69 | rename.intLookups | rename.intLookups |
| 70 | rename.renamedOperands | rename.renamedOperands |
| 71 | rename.fpLookups | rename.fpLookups |
| 72 | rename.renamedInsts | rename.renamedInsts |
| 73 | rename.runCycles | rename.runCycles |
| 74 | rename.blockCycles | rename.blockCycles |
| 75 | rename.committedMaps | rename.committedMaps |
| 76 | rob.reads | rob.reads |
| 77 | rob.writes | rob.writes |
| 78 | mem_ctrls.readReqs | mem_ctrls.readReqs |
| 79 | mem_ctrls.writeReqs | mem_ctrls.writeReqs |
| 80 | mem_ctrls.bytesReadSys | mem_ctrls.bytesReadSys |
| 81 | icache.overallAccesses | icache.overallAccesses |
| 82 | icache.overallMisses | icache.demandMisses |
| 83 | icache.ReadReq.mshrHits | icache.demandMshrHits |
| 84 | icache.ReadReq.mshrMisses | icache.demandMshrMisses |
| 85 | icache.tags.totalRefs | icache.tags.totalRefs |
| 86 | icache.tagAccesses | icache.tags.tagAccesses |
| 87 | dcache.ReadReq.accesses | dcache.ReadReq.accesses |
| 88 | dcache.WriteReq.accesses | dcache.WriteReq.accesses |
| 89 | dcache.ReadReq.misses | dcache.ReadReq.misses |
| 90 | dcache.WriteReq.misses | dcache.WriteReq.misses |
| 91 | dcache.overallAccesses | dcache.demandAccesses |
| 92 | dcache.overallMisses | dcache.demandMisses |
| 93 | dcache.MshrHits | dcache.demandMshrHits |
| 94 | dcache.MshrMisses | dcache.demandMshrMisses |
| 95 | dcache.tags.totalRefs | dcache.tags.totalRefs |
| 96 | dcache.tagAccesses | dcache.tags.tagAccesses |
| 97 | intRegfileReads | system.cpu.intRegfileReads |
| 98 | fpRegfileReads | system.cpu.fpRegfileReads |
| 99 | intRegfileWrites | system.cpu.intRegfileWrites |
| 100 | fpRegfileWrites | system.cpu.fpRegfileWrites |

## Component Feature Files (dataset/component_feature/)

Each component `.npy` file extracts columns from the global 101-D feature vector via `component_mask.npy`. `Others.npy` is identical to `feature.npy` (all 101 features).

### BP.npy (200, 6) — mask indices: [0, 7, 18, 19, 20, 29]

| Col | Global Index | Name | Type |
|-----|-------------|------|------|
| 0 | 0 | FetchWidth | Hardware |
| 1 | 7 | BranchCount | Hardware |
| 2 | 18 | BTBLookups | Event |
| 3 | 19 | condPredicted | Event |
| 4 | 20 | condIncorrect | Event |
| 5 | 29 | commit.branches | Event |

### ICache.npy (200, 11) — mask indices: [0, 1, 2, 10, 13, 81, 82, 83, 84, 85, 86]

| Col | Global Index | Name | Type |
|-----|-------------|------|------|
| 0 | 0 | FetchWidth | Hardware |
| 1 | 1 | DecodeWidth | Hardware |
| 2 | 2 | FetchBufferEntry | Hardware |
| 3 | 10 | DCache/ICacheWay | Hardware |
| 4 | 13 | ICacheFetchBytes | Hardware |
| 5 | 81 | icache.overallAccesses | Event |
| 6 | 82 | icache.overallMisses | Event |
| 7 | 83 | icache.ReadReq.mshrHits | Event |
| 8 | 84 | icache.ReadReq.mshrMisses | Event |
| 9 | 85 | icache.tags.totalRefs | Event |
| 10 | 86 | icache.tagAccesses | Event |

### IFU.npy (200, 21) — mask indices: [0, 1, 2, 13, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 44, 45, 46, 47, 48, 49, 52]

| Col | Global Index | Name | Type |
|-----|-------------|------|------|
| 0 | 0 | FetchWidth | Hardware |
| 1 | 1 | DecodeWidth | Hardware |
| 2 | 2 | FetchBufferEntry | Hardware |
| 3 | 13 | ICacheFetchBytes | Hardware |
| 4 | 29 | numBranches | Event |
| 5 | 30 | decode.runCycles | Event |
| 6 | 31 | decode.blockedCycles | Event |
| 7 | 32 | decode.decodedInsts | Event |
| 8 | 35 | fetch.insts | Event |
| 9 | 36 | fetch.branches | Event |
| 10 | 37 | fetch.cycles | Event |
| 11 | 38 | numRefs | Event |
| 12 | 39 | numStoreInsts | Event |
| 13 | 40 | numInsts | Event |
| 14 | 44 | intInstQueueReads | Event |
| 15 | 45 | intInstQueueWrites | Event |
| 16 | 46 | intInstQueueWakeupAccesses | Event |
| 17 | 47 | fpInstQueueReads | Event |
| 18 | 48 | fpInstQueueWrites | Event |
| 19 | 49 | fpInstQueueWakeupAccesses | Event |
| 20 | 52 | statIssuedInstType_0::total | Event |

### RNU.npy (200, 11) — mask indices: [0, 1, 2, 3, 69, 70, 71, 72, 73, 74, 75]

| Col | Global Index | Name | Type |
|-----|-------------|------|------|
| 0 | 0 | FetchWidth | Hardware |
| 1 | 1 | DecodeWidth | Hardware |
| 2 | 2 | FetchBufferEntry | Hardware |
| 3 | 3 | RobEntry | Hardware |
| 4 | 69 | rename.intLookups | Event |
| 5 | 70 | rename.renamedOperands | Event |
| 6 | 71 | rename.fpLookups | Event |
| 7 | 72 | rename.renamedInsts | Event |
| 8 | 73 | rename.runCycles | Event |
| 9 | 74 | rename.blockCycles | Event |
| 10 | 75 | rename.committedMaps | Event |

### ROB.npy (200, 4) — mask indices: [1, 3, 76, 77]

| Col | Global Index | Name | Type |
|-----|-------------|------|------|
| 0 | 1 | DecodeWidth | Hardware |
| 1 | 3 | RobEntry | Hardware |
| 2 | 76 | rob.reads | Event |
| 3 | 77 | rob.writes | Event |

### ISU.npy (200, 6) — mask indices: [1, 8, 9, 38, 55, 57]

| Col | Global Index | Name | Type |
|-----|-------------|------|------|
| 0 | 1 | DecodeWidth | Hardware |
| 1 | 8 | Mem/FpIssueWidth | Hardware |
| 2 | 9 | IntIssueWidth | Hardware |
| 3 | 38 | numRefs | Event |
| 4 | 55 | IssuedFloatMemRead | Event |
| 5 | 57 | IssuedIntAlu | Event |

### Regfile.npy (200, 8) — mask indices: [1, 4, 5, 24, 97, 98, 99, 100]

| Col | Global Index | Name | Type |
|-----|-------------|------|------|
| 0 | 1 | DecodeWidth | Hardware |
| 1 | 4 | IntPhyRegister | Hardware |
| 2 | 5 | FpPhyRegister | Hardware |
| 3 | 24 | functionCalls | Event |
| 4 | 97 | intRegfileReads | Event |
| 5 | 98 | fpRegfileReads | Event |
| 6 | 99 | intRegfileWrites | Event |
| 7 | 100 | fpRegfileWrites | Event |

### FU-Pool.npy (200, 4) — mask indices: [8, 9, 21, 22]

| Col | Global Index | Name | Type |
|-----|-------------|------|------|
| 0 | 8 | Mem/FpIssueWidth | Hardware |
| 1 | 9 | IntIssueWidth | Hardware |
| 2 | 21 | intAluAccesses | Event |
| 3 | 22 | fpAluAccesses | Event |

### LSU.npy (200, 5) — mask indices: [6, 8, 33, 53, 88]

| Col | Global Index | Name | Type |
|-----|-------------|------|------|
| 0 | 6 | LDQ/STQEntry | Hardware |
| 1 | 8 | Mem/FpIssueWidth | Hardware |
| 2 | 33 | InstPrefetch (all zeros) | Event |
| 3 | 53 | IssuedMemRead | Event |
| 4 | 88 | dcache.WriteReq.accesses (MemWrite) | Event |

### DCache.npy (200, 14) — mask indices: [8, 10, 11, 12, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96]

| Col | Global Index | Name | Type |
|-----|-------------|------|------|
| 0 | 8 | Mem/FpIssueWidth | Hardware |
| 1 | 10 | DCache/ICacheWay | Hardware |
| 2 | 11 | DCacheTLBEntry | Hardware |
| 3 | 12 | DCacheMSHR | Hardware |
| 4 | 87 | dcache.ReadReq.accesses | Event |
| 5 | 88 | dcache.WriteReq.accesses | Event |
| 6 | 89 | dcache.ReadReq.misses | Event |
| 7 | 90 | dcache.WriteReq.misses | Event |
| 8 | 91 | dcache.overallAccesses | Event |
| 9 | 92 | dcache.overallMisses | Event |
| 10 | 93 | dcache.MshrHits | Event |
| 11 | 94 | dcache.MshrMisses | Event |
| 12 | 95 | dcache.tags.totalRefs | Event |
| 13 | 96 | dcache.tagAccesses | Event |

### Others.npy (200, 101)

Identical to `feature.npy` — contains all 101 features (verified via `np.allclose`).

## Notes

- Mapping verified by cross-referencing gem5 raw statistics files (`dataset/statistics/`) against `feature.npy` values using multiple samples (`boom0_dhrystone`, `boom0_spmv`, `boom1_dhrystone`, `boom1_qsort`).
- Component feature files may have minor numerical differences from `feature.npy` due to independent normalization during generation.
- Indices 34, 60, 61, 63, 64 are all zeros across all 200 samples and are excluded from the Others component mask.
- Indices 42, 46, 50, 51 are duplicates of other features (29, 41, 21, 22 respectively) and are also excluded from the Others component mask.
